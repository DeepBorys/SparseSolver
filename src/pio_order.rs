use crate::{card_from_str, Card};
#[cfg(all(feature = "neural", feature = "rayon"))]
use rayon::prelude::*;
#[cfg(feature = "neural")]
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(feature = "neural")]
use std::time::Instant;

#[cfg(feature = "neural")]
struct NnWallTimer {
    active: u64,
    started: Option<Instant>,
    nanos: u64,
}

#[cfg(feature = "neural")]
impl NnWallTimer {
    const fn new() -> Self {
        Self {
            active: 0,
            started: None,
            nanos: 0,
        }
    }
    fn enter(&mut self) {
        if self.active == 0 {
            self.started = Some(Instant::now());
        }
        self.active += 1;
    }
    fn exit(&mut self) {
        assert!(self.active > 0);
        self.active -= 1;
        if self.active == 0 {
            self.nanos += self.started.take().unwrap().elapsed().as_nanos() as u64;
        }
    }
}

static NN_ZERO_MODE: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "neural")]
static NN_USE_CUDA: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "neural")]
static NN_SPARSE_CPU: AtomicBool = AtomicBool::new(true);
#[cfg(feature = "neural")]
static NN_DTYPE: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "neural")]
static NN_EXPLICIT_SYNC: AtomicBool = AtomicBool::new(true);
#[cfg(feature = "neural")]
static NN_NO_GRAD: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "neural")]
static NN_GLUED_RIVERS: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "neural")]
static NN_GROUPED_RIVERS: AtomicBool = AtomicBool::new(false);
static NN_CALLS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static NN_NANOS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static NN_FORWARD_NANOS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static NN_BATCHES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static NN_BATCH_REQUESTS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
#[cfg(feature = "neural")]
static NN_COMPACT_CALLS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
#[cfg(feature="neural")]
static NN_CHILD_PREP_CALLS:[AtomicU64;2]=[AtomicU64::new(0),AtomicU64::new(0)];
#[cfg(feature="neural")]
static NN_CHILD_PREP_NANOS:[AtomicU64;2]=[AtomicU64::new(0),AtomicU64::new(0)];
#[cfg(feature = "neural")]
static NN_CALL_WALL: std::sync::Mutex<NnWallTimer> = std::sync::Mutex::new(NnWallTimer::new());
#[cfg(feature = "neural")]
static NN_FORWARD_WALL: std::sync::Mutex<NnWallTimer> = std::sync::Mutex::new(NnWallTimer::new());

#[cfg(feature = "neural")]
fn nn_wall_timers_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = std::env::var("PFS_NN_WALL_TIMERS_OFF").as_deref() != Ok("1");
        if !enabled { println!("NN_WALL_TIMING disabled=true experiment=1 wall_metrics_unavailable=true"); }
        enabled
    })
}

#[cfg(feature = "neural")]
fn nn_wall_enter(timer: &std::sync::Mutex<NnWallTimer>) {
    if nn_wall_timers_enabled() { timer.lock().unwrap().enter(); }
}

#[cfg(feature = "neural")]
fn nn_wall_exit(timer: &std::sync::Mutex<NnWallTimer>) {
    if nn_wall_timers_enabled() { timer.lock().unwrap().exit(); }
}

#[cfg(feature = "neural")]
fn compact_pipeline_experiment() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| crate::sparse_nn::cpu_defaults::flag("PFS_NN_COMPACT_PIPELINE"))
}

pub fn set_nn_zero_mode(enabled: bool) {
    NN_ZERO_MODE.store(enabled, Ordering::Relaxed);
}

pub(crate) fn nn_zero_mode() -> bool {
    NN_ZERO_MODE.load(Ordering::Relaxed)
}
#[cfg(feature = "neural")]
pub fn set_nn_cuda(enabled: bool) {
    assert!(
        !enabled || tch::Cuda::is_available(),
        "CUDA neural inference requested, but CUDA is unavailable"
    );
    NN_USE_CUDA.store(enabled, Ordering::Relaxed);
    NN_SPARSE_CPU.store(false, Ordering::Relaxed);
}

#[cfg(feature = "neural")]
pub fn set_nn_sparse_cpu(enabled: bool) {
    NN_SPARSE_CPU.store(enabled, Ordering::Relaxed);
    if enabled {
        NN_USE_CUDA.store(false, Ordering::Relaxed);
    }
}

#[cfg(feature = "neural")]
pub(crate) fn nn_sparse_cpu() -> bool {
    NN_SPARSE_CPU.load(Ordering::Relaxed)
}

#[cfg(feature = "neural")]
fn nn_sparse_bf16() -> bool {
    match NN_DTYPE.load(Ordering::Relaxed) {
        0 => false,
        2 => true,
        _ => panic!("sparse CPU inference supports fp32 or bf16, not fp16"),
    }
}

#[cfg(feature = "neural")]
pub fn set_nn_dtype(dtype: &str) {
    let value = match dtype {
        "fp32" => 0,
        "fp16" => 1,
        "bf16" => 2,
        _ => panic!("NN dtype must be fp32, fp16, or bf16"),
    };
    NN_DTYPE.store(value, Ordering::Relaxed);
}

#[cfg(feature = "neural")]
pub fn set_nn_explicit_sync(enabled: bool) {
    NN_EXPLICIT_SYNC.store(enabled, Ordering::Relaxed);
}

#[cfg(feature = "neural")]
pub fn set_nn_no_grad(enabled: bool) {
    NN_NO_GRAD.store(enabled, Ordering::Relaxed);
}

#[cfg(feature = "neural")]
pub fn set_nn_glued_rivers(enabled: bool) {
    NN_GLUED_RIVERS.store(enabled, Ordering::Relaxed);
}

#[cfg(feature = "neural")]
pub fn set_nn_grouped_rivers(enabled: bool) {
    NN_GROUPED_RIVERS.store(enabled, Ordering::Relaxed);
    if enabled {
        NN_GLUED_RIVERS.store(true, Ordering::Relaxed);
    }
}

#[cfg(feature = "neural")]
pub(crate) fn nn_grouped_rivers() -> bool {
    NN_GROUPED_RIVERS.load(Ordering::Relaxed)
}

#[cfg(feature = "neural")]
pub(crate) fn nn_glued_rivers() -> bool {
    NN_GLUED_RIVERS.load(Ordering::Relaxed)
}

#[cfg(feature = "neural")]
fn forward_nn(module: &tch::CModule, inputs: &[tch::Tensor]) -> Result<tch::Tensor, tch::TchError> {
    if NN_NO_GRAD.load(Ordering::Relaxed) {
        tch::no_grad(|| module.forward_ts(inputs))
    } else {
        module.forward_ts(inputs)
    }
}

#[cfg(feature = "neural")]
pub(crate) fn nn_kind() -> tch::Kind {
    match NN_DTYPE.load(Ordering::Relaxed) {
        0 => tch::Kind::Float,
        1 => tch::Kind::Half,
        2 => tch::Kind::BFloat16,
        _ => unreachable!(),
    }
}

#[cfg(feature = "neural")]
pub(crate) fn load_nn_module(
    path: impl AsRef<std::path::Path>,
    device: tch::Device,
) -> Result<tch::CModule, tch::TchError> {
    let mut module = tch::CModule::load_on_device(path, device)?;
    module.to(device, nn_kind(), false);
    module.set_eval();
    Ok(module)
}

#[cfg(feature = "neural")]
pub(crate) fn nn_device() -> tch::Device {
    if NN_USE_CUDA.load(Ordering::Relaxed) {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    }
}
pub fn reset_nn_metrics() {
    for i in 0..2 {
        NN_CALLS[i].store(0, Ordering::Relaxed);
        NN_NANOS[i].store(0, Ordering::Relaxed);
        NN_FORWARD_NANOS[i].store(0, Ordering::Relaxed);
        NN_BATCHES[i].store(0, Ordering::Relaxed);
        NN_BATCH_REQUESTS[i].store(0, Ordering::Relaxed);
        #[cfg(feature = "neural")]
        NN_COMPACT_CALLS[i].store(0, Ordering::Relaxed);
        #[cfg(feature="neural")]
        {NN_CHILD_PREP_CALLS[i].store(0,Ordering::Relaxed);NN_CHILD_PREP_NANOS[i].store(0,Ordering::Relaxed);}
    }
    #[cfg(feature = "neural")]
    for timer in [&NN_CALL_WALL, &NN_FORWARD_WALL] {
        let mut timer = timer.lock().unwrap();
        assert_eq!(timer.active, 0, "cannot reset NN metrics during inference");
        timer.started = None;
        timer.nanos = 0;
    }
}
/// Per player: `(calls, total call nanoseconds, TorchScript forward nanoseconds)`.
pub fn nn_metrics() -> [(u64, u64, u64); 2] {
    #[cfg(feature="neural")]
    crate::sparse_nn::buffer_pool::report();
    #[cfg(feature="neural")]
    if NN_CHILD_PREP_CALLS[0].load(Ordering::Relaxed)>0 {
        println!("NN_CHILD_PREP blocks_oop={} blocks_ip={} prep_work_ns_oop={} prep_work_ns_ip={}",
            NN_CHILD_PREP_CALLS[0].load(Ordering::Relaxed),NN_CHILD_PREP_CALLS[1].load(Ordering::Relaxed),
            NN_CHILD_PREP_NANOS[0].load(Ordering::Relaxed),NN_CHILD_PREP_NANOS[1].load(Ordering::Relaxed));
    }
    #[cfg(feature = "neural")]
    if compact_pipeline_experiment() {
        println!("NN_COMPACT_PIPELINE calls_oop={} calls_ip={}",
            NN_COMPACT_CALLS[0].load(Ordering::Relaxed), NN_COMPACT_CALLS[1].load(Ordering::Relaxed));
    }
    [0, 1].map(|i| {
        (
            NN_CALLS[i].load(Ordering::Relaxed),
            NN_NANOS[i].load(Ordering::Relaxed),
            NN_FORWARD_NANOS[i].load(Ordering::Relaxed),
        )
    })
}

pub fn nn_batch_metrics() -> [(u64, u64); 2] {
    [0, 1].map(|i| {
        (
            NN_BATCHES[i].load(Ordering::Relaxed),
            NN_BATCH_REQUESTS[i].load(Ordering::Relaxed),
        )
    })
}

#[cfg(feature = "neural")]
pub fn nn_wall_metrics() -> (u64, u64) {
    (
        NN_CALL_WALL.lock().unwrap().nanos,
        NN_FORWARD_WALL.lock().unwrap().nanos,
    )
}

pub type CanonicalHand = (Card, Card);

fn canonical_hand(hand: (Card, Card)) -> CanonicalHand {
    if hand.0 < hand.1 {
        (hand.0, hand.1)
    } else {
        (hand.1, hand.0)
    }
}

fn parse_hand(text: &str) -> Result<CanonicalHand, String> {
    if text.len() != 4 {
        return Err(format!("invalid hand {text:?}"));
    }
    if !text.is_ascii() {
        return Err(format!("invalid hand {text:?}"));
    }
    let hand = (card_from_str(&text[..2])?, card_from_str(&text[2..])?);
    if hand.0 == hand.1 {
        return Err(format!("duplicate card in hand {text:?}"));
    }
    Ok(canonical_hand(hand))
}

pub fn load_pio_hand_order(
    path: impl AsRef<std::path::Path>,
) -> Result<Vec<CanonicalHand>, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let tokens: Vec<_> = text.split_whitespace().collect();
    if tokens.len() != 1326 {
        return Err(format!("expected 1326 hands, got {}", tokens.len()));
    }
    let mut result = Vec::with_capacity(1326);
    let mut seen = HashMap::with_capacity(1326);
    for token in tokens {
        let hand = parse_hand(token)?;
        if seen.insert(hand, result.len()).is_some() {
            return Err(format!("duplicate canonical hand {token}"));
        }
        result.push(hand);
    }
    Ok(result)
}

pub fn local_to_pio(
    pio_order: &[CanonicalHand],
    local_hands: &[(Card, Card)],
) -> Result<Vec<usize>, String> {
    if pio_order.len() != 1326 {
        return Err(format!(
            "expected 1326 canonical hands, got {}",
            pio_order.len()
        ));
    }
    let mut indices = HashMap::with_capacity(pio_order.len());
    for (index, &hand) in pio_order.iter().enumerate() {
        if indices.insert(hand, index).is_some() {
            return Err("duplicate canonical hand".into());
        }
    }
    let result: Vec<_> = local_hands
        .iter()
        .map(|&hand| {
            indices
                .get(&canonical_hand(hand))
                .copied()
                .ok_or_else(|| format!("local hand not in Pio order: {hand:?}"))
        })
        .collect::<Result<_, _>>()?;
    let mut unique = result.clone();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != result.len() {
        return Err("duplicate local-to-Pio mapping".into());
    }
    for (&local, &pio_index) in local_hands.iter().zip(&result) {
        if canonical_hand(local) != pio_order[pio_index] {
            return Err(format!(
                "mapping round-trip failed at local index {pio_index}"
            ));
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn export_preserves_hand_identity_and_blocked_zeros() {
        let output = export_pio_rows(&[3.0, 7.0, 0.0, -2.0], &[1000, 2]).unwrap();
        assert_eq!(output.len(), 2 * 1326);
        assert_eq!(output[1000], 3.0);
        assert_eq!(output[2], 7.0);
        assert_eq!(output[1326 + 2], -2.0);
        assert_eq!(output[0], 0.0);
        assert!(export_pio_rows(&[1.0, 2.0], &[2, 2]).is_err());
        assert!(parse_hand("AcAc").is_err());
    }

    #[test]
    fn local_weight_conversion_accepts_twelve_rows() {
        let canonical = [
            vec![1.0; 12 * WEIGHT_CANONICAL_HAND_COUNT],
            vec![2.0; 12 * WEIGHT_CANONICAL_HAND_COUNT],
        ];
        let local = convert_weights_to_local(&canonical, [&[0, 1325], &[7]]).unwrap();
        assert_eq!(local[0].len(), 24);
        assert_eq!(local[1].len(), 12);
        assert!(local[0].iter().all(|&value| value == 1.0));
        assert!(local[1].iter().all(|&value| value == 2.0));
    }

    #[test]
    fn bucket_model_combines_other_hands() {
        let conversion = TurnCfvConversion {
            node_id: 0,
            action_count: 49,
            use_in_solving: true,
            all_turn_nodes: false,
            basis_actions: vec![0],
            weights: [vec![0.0; 2], vec![0.0; 2]],
            bucket_model: Some(LocalBucketWeights {
                bucket_sizes: vec![2.0],
                assignments: [vec![0, 0], vec![0, 0]],
                weights: [vec![2.0, 3.0], vec![2.0, 3.0]],
            }),
            neural_model: None,
        };
        let mut cfvs = vec![0.0; 49 * 2];
        cfvs[0] = 4.0;
        cfvs[1] = 6.0;
        assert_eq!(conversion.estimate_sum(0, &cfvs), vec![10.0, 15.0]);
    }
    #[test]
    fn pio_order_is_valid_and_round_trips() {
        let order = load_pio_hand_order("piosolver_hand_order.txt").unwrap();
        assert_eq!(order.len(), 1326);
        let mapping = local_to_pio(&order, &order).unwrap();
        assert_eq!(mapping, (0..1326).collect::<Vec<_>>());
    }

    #[test]
    fn canonical_range_features_preserve_player_and_hand_alignment() {
        let mappings = [vec![5, 100], vec![7]];
        let weights = [vec![0.25, 0.75], vec![0.5]];
        let result = canonical_range_features(&mappings, &weights).unwrap();
        assert_eq!(result[2 * 5], 0.25);
        assert_eq!(result[2 * 100], 0.75);
        assert_eq!(result[2 * 7 + 1], 0.5);
        assert_eq!(result[0], 0.0);
        assert!(canonical_range_features(&[vec![1], vec![]], &[vec![], vec![]]).is_err());
        assert!(canonical_range_features(&[vec![1326], vec![]], &[vec![1.0], vec![]]).is_err());
    }
}

pub const WEIGHT_BASIS_COUNT: usize = 36;
pub const WEIGHT_CANONICAL_HAND_COUNT: usize = 1326;

#[cfg_attr(not(feature = "neural"), allow(dead_code))]
pub(crate) fn canonical_range_features(
    mappings: &[Vec<usize>; 2],
    weights: &[Vec<f32>; 2],
) -> Result<Vec<f32>, String> {
    let mut result = vec![0.0f32; WEIGHT_CANONICAL_HAND_COUNT * 2];
    for player in 0..2 {
        if mappings[player].len() != weights[player].len() {
            return Err("Private-hand mapping and initial range lengths differ".into());
        }
        for (&pio, &weight) in mappings[player].iter().zip(&weights[player]) {
            if pio >= WEIGHT_CANONICAL_HAND_COUNT {
                return Err("Private-hand mapping exceeds canonical hand count".into());
            }
            if !weight.is_finite() || weight < 0.0 {
                return Err("Initial range contains an invalid weight".into());
            }
            result[2 * pio + player] = weight;
        }
    }
    Ok(result)
}

pub fn load_canonical_weights(path: impl AsRef<std::path::Path>) -> Result<[Vec<f32>; 2], String> {
    load_canonical_weights_with_basis(path, WEIGHT_BASIS_COUNT)
}

pub fn load_canonical_weights_with_basis(
    path: impl AsRef<std::path::Path>,
    basis_count: usize,
) -> Result<[Vec<f32>; 2], String> {
    if basis_count == 0 {
        return Err("weight basis cannot be empty".into());
    }
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let per_player = basis_count * WEIGHT_CANONICAL_HAND_COUNT;
    let expected = 2 * per_player * std::mem::size_of::<f32>();
    if bytes.len() != expected {
        return Err(format!(
            "expected {expected} weight bytes, got {}",
            bytes.len()
        ));
    }
    let mut result = [
        Vec::with_capacity(per_player),
        Vec::with_capacity(per_player),
    ];
    for player in 0..2 {
        for chunk in bytes[player * per_player * 4..(player + 1) * per_player * 4].chunks_exact(4) {
            result[player].push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
    }
    Ok(result)
}

pub fn convert_weights_to_local(
    canonical: &[Vec<f32>; 2],
    mappings: [&[usize]; 2],
) -> Result<[Vec<f32>; 2], String> {
    let mut local = [Vec::new(), Vec::new()];
    if canonical[0].len() != canonical[1].len()
        || canonical[0].len() % WEIGHT_CANONICAL_HAND_COUNT != 0
    {
        return Err("invalid canonical matrix size".into());
    }
    let basis_count = canonical[0].len() / WEIGHT_CANONICAL_HAND_COUNT;
    for player in 0..2 {
        local[player].reserve(basis_count * mappings[player].len());
        for basis in 0..basis_count {
            for &pio in mappings[player] {
                if pio >= WEIGHT_CANONICAL_HAND_COUNT {
                    return Err("Pio index out of bounds".into());
                }
                local[player].push(canonical[player][basis * WEIGHT_CANONICAL_HAND_COUNT + pio]);
            }
        }
    }
    Ok(local)
}

pub struct CanonicalBucketWeights {
    pub bucket_sizes: Vec<f32>,
    pub assignments: Vec<u8>,
    pub self_weights: [Vec<f32>; 2],
    pub bucket_weights: [Vec<f32>; 2],
}

pub fn load_canonical_bucket_weights(
    assignments_path: impl AsRef<std::path::Path>,
    weights_path: impl AsRef<std::path::Path>,
    basis_count: usize,
    bucket_count: usize,
) -> Result<CanonicalBucketWeights, String> {
    if basis_count == 0 || bucket_count == 0 || bucket_count > u8::MAX as usize + 1 {
        return Err("invalid bucket model dimensions".into());
    }
    let assignments = fs::read(assignments_path).map_err(|e| e.to_string())?;
    if assignments.len() != WEIGHT_CANONICAL_HAND_COUNT
        || assignments
            .iter()
            .any(|&bucket| bucket as usize >= bucket_count)
    {
        return Err("invalid canonical bucket assignments".into());
    }
    let bytes = fs::read(weights_path).map_err(|e| e.to_string())?;
    if bytes.len() % 4 != 0 {
        return Err("bucket weight file is not f32 aligned".into());
    }
    let values: Vec<_> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let self_count = basis_count * WEIGHT_CANONICAL_HAND_COUNT;
    let bucket_weight_count = basis_count * bucket_count * WEIGHT_CANONICAL_HAND_COUNT;
    let expected = bucket_count + 2 * (self_count + bucket_weight_count);
    if values.len() != expected || values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "expected {expected} finite bucket weight values, got {}",
            values.len()
        ));
    }
    let bucket_sizes = values[..bucket_count].to_vec();
    let actual_sizes: Vec<_> = (0..bucket_count)
        .map(|bucket| {
            assignments
                .iter()
                .filter(|&&value| value as usize == bucket)
                .count() as f32
        })
        .collect();
    if bucket_sizes != actual_sizes || bucket_sizes.iter().any(|&size| size <= 0.0) {
        return Err("bucket sizes do not match assignments".into());
    }
    let mut offset = bucket_count;
    let mut self_weights = [Vec::new(), Vec::new()];
    let mut bucket_weights = [Vec::new(), Vec::new()];
    for player in 0..2 {
        self_weights[player] = values[offset..offset + self_count].to_vec();
        offset += self_count;
        bucket_weights[player] = values[offset..offset + bucket_weight_count].to_vec();
        offset += bucket_weight_count;
    }
    Ok(CanonicalBucketWeights {
        bucket_sizes,
        assignments,
        self_weights,
        bucket_weights,
    })
}

/// Export physical turn rows in the full Pio column order. Absent hands are zero.
pub fn export_pio_rows(values: &[f32], mapping: &[usize]) -> Result<Vec<f32>, String> {
    if mapping.is_empty() || values.len() % mapping.len() != 0 {
        return Err("invalid local CFV matrix size".into());
    }
    let mut seen = vec![false; 1326];
    for &index in mapping {
        if index >= 1326 || seen[index] {
            return Err("invalid local-to-Pio mapping".into());
        }
        seen[index] = true;
    }
    let mut output = vec![0.0; values.len() / mapping.len() * 1326];
    for (row, local) in values.chunks_exact(mapping.len()).enumerate() {
        for (&value, &index) in local.iter().zip(mapping) {
            output[row * 1326 + index] = value;
        }
    }
    Ok(output)
}

/// Runtime conversion for one physical-turn chance node. Columns use solver-local hands.
/// Installed explicitly on a game; it is not included in saved solver files.
#[derive(Debug)]
pub struct TurnCfvConversion {
    pub(crate) node_id: u64,
    pub(crate) action_count: usize,
    pub(crate) use_in_solving: bool,
    pub(crate) all_turn_nodes: bool,
    pub(crate) basis_actions: Vec<usize>,
    pub(crate) weights: [Vec<f32>; 2],
    pub(crate) bucket_model: Option<LocalBucketWeights>,
    #[cfg_attr(not(feature = "neural"), allow(dead_code))]
    pub(crate) neural_model: Option<NeuralTurnCfvModel>,
}

#[cfg(feature = "neural")]
pub(crate) struct NeuralTurnCfvModel {
    pub(crate) modules: Option<[std::sync::Mutex<tch::CModule>; 2]>,
    pub(crate) sparse_models: Option<[crate::sparse_nn::SparseCpuModel; 2]>,
    pub(crate) contexts: HashMap<u64, Vec<f32>>,
    pub(crate) direct_contexts: Option<DirectNnContexts>,
    pub(crate) mappings: [Vec<usize>; 2],
    /// Canonical `[hand, player]` raw initial range weights. `None` preserves
    /// compatibility with models whose forward signature is `(cfv, context)`.
    pub(crate) range_features: Option<Vec<f32>>,
    /// The module consumes current canonical OOP/IP reaches as its third input.
    pub(crate) current_reaches: bool,
    /// Sum sampled CFV rows on the CPU before tensor creation/device transfer.
    pub(crate) sum_cfvs: bool,
    pub(crate) iteration_scale: Option<f32>,
    /// If set, NN outputs are in DeepStack's pot/opponent-reach normalized units.
    pub(crate) deepstack_root_combinations: Option<f64>,
    /// Number of legal chance outcomes per private hand used by DeepStack
    /// normalization: 45 turns from the flop or 44 rivers from the turn.
    pub(crate) deepstack_chance_count: f64,
    pub(crate) pots: HashMap<u64, f32>,
    pub(crate) device: tch::Device,
    pub(crate) batch_state: std::sync::Mutex<NeuralBatchState>,
}

/// Optional bounded direct-address table built before solving. The source map
/// remains authoritative for legacy/Torch paths and diagnostics.
#[cfg(feature = "neural")]
pub(crate) struct DirectNnContexts {
    first: u64,
    width: usize,
    present: Vec<bool>,
    values: Vec<f32>,
}
#[cfg(feature="neural")]
impl NeuralTurnCfvModel{
    pub(crate) fn diagnostic_extra_vector_bytes(&self)->u64{
        let direct=self.direct_contexts.as_ref().map_or(0,|x|x.values.capacity()*4+x.present.capacity());
        let batch=self.batch_state.lock().unwrap();assert!(batch.pending.is_empty()&&batch.predictions.is_empty());direct as u64
    }
}
#[cfg(feature="neural")]
pub fn diagnostic_nn_worker_scratch_bytes()->u64{crate::sparse_nn::diagnostic_worker_scratch_bytes()}


#[cfg(feature = "neural")]
#[derive(Default)]
pub(crate) struct NeuralBatchState {
    collecting: bool,
    consuming: bool,
    pending: Vec<NeuralBatchRequest>,
    predictions: HashMap<(u64, usize, u32), Vec<f64>>,
}

#[cfg(feature = "neural")]
struct NeuralBatchRequest {
    node_id: u64,
    player: usize,
    outer_iteration: u32,
    canonical: Vec<f32>,
    compact: Vec<f32>,
    context: Vec<f32>,
    reaches: Vec<f32>,
    deferred_sparse_input: Option<DeferredSparseBatchInput>,
}

#[cfg(feature = "neural")]
#[derive(PartialEq)]
struct DeferredSparseBatchInput {
    basis_actions: Vec<usize>,
    action_count: usize,
    compact_input: bool,
    opponent_reach: Vec<f32>,
    player_reach: Option<Vec<f32>>,
}

#[cfg(feature = "neural")]
fn batch_adjacent_experiment() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("PFS_NN_BATCH_ADJACENT_EXPERIMENT").as_deref() == Ok("1")
    })
}

#[cfg(feature = "neural")]
fn deduplicate_batch_requests(pending: Vec<NeuralBatchRequest>) -> Vec<NeuralBatchRequest> {
    let mut positions: HashMap<(u64, usize, u32), usize> = HashMap::new();
    let mut unique: Vec<NeuralBatchRequest> = Vec::with_capacity(pending.len());
    for request in pending {
        let key = (request.node_id, request.player, request.outer_iteration);
        if let Some(&position) = positions.get(&key) {
            let previous = &unique[position];
            assert_eq!(
                request.canonical, previous.canonical,
                "duplicate neural request has different CFVs for node {}",
                request.node_id
            );
            assert_eq!(request.compact, previous.compact);
            assert!(request.deferred_sparse_input == previous.deferred_sparse_input);
            assert_eq!(
                request.context, previous.context,
                "duplicate neural request has different context for node {}",
                request.node_id
            );
            assert_eq!(
                request.reaches, previous.reaches,
                "duplicate neural request has different reaches for node {}",
                request.node_id
            );
        } else {
            positions.insert(key, unique.len());
            unique.push(request);
        }
    }
    unique
}

#[cfg(not(feature = "neural"))]
#[derive(Debug)]
pub(crate) struct NeuralTurnCfvModel;

#[cfg(feature = "neural")]
impl std::fmt::Debug for NeuralTurnCfvModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuralTurnCfvModel")
            .field("node_ids", &self.contexts.keys())
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub(crate) struct LocalBucketWeights {
    pub(crate) bucket_sizes: Vec<f32>,
    pub(crate) assignments: [Vec<u8>; 2],
    pub(crate) weights: [Vec<f32>; 2],
}

impl TurnCfvConversion {
    pub fn use_in_solving(&self) -> bool {
        self.use_in_solving
    }

    pub fn basis_actions(&self) -> &[usize] {
        &self.basis_actions
    }

    pub(crate) fn uses_current_reaches(&self) -> bool {
        #[cfg(feature = "neural")]
        {
            return self
                .neural_model
                .as_ref()
                .is_some_and(|model| model.current_reaches);
        }
        #[cfg(not(feature = "neural"))]
        false
    }

    #[cfg(feature="neural")]
    pub(crate) fn child_input_model(&self, player:usize) -> Option<&NeuralTurnCfvModel> {
        let m=self.neural_model.as_ref()?;
        if !compact_pipeline_experiment() || NN_ZERO_MODE.load(Ordering::Relaxed)
            || !m.current_reaches || m.sum_cfvs || m.deepstack_root_combinations.is_some() || nn_sparse_bf16() {
            return None;
        }
        let state=m.batch_state.lock().unwrap();
        if state.collecting || state.consuming {return None;}
        let models=m.sparse_models.as_ref()?;
        models[player].supports_child_inputs().then_some(m)
    }

    #[cfg(feature="neural")]
    pub(crate) fn prepare_child_input(&self, player:usize, basis:usize, raw:Vec<f32>) -> crate::sparse_nn::PreparedBasisInput {
        let started=Instant::now();
        nn_wall_enter(&NN_CALL_WALL);
        let m=self.neural_model.as_ref().unwrap();
        let result=m.sparse_models.as_ref().unwrap()[player].prepare_basis_input(basis,raw);
        NN_CHILD_PREP_CALLS[player].fetch_add(1,Ordering::Relaxed);
        let elapsed=started.elapsed().as_nanos() as u64;
        NN_CHILD_PREP_NANOS[player].fetch_add(elapsed,Ordering::Relaxed);
        NN_NANOS[player].fetch_add(elapsed,Ordering::Relaxed);
        nn_wall_exit(&NN_CALL_WALL);
        result
    }

    #[cfg(feature="neural")]
    pub(crate) fn prepare_common_input(&self,node_id:u64,player:usize,iteration:u32,
        opponent:&[f32],own:&[f32]) -> crate::sparse_nn::PreparedCommonInput {
        let started=Instant::now();nn_wall_enter(&NN_CALL_WALL);
        let m=self.neural_model.as_ref().unwrap();
        let saved=m.compact_context(node_id);
        let context=if let Some(scale)=m.iteration_scale {
            let mut c=saved.to_vec();c.push((iteration as f32).min(scale)/scale);Cow::Owned(c)
        } else {Cow::Borrowed(saved)};
        // These are the parent chance node's same-pass reaches, before chance scaling.
        let reaches=if player==0 {[own,opponent]} else {[opponent,own]};
        let result=m.sparse_models.as_ref().unwrap()[player].prepare_common_input(&context,reaches);
        let elapsed=started.elapsed().as_nanos() as u64;
        NN_CHILD_PREP_NANOS[player].fetch_add(elapsed,Ordering::Relaxed);
        NN_NANOS[player].fetch_add(elapsed,Ordering::Relaxed);
        nn_wall_exit(&NN_CALL_WALL);result
    }

    #[cfg(feature="neural")]
    pub(crate) fn estimate_child_inputs(&self,player:usize,blocks:&[crate::sparse_nn::PreparedBasisInput],
        common:crate::sparse_nn::PreparedCommonInput) -> Vec<f64> {
        let started=Instant::now();nn_wall_enter(&NN_CALL_WALL);nn_wall_enter(&NN_FORWARD_WALL);
        NN_CALLS[player].fetch_add(1,Ordering::Relaxed);NN_COMPACT_CALLS[player].fetch_add(1,Ordering::Relaxed);
        let m=self.neural_model.as_ref().unwrap();
        let result=m.sparse_models.as_ref().unwrap()[player].infer_child_inputs(blocks,common,m.mappings[player].len());
        let elapsed=started.elapsed().as_nanos() as u64;
        NN_NANOS[player].fetch_add(elapsed,Ordering::Relaxed);
        NN_FORWARD_NANOS[player].fetch_add(elapsed,Ordering::Relaxed);
        nn_wall_exit(&NN_FORWARD_WALL);nn_wall_exit(&NN_CALL_WALL);result
    }

    pub fn weights(&self, player: usize) -> &[f32] {
        &self.weights[player]
    }

    pub fn bucket_count(&self) -> usize {
        self.bucket_model
            .as_ref()
            .map_or(0, |model| model.bucket_sizes.len())
    }

    #[cfg(feature = "neural")]
    pub(crate) fn begin_neural_batch_collection(&self) -> bool {
        let Some(model) = &self.neural_model else {
            return false;
        };
        let mut state = model.batch_state.lock().unwrap();
        assert!(!state.collecting && !state.consuming);
        state.pending.clear();
        state.predictions.clear();
        state.collecting = true;
        true
    }

    #[cfg(feature = "neural")]
    pub(crate) fn neural_batch_is_collecting(&self) -> bool {
        self.neural_model
            .as_ref()
            .is_some_and(|model| model.batch_state.lock().unwrap().collecting)
    }

    #[cfg(not(feature = "neural"))]
    pub(crate) fn neural_batch_is_collecting(&self) -> bool {
        false
    }

    #[cfg(feature = "neural")]
    pub(crate) fn neural_batch_is_consuming(&self) -> bool {
        self.neural_model
            .as_ref()
            .is_some_and(|model| model.batch_state.lock().unwrap().consuming)
    }

    #[cfg(feature = "neural")]
    pub(crate) fn finish_neural_batch(&self, player: usize) {
        self.neural_model
            .as_ref()
            .expect("neural batch without model")
            .finish_batch(player);
    }

    #[cfg(feature = "neural")]
    pub(crate) fn end_neural_batch(&self) {
        let model = self
            .neural_model
            .as_ref()
            .expect("neural batch without model");
        let mut state = model.batch_state.lock().unwrap();
        assert!(state.consuming);
        state.consuming = false;
        state.predictions.clear();
    }

    #[cfg(feature = "neural")]
    pub(crate) fn cached_neural_batch_prediction(
        &self,
        node_id: u64,
        player: usize,
        outer_iteration: u32,
    ) -> Option<Vec<f64>> {
        let model = self.neural_model.as_ref()?;
        let state = model.batch_state.lock().unwrap();
        if !state.consuming {
            return None;
        }
        Some(
            state
                .predictions
                .get(&(node_id, player, outer_iteration))
                .unwrap_or_else(|| panic!("missing batched prediction for node {node_id}"))
                .clone(),
        )
    }

    /// Estimate the already chance-scaled 49-turn sum, using f64 accumulation.
    pub fn estimate_sum(&self, player: usize, cfv_actions: &[f32]) -> Vec<f64> {
        self.estimate_sum_at(self.node_id, player, cfv_actions)
    }

    pub(crate) fn estimate_sum_at(
        &self,
        node_id: u64,
        player: usize,
        cfv_actions: &[f32],
    ) -> Vec<f64> {
        self.estimate_sum_at_iteration(node_id, player, cfv_actions, 0, &[], None)
    }

    pub(crate) fn estimate_sum_at_iteration(
        &self,
        node_id: u64,
        player: usize,
        cfv_actions: &[f32],
        outer_iteration: u32,
        opponent_reach: &[f32],
        player_reach: Option<&[f32]>,
    ) -> Vec<f64> {
        #[cfg(not(feature = "neural"))]
        let _ = outer_iteration;
        #[cfg(not(feature = "neural"))]
        let _ = node_id;
        #[cfg(feature = "neural")]
        if let Some(model) = &self.neural_model {
            return model.estimate(
                node_id,
                player,
                &self.basis_actions,
                cfv_actions,
                self.action_count,
                outer_iteration,
                opponent_reach,
                player_reach,
                false,
            );
        }
        let basis_count = self.basis_actions.len();
        assert!(basis_count > 0);
        assert_eq!(self.weights[player].len() % basis_count, 0);
        let hands = self.weights[player].len() / basis_count;
        assert_eq!(cfv_actions.len(), self.action_count * hands);
        let mut result = vec![0.0; hands];
        for (basis, &action) in self.basis_actions.iter().enumerate() {
            for (hand, sum) in result.iter_mut().enumerate() {
                *sum += cfv_actions[action * hands + hand] as f64
                    * self.weights[player][basis * hands + hand] as f64;
            }
        }
        if let Some(model) = &self.bucket_model {
            let bucket_count = model.bucket_sizes.len();
            assert_eq!(model.assignments[player].len(), hands);
            assert_eq!(
                model.weights[player].len(),
                basis_count * bucket_count * hands
            );
            let mut bucket_values = vec![0.0f64; basis_count * bucket_count];
            for (basis, &action) in self.basis_actions.iter().enumerate() {
                let row = &cfv_actions[action * hands..(action + 1) * hands];
                for (hand, &value) in row.iter().enumerate() {
                    let bucket = model.assignments[player][hand] as usize;
                    bucket_values[basis * bucket_count + bucket] += value as f64;
                }
            }
            for (feature, value) in bucket_values.iter_mut().enumerate() {
                *value /= model.bucket_sizes[feature % bucket_count] as f64;
                for (hand, sum) in result.iter_mut().enumerate() {
                    *sum += *value * model.weights[player][feature * hands + hand] as f64;
                }
            }
        }
        result
    }

    /// Estimate from a compact matrix whose rows are exactly `basis_actions`.
    /// Unlike `estimate_sum_at_iteration`, this method never interprets rows
    /// using physical action indices.
    pub(crate) fn estimate_sum_from_basis_at_iteration(
        &self,
        node_id: u64,
        player: usize,
        compact_cfv: &[f32],
        basis_actions: &[usize],
        outer_iteration: u32,
        opponent_reach: &[f32],
        player_reach: Option<&[f32]>,
    ) -> Vec<f64> {
        assert_eq!(basis_actions, self.basis_actions.as_slice());
        assert!(basis_actions.windows(2).all(|pair| pair[0] != pair[1]));
        #[cfg(feature = "neural")]
        if let Some(model) = &self.neural_model {
            return model.estimate_compact(
                node_id,
                player,
                compact_cfv,
                basis_actions.len(),
                outer_iteration,
                opponent_reach,
                player_reach,
            );
        }
        panic!("compact basis conversion requires a neural model");
    }
}

#[cfg(feature = "neural")]
pub(crate) fn finish_glued_neural_batches(
    conversions: &[&TurnCfvConversion],
    player: usize,
    module: &std::sync::Mutex<tch::CModule>,
) {
    assert_eq!(conversions.len(), 6);
    let models: Vec<_> = conversions
        .iter()
        .map(|conversion| conversion.neural_model.as_ref().unwrap())
        .collect();
    let pending: Vec<Vec<_>> = models
        .iter()
        .map(|model| {
            let mut state = model.batch_state.lock().unwrap();
            assert!(state.collecting && !state.consuming);
            state.collecting = false;
            deduplicate_batch_requests(std::mem::take(&mut state.pending))
        })
        .collect();
    let batch = pending[0].len();
    assert!(batch > 0 && pending.iter().all(|requests| requests.len() == batch));
    let basis = pending[0][0].canonical.len() / WEIGHT_CANONICAL_HAND_COUNT;
    let context_count = pending[0][0].context.len();
    let mut inputs = Vec::with_capacity(batch * 6 * WEIGHT_CANONICAL_HAND_COUNT * basis);
    let mut contexts = Vec::with_capacity(batch * 6 * context_count);
    let mut reaches = Vec::with_capacity(batch * 6 * WEIGHT_CANONICAL_HAND_COUNT * 2);
    for batch_index in 0..batch {
        for model_index in 0..6 {
            let request = &pending[model_index][batch_index];
            assert_eq!(request.player, player);
            inputs.extend_from_slice(&request.canonical);
            contexts.extend_from_slice(&request.context);
            reaches.extend_from_slice(&request.reaches);
        }
    }
    NN_BATCHES[player].fetch_add(1, Ordering::Relaxed);
    NN_BATCH_REQUESTS[player].fetch_add((batch * 6) as u64, Ordering::Relaxed);
    let input = tch::Tensor::from_slice(&inputs)
        .reshape([
            batch as i64,
            6,
            WEIGHT_CANONICAL_HAND_COUNT as i64,
            basis as i64,
        ])
        .to_device(models[0].device)
        .to_kind(nn_kind());
    let context = tch::Tensor::from_slice(&contexts)
        .reshape([batch as i64, 6, context_count as i64])
        .to_device(models[0].device)
        .to_kind(nn_kind());
    let reach = tch::Tensor::from_slice(&reaches)
        .reshape([batch as i64, 6, WEIGHT_CANONICAL_HAND_COUNT as i64, 2])
        .to_device(models[0].device)
        .to_kind(nn_kind());
    let started = Instant::now();
    nn_wall_enter(&NN_CALL_WALL);
    let forward_started = Instant::now();
    nn_wall_enter(&NN_FORWARD_WALL);
    let output = forward_nn(&module.lock().unwrap(), &[input, context, reach])
        .expect("glued river NN inference failed");
    if NN_EXPLICIT_SYNC.load(Ordering::Relaxed) {
        tch::Cuda::synchronize(0);
    }
    NN_FORWARD_NANOS[player].fetch_add(
        forward_started.elapsed().as_nanos() as u64,
        Ordering::Relaxed,
    );
    nn_wall_exit(&NN_FORWARD_WALL);
    let output: Vec<f32> = Vec::<f32>::try_from(
        output
            .to_kind(tch::Kind::Float)
            .to_device(tch::Device::Cpu)
            .reshape([-1]),
    )
    .expect("invalid glued river output");
    assert!(output.iter().all(|value| value.is_finite()));
    NN_NANOS[player].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    nn_wall_exit(&NN_CALL_WALL);
    for model_index in 0..6 {
        let model = models[model_index];
        let mut state = model.batch_state.lock().unwrap();
        for batch_index in 0..batch {
            let request = &pending[model_index][batch_index];
            let offset = (batch_index * 6 + model_index) * WEIGHT_CANONICAL_HAND_COUNT;
            let row = &output[offset..offset + WEIGHT_CANONICAL_HAND_COUNT];
            let prediction = model.mappings[player]
                .iter()
                .map(|&pio| row[pio] as f64)
                .collect();
            assert!(state
                .predictions
                .insert(
                    (request.node_id, player, request.outer_iteration),
                    prediction
                )
                .is_none());
        }
        state.consuming = true;
    }
}

#[cfg(feature = "neural")]
impl NeuralTurnCfvModel {
    pub(crate) fn build_direct_contexts(contexts: &HashMap<u64, Vec<f32>>) -> Option<DirectNnContexts> {
        if !crate::sparse_nn::cpu_defaults::flag("PFS_NN_LINEAR_LAYOUT") || contexts.is_empty() {
            return None;
        }
        let first = *contexts.keys().min()?;
        let last = *contexts.keys().max()?;
        let width = contexts.values().next()?.len();
        if !contexts.values().all(|v| v.len() == width) { return None; }
        let span = usize::try_from(last.checked_sub(first)?.checked_add(1)?).ok()?;
        // Bound extra setup storage even on unusual, widely separated node IDs.
        if span.checked_mul(width.checked_mul(4)?.checked_add(1)?)? > 32 * 1024 * 1024 { return None; }
        let mut table = DirectNnContexts { first, width, present: vec![false; span], values: vec![0.; span * width] };
        for (&id, context) in contexts {
            let slot = (id-first) as usize;
            table.present[slot] = true;
            table.values[slot*width..(slot+1)*width].copy_from_slice(context);
        }
        Some(table)
    }

    fn compact_context(&self, node_id: u64) -> &[f32] {
        if let Some(table) = &self.direct_contexts {
            let slot = usize::try_from(node_id.checked_sub(table.first).expect("Missing parent NN context")).unwrap();
            assert!(table.present.get(slot).copied().unwrap_or(false), "Missing parent NN context");
            return &table.values[slot*table.width..(slot+1)*table.width];
        }
        self.contexts.get(&node_id).expect("Missing parent NN context")
    }

    fn estimate_compact_pipeline(
        &self, node_id: u64, player: usize, basis_actions: &[usize], values: &[f32],
        action_count: usize, outer_iteration: u32, opponent_reach: &[f32],
        player_reach: Option<&[f32]>, compact_input: bool,
    ) -> Option<Vec<f64>> {
        // The sampled-CFV path is already in native units. DeepStack targets,
        // BF16 and batched replay retain their existing implementation.
        if self.deepstack_root_combinations.is_some() || !self.current_reaches || nn_sparse_bf16() {
            return None;
        }
        let models = self.sparse_models.as_ref()?;
        let own = player_reach?;
        let hands = self.mappings[player].len();
        assert_eq!(values.len(), action_count * hands);
        let basis_count = if self.sum_cfvs { 1 } else if compact_input { action_count } else { basis_actions.len() };
        let compact = if compact_input && !self.sum_cfvs {
            Cow::Borrowed(values)
        } else {
            let mut compact = vec![0.0f32; basis_count * hands];
            for (basis, &action) in basis_actions.iter().enumerate() {
                for local in 0..hands {
                    if self.sum_cfvs { compact[local] += values[action * hands + local]; }
                    else { compact[basis * hands + local] = values[action * hands + local]; }
                }
            }
            Cow::Owned(compact)
        };
        let saved_context = self.compact_context(node_id);
        let context = if let Some(scale) = self.iteration_scale {
            let mut context = saved_context.to_vec();
            context.push((outer_iteration as f32).min(scale) / scale);
            Cow::Owned(context)
        } else { Cow::Borrowed(saved_context) };
        let reaches = if player == 0 { [own, opponent_reach] } else { [opponent_reach, own] };
        let started = Instant::now();
        nn_wall_enter(&NN_FORWARD_WALL);
        let output = models[player].infer_prepared_compact_f64(&compact, &self.mappings[player], &context, reaches);
        NN_FORWARD_NANOS[player].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        nn_wall_exit(&NN_FORWARD_WALL);
        output
    }

    fn output_scale(&self, node_id: u64, player: usize, reaches: &[f32]) -> f64 {
        let Some(root_combinations) = self.deepstack_root_combinations else {
            return 1.0;
        };
        let opponent_sum: f64 = reaches
            .chunks_exact(2)
            .map(|row| row[player ^ 1] as f64)
            .sum();
        let pot = *self
            .pots
            .get(&node_id)
            .expect("NN has no pot for chance node") as f64;
        opponent_sum * pot / (root_combinations * self.deepstack_chance_count)
    }

    fn finish_batch(&self, player: usize) {
        let pending = {
            let mut state = self.batch_state.lock().unwrap();
            assert!(state.collecting && !state.consuming);
            state.collecting = false;
            deduplicate_batch_requests(std::mem::take(&mut state.pending))
        };
        if pending.is_empty() {
            return;
        }
        assert!(pending.iter().all(|request| request.player == player));
        let batch = pending.len();
        NN_BATCHES[player].fetch_add(1, Ordering::Relaxed);
        NN_BATCH_REQUESTS[player].fetch_add(batch as u64, Ordering::Relaxed);
        if NN_ZERO_MODE.load(Ordering::Relaxed) {
            let mut state = self.batch_state.lock().unwrap();
            for request in pending {
                assert!(state
                    .predictions
                    .insert(
                        (request.node_id, request.player, request.outer_iteration),
                        vec![0.0; self.mappings[player].len()],
                    )
                    .is_none());
            }
            state.consuming = true;
            return;
        }
        let basis_count = self.sparse_models.as_ref().map_or_else(
            || pending[0].canonical.len() / WEIGHT_CANONICAL_HAND_COUNT,
            |models| models[player].basis_count(),
        );
        let context_count = pending[0].context.len();
        if self.sparse_models.is_some() {
            assert!(pending.iter().all(|request| {
                request.compact.len() == self.mappings[player].len()
                    * request.deferred_sparse_input.as_ref()
                        .map_or(basis_count, |input| input.action_count)
            }));
        } else {
            assert!(pending.iter().all(|request| {
                request.canonical.len() == WEIGHT_CANONICAL_HAND_COUNT * basis_count
            }));
        }
        assert!(pending
            .iter()
            .all(|request| request.context.len() == context_count));
        let started = Instant::now();
        nn_wall_enter(&NN_CALL_WALL);
        if let Some(models) = &self.sparse_models {
            let forward_started = Instant::now();
            nn_wall_enter(&NN_FORWARD_WALL);
            let infer = |request: &NeuralBatchRequest| {
                let prepared;
                let (compact, context, reaches) = if let Some(input) = &request.deferred_sparse_input {
                    let hands = self.mappings[player].len();
                    let compact = if input.compact_input && !self.sum_cfvs {
                        Cow::Borrowed(request.compact.as_slice())
                    } else {
                        let mut compact = vec![0.0f32; basis_count * hands];
                        for (basis, &action) in input.basis_actions.iter().enumerate() {
                            for local in 0..hands {
                                if self.sum_cfvs {
                                    compact[local] += request.compact[action * hands + local];
                                } else {
                                    compact[basis * hands + local] = request.compact[action * hands + local];
                                }
                            }
                        }
                        Cow::Owned(compact)
                    };
                    let mut context = self.contexts.get(&request.node_id)
                        .expect("NN has no context for chance node").clone();
                    if let Some(scale) = self.iteration_scale {
                        context.push((request.outer_iteration as f32).min(scale) / scale);
                    }
                    let reaches = self.canonical_current_reaches(player, &input.opponent_reach,
                        input.player_reach.as_deref());
                    prepared = (compact, context, reaches);
                    (prepared.0.as_ref(), prepared.1.as_slice(), prepared.2.as_slice())
                } else {
                    (request.compact.as_slice(), request.context.as_slice(), request.reaches.as_slice())
                };
                let output = models[player].infer(
                    compact,
                    &self.mappings[player],
                    context,
                    reaches,
                    nn_sparse_bf16(),
                );
                let scale = if batch_adjacent_experiment() {
                    self.output_scale(request.node_id, player, reaches)
                } else { 1.0 };
                output.into_iter().map(|value| value as f64 * scale).collect::<Vec<_>>()
            };
            #[cfg(feature = "rayon")]
            let outputs: Vec<_> = pending.par_iter().map(infer).collect();
            #[cfg(not(feature = "rayon"))]
            let outputs: Vec<_> = pending.iter().map(infer).collect();
            NN_FORWARD_NANOS[player].fetch_add(
                forward_started.elapsed().as_nanos() as u64,
                Ordering::Relaxed,
            );
            nn_wall_exit(&NN_FORWARD_WALL);
            let mut state = self.batch_state.lock().unwrap();
            for (request, output) in pending.into_iter().zip(outputs) {
                let result = if batch_adjacent_experiment() {
                    output
                } else {
                    let scale = self.output_scale(request.node_id, player, &request.reaches);
                    output.into_iter().map(|value| value * scale).collect()
                };
                assert!(state
                    .predictions
                    .insert(
                        (request.node_id, request.player, request.outer_iteration),
                        result
                    )
                    .is_none());
            }
            state.consuming = true;
            NN_NANOS[player].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            nn_wall_exit(&NN_CALL_WALL);
            return;
        }
        let mut inputs = Vec::with_capacity(batch * WEIGHT_CANONICAL_HAND_COUNT * basis_count);
        let mut contexts = Vec::with_capacity(batch * context_count);
        let mut current_reaches = Vec::with_capacity(batch * WEIGHT_CANONICAL_HAND_COUNT * 2);
        for request in &pending {
            inputs.extend_from_slice(&request.canonical);
            contexts.extend_from_slice(&request.context);
            current_reaches.extend_from_slice(&request.reaches);
        }
        let input = tch::Tensor::from_slice(&inputs)
            .reshape([
                batch as i64,
                WEIGHT_CANONICAL_HAND_COUNT as i64,
                basis_count as i64,
            ])
            .to_device(self.device)
            .to_kind(nn_kind());
        let context = tch::Tensor::from_slice(&contexts)
            .reshape([batch as i64, context_count as i64])
            .to_device(self.device)
            .to_kind(nn_kind());
        let module = self.modules.as_ref().expect("legacy NN module missing")[player]
            .lock()
            .unwrap();
        let forward_started = Instant::now();
        nn_wall_enter(&NN_FORWARD_WALL);
        let output = if self.current_reaches {
            assert_eq!(
                current_reaches.len(),
                batch * WEIGHT_CANONICAL_HAND_COUNT * 2
            );
            let ranges = tch::Tensor::from_slice(&current_reaches)
                .reshape([batch as i64, WEIGHT_CANONICAL_HAND_COUNT as i64, 2])
                .to_device(self.device)
                .to_kind(nn_kind());
            if let Some(range_features) = &self.range_features {
                let initial_ranges = tch::Tensor::from_slice(range_features)
                    .reshape([1, WEIGHT_CANONICAL_HAND_COUNT as i64, 2])
                    .expand([batch as i64, -1, -1], true)
                    .to_device(self.device)
                    .to_kind(nn_kind());
                forward_nn(&module, &[input, context, ranges, initial_ranges])
            } else {
                forward_nn(&module, &[input, context, ranges])
            }
        } else if let Some(range_features) = &self.range_features {
            let ranges = tch::Tensor::from_slice(range_features)
                .reshape([1, WEIGHT_CANONICAL_HAND_COUNT as i64, 2])
                .expand([batch as i64, -1, -1], true)
                .to_device(self.device)
                .to_kind(nn_kind());
            forward_nn(&module, &[input, context, ranges])
        } else {
            forward_nn(&module, &[input, context])
        };
        let output = output
            .expect("batched turn NN inference failed")
            .reshape([batch as i64, WEIGHT_CANONICAL_HAND_COUNT as i64]);
        if NN_EXPLICIT_SYNC.load(Ordering::Relaxed) && matches!(self.device, tch::Device::Cuda(_)) {
            tch::Cuda::synchronize(0);
        }
        let forward_nanos = forward_started.elapsed().as_nanos() as u64;
        NN_FORWARD_NANOS[player].fetch_add(forward_nanos, Ordering::Relaxed);
        nn_wall_exit(&NN_FORWARD_WALL);
        let canonical_output: Vec<f32> = Vec::<f32>::try_from(
            output
                .to_kind(tch::Kind::Float)
                .to_device(tch::Device::Cpu)
                .reshape([-1]),
        )
        .expect("invalid batched turn NN output");
        assert!(
            canonical_output.iter().all(|value| value.is_finite()),
            "batched turn NN produced non-finite output"
        );
        NN_NANOS[player].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        nn_wall_exit(&NN_CALL_WALL);
        let mut state = self.batch_state.lock().unwrap();
        for (batch_index, request) in pending.into_iter().enumerate() {
            let row = &canonical_output[batch_index * WEIGHT_CANONICAL_HAND_COUNT
                ..(batch_index + 1) * WEIGHT_CANONICAL_HAND_COUNT];
            let scale = self.output_scale(request.node_id, player, &request.reaches);
            let result = self.mappings[player]
                .iter()
                .map(|&pio| row[pio] as f64 * scale)
                .collect();
            assert!(state
                .predictions
                .insert(
                    (request.node_id, request.player, request.outer_iteration),
                    result
                )
                .is_none());
        }
        state.consuming = true;
    }

    fn estimate_compact(
        &self,
        node_id: u64,
        player: usize,
        values: &[f32],
        basis_count: usize,
        outer_iteration: u32,
        opponent_reach: &[f32],
        player_reach: Option<&[f32]>,
    ) -> Vec<f64> {
        let hands = self.mappings[player].len();
        assert_eq!(values.len(), basis_count * hands);
        assert!(values.iter().all(|value| value.is_finite()));
        self.estimate(
            node_id,
            player,
            &[],
            values,
            basis_count,
            outer_iteration,
            opponent_reach,
            player_reach,
            true,
        )
    }

    fn canonical_current_reaches(
        &self,
        player: usize,
        opponent_reach: &[f32],
        player_reach: Option<&[f32]>,
    ) -> Vec<f32> {
        if !self.current_reaches {
            return Vec::new();
        }
        let own = player_reach.expect("current-reach river model requires player reach");
        assert_eq!(own.len(), self.mappings[player].len());
        assert_eq!(opponent_reach.len(), self.mappings[player ^ 1].len());
        let mut canonical = vec![0.0; WEIGHT_CANONICAL_HAND_COUNT * 2];
        let own_column = player;
        let opponent_column = player ^ 1;
        for (local, &pio) in self.mappings[player].iter().enumerate() {
            canonical[2 * pio + own_column] = own[local];
        }
        for (local, &pio) in self.mappings[player ^ 1].iter().enumerate() {
            canonical[2 * pio + opponent_column] = opponent_reach[local];
        }
        canonical
    }

    fn estimate(
        &self,
        node_id: u64,
        player: usize,
        basis_actions: &[usize],
        values: &[f32],
        action_count: usize,
        outer_iteration: u32,
        opponent_reach: &[f32],
        player_reach: Option<&[f32]>,
        compact_input: bool,
    ) -> Vec<f64> {
        let started = Instant::now();
        nn_wall_enter(&NN_CALL_WALL);
        NN_CALLS[player].fetch_add(1, Ordering::Relaxed);
        // Opt-in: own the current pass's raw inputs before they can change,
        // then perform all NN-specific packing/scaling on the batch workers.
        if self.sparse_models.is_some() && batch_adjacent_experiment()
            && !NN_ZERO_MODE.load(Ordering::Relaxed)
        {
            let mut state = self.batch_state.lock().unwrap();
            if state.collecting {
                assert_eq!(values.len(), action_count * self.mappings[player].len());
                state.pending.push(NeuralBatchRequest {
                    node_id, player, outer_iteration,
                    canonical: Vec::new(), compact: values.to_vec(),
                    context: Vec::new(), reaches: Vec::new(),
                    deferred_sparse_input: Some(DeferredSparseBatchInput {
                        basis_actions: basis_actions.to_vec(), action_count, compact_input,
                        opponent_reach: opponent_reach.to_vec(),
                        player_reach: player_reach.map(|reach| reach.to_vec()),
                    }),
                });
                nn_wall_exit(&NN_CALL_WALL);
                return vec![0.0; self.mappings[player].len()];
            }
        }
        if compact_pipeline_experiment() && !NN_ZERO_MODE.load(Ordering::Relaxed)
            && !self.batch_state.lock().unwrap().collecting
        {
            if let Some(result) = self.estimate_compact_pipeline(node_id, player, basis_actions, values,
                    action_count, outer_iteration, opponent_reach, player_reach, compact_input) {
                NN_COMPACT_CALLS[player].fetch_add(1, Ordering::Relaxed);
                NN_NANOS[player].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                nn_wall_exit(&NN_CALL_WALL);
                return result;
            }
        }
        let reaches = self.canonical_current_reaches(player, opponent_reach, player_reach);
        if NN_ZERO_MODE.load(Ordering::Relaxed) {
            let mut state = self.batch_state.lock().unwrap();
            if state.collecting {
                state.pending.push(NeuralBatchRequest {
                    node_id,
                    player,
                    outer_iteration,
                    canonical: Vec::new(),
                    compact: Vec::new(),
                    context: Vec::new(),
                    reaches: Vec::new(),
                    deferred_sparse_input: None,
                });
            }
            let result = vec![0.0; self.mappings[player].len()];
            NN_NANOS[player].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            nn_wall_exit(&NN_CALL_WALL);
            return result;
        }
        let hands = self.mappings[player].len();
        assert_eq!(values.len(), action_count * hands);
        let mut context = self
            .contexts
            .get(&node_id)
            .expect("NN has no context for chance node")
            .clone();
        if let Some(scale) = self.iteration_scale {
            context.push((outer_iteration as f32).min(scale) / scale);
        }
        let basis_count = if self.sum_cfvs {
            1
        } else if compact_input {
            action_count
        } else {
            basis_actions.len()
        };
        let compact: Cow<'_, [f32]> = if compact_input && !self.sum_cfvs {
            assert_eq!(values.len(), basis_count * hands);
            Cow::Borrowed(values)
        } else {
            let mut compact = vec![0.0f32; basis_count * hands];
            for (basis, &action) in basis_actions.iter().enumerate() {
                for local in 0..hands {
                    if self.sum_cfvs {
                        compact[local] += values[action * hands + local];
                    } else {
                        compact[basis * hands + local] = values[action * hands + local];
                    }
                }
            }
            Cow::Owned(compact)
        };
        let mut canonical = Vec::new();
        if self.sparse_models.is_none() {
            canonical.resize(basis_count * WEIGHT_CANONICAL_HAND_COUNT, 0.0);
            for basis in 0..basis_count {
                for (local, &pio) in self.mappings[player].iter().enumerate() {
                    canonical[pio * basis_count + basis] = compact[basis * hands + local];
                }
            }
        }
        {
            let mut state = self.batch_state.lock().unwrap();
            if state.collecting {
                state.pending.push(NeuralBatchRequest {
                    node_id,
                    player,
                    outer_iteration,
                    canonical,
                    compact: compact.into_owned(),
                    context,
                    reaches,
                    deferred_sparse_input: None,
                });
                nn_wall_exit(&NN_CALL_WALL);
                return vec![0.0; hands];
            }
        }
        if let Some(models) = &self.sparse_models {
            let forward_started = Instant::now();
            nn_wall_enter(&NN_FORWARD_WALL);
            let output = models[player].infer(
                &compact,
                &self.mappings[player],
                &context,
                &reaches,
                nn_sparse_bf16(),
            );
            NN_FORWARD_NANOS[player].fetch_add(
                forward_started.elapsed().as_nanos() as u64,
                Ordering::Relaxed,
            );
            nn_wall_exit(&NN_FORWARD_WALL);
            let scale = self.output_scale(node_id, player, &reaches);
            let result: Vec<f64> = output
                .into_iter()
                .map(|value| value as f64 * scale)
                .collect();
            NN_NANOS[player].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            nn_wall_exit(&NN_CALL_WALL);
            return result;
        }
        let input = tch::Tensor::from_slice(&canonical)
            .reshape([1, WEIGHT_CANONICAL_HAND_COUNT as i64, basis_count as i64])
            .to_device(self.device)
            .to_kind(nn_kind());
        let context = tch::Tensor::from_slice(&context)
            .reshape([1, context.len() as i64])
            .to_device(self.device)
            .to_kind(nn_kind());
        let module = self.modules.as_ref().expect("legacy NN module missing")[player]
            .lock()
            .unwrap();
        let forward_started = Instant::now();
        nn_wall_enter(&NN_FORWARD_WALL);
        let output = if self.current_reaches {
            let reaches = tch::Tensor::from_slice(&reaches)
                .reshape([1, WEIGHT_CANONICAL_HAND_COUNT as i64, 2])
                .to_device(self.device)
                .to_kind(nn_kind());
            if let Some(range_features) = &self.range_features {
                let initial_ranges = tch::Tensor::from_slice(range_features)
                    .reshape([1, WEIGHT_CANONICAL_HAND_COUNT as i64, 2])
                    .to_device(self.device)
                    .to_kind(nn_kind());
                forward_nn(&module, &[input, context, reaches, initial_ranges])
            } else {
                forward_nn(&module, &[input, context, reaches])
            }
        } else if let Some(range_features) = &self.range_features {
            let ranges = tch::Tensor::from_slice(range_features)
                .reshape([1, WEIGHT_CANONICAL_HAND_COUNT as i64, 2])
                .to_device(self.device)
                .to_kind(nn_kind());
            forward_nn(&module, &[input, context, ranges])
        } else {
            forward_nn(&module, &[input, context])
        };
        let output = output
            .expect("turn NN inference failed")
            .reshape([WEIGHT_CANONICAL_HAND_COUNT as i64]);
        if NN_EXPLICIT_SYNC.load(Ordering::Relaxed) && matches!(self.device, tch::Device::Cuda(_)) {
            tch::Cuda::synchronize(0);
        }
        NN_FORWARD_NANOS[player].fetch_add(
            forward_started.elapsed().as_nanos() as u64,
            Ordering::Relaxed,
        );
        nn_wall_exit(&NN_FORWARD_WALL);
        let canonical_output: Vec<f32> =
            Vec::<f32>::try_from(output.to_kind(tch::Kind::Float).to_device(tch::Device::Cpu))
                .expect("invalid turn NN output");
        assert!(
            canonical_output.iter().all(|value| value.is_finite()),
            "turn NN produced non-finite output"
        );
        let scale = self.output_scale(node_id, player, &reaches);
        let result: Vec<f64> = self.mappings[player]
            .iter()
            .map(|&pio| canonical_output[pio] as f64 * scale)
            .collect();
        NN_NANOS[player].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        nn_wall_exit(&NN_CALL_WALL);
        result
    }
}
