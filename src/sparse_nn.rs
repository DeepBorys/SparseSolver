use std::cell::RefCell;
use std::fs;
use std::path::Path;

#[path = "sparse_nn_prepare.rs"]
mod preparation;
#[path = "sparse_nn_buffer_pool.rs"]
pub(crate) mod buffer_pool;
#[path = "nn_cpu_defaults.rs"]
pub(crate) mod cpu_defaults;
pub(crate) use preparation::{PreparedBasisInput, PreparedCommonInput};

const MAGIC_V1: &[u8; 8] = b"SPNNDU01";
const MAGIC_V2: &[u8; 8] = b"SPNNDU02";
const MAGIC_V3: &[u8; 8] = b"SPNNDU03";
const MAGIC_V4: &[u8; 8] = b"SPNNDU04";
const MAGIC_SILU: &[u8; 8] = b"SPNNDU05";
const MAGIC_BUCKET: &[u8; 8] = b"SPNNBU01";
const MAGIC_BUCKET_HYBRID: &[u8; 8] = b"SPNNBU02";
const CANONICAL_HANDS: usize = 1326;

fn supported_legal_hand_count(mappings: &[usize], legal: Option<&[u8]>) -> usize {
    mappings.iter().filter(|&&hand| legal.map_or(true,|mask|mask[hand]!=0)).count()
}

/// Opt-in diagnostic recovery for FP32 mean-square underflow/overflow. Ordinary
/// positive finite FP32 scales retain their exact historical bits. No CFV is
/// clipped or replaced; only the exceptional RMS reduction is widened.
fn recover_cfv_rms(values: &[f32], scale: f32, denominator: usize) -> f32 {
    if scale > 0.0 && scale.is_finite() {
        return scale;
    }
    if !cpu_defaults::flag("SPARSE_NN_RMS_F64_FALLBACK") {
        return scale;
    }
    assert!(denominator > 0 && values.iter().all(|v| v.is_finite()));
    let mean = values.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / denominator as f64;
    let repaired = mean.sqrt() as f32;
    assert!(repaired.is_finite() && repaired > 0.0, "Invalid exceptional CFV RMS");
    eprintln!("SPARSE_RMS_F64_FALLBACK fp32_scale={scale:e} f64_mean_square={mean:e} recovered_scale={repaired:e} denominator={denominator}");
    repaired
}

#[cfg(test)]
mod rms_recovery_tests {
    #[test]
    fn supported_rms_counts_initial_support_intersected_with_board_legality() {
        assert_eq!(super::supported_legal_hand_count(&[0,1,3],Some(&[1,0,1,1])),2);
        assert_eq!(super::supported_legal_hand_count(&[0,1,3],None),3);
        assert_eq!(super::supported_legal_hand_count(&[1],Some(&[1,0])),0);
    }
    #[test]
    fn positive_fp32_scales_are_bit_preserved() {
        for scale in [1e-20f32, 0.125, 1.0, 1e20] {
            assert_eq!(super::recover_cfv_rms(&[1.0],scale,1).to_bits(),scale.to_bits());
        }
    }
    #[test]
    fn exceptional_rms_uses_original_values_not_a_floor() {
        // Run alone: this diagnostic option is intentionally opt-in.
        std::env::set_var("SPARSE_NN_RMS_F64_FALLBACK","1");
        let values=[1e-21f32,2e-21f32];
        let squares: f32=values.iter().map(|v|v*v).sum();
        let old=(squares/100000.0).sqrt();
        assert_eq!(old,0.0);
        let fixed=super::recover_cfv_rms(&values,old,100000);
        let expected=(values.iter().map(|&v|(v as f64).powi(2)).sum::<f64>()/100000.).sqrt() as f32;
        assert_eq!(fixed,expected);
        assert!(values.iter().all(|v|(v/fixed).is_finite()));
        std::env::remove_var("SPARSE_NN_RMS_F64_FALLBACK");
    }
}

#[derive(Default)]
struct SparseScratch {
    input_indices: Vec<u32>,
    input_values: Vec<f32>,
    dense_inputs: Vec<f32>,
    output_positions: Vec<usize>,
    output_indices: Vec<u32>,
    output_values: Vec<f32>,
    reach_sums: [f32; 2],
}

thread_local! {
    /// Immediate inference is invoked many times per CFR iteration. Keep its
    /// variable-sized staging storage per worker thread so calls do not visit
    /// the allocator. The returned prediction remains caller-owned.
    static SPARSE_SCRATCH: RefCell<SparseScratch> = RefCell::new(SparseScratch::default());
}

unsafe extern "C" {
    fn sparse_nn_has_avx512_bf16() -> bool;
    fn sparse_nn_forward_bf16(
        w1_input_major: *const u16,
        b1: *const f32,
        norm_weight: *const f32,
        norm_bias: *const f32,
        prelu: f32,
        w2_output_major: *const u16,
        b2: *const f32,
        hidden: u32,
        input_indices: *const u32,
        input_values: *const f32,
        input_count: u32,
        output_indices: *const u32,
        output_count: u32,
        output: *mut f32,
    );
    fn sparse_nn_forward_bf16_dense_input(
        w1_input_major: *const u16,
        b1: *const f32,
        norm_weight: *const f32,
        norm_bias: *const f32,
        prelu: f32,
        w2_output_major: *const u16,
        b2: *const f32,
        hidden: u32,
        input_values: *const f32,
        input_count: u32,
        output_indices: *const u32,
        output_count: u32,
        output: *mut f32,
    );
    fn sparse_nn_forward_bf16_compact_cfv(
        w1_input_major: *const u16,
        b1: *const f32,
        norm_weight: *const f32,
        norm_bias: *const f32,
        prelu: f32,
        w2_output_major: *const u16,
        b2: *const f32,
        hidden: u32,
        cfv_values: *const f32,
        cfv_count: u32,
        cfv_scale: f32,
        extra_indices: *const u32,
        extra_values: *const f32,
        extra_count: u32,
        output_indices: *const u32,
        output_count: u32,
        output: *mut f32,
    );
    fn sparse_nn_forward_f32(
        w1_input_major: *const f32,
        b1: *const f32,
        norm_weight: *const f32,
        norm_bias: *const f32,
        prelu: f32,
        w2_output_major: *const f32,
        b2: *const f32,
        hidden: u32,
        input_indices: *const u32,
        input_values: *const f32,
        input_count: u32,
        output_indices: *const u32,
        output_count: u32,
        output: *mut f32,
    );
}

#[derive(Debug, Clone)]
struct SiluHiddenLayer {
    weights: Vec<f32>,
    bias: Vec<f32>,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
}

fn layer_norm_silu(values: &mut [f32], weight: &[f32], bias: &[f32]) {
    let mean=values.iter().map(|&v|v as f64).sum::<f64>()/values.len() as f64;
    let variance=values.iter().map(|&v|(v as f64-mean).powi(2)).sum::<f64>()/values.len() as f64;
    let inverse=(variance+1e-5).sqrt().recip();
    for (i,value) in values.iter_mut().enumerate() {
        let normalized=((*value as f64-mean)*inverse) as f32;
        let z=normalized*weight[i]+bias[i];
        *value=z/(1.0+(-z).exp());
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SparseCpuModel {
    silu_layers: Option<Vec<SiluHiddenLayer>>,
    bucket_projection: Option<(usize, Vec<u16>, bool)>,
    folded_buckets: bool,
    basis_count: usize,
    input_channels: usize,
    context_repeat: usize,
    context_count: usize,
    hand_major: bool,
    hidden: usize,
    output_count: usize,
    legal_count: usize,
    legal_mask: Option<Vec<u8>>,
    basis_cards: Option<Vec<u8>>,
    equity_matrix: Option<Vec<f32>>,
    compatibility_matrix: Option<Vec<u8>>,
    linear_weights: Vec<f32>,
    w1_f32_input_major: Vec<f32>,
    w1_input_major: Vec<u16>,
    b1: Vec<f32>,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    prelu: f32,
    w2_f32_output_major: Vec<f32>,
    w2_output_major: Vec<u16>,
    b2: Vec<f32>,
    input_row_remap: Vec<u32>,
    output_row_remap: Vec<u32>,
    dense_compact_inputs: bool,
    stream_compact_inputs: bool,
    supported_hand_rms: bool,
    preparation_mode: u8,
    prepared_plan: Option<preparation::PreparedPlan>,
}

fn take<const N: usize>(bytes: &[u8], cursor: &mut usize) -> Result<[u8; N], String> {
    let end = cursor
        .checked_add(N)
        .ok_or("sparse model offset overflow")?;
    let value = bytes
        .get(*cursor..end)
        .ok_or("truncated sparse CPU model")?
        .try_into()
        .unwrap();
    *cursor = end;
    Ok(value)
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<usize, String> {
    Ok(u32::from_le_bytes(take(bytes, cursor)?) as usize)
}

fn read_f32s(bytes: &[u8], cursor: &mut usize, count: usize) -> Result<Vec<f32>, String> {
    (0..count)
        .map(|_| Ok(f32::from_le_bytes(take(bytes, cursor)?)))
        .collect()
}

fn read_u16s(bytes: &[u8], cursor: &mut usize, count: usize) -> Result<Vec<u16>, String> {
    (0..count)
        .map(|_| Ok(u16::from_le_bytes(take(bytes, cursor)?)))
        .collect()
}

pub(crate) fn diagnostic_worker_scratch_bytes()->u64{
    SPARSE_SCRATCH.with_borrow(|x|((x.input_indices.capacity()+x.input_values.capacity()+x.dense_inputs.capacity()+x.output_indices.capacity()+x.output_values.capacity())*4+x.output_positions.capacity()*std::mem::size_of::<usize>()) as u64)+preparation::diagnostic_worker_scratch_bytes()
}
impl SparseCpuModel {
    pub(crate) fn diagnostic_numeric_bytes(&self)->(u64,u64){
        let mut used=0usize;let mut allocated=0usize;
        macro_rules! add {($v:expr,$bytes:expr)=>{{used+=$v.len()*$bytes;allocated+=$v.capacity()*$bytes;}}}
        for v in [&self.linear_weights,&self.w1_f32_input_major,&self.b1,&self.norm_weight,&self.norm_bias,&self.w2_f32_output_major,&self.b2]{add!(v,4);}
        for v in [&self.w1_input_major,&self.w2_output_major]{add!(v,2);}
        for v in [&self.input_row_remap,&self.output_row_remap]{add!(v,4);}
        if let Some(v)=&self.legal_mask{add!(v,1);}if let Some(v)=&self.basis_cards{add!(v,1);}
        if let Some(v)=&self.equity_matrix{add!(v,4);}if let Some(v)=&self.compatibility_matrix{add!(v,1);}
        if let Some((_,v,_))=&self.bucket_projection{add!(v,2);}
        if let Some(layers)=&self.silu_layers{for l in layers{for v in [&l.weights,&l.bias,&l.norm_weight,&l.norm_bias]{add!(v,4);}}}
        if let Some(p)=&self.prepared_plan{let n=p.diagnostic_numeric_bytes();used+=n;allocated+=n;}
        (used as u64,allocated as u64)
    }
    pub(crate) fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        // Explicit immutable-model experiment. Cache only unspecialized models;
        // each caller owns its mutable range-specific clone. Changing files at
        // the same path requires restarting the process, never live replacement.
        if std::env::var("SPARSE_NN_RESIDENT_MODELS").as_deref() == Ok("1") {
            use std::collections::HashMap;
            use std::sync::{Mutex, OnceLock};
            static CACHE: OnceLock<Mutex<HashMap<(std::path::PathBuf, String, String), SparseCpuModel>>> = OnceLock::new();
            let path = std::fs::canonicalize(path.as_ref()).map_err(|e| e.to_string())?;
            let key = (path.clone(),
                std::env::var("SPARSE_NN_PREP_EXPERIMENT").unwrap_or_default(),
                std::env::var("SPARSE_NN_PREP_WIDTH").unwrap_or_default());
            let mut cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()))
                .lock().map_err(|_| "resident sparse model cache poisoned")?;
            if let Some(model) = cache.get(&key) { return Ok(model.clone()); }
            let model = Self::load_uncached(&path)?;
            cache.insert(key, model.clone());
            return Ok(model);
        }
        Self::load_uncached(path)
    }

    fn load_uncached(path: impl AsRef<Path>) -> Result<Self, String> {
        let bytes = fs::read(path.as_ref()).map_err(|error| {
            format!(
                "failed to read sparse CPU model {}: {error}",
                path.as_ref().display()
            )
        })?;
        let mut cursor = 0;
        let magic = take::<8>(&bytes, &mut cursor)?;
        let hybrid = &magic == MAGIC_BUCKET_HYBRID;
        let bucketed = &magic == MAGIC_BUCKET || hybrid;
        let silu=&magic==MAGIC_SILU;
        if &magic != MAGIC_V1 && &magic != MAGIC_V2 && &magic != MAGIC_V3 && &magic != MAGIC_V4 && !bucketed && !silu {
            return Err("invalid sparse CPU model magic".into());
        }
        let version_two = &magic == MAGIC_V2 || &magic == MAGIC_V3 || &magic == MAGIC_V4 || bucketed || silu;
        let version_three = &magic == MAGIC_V3 || &magic == MAGIC_V4 || bucketed || silu;
        let basis_count = read_u32(&bytes, &mut cursor)?;
        let context_field = read_u32(&bytes, &mut cursor)?;
        let hidden = read_u32(&bytes, &mut cursor)?;
        let output_count = read_u32(&bytes, &mut cursor)?;
        let legal_count = read_u32(&bytes, &mut cursor)?;
        let flags = if version_two {
            read_u32(&bytes, &mut cursor)?
        } else {
            0
        };
        let input_channels = if version_three {
            read_u32(&bytes, &mut cursor)?
        } else {
            basis_count
        };
        let hand_major = flags & 1 != 0;
        let context_repeat = if version_two { 0 } else { context_field };
        let context_count = if version_two {
            context_field
        } else {
            basis_count * context_repeat
        };
        if hidden == 0
            || hidden > 1024
            || hidden % 16 != 0
            || input_channels < basis_count
            || output_count != CANONICAL_HANDS
            || legal_count == 0
        {
            return Err("unsupported sparse CPU model dimensions".into());
        }
        if version_two && !hand_major {
            return Err("unsupported sparse CPU v2 input layout".into());
        }
        let legal_mask = if version_two {
            let end = cursor
                .checked_add(CANONICAL_HANDS)
                .ok_or("sparse model offset overflow")?;
            let mask = bytes
                .get(cursor..end)
                .ok_or("truncated sparse CPU legal mask")?
                .to_vec();
            cursor = end;
            if mask.iter().filter(|&&value| value != 0).count() != legal_count {
                return Err("sparse CPU legal mask count mismatch".into());
            }
            Some(mask)
        } else {
            None
        };
        let basis_cards = if flags & 2 != 0 {
            let end = cursor
                .checked_add(basis_count)
                .ok_or("sparse model offset overflow")?;
            let cards = bytes
                .get(cursor..end)
                .ok_or("truncated sparse CPU basis cards")?
                .to_vec();
            cursor = end;
            Some(cards)
        } else {
            None
        };
        let equity_matrix = if flags & 4 != 0 {
            Some(read_f32s(
                &bytes,
                &mut cursor,
                CANONICAL_HANDS * CANONICAL_HANDS,
            )?)
        } else {
            None
        };
        let compatibility_matrix = if flags & 4 != 0 {
            let count = CANONICAL_HANDS * CANONICAL_HANDS;
            let end = cursor
                .checked_add(count)
                .ok_or("sparse model offset overflow")?;
            let values = bytes
                .get(cursor..end)
                .ok_or("truncated compatibility matrix")?
                .to_vec();
            cursor = end;
            Some(values)
        } else {
            None
        };
        let bucket_projection = if bucketed {
            let buckets = read_u32(&bytes, &mut cursor)?;
            if buckets == 0 || buckets >= u16::MAX as usize || context_count != 1 || input_channels != basis_count + 2 {
                return Err("unsupported bucketed sparse model dimensions".into());
            }
            let assignments = read_u16s(&bytes, &mut cursor, basis_count * CANONICAL_HANDS)?;
            if assignments.iter().any(|&b| b != u16::MAX && b as usize >= buckets) {
                return Err("invalid sparse bucket assignment".into());
            }
            Some((buckets, assignments, hybrid))
        } else { None };
        let input_count = bucket_projection.as_ref().map_or(
            input_channels * CANONICAL_HANDS + context_count,
            |(buckets, _, hybrid)| if *hybrid { basis_count * buckets + 2 * CANONICAL_HANDS + context_count }
                                  else { basis_count * 3 * buckets + context_count },
        );
        let linear_weights = read_f32s(&bytes, &mut cursor, basis_count * output_count)?;
        let w1_f32_input_major = read_f32s(&bytes, &mut cursor, input_count * hidden)?;
        let w1_input_major = read_u16s(&bytes, &mut cursor, input_count * hidden)?;
        let b1 = read_f32s(&bytes, &mut cursor, hidden)?;
        let norm_weight = read_f32s(&bytes, &mut cursor, hidden)?;
        let norm_bias = read_f32s(&bytes, &mut cursor, hidden)?;
        let prelu = f32::from_le_bytes(take(&bytes, &mut cursor)?);
        let w2_f32_output_major = read_f32s(&bytes, &mut cursor, output_count * hidden)?;
        let w2_output_major = read_u16s(&bytes, &mut cursor, output_count * hidden)?;
        let b2 = read_f32s(&bytes, &mut cursor, output_count)?;
        let silu_layers=if silu {
            if context_count!=1 || input_channels!=basis_count+2 || flags!=3 {
                return Err("unsupported SiLU turn-model input layout".into());
            }
            let count=read_u32(&bytes,&mut cursor)?;
            if count>7 {return Err("SiLU model supports at most eight hidden layers".into());}
            let mut layers=Vec::with_capacity(count);
            for _ in 0..count {
                layers.push(SiluHiddenLayer {
                    weights: read_f32s(&bytes,&mut cursor,hidden*hidden)?,
                    bias: read_f32s(&bytes,&mut cursor,hidden)?,
                    norm_weight: read_f32s(&bytes,&mut cursor,hidden)?,
                    norm_bias: read_f32s(&bytes,&mut cursor,hidden)?,
                });
            }
            Some(layers)
        } else {None};
        if cursor != bytes.len() {
            return Err("sparse CPU model has trailing data".into());
        }
        Ok(Self {
            silu_layers,
            bucket_projection,
            folded_buckets: false,
            basis_count,
            input_channels,
            context_repeat,
            context_count,
            hand_major,
            hidden,
            output_count,
            legal_count,
            legal_mask,
            basis_cards,
            equity_matrix,
            compatibility_matrix,
            linear_weights,
            w1_f32_input_major,
            w1_input_major,
            b1,
            norm_weight,
            norm_bias,
            prelu,
            w2_f32_output_major,
            w2_output_major,
            b2,
            input_row_remap: Vec::new(),
            output_row_remap: Vec::new(),
            dense_compact_inputs: false,
            stream_compact_inputs: false,
            supported_hand_rms: false,
            // Validated supported-hand FP32 defaults; explicit overrides remain available.
            preparation_mode: {
                let mode = match std::env::var("SPARSE_NN_PREP_EXPERIMENT").as_deref() {
                    Err(_) => if cpu_defaults::available() { 4 } else { 0 },
                    Ok("0") | Ok("off") => 0,
                    Ok("1") | Ok("prepared") => 1,
                    Ok("2") | Ok("reciprocal") => 2,
                    Ok("3") | Ok("simd") => 3,
                    Ok("4") | Ok("simd_divide") => 4,
                    Ok(value) => return Err(format!("unknown SPARSE_NN_PREP_EXPERIMENT: {value}")),
                };
                let widths = std::env::var("SPARSE_NN_PREP_WIDTH").unwrap_or_else(|_| "64,128".into())
                    .split(',').map(|v| v.parse::<usize>())
                    .collect::<Result<Vec<_>, _>>().map_err(|_| "invalid SPARSE_NN_PREP_WIDTH")?;
                if widths.contains(&hidden) && !silu { mode } else { 0 }
            },
            prepared_plan: None,
        })
    }

    pub(crate) fn basis_count(&self) -> usize {
        self.basis_count
    }

    pub(crate) fn basis_cards(&self) -> Option<&[u8]> {
        self.basis_cards.as_deref()
    }

    pub(crate) fn set_supported_hand_rms(&mut self, enabled: bool) {
        self.supported_hand_rms = enabled;
    }

    /// Repack the two linear layers for the fixed solver ranges. This removes
    /// only rows that are provably zero for the lifetime of the solve; dynamic
    /// board blockers continue to be handled by the sparse request itself.
    pub(crate) fn specialize_for_ranges(
        &mut self,
        mappings: &[Vec<usize>; 2],
        player: usize,
    ) -> Result<(usize, usize), String> {
        self.fold_bucket_projection(mappings, player)?;
        if !self.hand_major {
            return Ok((self.w1_input_major.len() / self.hidden, self.b2.len()));
        }
        let feature_channels = self.input_channels - self.basis_count;
        if feature_channels != 0 && feature_channels != 2 && feature_channels != 4 {
            return Err("unsupported sparse reach/equity channel count".into());
        }
        if feature_channels == 4 && self.equity_matrix.is_none() {
            return Err("equity feature channels require embedded matrices".into());
        }
        let mut active = [vec![false; CANONICAL_HANDS], vec![false; CANONICAL_HANDS]];
        for side in 0..2 {
            for &canonical in &mappings[side] {
                active[side][canonical] = true;
            }
        }

        let original_input_rows = self.input_channels * CANONICAL_HANDS + self.context_count;
        let mut input_row_remap = vec![u32::MAX; original_input_rows];
        let mut retained_inputs = Vec::new();
        let layout = std::env::var("SPARSE_NN_RANGE_LAYOUT").unwrap_or_else(|_| "stream".into());
        if layout == "canonical" {
            for canonical in 0..CANONICAL_HANDS {
                for channel in 0..self.input_channels {
                    let keep = if channel < self.basis_count {
                        active[player][canonical]
                    } else {
                        active[(channel - self.basis_count) & 1][canonical]
                    };
                    if keep {
                        retained_inputs.push(canonical * self.input_channels + channel);
                    }
                }
            }
        } else if layout == "stream" || layout == "stream-staged" || layout == "dense" {
            for channel in 0..self.basis_count {
                retained_inputs.extend(
                    mappings[player]
                        .iter()
                        .map(|&canonical| canonical * self.input_channels + channel),
                );
            }
            for canonical in 0..CANONICAL_HANDS {
                for channel in 0..feature_channels {
                    if active[channel & 1][canonical] {
                        retained_inputs
                            .push(canonical * self.input_channels + self.basis_count + channel);
                    }
                }
            }
        } else {
            return Err(format!("unknown SPARSE_NN_RANGE_LAYOUT: {layout}"));
        }
        for (compact, &original) in retained_inputs.iter().enumerate() {
            input_row_remap[original] = compact as u32;
        }
        let context_start = self.input_channels * CANONICAL_HANDS;
        for original in context_start..original_input_rows {
            input_row_remap[original] = retained_inputs.len() as u32;
            retained_inputs.push(original);
        }

        let mut output_row_remap = vec![u32::MAX; self.output_count];
        let mut retained_outputs = Vec::new();
        for &canonical in &mappings[player] {
            if self
                .legal_mask
                .as_ref()
                .map_or(true, |mask| mask[canonical] != 0)
            {
                output_row_remap[canonical] = retained_outputs.len() as u32;
                retained_outputs.push(canonical);
            }
        }

        fn repack_rows<T: Copy>(source: &[T], rows: &[usize], width: usize) -> Vec<T> {
            let mut result = Vec::with_capacity(rows.len() * width);
            for &row in rows {
                result.extend_from_slice(&source[row * width..(row + 1) * width]);
            }
            result
        }
        self.w1_f32_input_major =
            repack_rows(&self.w1_f32_input_major, &retained_inputs, self.hidden);
        self.w1_input_major = repack_rows(&self.w1_input_major, &retained_inputs, self.hidden);
        self.w2_f32_output_major =
            repack_rows(&self.w2_f32_output_major, &retained_outputs, self.hidden);
        self.w2_output_major = repack_rows(&self.w2_output_major, &retained_outputs, self.hidden);
        self.b2 = retained_outputs
            .iter()
            .map(|&canonical| self.b2[canonical])
            .collect();
        self.input_row_remap = input_row_remap;
        self.output_row_remap = output_row_remap;
        self.dense_compact_inputs = layout == "dense";
        // The compact-CFV kernel currently does not reproduce the established
        // indexed sparse path numerically in solver benchmarks. Keep `stream`
        // on the verified indexed implementation until that kernel has a
        // prediction-level equivalence test and passes it.
        self.stream_compact_inputs = false;
        // Historical checkpoint RMS conventions retain the established path by
        // default. Explicit diagnostic preparation overrides still work.
        if !self.supported_hand_rms && std::env::var_os("SPARSE_NN_PREP_EXPERIMENT").is_none() {
            self.preparation_mode = 0;
        }
        self.prepared_plan = if self.preparation_mode != 0 {
            preparation::PreparedPlan::new(self, &mappings[player], mappings)
        } else {
            None
        };
        if self.preparation_mode != 0 {
            println!("SPARSE_PREPARATION_EXPERIMENT mode={} hidden={} enabled={}",
                     self.preparation_mode, self.hidden, self.prepared_plan.is_some());
        }
        Ok((retained_inputs.len(), retained_outputs.len()))
    }

    /// Bucket means and range masses are linear before the first hidden layer.
    /// Compose them once for the fixed initial supports, including the fixed
    /// CFV bucket denominators. Inference can then use the established sparse
    /// hand-input kernel without constructing any bucket vectors.
    fn fold_bucket_projection(&mut self, mappings: &[Vec<usize>; 2], player: usize) -> Result<(), String> {
        let Some((buckets, assignments, hybrid)) = self.bucket_projection.take() else { return Ok(()); };
        if player > 1 || !self.input_row_remap.is_empty() {
            return Err("bucket projection requires unspecialized model and valid player".into());
        }
        let legal = self.legal_mask.as_ref().ok_or("bucket projection requires legal mask")?;
        let mut counts = vec![0usize; self.basis_count * buckets];
        for basis in 0..self.basis_count {
            for &hand in &mappings[player] {
                let bucket = assignments[basis * CANONICAL_HANDS + hand];
                if legal[hand] != 0 && bucket != u16::MAX {
                    counts[basis * buckets + bucket as usize] += 1;
                }
            }
        }
        let input_rows = self.input_channels * CANONICAL_HANDS + self.context_count;
        let mut folded = vec![0.0; input_rows * self.hidden];
        for basis in 0..self.basis_count {
            for &hand in &mappings[player] {
                let bucket = assignments[basis * CANONICAL_HANDS + hand];
                if legal[hand] == 0 || bucket == u16::MAX { continue; }
                let divisor = counts[basis * buckets + bucket as usize] as f32;
                let source = (basis * if hybrid { 1 } else { 3 } * buckets + bucket as usize) * self.hidden;
                let dest = (hand * self.input_channels + basis) * self.hidden;
                for column in 0..self.hidden {
                    folded[dest + column] = self.w1_f32_input_major[source + column] / divisor;
                }
            }
            for side in 0..2 {
                if hybrid { continue; }
                for &hand in &mappings[side] {
                    let bucket = assignments[basis * CANONICAL_HANDS + hand];
                    if legal[hand] == 0 || bucket == u16::MAX { continue; }
                    let source = ((basis * 3 + 1 + side) * buckets + bucket as usize) * self.hidden;
                    let dest = (hand * self.input_channels + self.basis_count + side) * self.hidden;
                    for column in 0..self.hidden {
                        folded[dest + column] += self.w1_f32_input_major[source + column];
                    }
                }
            }
        }
        if hybrid {
            for side in 0..2 {
                for &hand in &mappings[side] {
                    if legal[hand] == 0 { continue; }
                    let source = (self.basis_count * buckets + hand * 2 + side) * self.hidden;
                    let dest = (hand * self.input_channels + self.basis_count + side) * self.hidden;
                    folded[dest..dest + self.hidden].copy_from_slice(&self.w1_f32_input_major[source..source + self.hidden]);
                }
            }
        }
        let source = (if hybrid { self.basis_count * buckets + 2 * CANONICAL_HANDS }
                     else { self.basis_count * 3 * buckets }) * self.hidden;
        let dest = self.input_channels * CANONICAL_HANDS * self.hidden;
        folded[dest..].copy_from_slice(&self.w1_f32_input_major[source..]);
        self.w1_input_major = folded.iter().map(|value| {
            let bits = value.to_bits();
            ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
        }).collect();
        self.w1_f32_input_major = folded;
        self.folded_buckets = true;
        Ok(())
    }

    pub(crate) fn supports_child_inputs(&self) -> bool {
        self.preparation_mode==4 && self.prepared_plan.as_ref().is_some_and(|p|p.supports_child_inputs())
    }
    pub(crate) fn prepare_basis_input(&self, basis:usize, raw:Vec<f32>) -> PreparedBasisInput {
        self.prepared_plan.as_ref().expect("missing prepared plan").prepare_basis(self,basis,raw)
    }
    pub(crate) fn prepare_common_input(&self, context:&[f32], reaches:[&[f32];2]) -> PreparedCommonInput {
        self.prepared_plan.as_ref().expect("missing prepared plan").prepare_common(context,reaches)
    }
    pub(crate) fn infer_child_inputs(&self, blocks:&[PreparedBasisInput], common:PreparedCommonInput, hands:usize) -> Vec<f64> {
        self.prepared_plan.as_ref().expect("missing prepared plan").infer_children(self,blocks,common,hands)
    }

    pub(crate) fn infer_prepared_compact_f64(
        &self, values: &[f32], mappings: &[usize], context: &[f32], reaches: [&[f32]; 2],
    ) -> Option<Vec<f64>> {
        let plan = self.prepared_plan.as_ref()?;
        if self.preparation_mode != 4 || !plan.matches(mappings) { return None; }
        Some(plan.infer_compact_f64(self, values, context, reaches))
    }

    pub(crate) fn infer(
        &self,
        values: &[f32],
        mappings: &[usize],
        context: &[f32],
        reaches: &[f32],
        use_bf16: bool,
    ) -> Vec<f32> {
        assert!(!use_bf16 || unsafe { sparse_nn_has_avx512_bf16() },
            "BF16 sparse inference requires AVX512 BF16; use FP32 on this CPU");
        // Legacy models are unchanged. New supported-hand models must exclude
        // dealt-board collisions even when mappings retain flop-level hands.
        let mut masked_values=None;
        let mut masked_reaches=None;
        if self.supported_hand_rms {
            if let Some(legal)=&self.legal_mask {
                let hands=mappings.len();
                assert_eq!(values.len(),self.basis_count*hands);
                if values.iter().enumerate().any(|(i,&v)|v!=0. && legal[mappings[i%hands]]==0) {
                    let mut copy=values.to_vec();
                    for (i,v) in copy.iter_mut().enumerate() {
                        if legal[mappings[i%hands]]==0 {*v=0.;}
                    }
                    masked_values=Some(copy);
                }
                if reaches.chunks_exact(2).enumerate().any(|(h,v)|legal[h]==0 && (v[0]!=0. || v[1]!=0.)) {
                    let mut copy=reaches.to_vec();
                    for (h,v) in copy.chunks_exact_mut(2).enumerate() {
                        if legal[h]==0 {v.fill(0.);}
                    }
                    masked_reaches=Some(copy);
                }
            }
        }
        let values=masked_values.as_deref().unwrap_or(values);
        let reaches=masked_reaches.as_deref().unwrap_or(reaches);
        if !use_bf16 {
            if let Some(plan) = &self.prepared_plan {
                if plan.matches(mappings) {
                    return plan.infer(self, values, context, reaches, self.preparation_mode);
                }
            }
        }
        let hands = mappings.len();
        assert!(self.bucket_projection.is_none(), "bucket model must be specialized before inference");
        assert_eq!(values.len(), self.basis_count * hands);
        let feature_channels = self.input_channels - self.basis_count;
        assert!(feature_channels == 0 || feature_channels == 2 || feature_channels == 4);
        let reach_channels = if feature_channels == 0 { 0 } else { 2 };
        assert_eq!(reaches.len(), reach_channels * CANONICAL_HANDS);
        let sum_squares: f32 = values.iter().map(|value| value * value).sum();
        if self.basis_count != 0 && (if self.silu_layers.is_some() {
            values.iter().all(|&v|v==0.0)
        } else {sum_squares==0.0}) {
            return vec![0.0; hands];
        }
        // A zero-basis DeepStack model consumes only normalized reaches and
        // context. Its output is already normalized and is rescaled by the
        // caller using opponent reach, pot, and root combinations.
        let scale = if self.basis_count == 0 {
            1.0
        } else {
            let hand_count = if self.supported_hand_rms {
                // New checkpoints explicitly use the current player's
                // initially supported canonical hands.
                supported_legal_hand_count(mappings,self.legal_mask.as_deref()).max(1)
            } else {
                // Legacy checkpoints were trained with every flop-legal hand
                // in the denominator, including range-absent zero entries.
                self.legal_count
            };
            recover_cfv_rms(values,
                (sum_squares / (hand_count * self.basis_count) as f32).sqrt(),
                hand_count * self.basis_count)
        };
        let silu_scale=self.silu_layers.as_ref().map(|_| {
            let count=supported_legal_hand_count(mappings,self.legal_mask.as_deref()).max(1)*self.basis_count;
            (values.iter().map(|&v|(v as f64).powi(2)).sum::<f64>()/count as f64).sqrt()
        });
        assert_eq!(
            context.len(),
            if self.hand_major {
                self.context_count
            } else {
                1
            }
        );
        let mut result = vec![0.0; hands];
        SPARSE_SCRATCH.with_borrow_mut(|scratch| {
            scratch.input_indices.clear();
            scratch.input_values.clear();
            scratch.output_positions.clear();
            scratch.output_indices.clear();
            scratch.reach_sums = [0.0; 2];
            let required_inputs = self.basis_count * hands + reaches.len() + self.context_count;
            scratch.input_indices.reserve(required_inputs);
            scratch.input_values.reserve(required_inputs);

            let block = CANONICAL_HANDS + self.context_repeat;
            for basis in 0..self.basis_count {
                for (local, &canonical) in mappings.iter().enumerate() {
                    let value = values[basis * hands + local];
                    if value != 0.0 && !self.stream_compact_inputs {
                        let index = if self.hand_major {
                            canonical * self.input_channels + basis
                        } else {
                            basis * block + canonical
                        };
                        let index = if self.input_row_remap.is_empty() {
                            index as u32
                        } else {
                            self.input_row_remap[index]
                        };
                        assert_ne!(index, u32::MAX, "active CFV input was removed at startup");
                        scratch.input_indices.push(index);
                        scratch.input_values.push(silu_scale.map_or_else(||value/scale,|s|(value as f64/s) as f32));
                    }
                }
                if !self.hand_major && context[0] != 0.0 {
                    for context_index in 0..self.context_repeat {
                        scratch
                            .input_indices
                            .push((basis * block + CANONICAL_HANDS + context_index) as u32);
                        scratch.input_values.push(context[0]);
                    }
                }
            }
            if self.hand_major {
                if reach_channels != 0 {
                    for (canonical, row) in reaches.chunks_exact(reach_channels).enumerate() {
                        if self.folded_buckets && self.legal_mask.as_ref().unwrap()[canonical] == 0 { continue; }
                        for (channel, &value) in row.iter().enumerate() {
                            scratch.reach_sums[channel] += value;
                        }
                    }
                    for canonical in 0..CANONICAL_HANDS {
                        if self.folded_buckets && self.legal_mask.as_ref().unwrap()[canonical] == 0 { continue; }
                        for channel in 0..reach_channels {
                            let sum=scratch.reach_sums[channel];
                            let denominator=if self.silu_layers.is_some() {
                                if sum>0.0 {sum} else {1.0}
                            } else {sum.max(if self.folded_buckets { 1e-30 } else { 1e-12 })};
                            let value = reaches[canonical * reach_channels + channel]/denominator;
                            if value != 0.0 {
                                let original =
                                    canonical * self.input_channels + self.basis_count + channel;
                                let index = if self.input_row_remap.is_empty() {
                                    original as u32
                                } else {
                                    self.input_row_remap[original]
                                };
                                assert_ne!(
                                    index,
                                    u32::MAX,
                                    "active reach input was removed at startup"
                                );
                                scratch.input_indices.push(index);
                                scratch.input_values.push(value);
                            }
                        }
                    }
                }
                if feature_channels == 4 {
                    let equity = self.equity_matrix.as_ref().unwrap();
                    let compatible = self.compatibility_matrix.as_ref().unwrap();
                    for hero in 0..2 {
                        let channel = self.basis_count + 2 + hero;
                        let opponent = hero ^ 1;
                        for canonical in 0..CANONICAL_HANDS {
                            let original = canonical * self.input_channels + channel;
                            let index = if self.input_row_remap.is_empty() {
                                original as u32
                            } else {
                                self.input_row_remap[original]
                            };
                            if index == u32::MAX {
                                continue;
                            }
                            let row = canonical * CANONICAL_HANDS;
                            let mut numerator = 0.0f32;
                            let mut denominator = 0.0f32;
                            for villain in 0..CANONICAL_HANDS {
                                let weight = reaches[villain * 2 + opponent];
                                if weight != 0.0 && compatible[row + villain] != 0 {
                                    numerator += weight * equity[row + villain];
                                    denominator += weight;
                                }
                            }
                            if denominator > 0.0 {
                                scratch.input_indices.push(index);
                                scratch.input_values.push(numerator / denominator);
                            }
                        }
                    }
                }
                let start = self.input_channels * CANONICAL_HANDS;
                for (offset, &value) in context.iter().enumerate() {
                    if value != 0.0 {
                        let original = start + offset;
                        let index = if self.input_row_remap.is_empty() {
                            original as u32
                        } else {
                            self.input_row_remap[original]
                        };
                        assert_ne!(index, u32::MAX);
                        scratch.input_indices.push(index);
                        scratch.input_values.push(value);
                    }
                }
            }
            if self.dense_compact_inputs {
                let rows = self.w1_input_major.len() / self.hidden;
                scratch.dense_inputs.resize(rows, 0.0);
                scratch.dense_inputs.fill(0.0);
                for (&index, &value) in scratch.input_indices.iter().zip(&scratch.input_values) {
                    scratch.dense_inputs[index as usize] = value;
                }
            }
            for (local, &canonical) in mappings.iter().enumerate() {
                if self
                    .legal_mask
                    .as_ref()
                    .map_or(true, |mask| mask[canonical] != 0)
                {
                    scratch.output_positions.push(local);
                    let index = if self.output_row_remap.is_empty() {
                        canonical as u32
                    } else {
                        self.output_row_remap[canonical]
                    };
                    assert_ne!(index, u32::MAX, "active output was removed at startup");
                    scratch.output_indices.push(index);
                }
            }
            scratch
                .output_values
                .resize(scratch.output_indices.len(), 0.0);
            if let Some(layers)=&self.silu_layers {
                assert!(!use_bf16,"Multilayer SiLU CPU inference currently requires FP32");
                // Separate reference implementation: FP32 weights/activations,
                // wider dot reductions. Existing PReLU SIMD kernels are untouched.
                let mut sums: Vec<f64>=self.b1.iter().map(|&v|v as f64).collect();
                for (&row,&value) in scratch.input_indices.iter().zip(&scratch.input_values) {
                    let weights=&self.w1_f32_input_major[row as usize*self.hidden..(row as usize+1)*self.hidden];
                    for (sum,&weight) in sums.iter_mut().zip(weights) {*sum+=weight as f64*value as f64;}
                }
                let mut hidden: Vec<f32>=sums.into_iter().map(|v|v as f32).collect();
                layer_norm_silu(&mut hidden,&self.norm_weight,&self.norm_bias);
                for layer in layers {
                    let mut next=vec![0.0f32;self.hidden];
                    for (row,out) in next.iter_mut().enumerate() {
                        let sum=layer.weights[row*self.hidden..(row+1)*self.hidden].iter().zip(&hidden)
                            .map(|(&w,&v)|w as f64*v as f64).sum::<f64>();
                        *out=(sum+layer.bias[row] as f64) as f32;
                    }
                    layer_norm_silu(&mut next,&layer.norm_weight,&layer.norm_bias);
                    hidden=next;
                }
                for (i,&row) in scratch.output_indices.iter().enumerate() {
                    let row=row as usize;
                    let sum=self.w2_f32_output_major[row*self.hidden..(row+1)*self.hidden].iter().zip(&hidden)
                        .map(|(&w,&v)|w as f64*v as f64).sum::<f64>();
                    scratch.output_values[i]=(sum+self.b2[row] as f64) as f32;
                }
            } else { unsafe {
                if use_bf16 && self.dense_compact_inputs {
                    sparse_nn_forward_bf16_dense_input(
                        self.w1_input_major.as_ptr(),
                        self.b1.as_ptr(),
                        self.norm_weight.as_ptr(),
                        self.norm_bias.as_ptr(),
                        self.prelu,
                        self.w2_output_major.as_ptr(),
                        self.b2.as_ptr(),
                        self.hidden as u32,
                        scratch.dense_inputs.as_ptr(),
                        scratch.dense_inputs.len() as u32,
                        scratch.output_indices.as_ptr(),
                        scratch.output_indices.len() as u32,
                        scratch.output_values.as_mut_ptr(),
                    );
                } else if use_bf16 && self.stream_compact_inputs {
                    sparse_nn_forward_bf16_compact_cfv(
                        self.w1_input_major.as_ptr(),
                        self.b1.as_ptr(),
                        self.norm_weight.as_ptr(),
                        self.norm_bias.as_ptr(),
                        self.prelu,
                        self.w2_output_major.as_ptr(),
                        self.b2.as_ptr(),
                        self.hidden as u32,
                        values.as_ptr(),
                        values.len() as u32,
                        scale,
                        scratch.input_indices.as_ptr(),
                        scratch.input_values.as_ptr(),
                        scratch.input_indices.len() as u32,
                        scratch.output_indices.as_ptr(),
                        scratch.output_indices.len() as u32,
                        scratch.output_values.as_mut_ptr(),
                    );
                } else if use_bf16 {
                    sparse_nn_forward_bf16(
                        self.w1_input_major.as_ptr(),
                        self.b1.as_ptr(),
                        self.norm_weight.as_ptr(),
                        self.norm_bias.as_ptr(),
                        self.prelu,
                        self.w2_output_major.as_ptr(),
                        self.b2.as_ptr(),
                        self.hidden as u32,
                        scratch.input_indices.as_ptr(),
                        scratch.input_values.as_ptr(),
                        scratch.input_indices.len() as u32,
                        scratch.output_indices.as_ptr(),
                        scratch.output_indices.len() as u32,
                        scratch.output_values.as_mut_ptr(),
                    );
                } else {
                    sparse_nn_forward_f32(
                        self.w1_f32_input_major.as_ptr(),
                        self.b1.as_ptr(),
                        self.norm_weight.as_ptr(),
                        self.norm_bias.as_ptr(),
                        self.prelu,
                        self.w2_f32_output_major.as_ptr(),
                        self.b2.as_ptr(),
                        self.hidden as u32,
                        scratch.input_indices.as_ptr(),
                        scratch.input_values.as_ptr(),
                        scratch.input_indices.len() as u32,
                        scratch.output_indices.as_ptr(),
                        scratch.output_indices.len() as u32,
                        scratch.output_values.as_mut_ptr(),
                    );
                }
            } }
            for index in 0..scratch.output_positions.len() {
                let local = scratch.output_positions[index];
                let canonical = mappings[local];
                let mut baseline = 0.0;
                for basis in 0..self.basis_count {
                    let value = values[basis * hands + local];
                    baseline +=
                        value * (if self.folded_buckets { 0.0 } else { 1.0 }
                            + self.linear_weights[basis * self.output_count + canonical]);
                }
                result[local] = if let Some(scale)=silu_scale {
                    let baseline=(0..self.basis_count).map(|b|values[b*hands+local] as f64*
                        (1.0+self.linear_weights[b*self.output_count+canonical] as f64)).sum::<f64>();
                    (baseline+scratch.output_values[index] as f64*scale) as f32
                } else {baseline+scratch.output_values[index]*scale};
            }
        });
        result
    }
}
