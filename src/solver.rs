use crate::card::NOT_DEALT;
#[path = "solver_operation_profile.rs"]
mod operation_profile;
#[path = "cfr_action_buffer.rs"]
mod cfr_action_buffer;
#[path = "leaf_path_reach.rs"]
mod leaf_path_reach;
#[path = "direct_terminal_dispatch.rs"]
mod direct_terminal_dispatch;
#[doc(hidden)]
pub use direct_terminal_dispatch::CompiledTerminalPlan;
pub fn fused_regret_updates_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    // Retain the original experiment switch as an explicit comparison override.
    // The validated FP32 update is now enabled unless explicitly disabled.
    *ENABLED.get_or_init(|| std::env::var("PFS_FUSED_REGRET_UPDATE_EXPERIMENT").as_deref() != Ok("0"))
}
pub fn reset_solver_operation_profile() { operation_profile::reset(); }
pub fn report_solver_operation_profile() { operation_profile::report(); }
use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;
use crate::utility::*;
use std::collections::BTreeMap;
use std::io::{self, Write};
use std::mem::MaybeUninit;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

#[path = "paired_cfv_diagnostic.rs"]
mod paired_cfv_diagnostic;
pub use paired_cfv_diagnostic::{paired_cfv_snapshot, PairedCfvState};

#[path = "frozen_river_diagnostic.rs"]
mod frozen_river_diagnostic;
pub use frozen_river_diagnostic::{set_diagnostic_frozen_river_roots, diagnostic_frozen_river_calls,
    set_diagnostic_frozen_river_cached_values, diagnostic_frozen_river_value_cache_stats, set_diagnostic_frozen_river_current_policy};

#[cfg(feature="neural")]
#[path="nn_child_input_traversal.rs"]
mod nn_child_input_traversal;

#[cfg(feature = "custom-alloc")]
use crate::alloc::*;

#[derive(Debug, Clone)]
pub struct CfvCapture {
    pub outer_iteration: u32,
    pub node_id: u64,
    pub player: usize,
    pub values: Vec<f32>,
    /// Opponent reach at this chance node before chance-factor scaling.
    pub opponent_reach: Vec<f32>,
    /// Updating player's reach at the same node and solver state.
    pub player_reach: Vec<f32>,
    /// Dealt turn card, or `NOT_DEALT` for flop-to-turn captures.
    pub turn_card: u8,
    /// Physical chance cards in matrix row order.
    pub chance_cards: Vec<u8>,
    /// Optional estimate of the 49-turn sum in solver-local hand order.
    pub converted_sum: Option<Vec<f64>>,
    /// Solver-local output-player hands mapped to canonical Pio indices.
    pub player_canonical_hands: Vec<usize>,
    /// Solver-local opponent hands mapped to canonical Pio indices.
    pub opponent_canonical_hands: Vec<usize>,
}

static CFV_CAPTURES: OnceLock<Mutex<Vec<CfvCapture>>> = OnceLock::new();
const ALL_CAPTURE_SNAPSHOTS: u64 = u64::MAX;
static CFV_CAPTURE_SNAPSHOTS: AtomicU64 = AtomicU64::new(ALL_CAPTURE_SNAPSHOTS);
static RIVER_CFV_CAPTURE_SNAPSHOTS: AtomicU64 = AtomicU64::new(ALL_CAPTURE_SNAPSHOTS);
static CAPTURE_PLAYER_REACH: AtomicBool = AtomicBool::new(false);
static CAPTURE_ALL_TURN_ITERATIONS: AtomicBool = AtomicBool::new(false);
static CAPTURE_ALL_RIVER_ITERATIONS: AtomicBool = AtomicBool::new(false);
static CHANCE_CHILD_COUNTS: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
// Full traversal is the production/default behavior. Pruning is an explicit
// diagnostic experiment enabled by the benchmark with `set_prune_chance_branches`.
static PRUNE_CHANCE_BRANCHES: AtomicBool = AtomicBool::new(false);
// Opt-in warm-start diagnostic: retain signed flop regrets while downstream
// streets learn. Average-strategy updates and all evaluation remain unchanged.
static HOLD_FLOP_REGRETS: AtomicBool = AtomicBool::new(false);
static HOLD_TURN_REGRETS: AtomicBool = AtomicBool::new(false);
// 2026-11-10 opt-in corridor experiment; default mask0 retains normal solving.
static HOLD_TURN_CARD_MASK: AtomicU64 = AtomicU64::new(0);
pub fn set_diagnostic_hold_turn_cards(cards:&[u8]){let mask=cards.iter().fold(0u64,|m,&c|{assert!(c<52);m|(1u64<<c)});HOLD_TURN_CARD_MASK.store(mask,Ordering::Relaxed);HOLD_TURN_REGRETS.store(!cards.is_empty(),Ordering::Relaxed);}
pub fn diagnostic_average_discount(iteration:u32)->f32{DiscountParams::new(iteration).gamma_t}
pub fn set_diagnostic_hold_turn_policy(enabled: bool) {
    HOLD_TURN_CARD_MASK.store(0,Ordering::Relaxed);
    HOLD_TURN_REGRETS.store(enabled, Ordering::Relaxed);
}

pub fn set_diagnostic_hold_flop_policy(enabled: bool) {
    HOLD_FLOP_REGRETS.store(enabled, Ordering::Relaxed);
}

// Exact called-all-in traversal is the production default. Benchmarks may
// disable it explicitly to measure the neural all-in approximation.
static EXACT_CALLED_ALLINS: AtomicBool = AtomicBool::new(true);
static FLOP_ALLIN_MATRIX_CHECKS: AtomicU64 = AtomicU64::new(0);
static FLOP_ALLIN_MATRIX_MAX_DIFF_BITS: AtomicU64 = AtomicU64::new(0);
static EXACT_ALLIN_MATRIX_EVALS: AtomicU64 = AtomicU64::new(0);
static REPORT_EXACT_ALLIN_RANGE_ZEROS: AtomicBool = AtomicBool::new(false);
static EXACT_ALLIN_RANGE_ZERO_STATS: OnceLock<
    Mutex<BTreeMap<(u64, usize), ExactAllinRangeZeroStat>>,
> = OnceLock::new();
static REGRET_FLOOR_BITS: AtomicU64 = AtomicU64::new(0);
static RECURSION_COUNTS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static TERMINAL_EVAL_NANOS: AtomicU64 = AtomicU64::new(0);
static NEURAL_BATCH_PHASE_NANOS: [AtomicU64; 5] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static ZERO_TERMINAL_PAYOUTS: AtomicBool = AtomicBool::new(false);
static BATCH_NEURAL_TURN_FRONTIERS: AtomicBool = AtomicBool::new(false);
static QRE_INITIAL_TEMPERATURE_BITS: AtomicU64 = AtomicU64::new(0);
static QRE_ANNEAL_ITERATIONS: AtomicU64 = AtomicU64::new(0);
// Opt-in diagnostic: retain the average-strategy epoch anchored at 16 until
// the next normal reset at 256. Regret discounting is unaffected.
static SKIP_STRATEGY_RESET64: AtomicBool = AtomicBool::new(false);
// Explicit warm-transfer experiment: preserve the averaging epoch that existed
// just before this iteration. Regret discounts and default behavior are unchanged.
static PRESERVE_AVERAGE_FROM: AtomicU64 = AtomicU64::new(u64::MAX);
pub fn set_preserve_average_strategy_from(iteration:Option<u32>){
    PRESERVE_AVERAGE_FROM.store(iteration.map(u64::from).unwrap_or(u64::MAX),Ordering::Relaxed);
}


pub fn set_skip_strategy_reset64(enabled: bool) {
    SKIP_STRATEGY_RESET64.store(enabled, Ordering::Relaxed);
}
pub fn set_prune_chance_branches(enabled: bool) {
    PRUNE_CHANCE_BRANCHES.store(enabled, Ordering::Relaxed);
}

pub fn set_exact_called_allins(enabled: bool) {
    EXACT_CALLED_ALLINS.store(enabled, Ordering::Relaxed);
}

pub fn reset_flop_allin_matrix_check_stats() {
    FLOP_ALLIN_MATRIX_CHECKS.store(0, Ordering::Relaxed);
    FLOP_ALLIN_MATRIX_MAX_DIFF_BITS.store(0, Ordering::Relaxed);
}

pub fn flop_allin_matrix_check_stats() -> (u64, f64) {
    (
        FLOP_ALLIN_MATRIX_CHECKS.load(Ordering::Relaxed),
        f64::from_bits(FLOP_ALLIN_MATRIX_MAX_DIFF_BITS.load(Ordering::Relaxed)),
    )
}

pub fn reset_exact_allin_matrix_eval_count() {
    EXACT_ALLIN_MATRIX_EVALS.store(0, Ordering::Relaxed);
}

pub fn exact_allin_matrix_eval_count() -> u64 {
    EXACT_ALLIN_MATRIX_EVALS.load(Ordering::Relaxed)
}

#[derive(Clone, Debug)]
pub struct ExactAllinRangeZeroStat {
    pub node_id: u64,
    pub output_player: usize,
    pub turn_card: u8,
    pub outer_iteration: u32,
    pub hands: usize,
    pub zero_hands: usize,
    pub reach_sum: f64,
}

pub fn set_report_exact_allin_range_zeros(enabled: bool) {
    REPORT_EXACT_ALLIN_RANGE_ZEROS.store(enabled, Ordering::Relaxed);
}

pub fn reset_exact_allin_range_zero_stats() {
    EXACT_ALLIN_RANGE_ZERO_STATS
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .unwrap()
        .clear();
}

pub fn exact_allin_range_zero_stats() -> Vec<ExactAllinRangeZeroStat> {
    EXACT_ALLIN_RANGE_ZERO_STATS
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .unwrap()
        .values()
        .cloned()
        .collect()
}

fn record_exact_allin_range_zeros<T: Game>(
    node: &T::Node,
    output_player: usize,
    outer_iteration: u32,
    cfreach: &[f32],
) {
    if !REPORT_EXACT_ALLIN_RANGE_ZEROS.load(Ordering::Relaxed) {
        return;
    }
    let stat = ExactAllinRangeZeroStat {
        node_id: node.id(),
        output_player,
        turn_card: node.board_turn(),
        outer_iteration,
        hands: cfreach.len(),
        zero_hands: cfreach.iter().filter(|&&value| value == 0.0).count(),
        reach_sum: cfreach.iter().map(|&value| value as f64).sum(),
    };
    EXACT_ALLIN_RANGE_ZERO_STATS
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .unwrap()
        .insert((stat.node_id, stat.output_player), stat);
}

fn record_flop_allin_matrix_check(diff: f64) {
    FLOP_ALLIN_MATRIX_CHECKS.fetch_add(1, Ordering::Relaxed);
    let mut current = FLOP_ALLIN_MATRIX_MAX_DIFF_BITS.load(Ordering::Relaxed);
    loop {
        let current_value = f64::from_bits(current);
        if diff <= current_value {
            break;
        }
        match FLOP_ALLIN_MATRIX_MAX_DIFF_BITS.compare_exchange_weak(
            current,
            diff.to_bits(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(updated) => current = updated,
        }
    }
}

/// Sets a scale-invariant positive floor for regret-matching weights.
/// A value of zero restores ordinary regret matching.
pub fn set_regret_floor_epsilon(epsilon: f32) {
    assert!(epsilon.is_finite() && (0.0..1.0).contains(&epsilon));
    REGRET_FLOOR_BITS.store(epsilon.to_bits() as u64, Ordering::Relaxed);
}

#[inline]
pub(crate) fn regret_floor_epsilon() -> f32 {
    f32::from_bits(REGRET_FLOOR_BITS.load(Ordering::Relaxed) as u32)
}

/// Floors each legal action's positive-regret weight relative to the largest
/// weight for the same hand. This is scale invariant; hands with no positive
/// regret remain uniform, as in ordinary regret matching.
fn apply_regret_floor(strategy: &mut [f32], num_actions: usize, epsilon: f32) {
    if epsilon == 0.0 {
        return;
    }
    let row_size = strategy.len() / num_actions;
    // Separate action slices let LLVM vectorize across contiguous hands. Keep
    // the same maximum reduction order and floor arithmetic as the general path.
    macro_rules! floor_rows {
        ($($row:ident),+) => {
            for hand in 0..row_size {
                let maximum = 0.0f32$(.max($row[hand]))+;
                if maximum > 0.0 {
                    let floor = epsilon * maximum;
                    $($row[hand] = $row[hand].max(floor);)+
                }
            }
        };
    }
    match num_actions {
        2 => {
            let (a, b) = strategy.split_at_mut(row_size);
            floor_rows!(a, b);
            return;
        }
        3 => {
            let (a, rest) = strategy.split_at_mut(row_size);
            let (b, c) = rest.split_at_mut(row_size);
            floor_rows!(a, b, c);
            return;
        }
        4 => {
            let (a, rest) = strategy.split_at_mut(row_size);
            let (b, rest) = rest.split_at_mut(row_size);
            let (c, d) = rest.split_at_mut(row_size);
            floor_rows!(a, b, c, d);
            return;
        }
        _ => {}
    }
    for hand in 0..row_size {
        let maximum = (0..num_actions)
            .map(|action| strategy[action * row_size + hand])
            .fold(0.0f32, f32::max);
        if maximum > 0.0 {
            let floor = epsilon * maximum;
            for action in 0..num_actions {
                let index = action * row_size + hand;
                strategy[index] = strategy[index].max(floor);
            }
        }
    }
}

/// Returns (turn traversed, turn skipped, river traversed, river skipped).
pub fn chance_child_counts() -> [u64; 4] {
    std::array::from_fn(|index| CHANCE_CHILD_COUNTS[index].load(Ordering::Relaxed))
}

pub fn reset_chance_child_counts() {
    for value in &CHANCE_CHILD_COUNTS {
        value.store(0, Ordering::Relaxed);
    }
}
pub fn reset_recursion_counts() {
    for value in &RECURSION_COUNTS {
        value.store(0, Ordering::Relaxed);
    }
    for value in &NEURAL_BATCH_PHASE_NANOS {
        value.store(0, Ordering::Relaxed);
    }
}
/// (terminal evaluations, decision-node visits, chance-node visits)
pub fn recursion_counts() -> [u64; 3] {
    std::array::from_fn(|i| RECURSION_COUNTS[i].load(Ordering::Relaxed))
}
pub fn reset_terminal_timing() {
    TERMINAL_EVAL_NANOS.store(0, Ordering::Relaxed);
}
pub fn terminal_eval_nanos() -> u64 {
    TERMINAL_EVAL_NANOS.load(Ordering::Relaxed)
}
/// Wall time for river collection/inference, turn collection/inference, and flop replay.
pub fn neural_batch_phase_nanos() -> [u64; 5] {
    std::array::from_fn(|i| NEURAL_BATCH_PHASE_NANOS[i].load(Ordering::Relaxed))
}
pub fn set_zero_terminal_payouts(enabled: bool) {
    ZERO_TERMINAL_PAYOUTS.store(enabled, Ordering::Relaxed);
}
pub fn set_batch_neural_turn_frontiers(enabled: bool) {
    BATCH_NEURAL_TURN_FRONTIERS.store(enabled, Ordering::Relaxed);
}

/// Enables an experimental quantal-response strategy during the first
/// `anneal_iterations` solver iterations. A temperature of zero disables it.
pub fn set_qre_annealing(initial_temperature: f32, anneal_iterations: u32) {
    assert!(initial_temperature.is_finite() && initial_temperature >= 0.0);
    assert!(initial_temperature == 0.0 || anneal_iterations > 0);
    QRE_INITIAL_TEMPERATURE_BITS.store(initial_temperature.to_bits() as u64, Ordering::Relaxed);
    QRE_ANNEAL_ITERATIONS.store(anneal_iterations as u64, Ordering::Relaxed);
}

pub(crate) fn qre_annealing_enabled() -> bool {
    QRE_INITIAL_TEMPERATURE_BITS.load(Ordering::Relaxed) != 0
}

fn capture_snapshot_bit(iteration: u32) -> u64 {
    match iteration {
        5 => 1 << 0,
        10 => 1 << 1,
        25 => 1 << 2,
        50 => 1 << 3,
        100 => 1 << 4,
        180 => 1 << 5,
        200 => 1 << 6,
        500 => 1 << 7,
        1000 => 1 << 8,
        20 => 1 << 9,
        30 => 1 << 10,
        150 => 1 << 11,
        40 => 1 << 12,
        60 => 1 << 13,
        300 => 1 << 14,
        400 => 1 << 15,
        45 => 1 << 16,
        55 => 1 << 17,
        41..=59 => 1u64 << (18 + iteration - 41),
        _ => 0,
    }
}

#[cfg(test)]
#[test]
fn capture_snapshot_masks_include_late_five_without_aliasing_legacy() {
    let legacy=[5,10,25,50,100,180,200,500,1000,20,30,150,40,60,300,400];
    for (bit,iteration) in legacy.into_iter().enumerate() {
        assert_eq!(capture_snapshot_bit(iteration),1u64<<bit);
    }
    let requested=[40,45,50,55,60];
    let mask=requested.into_iter().fold(0,|mask,i|mask|capture_snapshot_bit(i));
    assert_eq!(mask.count_ones(),5);
    for i in requested {assert_ne!(mask&capture_snapshot_bit(i),0);}
    assert_eq!(mask&capture_snapshot_bit(30),0);
    assert_eq!(capture_snapshot_bit(39),0);
}

#[cfg(test)]
#[test]
fn capture_snapshot_masks_support_every_iteration_40_through_60() {
    let mask = (40..=60).fold(0u64, |mask, i| mask | capture_snapshot_bit(i));
    assert_eq!(mask.count_ones(), 21);
    for i in 40..=60 { assert_ne!(mask & capture_snapshot_bit(i), 0); }
    for i in [5, 10, 20, 25, 30, 100, 150, 180, 200, 300, 400, 500, 1000] {
        assert_eq!(mask & capture_snapshot_bit(i), 0);
    }
}

/// Selects which standard diagnostic snapshots are retained in memory.
pub fn set_cfv_capture_snapshots(snapshots: &[u32]) {
    let mask = snapshots
        .iter()
        .fold(0, |mask, &iteration| mask | capture_snapshot_bit(iteration));
    assert_eq!(mask.count_ones() as usize, snapshots.len());
    CFV_CAPTURE_SNAPSHOTS.store(mask, Ordering::Relaxed);
}

/// Selects river diagnostic snapshots independently of flop-to-turn captures.
pub fn set_river_cfv_capture_snapshots(snapshots: &[u32]) {
    let mask = snapshots
        .iter()
        .fold(0, |mask, &iteration| mask | capture_snapshot_bit(iteration));
    assert_eq!(mask.count_ones() as usize, snapshots.len());
    RIVER_CFV_CAPTURE_SNAPSHOTS.store(mask, Ordering::Relaxed);
}

/// Enables propagation of the updating player's reach for diagnostic exports.
/// Disabled by default so normal solving pays no reach-propagation cost.
pub fn set_cfv_capture_player_reach(enabled: bool) {
    CAPTURE_PLAYER_REACH.store(enabled, Ordering::Relaxed);
}

/// Captures every flop-to-turn diagnostic iteration instead of the fixed
/// snapshot set. Intended only for short input-drift experiments.
pub fn set_capture_all_turn_iterations(enabled: bool) {
    CAPTURE_ALL_TURN_ITERATIONS.store(enabled, Ordering::Relaxed);
}

/// Captures every turn-to-river diagnostic iteration instead of the fixed
/// snapshot set. Intended only for short input-drift experiments.
pub fn set_capture_all_river_iterations(enabled: bool) {
    CAPTURE_ALL_RIVER_ITERATIONS.store(enabled, Ordering::Relaxed);
}

/// Clears raw diagnostic captures.
pub fn clear_cfv_captures() {
    CFV_CAPTURES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap()
        .clear();
}

/// Returns raw diagnostic captures collected at the configured turn node.
pub fn take_cfv_captures() -> Vec<CfvCapture> {
    std::mem::take(
        &mut *CFV_CAPTURES
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .unwrap(),
    )
}

fn capture_cfv_actions(
    node_id: u64,
    outer_iteration: u32,
    player: usize,
    values: &[f32],
    opponent_reach: &[f32],
    player_reach: Option<&[f32]>,
    turn_card: u8,
    chance_cards: Vec<u8>,
    conversion: Option<&crate::pio_order::TurnCfvConversion>,
    compact_basis_actions: Option<&[usize]>,
) {
    let snapshot_mask = if turn_card == NOT_DEALT {
        CFV_CAPTURE_SNAPSHOTS.load(Ordering::Relaxed)
    } else {
        RIVER_CFV_CAPTURE_SNAPSHOTS.load(Ordering::Relaxed)
    };
    let capture_all_turns =
        turn_card == NOT_DEALT && CAPTURE_ALL_TURN_ITERATIONS.load(Ordering::Relaxed);
    let capture_all_rivers =
        turn_card != NOT_DEALT && CAPTURE_ALL_RIVER_ITERATIONS.load(Ordering::Relaxed);
    if capture_all_turns
        || capture_all_rivers
        || snapshot_mask & capture_snapshot_bit(outer_iteration) != 0
    {
        assert!(values.iter().all(|value| value.is_finite()));
        #[cfg(feature = "neural")]
        let (player_canonical_hands, opponent_canonical_hands) = conversion
            .and_then(|matrix| matrix.neural_model.as_ref())
            .map_or_else(
                || (Vec::new(), Vec::new()),
                |model| {
                    (
                        model.mappings[player].clone(),
                        model.mappings[player ^ 1].clone(),
                    )
                },
            );
        #[cfg(not(feature = "neural"))]
        let (player_canonical_hands, opponent_canonical_hands) = (Vec::new(), Vec::new());
        CFV_CAPTURES
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .unwrap()
            .push(CfvCapture {
                node_id,
                outer_iteration,
                player,
                values: values.to_vec(),
                opponent_reach: opponent_reach.to_vec(),
                player_reach: player_reach.unwrap_or_default().to_vec(),
                turn_card,
                chance_cards,
                converted_sum: conversion.map(|matrix| match compact_basis_actions {
                    Some(actions) => matrix.estimate_sum_from_basis_at_iteration(
                        node_id,
                        player,
                        values,
                        actions,
                        outer_iteration,
                        opponent_reach,
                        player_reach,
                    ),
                    None => matrix.estimate_sum_at_iteration(
                        node_id,
                        player,
                        values,
                        outer_iteration,
                        opponent_reach,
                        player_reach,
                    ),
                }),
                player_canonical_hands,
                opponent_canonical_hands,
            });
    }
}

fn sum_chance_cfvs(
    result: &mut [MaybeUninit<f64>],
    values: &[f32],
    node_id: u64,
    player: usize,
    outer_iteration: u32,
    conversion: Option<&crate::pio_order::TurnCfvConversion>,
    opponent_reach: &[f32],
    player_reach: Option<&[f32]>,
) {
    if let Some(matrix) = conversion.filter(|matrix| matrix.use_in_solving()) {
        let estimate = matrix.estimate_sum_at_iteration(
            node_id,
            player,
            values,
            outer_iteration,
            opponent_reach,
            player_reach,
        );
        assert_eq!(result.len(), estimate.len());
        for (output, value) in result.iter_mut().zip(estimate) {
            output.write(value);
        }
    } else {
        sum_slices_f64_uninit(result, values);
    }
}

/// Read-only average-strategy discount used by the current DCFR clock.
/// Allows an external gadget root to use the same averaging schedule.
#[doc(hidden)]
pub fn diagnostic_dcfr_average_discount(current_iteration:u32)->f32{DiscountParams::new(current_iteration).gamma_t}

struct DiscountParams {
    alpha_t: f32,
    beta_t: f32,
    gamma_t: f32,
    qre_temperature: f32,
}

impl DiscountParams {
    pub fn new(current_iteration: u32) -> Self {
        let start=PRESERVE_AVERAGE_FROM.load(Ordering::Relaxed);
        Self::with_average_epoch_policy(current_iteration, SKIP_STRATEGY_RESET64.load(Ordering::Relaxed),if start==u64::MAX{None}else{Some(start as u32)})
    }

    fn with_reset64_policy(current_iteration: u32, skip_reset64: bool) -> Self {
        Self::with_average_epoch_policy(current_iteration,skip_reset64,None)
    }
    fn with_average_epoch_policy(current_iteration:u32,skip_reset64:bool,preserve_from:Option<u32>)->Self{
        // 0, 1, 4, 16, 64, 256, ...
        let mut nearest_lower_power_of_4 = match current_iteration {
            0 => 0,
            x => 1 << ((x.leading_zeros() ^ 31) & !1),
        };
        if skip_reset64 && nearest_lower_power_of_4 == 64 {
            nearest_lower_power_of_4 = 16;
        }

        if let Some(start)=preserve_from{
            if current_iteration>=start{
                let before=start.saturating_sub(1);
                nearest_lower_power_of_4=if before==0{0}else{1 << ((before.leading_zeros() ^ 31) & !1)};
            }
        }
        let t_alpha = (current_iteration as i32 - 1).max(0) as f64;
        let t_gamma = (current_iteration - nearest_lower_power_of_4) as f64;

        // Explicit experimental overrides; the default arithmetic remains identical.
        static EXPONENTS: OnceLock<(f64, f64, f64)> = OnceLock::new();
        let &(alpha, beta, gamma) = EXPONENTS.get_or_init(|| {
            let read = |name: &str, default: f64| {
                std::env::var(name).map(|s| {
                    let x: f64 = s.parse().expect("invalid DCFR exponent");
                    assert!(x.is_finite(), "DCFR exponent must be finite");
                    x
                }).unwrap_or(default)
            };
            (read("PFS_DCFR_ALPHA_EXPERIMENT", 1.5),
             read("PFS_DCFR_BETA_EXPERIMENT", 0.0),
             read("PFS_DCFR_GAMMA_EXPERIMENT", 3.0))
        });
        let pow_alpha = if alpha == 1.5 { t_alpha * t_alpha.sqrt() } else { t_alpha.powf(alpha) };
        let pow_gamma = if gamma == 3.0 { (t_gamma / (t_gamma + 1.0)).powi(3) }
            else { assert!(gamma > 0.0); (t_gamma / (t_gamma + 1.0)).powf(gamma) };
        let beta_discount = if beta == 0.0 { 0.5 } else {
            let power = t_alpha.max(1.0).powf(beta);
            power / (power + 1.0)
        };

        Self {
            alpha_t: (pow_alpha / (pow_alpha + 1.0)) as f32,
            beta_t: beta_discount as f32,
            gamma_t: pow_gamma as f32,
            qre_temperature: {
                let initial =
                    f32::from_bits(QRE_INITIAL_TEMPERATURE_BITS.load(Ordering::Relaxed) as u32);
                let anneal = QRE_ANNEAL_ITERATIONS.load(Ordering::Relaxed) as u32;
                if initial > 0.0 && current_iteration < anneal {
                    let remaining = 1.0 - current_iteration as f32 / anneal as f32;
                    initial * remaining * remaining
                } else {
                    0.0
                }
            },
        }
    }
}

/// Performs Discounted CFR algorithm until the given number of iterations or exploitability is
/// satisfied.
///
/// This method returns the exploitability of the obtained strategy.
pub fn solve<T: Game>(
    game: &mut T,
    max_num_iterations: u32,
    target_exploitability: f32,
    print_progress: bool,
) -> f32 {
    if game.is_solved() {
        panic!("Game is already solved");
    }

    if !game.is_ready() {
        panic!("Game is not ready");
    }

    let mut root = game.root();
    let mut exploitability = compute_exploitability(game);

    if print_progress {
        print!("iteration: 0 / {max_num_iterations} ");
        print!("(exploitability = {exploitability:.4e})");
        io::stdout().flush().unwrap();
    }

    for t in 0..max_num_iterations {
        if exploitability <= target_exploitability {
            break;
        }

        let params = DiscountParams::new(t);

        // alternating updates
        for player in 0..2 {
            let mut result = Vec::with_capacity(game.num_private_hands(player));
            solve_player_update(
                result.spare_capacity_mut(),
                game,
                &mut root,
                player,
                game.initial_weights(player ^ 1),
                &params,
                t + 1,
            );
        }

        if (t + 1) % 10 == 0 || t + 1 == max_num_iterations {
            exploitability = compute_exploitability(game);
        }

        if print_progress {
            print!("\riteration: {} / {} ", t + 1, max_num_iterations);
            print!("(exploitability = {exploitability:.4e})");
            io::stdout().flush().unwrap();
        }
    }

    if print_progress {
        println!();
        io::stdout().flush().unwrap();
    }

    finalize(game);

    exploitability
}

/// Proceeds Discounted CFR algorithm for one iteration.
#[inline]
pub fn solve_step<T: Game>(game: &T, current_iteration: u32) {
    let _profile = operation_profile::enter(0);
    if game.is_solved() {
        panic!("Game is already solved");
    }

    if !game.is_ready() {
        panic!("Game is not ready");
    }

    let mut root = game.root();
    let params = DiscountParams::new(current_iteration);

    // alternating updates
    for player in 0..2 {
        let mut result = Vec::with_capacity(game.num_private_hands(player));
        solve_player_update(
            result.spare_capacity_mut(),
            game,
            &mut root,
            player,
            game.initial_weights(player ^ 1),
            &params,
            current_iteration + 1,
        );
    }
}

/// Experimental warm continuation of one decision subtree with fixed incoming
/// reaches. Everything outside `path` is untouched. Reaches must use the same
/// private-hand order and chance scaling as the original traversal. Retains
/// regrets, cumulative strategies and the caller's global DCFR iteration clock.
/// This is an unsafe (in the game-theoretic sense) local solve, not a safe
/// re-solving gadget or a whole-game exploitability calculation.
#[doc(hidden)]
pub fn solve_subgame_step<T: Game>(
    game: &T,
    path: &[usize],
    frozen_reaches: [&[f32]; 2],
    current_iteration: u32,
) {
    assert!(game.is_ready() && !game.is_solved());
    assert!(!path.is_empty(), "local continuation requires a non-root subtree");
    for player in 0..2 {
        assert_eq!(frozen_reaches[player].len(), game.num_private_hands(player));
        assert!(frozen_reaches[player].iter().all(|v| v.is_finite() && *v >= 0.));
    }
    fn descend<T: Game>(game: &T, node: &mut T::Node, path: &[usize],
                        reaches: [&[f32]; 2], params: &DiscountParams, iteration: u32) {
        if let Some((&action, rest)) = path.split_first() {
            assert!(action < node.num_actions());
            descend(game, &mut node.play(action), rest, reaches, params, iteration);
        } else {
            assert!(!node.is_terminal() && !node.is_chance());
            for player in 0..2 {
                let mut result = Vec::with_capacity(game.num_private_hands(player));
                solve_recursive(result.spare_capacity_mut(), game, node, player,
                    reaches[player ^ 1], Some(reaches[player]), params, iteration + 1,
                    true, true, false);
            }
        }
    }
    descend(game, &mut game.root(), path, frozen_reaches,
            &DiscountParams::new(current_iteration), current_iteration);
}

/// Experimental local CFR traversal returning each alternating update's native
/// counterfactual values. No EV normalization or own-hand reach division is
/// applied. The two vectors belong to successive alternating policy states.
/// Used by an external CFR-D enter/terminate gadget; does not add a gadget itself.
#[doc(hidden)]
pub fn solve_subgame_step_with_values<T: Game>(
    game: &T,
    path: &[usize],
    frozen_reaches: [&[f32]; 2],
    current_iteration: u32,
) -> [Vec<f32>; 2] {
    assert!(game.is_ready() && !game.is_solved());
    assert!(!path.is_empty(), "local continuation requires a non-root subtree");
    for player in 0..2 {
        assert_eq!(frozen_reaches[player].len(), game.num_private_hands(player));
        assert!(frozen_reaches[player].iter().all(|v| v.is_finite() && *v >= 0.));
    }
    fn descend<T: Game>(game: &T, node: &mut T::Node, path: &[usize],
                        reaches: [&[f32]; 2], params: &DiscountParams, iteration: u32) -> [Vec<f32>;2] {
        if let Some((&action, rest)) = path.split_first() {
            assert!(action < node.num_actions());
            descend(game, &mut node.play(action), rest, reaches, params, iteration)
        } else {
            assert!(!node.is_terminal() && !node.is_chance());
            let mut values: [Vec<f32>;2] = std::array::from_fn(|_|Vec::new());
            for player in 0..2 {
                let mut result = Vec::with_capacity(game.num_private_hands(player));
                solve_recursive(result.spare_capacity_mut(), game, node, player,
                    reaches[player ^ 1], Some(reaches[player]), params, iteration + 1,
                    true, true, false);
                unsafe {result.set_len(game.num_private_hands(player))};
                values[player]=result;
            }
            values
        }
    }
    descend(game, &mut game.root(), path, frozen_reaches,
            &DiscountParams::new(current_iteration), current_iteration)
}

fn solve_player_update<T: Game>(
    result: &mut [MaybeUninit<f32>],
    game: &T,
    root: &mut T::Node,
    player: usize,
    cfreach: &[f32],
    params: &DiscountParams,
    outer_iteration: u32,
) {
    let player_reach = (game.requires_player_reach()
        || CAPTURE_ALL_TURN_ITERATIONS.load(Ordering::Relaxed)
        || CAPTURE_ALL_RIVER_ITERATIONS.load(Ordering::Relaxed)
        || (CAPTURE_PLAYER_REACH.load(Ordering::Relaxed)
            && CFV_CAPTURE_SNAPSHOTS.load(Ordering::Relaxed)
                & capture_snapshot_bit(outer_iteration)
                != 0))
        .then(|| game.initial_weights(player));
    let batching = BATCH_NEURAL_TURN_FRONTIERS.load(Ordering::Relaxed);
    let river_batched = batching && game.begin_neural_river_batch();
    if river_batched {
        let started = std::time::Instant::now();
        solve_recursive(
            result,
            game,
            root,
            player,
            cfreach,
            player_reach,
            params,
            outer_iteration,
            false,
            false,
            false,
        );
        NEURAL_BATCH_PHASE_NANOS[0]
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let started = std::time::Instant::now();
        game.finish_neural_turn_batch(player);
        NEURAL_BATCH_PHASE_NANOS[1]
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        // River models use a whole-range batch. Keep the shallower turn model
        // on its direct path during the replay; legacy turn exports squeeze the
        // batch dimension and cannot share this river batch safely.
        let turn_batched = false;
        let started = std::time::Instant::now();
        solve_recursive(
            result,
            game,
            root,
            player,
            cfreach,
            player_reach,
            params,
            outer_iteration,
            !turn_batched,
            false,
            true,
        );
        NEURAL_BATCH_PHASE_NANOS[2]
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if turn_batched {
            let started = std::time::Instant::now();
            game.finish_neural_turn_batch(player);
            NEURAL_BATCH_PHASE_NANOS[3]
                .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            let started = std::time::Instant::now();
            solve_recursive(
                result,
                game,
                root,
                player,
                cfreach,
                player_reach,
                params,
                outer_iteration,
                true,
                false,
                true,
            );
            NEURAL_BATCH_PHASE_NANOS[4]
                .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        game.end_neural_turn_batch();
        return;
    }

    let turn_batched = batching && game.begin_neural_turn_batch();
    if turn_batched {
        let started = std::time::Instant::now();
        solve_recursive(
            result,
            game,
            root,
            player,
            cfreach,
            player_reach,
            params,
            outer_iteration,
            false,
            false,
            false,
        );
        NEURAL_BATCH_PHASE_NANOS[2]
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let started = std::time::Instant::now();
        game.finish_neural_turn_batch(player);
        NEURAL_BATCH_PHASE_NANOS[3]
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let started = std::time::Instant::now();
    solve_recursive(
        result,
        game,
        root,
        player,
        cfreach,
        player_reach,
        params,
        outer_iteration,
        true,
        false,
        turn_batched,
    );
    if turn_batched {
        NEURAL_BATCH_PHASE_NANOS[4]
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        game.end_neural_turn_batch();
    }
}

// Solve this continuation once per updating-player pass, under the same live
// flop reach history. Its return value deliberately has no parent consumer.
fn solve_auxiliary_turn<T: Game>(
    game: &T, node: &mut T::Node, player: usize, cfreach: &[f32],
    player_reach: Option<&[f32]>, params: &DiscountParams, outer_iteration: u32,
    update: bool,
) {
    if !update { return; }
    let Some(action) = game.auxiliary_chance_action(node) else { return; };
    let factor = 1.0 / game.chance_factor(node) as f32;
    let opponent: Vec<_> = cfreach.iter().map(|value| value * factor).collect();
    let own: Option<Vec<_>> = player_reach.map(|reach| reach.iter().map(|value| value * factor).collect());
    let mut discarded = Vec::with_capacity(game.num_private_hands(player));
    solve_recursive(
        discarded.spare_capacity_mut(), game, &mut node.play(action), player,
        &opponent, own.as_deref(), params, outer_iteration, true, true, false,
    );
}

/// Recursively solves the counterfactual values.
fn solve_recursive<T: Game>(
    result: &mut [MaybeUninit<f32>], game: &T, node: &mut T::Node,
    player: usize, cfreach: &[f32], player_reach: Option<&[f32]>,
    params: &DiscountParams, outer_iteration: u32, update: bool,
    below_batch_frontier: bool, use_cached_predictions: bool,
) {
    if direct_terminal_dispatch::enabled() && !game.is_compression_enabled()
        && !BATCH_NEURAL_TURN_FRONTIERS.load(Ordering::Relaxed)
        && !HOLD_FLOP_REGRETS.load(Ordering::Relaxed)
        && !HOLD_TURN_REGRETS.load(Ordering::Relaxed)
        && !node.is_chance() && !node.is_terminal()
        && direct_terminal_dispatch::try_solve(result, game, node, player,
            cfreach, player_reach, params, outer_iteration, update, below_batch_frontier,
            use_cached_predictions)
    { return; }
    if leaf_path_reach::enabled() && !game.is_compression_enabled()
        && !BATCH_NEURAL_TURN_FRONTIERS.load(Ordering::Relaxed)
        && !HOLD_FLOP_REGRETS.load(Ordering::Relaxed)
        && !HOLD_TURN_REGRETS.load(Ordering::Relaxed)
        && !node.is_chance() && !node.is_terminal()
    {
        let opponent = leaf_path_reach::Reach::Initial(cfreach);
        let own = player_reach.map(leaf_path_reach::Reach::Initial);
        leaf_path_reach::solve(result, game, node, player, &opponent,
            own.as_ref(), params, outer_iteration, update, below_batch_frontier,
            use_cached_predictions);
    } else {
        solve_recursive_standard(result, game, node, player, cfreach, player_reach,
            params, outer_iteration, update, below_batch_frontier, use_cached_predictions);
    }
}

fn solve_recursive_standard<T: Game>(
    result: &mut [MaybeUninit<f32>],
    game: &T,
    node: &mut T::Node,
    player: usize,
    cfreach: &[f32],
    player_reach: Option<&[f32]>,
    params: &DiscountParams,
    outer_iteration: u32,
    update: bool,
    below_batch_frontier: bool,
    use_cached_predictions: bool,
) {
    let _profile = operation_profile::enter(0);
    // return the counterfactual values when the `node` is terminal
    if node.is_terminal() {
        let _terminal_profile = operation_profile::enter(1);
        RECURSION_COUNTS[0].fetch_add(1, Ordering::Relaxed);
        let started = std::time::Instant::now();
        if ZERO_TERMINAL_PAYOUTS.load(Ordering::Relaxed) {
            for value in result.iter_mut() {
                value.write(0.0);
            }
        } else {
            game.evaluate(result, node, player, cfreach);
        }
        TERMINAL_EVAL_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        return;
    }

    // A chance node after a called all-in has no remaining strategic choices.
    // Traverse every physical runout so its showdown payout is always exact,
    // even when a neural or compact chance conversion is active elsewhere.
    if frozen_river_diagnostic::try_evaluate(result, game, node, player, cfreach) {
        return;
    }

    let exact_chance = EXACT_CALLED_ALLINS.load(Ordering::Relaxed)
        && node.is_chance()
        && game.should_traverse_chance_exactly(node);

    // An exact matrix leaf replaces only a called flop all-in chance subtree.
    // The opponent reach is still unscaled here; the matrix already averages
    // the complete 45x44 ordered runout distribution.
    let capture_allin_runouts = exact_chance && node.num_actions() > 0 && game.capture_turn_cfvs(node);
    if exact_chance && !capture_allin_runouts && {
        let _allin_profile = operation_profile::enter(2);
        game.evaluate_called_allin(result, node, player, cfreach)
    } {
        record_exact_allin_range_zeros::<T>(node, player, outer_iteration, cfreach);
        EXACT_ALLIN_MATRIX_EVALS.fetch_add(1, Ordering::Relaxed);
        return;
    }

    // In the batched update pass, a cached turn-frontier value is a leaf.
    // Return before counting or inspecting the chance node and never recurse
    // into any of its physical-card children.
    #[cfg(feature = "neural")]
    if use_cached_predictions && node.is_chance() && !exact_chance {
        let active_conversion = game
            .turn_cfv_conversion(node)
            .filter(|conversion| conversion.use_in_solving());
        if let Some(prediction) = active_conversion.and_then(|conversion| {
            conversion.cached_neural_batch_prediction(node.id(), player, outer_iteration)
        }) {
            solve_auxiliary_turn(game, node, player, cfreach, player_reach, params, outer_iteration, update);
            if update && game.auxiliary_chance_action(node).is_some() {
                CHANCE_CHILD_COUNTS[0].fetch_add(1, Ordering::Relaxed);
            }
            assert_eq!(prediction.len(), result.len());
            for (output, value) in result.iter_mut().zip(prediction) {
                output.write(value as f32);
            }
            return;
        }
    }

    let num_actions = node.num_actions();
    let num_hands = result.len();
    if node.is_chance() {
        RECURSION_COUNTS[2].fetch_add(1, Ordering::Relaxed);
    } else {
        RECURSION_COUNTS[1].fetch_add(1, Ordering::Relaxed);
    }

    // simply recurse when the number of actions is one
    if num_actions == 1 && !node.is_chance() {
        let child = &mut node.play(0);
        solve_recursive(
            result,
            game,
            child,
            player,
            cfreach,
            player_reach,
            params,
            outer_iteration,
            update,
            below_batch_frontier,
            use_cached_predictions,
        );
        return;
    }

    let active_conversion = if node.is_chance() && !exact_chance {
        game.turn_cfv_conversion(node).filter(|matrix| {
            PRUNE_CHANCE_BRANCHES.load(Ordering::Relaxed) && matrix.use_in_solving()
        })
    } else {
        None
    };
    let compact_basis = active_conversion.map(|matrix| matrix.basis_actions());
    let storage_actions = compact_basis.map_or(num_actions, |basis| basis.len());
    // allocate memory for storing the counterfactual values
    #[cfg(feature = "custom-alloc")]
    let cfv_actions = MutexLike::new(Vec::with_capacity_in(
        storage_actions * num_hands,
        StackAlloc,
    ));
    #[cfg(not(feature = "custom-alloc"))]
    let cfv_actions = {
        let _allocation_profile = operation_profile::enter(8);
        cfr_action_buffer::ActionBuffer::new(node.id(), storage_actions * num_hands)
    };

    // if the `node` is chance
    if node.is_chance() {
        // update the reach probabilities
        let reach_profile = operation_profile::enter(4);
        #[cfg(feature = "custom-alloc")]
        let mut cfreach_updated = Vec::with_capacity_in(cfreach.len(), StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut cfreach_updated = Vec::with_capacity(cfreach.len());
        mul_slice_scalar_uninit(
            cfreach_updated.spare_capacity_mut(),
            cfreach,
            1.0 / game.chance_factor(node) as f32,
        );
        unsafe { cfreach_updated.set_len(cfreach.len()) };
        let player_reach_updated = player_reach.map(|reach| {
            let mut updated = Vec::with_capacity(reach.len());
            mul_slice_scalar_uninit(
                updated.spare_capacity_mut(),
                reach,
                1.0 / game.chance_factor(node) as f32,
            );
            unsafe { updated.set_len(reach.len()) };
            updated
        });

        drop(reach_profile);
        let conversion = (!exact_chance)
            .then(|| game.turn_cfv_conversion(node))
            .flatten();
        let basis_actions = conversion
            .filter(|matrix| {
                PRUNE_CHANCE_BRANCHES.load(Ordering::Relaxed) && matrix.use_in_solving()
            })
            .map(|matrix| matrix.basis_actions());
        let enters_batch_frontier =
            conversion.is_some_and(|matrix| matrix.neural_batch_is_collecting());
        let pruned = basis_actions.is_some();
        if let Some(extra) = game.auxiliary_chance_action(node) {
            assert!(basis_actions.is_some_and(|basis| !basis.contains(&extra)),
                "Auxiliary exact turn requires an active pruned turn conversion excluding its card");
            solve_auxiliary_turn(game, node, player, cfreach, player_reach, params, outer_iteration, update);
        }
        let physical = node.num_actions();
        let is_turn = node.board_turn() == NOT_DEALT;
        let selected = basis_actions.map_or(physical, |actions| actions.len()
            + usize::from(update && game.auxiliary_chance_action(node).is_some()));
        CHANCE_CHILD_COUNTS[if is_turn { 0 } else { 2 }]
            .fetch_add(selected as u64, Ordering::Relaxed);
        CHANCE_CHILD_COUNTS[if is_turn { 1 } else { 3 }]
            .fetch_add((physical - selected) as u64, Ordering::Relaxed);
        #[cfg(feature="neural")]
        if let Some(actions)=basis_actions {
            if nn_child_input_traversal::try_solve(result,game,node,conversion.unwrap(),actions,player,
                cfreach,player_reach,&cfreach_updated,player_reach_updated.as_deref(),params,outer_iteration,
                update,below_batch_frontier||enters_batch_frontier,use_cached_predictions) {
                return;
            }
        }
        if let Some(basis_actions) = basis_actions {
            for (compact_index, &action) in basis_actions.iter().enumerate() {
                solve_recursive(
                    row_mut(
                        cfv_actions.lock().spare_capacity_mut(),
                        compact_index,
                        num_hands,
                    ),
                    game,
                    &mut node.play(action),
                    player,
                    &cfreach_updated,
                    player_reach_updated.as_deref(),
                    params,
                    outer_iteration,
                    update,
                    below_batch_frontier || enters_batch_frontier,
                    use_cached_predictions,
                );
            }
        } else {
            for_each_child(node, |action| {
                solve_recursive(
                    row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                    game,
                    &mut node.play(action),
                    player,
                    &cfreach_updated,
                    player_reach_updated.as_deref(),
                    params,
                    outer_iteration,
                    update,
                    below_batch_frontier || enters_batch_frontier,
                    use_cached_predictions,
                );
            });
        }

        let _chance_profile = operation_profile::enter(6);
        // use 64-bit floating point values
        #[cfg(feature = "custom-alloc")]
        let mut result_f64 = Vec::with_capacity_in(num_hands, StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut result_f64 = Vec::with_capacity(num_hands);

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(storage_actions * num_hands) };
        if game.capture_turn_cfvs(node) {
            assert!(node.is_chance());
            let physical_cards = node.chance_cards();
            assert_eq!(physical_cards.len(), num_actions);
            let chance_cards = if let Some(actions) = basis_actions {
                assert_eq!(cfv_actions.len(), actions.len() * num_hands);
                actions
                    .iter()
                    .map(|&action| physical_cards[action])
                    .collect()
            } else {
                assert_eq!(cfv_actions.len(), num_actions * num_hands);
                physical_cards
            };
            let expanded=(!game.isomorphic_chances(node).is_empty()).then(||{
                assert!(basis_actions.is_none());game.expand_physical_chance_values(node,player,&cfv_actions)
            });
            let capture_values=expanded.as_ref().map_or(&cfv_actions[..],|x|x.1.as_slice());
            let chance_cards=expanded.as_ref().map_or(chance_cards,|x|x.0.clone());
            capture_cfv_actions(
                node.id(),
                outer_iteration,
                player,
                capture_values,
                cfreach,
                player_reach,
                node.board_turn(),
                chance_cards,
                (!pruned && !exact_chance)
                    .then(|| game.turn_cfv_conversion(node))
                    .flatten(),
                basis_actions,
            );
        }

        // A collection pass only needs to descend through the frontier that is
        // currently collecting. Values above that frontier are placeholders:
        // the progressively shallower replay computes them after inference.
        if !update && !below_batch_frontier && !enters_batch_frontier {
            for output in result.iter_mut() {
                output.write(0.0);
            }
            return;
        }

        let conversion = (!exact_chance)
            .then(|| game.turn_cfv_conversion(node))
            .flatten();
        if conversion.is_some_and(|matrix| matrix.use_in_solving()) {
            assert!(game.isomorphic_chances(node).is_empty());
        }
        if let Some(basis_actions) = basis_actions {
            let _frontier_profile = operation_profile::enter(7);
            let estimate = conversion
                .expect("active compact conversion")
                .estimate_sum_from_basis_at_iteration(
                    node.id(),
                    player,
                    &cfv_actions,
                    basis_actions,
                    outer_iteration,
                    cfreach,
                    player_reach,
                );
            for (output, value) in result_f64.spare_capacity_mut().iter_mut().zip(estimate) {
                output.write(value);
            }
            unsafe {
                result_f64.set_len(num_hands);
            }
        } else {
            sum_chance_cfvs(
                result_f64.spare_capacity_mut(),
                &cfv_actions,
                node.id(),
                player,
                outer_iteration,
                conversion,
                cfreach,
                player_reach,
            );
            unsafe { result_f64.set_len(num_hands) };
        }

        // get information about isomorphic chances
        let isomorphic_chances = if pruned {
            &[][..]
        } else {
            game.isomorphic_chances(node)
        };

        // process isomorphic chances
        for (i, &isomorphic_index) in isomorphic_chances.iter().enumerate() {
            let swap_list = &game.isomorphic_swap(node, i)[player];
            let tmp = row_mut(&mut cfv_actions, isomorphic_index as usize, num_hands);

            apply_swap(tmp, swap_list);

            result_f64.iter_mut().zip(&*tmp).for_each(|(r, &v)| {
                *r += v as f64;
            });

            apply_swap(tmp, swap_list);
        }

        if exact_chance {
            if let Some(diff) = game.compare_called_allin(&result_f64, node, player, cfreach) {
                record_flop_allin_matrix_check(diff);
            }
        }

        result.iter_mut().zip(&result_f64).for_each(|(r, &v)| {
            r.write(v as f32);
        });
        // Keep physical per-card labels, but return the selected exact matrix
        // backend to the regret update just as in a non-capturing solve.
        if capture_allin_runouts && game.evaluate_called_allin(result, node, player, cfreach) {
            EXACT_ALLIN_MATRIX_EVALS.fetch_add(1, Ordering::Relaxed);
        }
    }
    // if the current player is `player`
    else if node.player() == player {
        let player_reach_actions = player_reach.map(|reach| {
            let mut actions = if game.is_compression_enabled() {
                regret_matching_compressed(
                    node.regrets_compressed(),
                    num_actions,
                    params.qre_temperature,
                )
            } else {
                regret_matching(node.regrets(), num_actions, params.qre_temperature)
            };
            apply_locking_strategy(&mut actions, game.locking_strategy(node));
            actions
                .chunks_exact_mut(reach.len())
                .for_each(|row| mul_slice(row, reach));
            actions
        });
        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            solve_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                cfreach,
                player_reach_actions
                    .as_deref()
                    .map(|actions| row(actions, action, num_hands)),
                params,
                outer_iteration,
                update,
                below_batch_frontier,
                use_cached_predictions,
            );
        });

        // compute the strategy by regret-maching algorithm
        let mut strategy = if game.is_compression_enabled() {
            regret_matching_compressed(
                node.regrets_compressed(),
                num_actions,
                params.qre_temperature,
            )
        } else {
            regret_matching(node.regrets(), num_actions, params.qre_temperature)
        };

        // node-locking
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut strategy, locking);

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        let result = {
            let _sum_profile = operation_profile::enter(9);
            fma_slices_uninit(result, &strategy, &cfv_actions)
        };

        if !update && !below_batch_frontier {
            return;
        }

        let _update_profile = operation_profile::enter(5);
        if game.is_compression_enabled() {
            // update the cumulative strategy
            let average_profile = operation_profile::enter(10);
            let scale = node.strategy_scale();
            let decoder = params.gamma_t * scale / u16::MAX as f32;
            let cum_strategy = node.strategy_compressed_mut();

            strategy.iter_mut().zip(&*cum_strategy).for_each(|(x, y)| {
                *x += (*y as f32) * decoder;
            });

            if !locking.is_empty() {
                strategy.iter_mut().zip(locking).for_each(|(d, s)| {
                    if s.is_sign_positive() {
                        *d = 0.0;
                    }
                })
            }

            let new_scale = encode_unsigned_slice(cum_strategy, &strategy);
            node.set_strategy_scale(new_scale);
            drop(average_profile);

            // update the cumulative regret
            let scale = node.regret_scale();
            let alpha_decoder = params.alpha_t * scale / i16::MAX as f32;
            let beta_decoder = params.beta_t * scale / i16::MAX as f32;
            let cum_regret = node.regrets_compressed_mut();

            cfv_actions.iter_mut().zip(&*cum_regret).for_each(|(x, y)| {
                *x += *y as f32 * if *y >= 0 { alpha_decoder } else { beta_decoder };
            });

            cfv_actions.chunks_exact_mut(num_hands).for_each(|row| {
                sub_slice(row, result);
            });

            if !locking.is_empty() {
                cfv_actions.iter_mut().zip(locking).for_each(|(d, s)| {
                    if s.is_sign_positive() {
                        *d = 0.0;
                    }
                })
            }

            let new_scale = encode_signed_slice(cum_regret, &cfv_actions);
            node.set_regret_scale(new_scale);
        } else {
            // update the cumulative strategy
            let average_profile = operation_profile::enter(10);
            let gamma = params.gamma_t;
            let cum_strategy = node.strategy_mut();
            cum_strategy.iter_mut().zip(&strategy).for_each(|(x, y)| {
                *x = *x * gamma + *y;
            });

            drop(average_profile);

            // Keep the seeded regret-matched policy for BOTH alternating passes.
            // No policy locks: full-tree best responses remain unconstrained.
            if (node.board_turn() == NOT_DEALT && HOLD_FLOP_REGRETS.load(Ordering::Relaxed))
                || (node.board_turn() != NOT_DEALT && node.board_river() == NOT_DEALT
                    && (HOLD_TURN_REGRETS.load(Ordering::Relaxed) && {let mask=HOLD_TURN_CARD_MASK.load(Ordering::Relaxed);mask==0 || (mask&(1u64<<node.board_turn()))!=0})) {
                return;
            }

            // update the cumulative regret
            let (alpha, beta) = (params.alpha_t, params.beta_t);
            let cum_regret = node.regrets_mut();
            if fused_regret_updates_enabled() {
                for (row, action_values) in cum_regret.chunks_exact_mut(num_hands).zip(cfv_actions.chunks_exact(num_hands)) {
                    for ((x, y), expected) in row.iter_mut().zip(action_values).zip(result.iter()) {
                        let coef = if x.is_sign_positive() { alpha } else { beta };
                        let discounted = *x * coef + *y;
                        *x = discounted - *expected;
                    }
                }
            } else {
            cum_regret.iter_mut().zip(&*cfv_actions).for_each(|(x, y)| {
                let coef = if x.is_sign_positive() { alpha } else { beta };
                *x = *x * coef + *y;
            });
            cum_regret.chunks_exact_mut(num_hands).for_each(|row| {
                sub_slice(row, result);
            });
            }
        }
    }
    // if the current player is not `player`
    else {
        let reach_profile = operation_profile::enter(4);
        // compute the strategy by regret-matching algorithm
        let mut cfreach_actions = if game.is_compression_enabled() {
            regret_matching_compressed(
                node.regrets_compressed(),
                num_actions,
                params.qre_temperature,
            )
        } else {
            regret_matching(node.regrets(), num_actions, params.qre_temperature)
        };

        // node-locking
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut cfreach_actions, locking);

        // update the reach probabilities
        let row_size = cfreach.len();
        cfreach_actions.chunks_exact_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        drop(reach_profile);
        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            solve_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
                player_reach,
                params,
                outer_iteration,
                update,
                below_batch_frontier,
                use_cached_predictions,
            );
        });

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_uninit(result, &cfv_actions);
    }
}

/// Computes the strategy by regret-matching algorithm.
#[cfg(feature = "custom-alloc")]
#[inline]
pub(crate) fn regret_matching(
    regret: &[f32],
    num_actions: usize,
    qre_temperature: f32,
) -> Vec<f32, StackAlloc> {
    let _regret_profile = operation_profile::enter(3);
    let mut strategy = Vec::with_capacity_in(regret.len(), StackAlloc);
    let uninit = strategy.spare_capacity_mut();
    uninit.iter_mut().zip(regret).for_each(|(s, r)| {
        s.write(max(*r, 0.0));
    });
    unsafe { strategy.set_len(regret.len()) };

    if qre_temperature > 0.0 {
        apply_quantal_response(&mut strategy, regret, num_actions, qre_temperature);
        return strategy;
    }

    apply_regret_floor(&mut strategy, num_actions, regret_floor_epsilon());

    let row_size = regret.len() / num_actions;
    let mut denom = Vec::with_capacity_in(row_size, StackAlloc);
    sum_slices_uninit(denom.spare_capacity_mut(), &strategy);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    strategy.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    strategy
}

/// Computes the strategy by regret-matching algorithm.
#[cfg(not(feature = "custom-alloc"))]
#[inline]
pub(crate) fn regret_matching(regret: &[f32], num_actions: usize, qre_temperature: f32) -> Vec<f32> {
    let _regret_profile = operation_profile::enter(3);
    let mut strategy = Vec::with_capacity(regret.len());
    let uninit = strategy.spare_capacity_mut();
    uninit.iter_mut().zip(regret).for_each(|(s, r)| {
        s.write(max(*r, 0.0));
    });
    unsafe { strategy.set_len(regret.len()) };

    if qre_temperature > 0.0 {
        apply_quantal_response(&mut strategy, regret, num_actions, qre_temperature);
        return strategy;
    }

    apply_regret_floor(&mut strategy, num_actions, regret_floor_epsilon());

    let row_size = regret.len() / num_actions;
    let mut denom = Vec::with_capacity(row_size);
    sum_slices_uninit(denom.spare_capacity_mut(), &strategy);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    strategy.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    strategy
}

/// Computes the strategy by regret-matching algorithm.
#[cfg(feature = "custom-alloc")]
#[inline]
fn regret_matching_compressed(
    regret: &[i16],
    num_actions: usize,
    qre_temperature: f32,
) -> Vec<f32, StackAlloc> {
    let _regret_profile = operation_profile::enter(3);
    let mut strategy = Vec::with_capacity_in(regret.len(), StackAlloc);
    strategy.extend(regret.iter().map(|&r| r.max(0) as f32));

    if qre_temperature > 0.0 {
        let mut decoded = Vec::with_capacity_in(regret.len(), StackAlloc);
        decoded.extend(regret.iter().map(|&r| r as f32));
        apply_quantal_response(&mut strategy, &decoded, num_actions, qre_temperature);
        return strategy;
    }

    apply_regret_floor(&mut strategy, num_actions, regret_floor_epsilon());

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity_in(row_size, StackAlloc);
    sum_slices_uninit(denom.spare_capacity_mut(), &strategy);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    strategy.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    strategy
}

/// Computes the strategy by regret-matching algorithm.
#[cfg(not(feature = "custom-alloc"))]
#[inline]
fn regret_matching_compressed(
    regret: &[i16],
    num_actions: usize,
    qre_temperature: f32,
) -> Vec<f32> {
    let _regret_profile = operation_profile::enter(3);
    let mut strategy = Vec::with_capacity(regret.len());
    strategy.extend(regret.iter().map(|&r| r.max(0) as f32));

    if qre_temperature > 0.0 {
        let decoded: Vec<f32> = regret.iter().map(|&r| r as f32).collect();
        apply_quantal_response(&mut strategy, &decoded, num_actions, qre_temperature);
        return strategy;
    }

    apply_regret_floor(&mut strategy, num_actions, regret_floor_epsilon());

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity(row_size);
    sum_slices_uninit(denom.spare_capacity_mut(), &strategy);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    strategy.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    strategy
}

/// Applies a per-hand softmax to cumulative regrets. Dividing by the action
/// range makes temperature independent of regret encoding and pot scale.
fn apply_quantal_response(
    strategy: &mut [f32],
    regret: &[f32],
    num_actions: usize,
    temperature: f32,
) {
    let row_size = regret.len() / num_actions;
    for hand in 0..row_size {
        let mut minimum = f32::INFINITY;
        let mut maximum = f32::NEG_INFINITY;
        for action in 0..num_actions {
            let value = regret[action * row_size + hand];
            minimum = minimum.min(value);
            maximum = maximum.max(value);
        }
        let scale = (maximum - minimum).max(f32::EPSILON) * temperature;
        let mut denominator = 0.0;
        for action in 0..num_actions {
            let index = action * row_size + hand;
            let weight = ((regret[index] - maximum) / scale).exp();
            strategy[index] = weight;
            denominator += weight;
        }
        for action in 0..num_actions {
            strategy[action * row_size + hand] /= denominator;
        }
    }
}

#[cfg(test)]
mod weighted_turn_tests {
    use super::*;
    use crate::pio_order::TurnCfvConversion;

    #[test]
    fn default_dcfr_discounts_preserve_historical_arithmetic() {
        for t in 0..300 {
            let actual = DiscountParams::with_reset64_policy(t, false);
            let x = (t as i32 - 1).max(0) as f64;
            let power = x * x.sqrt();
            assert_eq!(actual.alpha_t, (power / (power + 1.0)) as f32);
            assert_eq!(actual.beta_t, 0.5);
            let reset = if t == 0 { 0 } else { 1 << ((t.leading_zeros() ^ 31) & !1) };
            let y = (t - reset) as f64;
            assert_eq!(actual.gamma_t, (y / (y + 1.0)).powi(3) as f32);
        }
    }

    #[test]
    fn strategy_reset64_override_only_changes_average_discount_between_64_and_255() {
        for iteration in [0, 1, 4, 16, 63, 64, 65, 128, 255, 256, 257, 1024] {
            let normal = DiscountParams::with_reset64_policy(iteration, false);
            let skipped = DiscountParams::with_reset64_policy(iteration, true);
            assert_eq!(normal.alpha_t, skipped.alpha_t);
            assert_eq!(normal.beta_t, skipped.beta_t);
            if (64..256).contains(&iteration) {
                let t = (iteration - 16) as f64;
                assert_eq!(skipped.gamma_t, (t / (t + 1.0)).powi(3) as f32);
                assert!(skipped.gamma_t > normal.gamma_t);
            } else {
                assert_eq!(normal.gamma_t, skipped.gamma_t);
            }
        }
        assert_eq!(DiscountParams::with_reset64_policy(64, false).gamma_t, 0.0);
        assert_eq!(DiscountParams::with_reset64_policy(256, true).gamma_t, 0.0);
    }

    #[test]
    fn preserve_average_epoch_skips64_and256_without_changing_regret_discounts(){
        for t in 0..1100{
            let normal=DiscountParams::with_reset64_policy(t,false);let kept=DiscountParams::with_average_epoch_policy(t,false,Some(50));
            assert_eq!(normal.alpha_t,kept.alpha_t);assert_eq!(normal.beta_t,kept.beta_t);
            if t<50{assert_eq!(normal.gamma_t,kept.gamma_t);}else{let y=(t-16) as f64;assert_eq!(kept.gamma_t,(y/(y+1.)).powi(3) as f32);assert!(kept.gamma_t>0.);}
        }
    }

    #[test]
    fn regret_floor_is_relative_and_preserves_uniform_fallback() {
        // Action-major layout: action 0 has [10, 0], action 1 has [0, 0].
        let mut strategy = vec![10.0, 0.0, 0.0, 0.0];
        apply_regret_floor(&mut strategy, 2, 0.001);
        assert_eq!(strategy[0], 10.0);
        assert_eq!(strategy[1], 0.0);
        assert!((strategy[2] - 0.01).abs() < 1.0e-7);
        assert_eq!(strategy[3], 0.0);

        // Normalization produces a 0.001 / 1.001 exploration probability.
        assert!((strategy[2] / (strategy[0] + strategy[2]) - 0.01 / 10.01).abs() < 1.0e-7);

        let uniform = regret_matching(&[0.0, 0.0, 0.0, 0.0], 2, 0.0);
        assert_eq!(uniform, vec![0.5; 4]);
    }

    #[test]
    fn regret_floor_action_slices_preserve_bits() {
        for actions in 1..=8 {
            for hands in [0, 1, 7, 273, 416, 1326] {
                let mut state = 17u32;
                let values: Vec<f32> = (0..actions * hands).map(|i| {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    match i % 19 {
                        0 => 0.0,
                        1 => -0.0,
                        2 => f32::from_bits(1),
                        3 => f32::NAN,
                        4 => f32::INFINITY,
                        _ => (state as i32 as f32) * 1.0e-6,
                    }
                }).collect();
                for epsilon in [0.0, 1.0e-6, 0.001, 0.99] {
                    let mut expected = values.clone();
                    if epsilon != 0.0 {
                        for hand in 0..hands {
                            let maximum = (0..actions).map(|a| expected[a * hands + hand])
                                .fold(0.0f32, f32::max);
                            if maximum > 0.0 {
                                let floor = epsilon * maximum;
                                for a in 0..actions {
                                    let index = a * hands + hand;
                                    expected[index] = expected[index].max(floor);
                                }
                            }
                        }
                    }
                    let mut actual = values.clone();
                    apply_regret_floor(&mut actual, actions, epsilon);
                    assert!(actual.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "actions={actions} hands={hands} epsilon={epsilon}");
                }
            }
        }
    }

    #[test]
    fn weighted_sum_replaces_exact_sum_only_when_enabled() {
        let mut matrix = TurnCfvConversion {
            node_id: 4,
            action_count: 49,
            use_in_solving: false,
            all_turn_nodes: false,
            basis_actions: (13..49).collect(),
            weights: [vec![2.0; 36 * 2], vec![3.0; 36 * 2]],
            bucket_model: None,
            neural_model: None,
        };
        let mut values = vec![1.0; 49 * 2];
        // A blocked hand must remain zero; omitted turn values must not enter the estimate.
        for row in values.chunks_exact_mut(2) {
            row[1] = 0.0;
        }
        values[0] = 100.0;
        let mut result = [MaybeUninit::uninit(); 2];
        sum_chance_cfvs(&mut result, &values, 0, 0, 0, Some(&matrix), &[], None);
        assert_eq!(unsafe { result[0].assume_init() }, 148.0);
        matrix.use_in_solving = true;
        sum_chance_cfvs(&mut result, &values, 0, 0, 0, Some(&matrix), &[], None);
        assert_eq!(unsafe { result[0].assume_init() }, 72.0);
        assert_eq!(unsafe { result[1].assume_init() }, 0.0);
        sum_chance_cfvs(&mut result, &values, 0, 1, 0, Some(&matrix), &[], None);
        assert_eq!(unsafe { result[0].assume_init() }, 108.0);
        sum_chance_cfvs(&mut result, &values, 0, 0, 0, None, &[], None);
        assert_eq!(unsafe { result[0].assume_init() }, 148.0);
    }
}
