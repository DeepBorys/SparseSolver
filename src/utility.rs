use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;
use std::mem::{self, MaybeUninit};
use std::ptr;
use std::sync::atomic::{AtomicU32, Ordering};

#[cfg(feature = "custom-alloc")]
use crate::alloc::*;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

static STRATEGY_EVALUATION_THRESHOLD_BITS: AtomicU32 = AtomicU32::new(0);

/// Sets an evaluation-only probability threshold. Normalized action
/// probabilities below `threshold` are zeroed and the remaining actions are
/// renormalized per hand. Zero disables the threshold.
pub fn set_strategy_evaluation_threshold(threshold: f32) {
    assert!(threshold.is_finite() && (0.0..1.0).contains(&threshold));
    STRATEGY_EVALUATION_THRESHOLD_BITS.store(threshold.to_bits(), Ordering::Relaxed);
}

#[inline]
fn apply_strategy_evaluation_threshold(strategy: &mut [f32], num_actions: usize) {
    let threshold = f32::from_bits(STRATEGY_EVALUATION_THRESHOLD_BITS.load(Ordering::Relaxed));
    if threshold == 0.0 {
        return;
    }
    let row_size = strategy.len() / num_actions;
    for hand in 0..row_size {
        let mut denominator = 0.0;
        for action in 0..num_actions {
            let value = &mut strategy[action * row_size + hand];
            if *value < threshold {
                *value = 0.0;
            }
            denominator += *value;
        }
        if denominator > 0.0 {
            for action in 0..num_actions {
                strategy[action * row_size + hand] /= denominator;
            }
        }
    }
}

/// Executes `op` for each child potentially in parallel.
#[cfg(feature = "rayon")]
#[inline]
pub(crate) fn for_each_child<T: GameNode, OP: Fn(usize) + Sync + Send>(node: &T, op: OP) {
    // Timing experiment: bypass Rayon entirely, rather than scheduling onto
    // a one-worker pool. Normal traversal behavior remains unchanged.
    static SERIAL_TRAVERSAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let serial = *SERIAL_TRAVERSAL.get_or_init(|| {
        std::env::var("PFS_SERIAL_TRAVERSAL_EXPERIMENT").as_deref() == Ok("1")
    });
    if !serial && node.enable_parallelization() {
        node.action_indices().into_par_iter().for_each(op);
    } else {
        node.action_indices().for_each(op);
    }
}

/// Executes `op` for each child.
#[cfg(not(feature = "rayon"))]
#[inline]
pub(crate) fn for_each_child<T: GameNode, OP: Fn(usize) + Sync + Send>(node: &T, op: OP) {
    node.action_indices().for_each(op);
}

#[cfg(feature = "rayon")]
pub(crate) fn into_par_iter(range: std::ops::Range<usize>) -> rayon::range::Iter<usize> {
    range.into_par_iter()
}

#[cfg(not(feature = "rayon"))]
pub(crate) fn into_par_iter(range: std::ops::Range<usize>) -> std::ops::Range<usize> {
    range
}

#[inline]
pub(crate) fn max(x: f32, y: f32) -> f32 {
    if x > y {
        x
    } else {
        y
    }
}

#[inline]
pub(crate) fn is_zero(x: f32) -> bool {
    x.to_bits() == 0
}

#[inline]
pub(crate) fn vec_memory_usage<T>(vec: &Vec<T>) -> u64 {
    vec.capacity() as u64 * mem::size_of::<T>() as u64
}

/// Computes the average with given weights.
#[inline]
pub fn compute_average(slice: &[f32], weights: &[f32]) -> f32 {
    let mut weight_sum = 0.0;
    let mut value_sum = 0.0;
    for (&v, &w) in slice.iter().zip(weights.iter()) {
        weight_sum += w as f64;
        value_sum += v as f64 * w as f64;
    }
    (value_sum / weight_sum) as f32
}

#[inline]
fn weighted_sum(values: &[f32], weights: &[f32]) -> f32 {
    let f = |sum: f64, (&v, &w): (&f32, &f32)| sum + v as f64 * w as f64;
    values.iter().zip(weights).fold(0.0, f) as f32
}

/// Obtains the maximum absolute value of the given slice.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn slice_absolute_max(slice: &[f32]) -> f32 {
    if slice.len() < 32 {
        slice.iter().fold(0.0, |a, x| max(a, x.abs()))
    } else {
        use std::arch::wasm32::*;

        unsafe {
            let slice_ptr = slice.as_ptr() as *const v128;
            let mut tmp: [v128; 4] = [
                f32x4_abs(v128_load(slice_ptr)),
                f32x4_abs(v128_load(slice_ptr.add(1))),
                f32x4_abs(v128_load(slice_ptr.add(2))),
                f32x4_abs(v128_load(slice_ptr.add(3))),
            ];

            let mut iter = slice[16..].chunks_exact(16);
            for chunk in iter.by_ref() {
                let chunk_ptr = chunk.as_ptr() as *const v128;
                tmp[0] = f32x4_max(tmp[0], f32x4_abs(v128_load(chunk_ptr)));
                tmp[1] = f32x4_max(tmp[1], f32x4_abs(v128_load(chunk_ptr.add(1))));
                tmp[2] = f32x4_max(tmp[2], f32x4_abs(v128_load(chunk_ptr.add(2))));
                tmp[3] = f32x4_max(tmp[3], f32x4_abs(v128_load(chunk_ptr.add(3))));
            }

            tmp[0] = f32x4_max(tmp[0], tmp[1]);
            tmp[2] = f32x4_max(tmp[2], tmp[3]);
            tmp[0] = f32x4_max(tmp[0], tmp[2]);
            let tmpmax = max(
                max(
                    f32x4_extract_lane::<0>(tmp[0]),
                    f32x4_extract_lane::<1>(tmp[0]),
                ),
                max(
                    f32x4_extract_lane::<2>(tmp[0]),
                    f32x4_extract_lane::<3>(tmp[0]),
                ),
            );

            iter.remainder().iter().fold(tmpmax, |a, x| max(a, x.abs()))
        }
    }
}

/// Obtains the maximum absolute value of the given slice.
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn slice_absolute_max(slice: &[f32]) -> f32 {
    if slice.len() < 16 {
        slice.iter().fold(0.0, |a, x| max(a, x.abs()))
    } else {
        let mut tmp: [f32; 8] = slice[..8].try_into().unwrap();
        tmp.iter_mut().for_each(|x| *x = x.abs());
        let mut iter = slice[8..].chunks_exact(8);
        for chunk in iter.by_ref() {
            for i in 0..8 {
                tmp[i] = max(tmp[i], chunk[i].abs());
            }
        }
        let tmpmax = tmp.iter().fold(0.0f32, |a, &x| max(a, x));
        iter.remainder().iter().fold(tmpmax, |a, x| max(a, x.abs()))
    }
}

/// Obtains the maximum value of the given non-negative slice.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn slice_nonnegative_max(slice: &[f32]) -> f32 {
    if slice.len() < 32 {
        slice.iter().fold(0.0, |a, &x| max(a, x))
    } else {
        use std::arch::wasm32::*;

        unsafe {
            let slice_ptr = slice.as_ptr() as *const v128;
            let mut tmp: [v128; 4] = [
                v128_load(slice_ptr),
                v128_load(slice_ptr.add(1)),
                v128_load(slice_ptr.add(2)),
                v128_load(slice_ptr.add(3)),
            ];

            let mut iter = slice[16..].chunks_exact(16);
            for chunk in iter.by_ref() {
                let chunk_ptr = chunk.as_ptr() as *const v128;
                tmp[0] = f32x4_max(tmp[0], v128_load(chunk_ptr));
                tmp[1] = f32x4_max(tmp[1], v128_load(chunk_ptr.add(1)));
                tmp[2] = f32x4_max(tmp[2], v128_load(chunk_ptr.add(2)));
                tmp[3] = f32x4_max(tmp[3], v128_load(chunk_ptr.add(3)));
            }

            tmp[0] = f32x4_max(tmp[0], tmp[1]);
            tmp[2] = f32x4_max(tmp[2], tmp[3]);
            tmp[0] = f32x4_max(tmp[0], tmp[2]);
            let tmpmax = max(
                max(
                    f32x4_extract_lane::<0>(tmp[0]),
                    f32x4_extract_lane::<1>(tmp[0]),
                ),
                max(
                    f32x4_extract_lane::<2>(tmp[0]),
                    f32x4_extract_lane::<3>(tmp[0]),
                ),
            );

            iter.remainder().iter().fold(tmpmax, |a, &x| max(a, x))
        }
    }
}

/// Obtains the maximum value of the given non-negative slice.
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn slice_nonnegative_max(slice: &[f32]) -> f32 {
    if slice.len() < 16 {
        slice.iter().fold(0.0, |a, &x| max(a, x))
    } else {
        let mut tmp: [f32; 8] = slice[..8].try_into().unwrap();
        let mut iter = slice[8..].chunks_exact(8);
        for chunk in iter.by_ref() {
            for i in 0..8 {
                tmp[i] = max(tmp[i], chunk[i]);
            }
        }
        let tmpmax = tmp.iter().fold(0.0f32, |a, &x| max(a, x));
        iter.remainder().iter().fold(tmpmax, |a, &x| max(a, x))
    }
}

/// Encodes the `f32` slice to the `i16` slice, and returns the scale.
#[inline]
pub(crate) fn encode_signed_slice(dst: &mut [i16], slice: &[f32]) -> f32 {
    let scale = slice_absolute_max(slice);
    let scale_nonzero = if scale == 0.0 { 1.0 } else { scale };
    let encoder = i16::MAX as f32 / scale_nonzero;
    dst.iter_mut()
        .zip(slice)
        .for_each(|(d, s)| *d = unsafe { (s * encoder).round().to_int_unchecked::<i32>() as i16 });
    scale
}

/// Encodes the `f32` slice to the `u16` slice, and returns the scale.
#[inline]
pub(crate) fn encode_unsigned_slice(dst: &mut [u16], slice: &[f32]) -> f32 {
    let scale = slice_nonnegative_max(slice);
    let scale_nonzero = if scale == 0.0 { 1.0 } else { scale };
    let encoder = u16::MAX as f32 / scale_nonzero;
    // note: 0.49999997 + 0.49999997 = 0.99999994 < 1.0 | 0.5 + 0.49999997 = 1.0
    dst.iter_mut().zip(slice).for_each(|(d, s)| {
        *d = unsafe { (s * encoder + 0.49999997).to_int_unchecked::<i32>() as u16 }
    });
    scale
}

/// Applies the given swap to the given slice.
#[inline]
pub(crate) fn apply_swap<T>(slice: &mut [T], swap_list: &[(u16, u16)]) {
    for &(i, j) in swap_list {
        unsafe {
            ptr::swap(
                slice.get_unchecked_mut(i as usize),
                slice.get_unchecked_mut(j as usize),
            );
        }
    }
}

/// Finalizes the solving process.
#[inline]
pub fn finalize<T: Game>(game: &mut T) {
    finalize_impl(game, false, 0);
}

/// Stores display EVs using the averaged strategy and the configured neural
/// chance-frontier conversions. All-ins remain exact. This is for approximate
/// solution exports, NOT exploitability or exact best-response evaluation.
/// Unlike a physical-only finalization of a pruned tree, omitted chance cards
/// are accounted for by the same models used during solving. Both live reaches
/// follow the averaged policy; regrets and cumulative strategies are unchanged.
pub fn finalize_with_neural_frontiers<T: Game>(game: &mut T, completed_iterations: u32) {
    finalize_impl(game, true, completed_iterations);
}

fn finalize_impl<T: Game>(game: &mut T, neural_frontiers: bool, iteration: u32) {
    if game.is_solved() {
        panic!("Game is already solved");
    }

    if !game.is_ready() {
        panic!("Game is not ready");
    }

    // compute the expected values and save them
    for player in 0..2 {
        let mut cfvalues = Vec::with_capacity(game.num_private_hands(player));
        compute_cfvalue_for_display_recursive(
            cfvalues.spare_capacity_mut(),
            game,
            &mut game.root(),
            player,
            game.initial_weights(player ^ 1),
            true,
            neural_frontiers.then(|| game.initial_weights(player)),
            neural_frontiers,
            iteration,
        );
    }

    // set the game solved
    game.set_solved();

    // free buffer
    #[cfg(all(feature = "custom-alloc", feature = "rayon"))]
    rayon::broadcast(|_| free_custom_alloc_buffer());
    #[cfg(all(feature = "custom-alloc", not(feature = "rayon")))]
    free_custom_alloc_buffer();
}

/// Computes the exploitability of the current strategy.
#[inline]
pub fn compute_exploitability<T: Game>(game: &T) -> f32 {
    assert!(!game.has_auxiliary_subgame(), "Exploitability requires a complete physical tree, not an auxiliary pruned continuation");
    if !game.is_ready() && !game.is_solved() {
        panic!("Game is not ready");
    }

    let mes_ev = compute_mes_ev(game);
    if !game.is_raked() {
        (mes_ev[0] + mes_ev[1]) * 0.5
    } else {
        let current_ev = compute_current_ev(game);
        ((mes_ev[0] - current_ev[0]) + (mes_ev[1] - current_ev[1])) * 0.5
    }
}

/// Computes the expected values of the current strategy of each player.
///
/// The bias, i.e., (starting pot) / 2, is already subtracted to increase the significant figures.
/// This treatment makes the return value zero-sum when not raked.
#[inline]
pub fn compute_current_ev<T: Game>(game: &T) -> [f32; 2] {
    if !game.is_ready() && !game.is_solved() {
        panic!("Game is not ready");
    }

    let mut cfvalues = [
        Vec::with_capacity(game.num_private_hands(0)),
        Vec::with_capacity(game.num_private_hands(1)),
    ];

    let reach = [game.initial_weights(0), game.initial_weights(1)];

    for player in 0..2 {
        compute_cfvalue_recursive(
            cfvalues[player].spare_capacity_mut(),
            game,
            &mut game.root(),
            player,
            reach[player ^ 1],
            false,
        );
        unsafe { cfvalues[player].set_len(game.num_private_hands(player)) };
    }

    let get_sum = |player: usize| weighted_sum(&cfvalues[player], reach[player]);
    [get_sum(0), get_sum(1)]
}

/// Computes the expected values of the MES (Maximally Exploitative Strategy) of each player.
///
/// The bias, i.e., (starting pot) / 2, is already subtracted to increase the significant figures.
/// Therefore, the average of the return value corresponds to the exploitability value if not raked.
#[inline]
pub fn compute_mes_ev<T: Game>(game: &T) -> [f32; 2] {
    assert!(!game.has_auxiliary_subgame(), "Best-response evaluation requires a complete physical tree");
    if !game.is_ready() && !game.is_solved() {
        panic!("Game is not ready");
    }

    let mut cfvalues = [
        Vec::with_capacity(game.num_private_hands(0)),
        Vec::with_capacity(game.num_private_hands(1)),
    ];

    let reach = [game.initial_weights(0), game.initial_weights(1)];

    for player in 0..2 {
        compute_best_cfv_recursive(
            cfvalues[player].spare_capacity_mut(),
            game,
            &game.root(),
            player,
            reach[player ^ 1],
        );
        unsafe { cfvalues[player].set_len(game.num_private_hands(player)) };
    }

    let get_sum = |player: usize| weighted_sum(&cfvalues[player], reach[player]);
    [get_sum(0), get_sum(1)]
}

/// Reach-weighted exploitability of the complete subtree below one flop node.
/// Parent and descendant rows overlap by design.
#[derive(Debug, Clone)]
pub struct FlopExploitabilityContribution {
    pub node_id: u64,
    pub acting_player: usize,
    pub action_path: Vec<usize>,
    pub oop_br_gain_chips: f32,
    pub ip_br_gain_chips: f32,
    pub subtree_exploitability_chips: f32,
}

/// Computes cumulative best-response gain below every flop decision node,
/// including all downstream turn and river decisions. Values are weighted by
/// current-strategy reach from the root. The root row equals exploitability in
/// an unraked game; other rows identify strategically expensive subtrees.
pub fn compute_flop_exploitability_contributions<T: Game>(
    game: &T,
) -> Vec<FlopExploitabilityContribution> {
    if !game.is_ready() && !game.is_solved() {
        panic!("Game is not ready");
    }

    fn recurse<T: Game>(
        game: &T,
        node: &mut T::Node,
        reaches: [&[f32]; 2],
        _initial_weight_sums: [f64; 2],
        path: &mut Vec<usize>,
        output: &mut Vec<FlopExploitabilityContribution>,
    ) {
        if node.is_terminal() {
            return;
        }
        let num_actions = node.num_actions();
        let acting = if node.is_chance() { 2 } else { node.player() };
        let mut gains = [0.0f32; 2];
        for player in 0..2 {
            let num_hands = game.num_private_hands(player);
            let mut best = Vec::with_capacity(num_hands);
            compute_best_cfv_recursive(
                best.spare_capacity_mut(),
                game,
                node,
                player,
                reaches[player ^ 1],
            );
            unsafe { best.set_len(num_hands) };
            let mut current = Vec::with_capacity(num_hands);
            compute_cfvalue_recursive(
                current.spare_capacity_mut(),
                game,
                node,
                player,
                reaches[player ^ 1],
                false,
            );
            unsafe { current.set_len(num_hands) };
            let raw_gain: f64 = best
                .iter()
                .zip(&current)
                .zip(reaches[player])
                .map(|((&br, &ev), &reach)| (br - ev) as f64 * reach as f64)
                .sum();
            gains[player] = raw_gain as f32;
        }
        output.push(FlopExploitabilityContribution {
            node_id: node.id(),
            acting_player: acting,
            action_path: path.clone(),
            oop_br_gain_chips: gains[0],
            ip_br_gain_chips: gains[1],
            subtree_exploitability_chips: 0.5 * (gains[0] + gains[1]),
        });

        // This is the end-of-flop frontier. Its row intentionally includes all
        // downstream turn and river exploitability, but traversal stops here so
        // the report remains flop-focused.
        if node.is_chance() {
            return;
        }

        let num_hands = game.num_private_hands(acting);
        let mut strategy = if game.is_compression_enabled() {
            normalized_strategy_compressed(node.strategy_compressed(), num_actions)
        } else {
            normalized_strategy(node.strategy(), num_actions)
        };
        apply_locking_strategy(&mut strategy, game.locking_strategy(node));

        for action in 0..num_actions {
            let mut updated = reaches[acting].to_vec();
            let probabilities = row(&strategy, action, num_hands);
            mul_slice(&mut updated, probabilities);
            path.push(action);
            let mut child = node.play(action);
            if acting == 0 {
                recurse(
                    game,
                    &mut child,
                    [&updated, reaches[1]],
                    _initial_weight_sums,
                    path,
                    output,
                );
            } else {
                recurse(
                    game,
                    &mut child,
                    [reaches[0], &updated],
                    _initial_weight_sums,
                    path,
                    output,
                );
            }
            path.pop();
        }
    }

    let initial = [game.initial_weights(0), game.initial_weights(1)];
    let sums = [
        initial[0].iter().map(|&v| v as f64).sum(),
        initial[1].iter().map(|&v| v as f64).sum(),
    ];
    let mut output = Vec::new();
    recurse(
        game,
        &mut game.root(),
        initial,
        sums,
        &mut Vec::new(),
        &mut output,
    );
    output
}

/// The recursive helper function for computing the counterfactual values of the given strategy.
/// Evaluate the averaged continuation at a decision-subtree root using an
/// explicitly supplied opponent reach vector. Returns native counterfactual
/// values (no division by own reach, no normalization, no payout offset).
/// Pure evaluation: does not save CFVs or alter regret/average buffers.
#[doc(hidden)]
pub fn compute_subgame_policy_cfvs<T: Game>(game:&T,path:&[usize],player:usize,opponent_reach:&[f32])->Vec<f32>{
    assert!(game.is_ready()&&!game.is_solved()&&player<2);
    assert_eq!(opponent_reach.len(),game.num_private_hands(player^1));
    assert!(opponent_reach.iter().all(|v|v.is_finite()&&*v>=0.));
    fn descend<T:Game>(game:&T,node:&mut T::Node,path:&[usize],player:usize,reach:&[f32],out:&mut Vec<f32>){
        if let Some((&a,rest))=path.split_first(){assert!(a<node.num_actions());descend(game,&mut node.play(a),rest,player,reach,out);}
        else{assert!(!node.is_terminal()&&!node.is_chance());compute_cfvalue_recursive(out.spare_capacity_mut(),game,node,player,reach,false);unsafe{out.set_len(game.num_private_hands(player));}}
    }
    let mut out=Vec::with_capacity(game.num_private_hands(player));descend(game,&mut game.root(),path,player,opponent_reach,&mut out);assert!(out.iter().all(|v|v.is_finite()));out
}

/// Exact per-hand counterfactual best-response values at a decision subtree,
/// against the fixed average continuation and explicit opponent reaches.
#[doc(hidden)]
pub fn compute_subgame_best_response_cfvs<T: Game>(game:&T,path:&[usize],player:usize,opponent_reach:&[f32])->Vec<f32>{
    assert!(game.is_ready()&&!game.is_solved()&&player<2);
    assert_eq!(opponent_reach.len(),game.num_private_hands(player^1));
    assert!(opponent_reach.iter().all(|v|v.is_finite()&&*v>=0.));
    fn descend<T:Game>(game:&T,node:&T::Node,path:&[usize],player:usize,reach:&[f32],out:&mut Vec<f32>){
        if let Some((&a,rest))=path.split_first(){assert!(a<node.num_actions());descend(game,&node.play(a),rest,player,reach,out);}
        else{assert!(!node.is_terminal()&&!node.is_chance());compute_best_cfv_recursive(out.spare_capacity_mut(),game,node,player,reach);unsafe{out.set_len(game.num_private_hands(player));}}
    }
    let mut out=Vec::with_capacity(game.num_private_hands(player));descend(game,&game.root(),path,player,opponent_reach,&mut out);assert!(out.iter().all(|v|v.is_finite()));out
}

fn compute_cfvalue_recursive<T: Game>(
    result: &mut [MaybeUninit<f32>],
    game: &T,
    node: &mut T::Node,
    player: usize,
    cfreach: &[f32],
    save_cfvalues: bool,
) {
    compute_cfvalue_for_display_recursive(result, game, node, player, cfreach,
        save_cfvalues, None, false, 0);
}

pub(crate) fn compute_cfvalue_for_display_recursive<T: Game>(
    result: &mut [MaybeUninit<f32>],
    game: &T,
    node: &mut T::Node,
    player: usize,
    cfreach: &[f32],
    save_cfvalues: bool,
    player_reach: Option<&[f32]>,
    neural_frontiers: bool,
    iteration: u32,
) {
    // terminal node
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    // Exact matrix leaves have no physical chance children. The matrix
    // already averages the complete flop all-in runout distribution, so pass
    // the unscaled opponent reach just as the solver recursion does.
    if node.is_chance() && game.evaluate_called_allin(result, node, player, cfreach) {
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = result.len();

    // allocate memory for storing the counterfactual values
    #[cfg(feature = "custom-alloc")]
    let cfv_actions = MutexLike::new(Vec::with_capacity_in(num_actions * num_hands, StackAlloc));
    #[cfg(not(feature = "custom-alloc"))]
    let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));

    // chance node
    if node.is_chance() {
        // update the reach probabilities
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
            let factor = 1.0 / game.chance_factor(node) as f32;
            reach.iter().map(|v| v * factor).collect::<Vec<_>>()
        });

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            compute_cfvalue_for_display_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                &cfreach_updated,
                save_cfvalues,
                player_reach_updated.as_deref(),
                neural_frontiers,
                iteration,
            );
        });

        // use 64-bit floating point values
        #[cfg(feature = "custom-alloc")]
        let mut result_f64 = Vec::with_capacity_in(num_hands, StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut result_f64 = Vec::with_capacity(num_hands);

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        // The subtree's own EVs were saved during recursion above. Do not add
        // its extra chance row to the parent aggregate during finalization.
        if let Some(action) = game.auxiliary_chance_action(node) {
            row_mut(&mut cfv_actions, action, num_hands).fill(0.0);
        }
        let conversion = neural_frontiers.then(|| game.turn_cfv_conversion(node)).flatten()
            .filter(|matrix| matrix.use_in_solving() && !game.should_traverse_chance_exactly(node));
        if let Some(matrix) = conversion {
            assert!(game.isomorphic_chances(node).is_empty(),
                "Neural display finalization requires explicit chance cards");
            let estimate = if matrix.neural_model.is_some() {
                let basis = matrix.basis_actions();
                let compact: Vec<f32> = basis.iter().flat_map(|&action|
                    row(&cfv_actions, action, num_hands).iter().copied()).collect();
                matrix.estimate_sum_from_basis_at_iteration(node.id(), player, &compact,
                    basis, iteration, cfreach, player_reach)
            } else {
                matrix.estimate_sum_at_iteration(node.id(), player,
                    &cfv_actions[..matrix.action_count * num_hands], iteration, cfreach, player_reach)
            };
            assert_eq!(estimate.len(), num_hands);
            for (output, value) in result_f64.spare_capacity_mut().iter_mut().zip(estimate) {
                output.write(value);
            }
        } else {
            sum_slices_f64_uninit(result_f64.spare_capacity_mut(), &cfv_actions);
        }
        unsafe { result_f64.set_len(num_hands) };

        // get information about isomorphic chances
        let isomorphic_chances = game.isomorphic_chances(node);

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

        result.iter_mut().zip(&result_f64).for_each(|(r, &v)| {
            r.write(v as f32);
        });

        // save the counterfactual values
        if save_cfvalues && node.cfvalue_storage_player() == Some(player) {
            let result = unsafe { &*(result as *const _ as *const [f32]) };
            if game.is_compression_enabled() {
                let cfv_scale = encode_signed_slice(node.cfvalues_chance_compressed_mut(), result);
                node.set_cfvalue_chance_scale(cfv_scale);
            } else {
                node.cfvalues_chance_mut().copy_from_slice(result);
            }
        }
    }
    // player node
    else if node.player() == player {
        let player_reach_actions = player_reach.map(|reach| {
            let mut probabilities = if game.is_compression_enabled() {
                normalized_strategy_compressed(node.strategy_compressed(), num_actions)
            } else {
                normalized_strategy(node.strategy(), num_actions)
            };
            apply_locking_strategy(&mut probabilities, game.locking_strategy(node));
            probabilities.chunks_exact_mut(num_hands).for_each(|row| mul_slice(row, reach));
            probabilities
        });
        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            compute_cfvalue_for_display_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                cfreach,
                save_cfvalues,
                player_reach_actions.as_deref().map(|values| row(values, action, num_hands)),
                neural_frontiers,
                iteration,
            );
        });

        // obtain the strategy
        #[cfg(feature = "custom-alloc")]
        let mut strategy = if game.is_compression_enabled() {
            normalized_strategy_compressed_custom_alloc(node.strategy_compressed(), num_actions)
        } else {
            normalized_strategy_custom_alloc(node.strategy(), num_actions)
        };
        #[cfg(not(feature = "custom-alloc"))]
        let mut strategy = if game.is_compression_enabled() {
            normalized_strategy_compressed(node.strategy_compressed(), num_actions)
        } else {
            normalized_strategy(node.strategy(), num_actions)
        };

        // node-locking
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut strategy, locking);

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        fma_slices_uninit(result, &strategy, &cfv_actions);

        // save the counterfactual values
        if save_cfvalues {
            if game.is_compression_enabled() {
                let cfv_scale = encode_signed_slice(node.cfvalues_compressed_mut(), &cfv_actions);
                node.set_cfvalue_scale(cfv_scale);
            } else {
                node.cfvalues_mut().copy_from_slice(&cfv_actions);
            }
        }
    }
    // opponent node
    else if num_actions == 1 {
        // simply recurse when the number of actions is one
        compute_cfvalue_for_display_recursive(
            result,
            game,
            &mut node.play(0),
            player,
            cfreach,
            save_cfvalues,
            player_reach,
            neural_frontiers,
            iteration,
        );
    } else {
        // obtain the strategy
        #[cfg(feature = "custom-alloc")]
        let mut cfreach_actions = if game.is_compression_enabled() {
            normalized_strategy_compressed_custom_alloc(node.strategy_compressed(), num_actions)
        } else {
            normalized_strategy_custom_alloc(node.strategy(), num_actions)
        };
        #[cfg(not(feature = "custom-alloc"))]
        let mut cfreach_actions = if game.is_compression_enabled() {
            normalized_strategy_compressed(node.strategy_compressed(), num_actions)
        } else {
            normalized_strategy(node.strategy(), num_actions)
        };

        // node-locking
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut cfreach_actions, locking);

        // update the reach probabilities
        let row_size = cfreach.len();
        cfreach_actions.chunks_exact_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            compute_cfvalue_for_display_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
                save_cfvalues,
                player_reach,
                neural_frontiers,
                iteration,
            );
        });

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_uninit(result, &cfv_actions);
    }

    // save the counterfactual values for IP
    if save_cfvalues && node.has_cfvalues_ip() && player == 1 {
        let result = unsafe { &*(result as *const _ as *const [f32]) };
        if game.is_compression_enabled() {
            let cfv_scale = encode_signed_slice(node.cfvalues_ip_compressed_mut(), result);
            node.set_cfvalue_ip_scale(cfv_scale);
        } else {
            node.cfvalues_ip_mut().copy_from_slice(result);
        }
    }
}

fn compute_best_cfv_recursive<T: Game>(
    result: &mut [MaybeUninit<f32>], game: &T, node: &T::Node,
    player: usize, cfreach: &[f32],
) {
    compute_best_cfv_with_policy_recursive(result, game, node, player, cfreach, None);
}

/// The recursive helper function for computing the counterfactual values of best response.
pub(crate) fn compute_best_cfv_with_policy_recursive<T: Game>(
    result: &mut [MaybeUninit<f32>],
    game: &T,
    node: &T::Node,
    player: usize,
    cfreach: &[f32],
    policies: Option<&std::sync::Mutex<std::collections::BTreeMap<u64, Vec<u32>>>>,
) {
    // terminal node
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    if node.is_chance() && game.evaluate_called_allin(result, node, player, cfreach) {
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = game.num_private_hands(player);

    // simply recurse when the number of actions is one
    if num_actions == 1 && !node.is_chance() {
        let child = &node.play(0);
        compute_best_cfv_with_policy_recursive(result, game, child, player, cfreach, policies);
        return;
    }

    // allocate memory for storing the counterfactual values
    #[cfg(feature = "custom-alloc")]
    let cfv_actions = MutexLike::new(Vec::with_capacity_in(num_actions * num_hands, StackAlloc));
    #[cfg(not(feature = "custom-alloc"))]
    let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));

    // chance node
    if node.is_chance() {
        // update the reach probabilities
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

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            compute_best_cfv_with_policy_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &node.play(action),
                player,
                &cfreach_updated,
                policies,
            )
        });

        // use 64-bit floating point values
        #[cfg(feature = "custom-alloc")]
        let mut result_f64 = Vec::with_capacity_in(num_hands, StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut result_f64 = Vec::with_capacity(num_hands);

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_f64_uninit(result_f64.spare_capacity_mut(), &cfv_actions);
        unsafe { result_f64.set_len(num_hands) };

        // get information about isomorphic chances
        let isomorphic_chances = game.isomorphic_chances(node);

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

        result.iter_mut().zip(&result_f64).for_each(|(r, &v)| {
            r.write(v as f32);
        });
    }
    // player node
    else if node.player() == player {
        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            compute_best_cfv_with_policy_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &node.play(action),
                player,
                cfreach,
                policies,
            )
        });

        let locking = game.locking_strategy(node);
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };

        if let Some(policies) = policies {
            let choices = (0..num_hands).map(|hand| {
                let mut best = 0;
                for action in 1..num_actions {
                    if cfv_actions[action * num_hands + hand] > cfv_actions[best * num_hands + hand] {
                        best = action;
                    }
                }
                best as u32
            }).collect();
            assert!(policies.lock().unwrap().insert(node.id(), choices).is_none(),
                "BR capture requires unique decision-node IDs");
        }

        if locking.is_empty() {
            // compute element-wise maximum (take the best response)
            max_slices_uninit(result, &cfv_actions);
        } else {
            // when the node is locked
            max_fma_slices_uninit(result, &cfv_actions, locking);
        }
    }
    // opponent node
    else {
        // obtain the strategy
        #[cfg(feature = "custom-alloc")]
        let mut cfreach_actions = if game.is_compression_enabled() {
            normalized_strategy_compressed_custom_alloc(node.strategy_compressed(), num_actions)
        } else {
            normalized_strategy_custom_alloc(node.strategy(), num_actions)
        };
        #[cfg(not(feature = "custom-alloc"))]
        let mut cfreach_actions = if game.is_compression_enabled() {
            normalized_strategy_compressed(node.strategy_compressed(), num_actions)
        } else {
            normalized_strategy(node.strategy(), num_actions)
        };

        // node-locking
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut cfreach_actions, locking);

        // update the reach probabilities
        let row_size = cfreach.len();
        cfreach_actions.chunks_exact_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            compute_best_cfv_with_policy_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
                policies,
            );
        });

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_uninit(result, &cfv_actions);
    }
}

/// Temporary responder CFVs at the flop-to-turn frontier, under native BR choices.
pub(crate) struct TurnBrCache{pub cards:Vec<u8>,pub samples:Vec<f32>,pub aggregate:Vec<f64>}

pub(crate) fn compute_best_cfv_with_turn_cache_recursive<T: Game>(
    result: &mut [MaybeUninit<f32>],
    game: &T,
    node: &T::Node,
    player: usize,
    cfreach: &[f32],
    policies: Option<&std::sync::Mutex<std::collections::BTreeMap<u64, Vec<u32>>>>,
    shadow: &mut [MaybeUninit<f32>],
    cache: &std::sync::Mutex<std::collections::BTreeMap<u64, TurnBrCache>>,
) {
    // terminal node
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        for (dst,src) in shadow.iter_mut().zip(result.iter()){dst.write(unsafe{src.assume_init()});}
        return;
    }

    if node.is_chance() && game.evaluate_called_allin(result, node, player, cfreach) {
        if node.num_actions()>0{
            let values=crate::best_response_capture::prepare_turn_allin_cache(game,node,player,cfreach,cache);
            for (dst,&v) in shadow.iter_mut().zip(&values){dst.write(v);}
        }else{for (dst,src) in shadow.iter_mut().zip(result.iter()){dst.write(unsafe{src.assume_init()});}}
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = game.num_private_hands(player);
    let shadow_actions = MutexLike::new(Vec::<f32>::with_capacity(num_actions * num_hands));

    // simply recurse when the number of actions is one
    if num_actions == 1 && !node.is_chance() {
        let child = &node.play(0);
        compute_best_cfv_with_turn_cache_recursive(result, game, child, player, cfreach, policies, shadow, cache);
        return;
    }

    // allocate memory for storing the counterfactual values
    #[cfg(feature = "custom-alloc")]
    let cfv_actions = MutexLike::new(Vec::with_capacity_in(num_actions * num_hands, StackAlloc));
    #[cfg(not(feature = "custom-alloc"))]
    let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));

    // chance node
    if node.is_chance() {
        // update the reach probabilities
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

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            compute_best_cfv_with_turn_cache_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &node.play(action),
                player,
                &cfreach_updated,
                policies,
                row_mut(shadow_actions.lock().spare_capacity_mut(), action, num_hands),
                cache,
            )
        });

        // use 64-bit floating point values
        #[cfg(feature = "custom-alloc")]
        let mut result_f64 = Vec::with_capacity_in(num_hands, StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut result_f64 = Vec::with_capacity(num_hands);

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_f64_uninit(result_f64.spare_capacity_mut(), &cfv_actions);
        unsafe { result_f64.set_len(num_hands) };

        // get information about isomorphic chances
        let isomorphic_chances = game.isomorphic_chances(node);

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

        result.iter_mut().zip(&result_f64).for_each(|(r, &v)| {
            r.write(v as f32);
        });
    }
    // player node
    else if node.player() == player {
        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            compute_best_cfv_with_turn_cache_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &node.play(action),
                player,
                cfreach,
                policies,
                row_mut(shadow_actions.lock().spare_capacity_mut(), action, num_hands),
                cache,
            )
        });

        let locking = game.locking_strategy(node);
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };

        if let Some(policies) = policies {
            let choices = (0..num_hands).map(|hand| {
                let mut best = 0;
                for action in 1..num_actions {
                    if cfv_actions[action * num_hands + hand] > cfv_actions[best * num_hands + hand] {
                        best = action;
                    }
                }
                best as u32
            }).collect();
            assert!(policies.lock().unwrap().insert(node.id(), choices).is_none(),
                "BR capture requires unique decision-node IDs");
        }

        if locking.is_empty() {
            // compute element-wise maximum (take the best response)
            max_slices_uninit(result, &cfv_actions);
        } else {
            // when the node is locked
            max_fma_slices_uninit(result, &cfv_actions, locking);
        }
    }
    // opponent node
    else {
        // obtain the strategy
        #[cfg(feature = "custom-alloc")]
        let mut cfreach_actions = if game.is_compression_enabled() {
            normalized_strategy_compressed_custom_alloc(node.strategy_compressed(), num_actions)
        } else {
            normalized_strategy_custom_alloc(node.strategy(), num_actions)
        };
        #[cfg(not(feature = "custom-alloc"))]
        let mut cfreach_actions = if game.is_compression_enabled() {
            normalized_strategy_compressed(node.strategy_compressed(), num_actions)
        } else {
            normalized_strategy(node.strategy(), num_actions)
        };

        // node-locking
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut cfreach_actions, locking);

        // update the reach probabilities
        let row_size = cfreach.len();
        cfreach_actions.chunks_exact_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            compute_best_cfv_with_turn_cache_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
                policies,
                row_mut(shadow_actions.lock().spare_capacity_mut(), action, num_hands),
                cache,
            );
        });

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_uninit(result, &cfv_actions);
    }

    // Match the historical fixed-profile evaluator's FP64 reduction at every
    // node while keeping native BR arithmetic and argmax/tie choices unchanged.
    let mut actions=shadow_actions.lock();
    unsafe{actions.set_len(num_actions*num_hands);}
    let mut sums=vec![0f64;num_hands];
    if node.is_chance(){
        let (cards,physical)=game.expand_physical_chance_values(node,player,&actions);
        for row in physical.chunks_exact(num_hands){for (out,&v) in sums.iter_mut().zip(row){*out+=v as f64;}}
        if node.board_turn()==crate::NOT_DEALT{
            assert_eq!(cards.len(),49);
            assert!(cache.lock().unwrap().insert(node.id(),TurnBrCache{cards,samples:physical,aggregate:sums.clone()}).is_none());
        }
    }else if node.player()==player{
        let mut strategy=if num_actions==1{vec![1.;num_hands]}else{
            let guard=policies.unwrap().lock().unwrap();let choices=&guard[&node.id()];let mut probabilities=vec![0.;num_actions*num_hands];
            for (h,&a) in choices.iter().enumerate(){probabilities[a as usize*num_hands+h]=1.;}probabilities
        };
        apply_locking_strategy(&mut strategy,game.locking_strategy(node));
        for (values,probabilities) in actions.chunks_exact(num_hands).zip(strategy.chunks_exact(num_hands)){
            for ((out,&v),&probability) in sums.iter_mut().zip(values).zip(probabilities){*out+=v as f64*probability as f64;}
        }
    }else{
        for row in actions.chunks_exact(num_hands){for (out,&v) in sums.iter_mut().zip(row){*out+=v as f64;}}
    }
    for (out,&v) in shadow.iter_mut().zip(&sums){out.write(v as f32);}
}

#[cfg(feature = "custom-alloc")]
#[inline]
pub(crate) fn normalized_strategy_custom_alloc(
    strategy: &[f32],
    num_actions: usize,
) -> Vec<f32, StackAlloc> {
    let mut normalized = Vec::with_capacity_in(strategy.len(), StackAlloc);
    let uninit = normalized.spare_capacity_mut();

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity_in(row_size, StackAlloc);
    sum_slices_uninit(denom.spare_capacity_mut(), strategy);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    uninit
        .chunks_exact_mut(row_size)
        .zip(strategy.chunks_exact(row_size))
        .for_each(|(n, s)| {
            div_slice_uninit(n, s, &denom, default);
        });

    unsafe { normalized.set_len(strategy.len()) };
    apply_strategy_evaluation_threshold(&mut normalized, num_actions);
    normalized
}

#[inline]
pub(crate) fn normalized_strategy(strategy: &[f32], num_actions: usize) -> Vec<f32> {
    let mut normalized = Vec::with_capacity(strategy.len());
    let uninit = normalized.spare_capacity_mut();

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity(row_size);
    sum_slices_uninit(denom.spare_capacity_mut(), strategy);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    uninit
        .chunks_exact_mut(row_size)
        .zip(strategy.chunks_exact(row_size))
        .for_each(|(n, s)| {
            div_slice_uninit(n, s, &denom, default);
        });

    unsafe { normalized.set_len(strategy.len()) };
    apply_strategy_evaluation_threshold(&mut normalized, num_actions);
    normalized
}

#[cfg(feature = "custom-alloc")]
#[inline]
pub(crate) fn normalized_strategy_compressed_custom_alloc(
    strategy: &[u16],
    num_actions: usize,
) -> Vec<f32, StackAlloc> {
    let mut normalized = Vec::with_capacity_in(strategy.len(), StackAlloc);
    let uninit = normalized.spare_capacity_mut();

    uninit.iter_mut().zip(strategy).for_each(|(n, s)| {
        n.write(*s as f32);
    });
    unsafe { normalized.set_len(strategy.len()) };

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity_in(row_size, StackAlloc);
    sum_slices_uninit(denom.spare_capacity_mut(), &normalized);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    normalized.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });
    apply_strategy_evaluation_threshold(&mut normalized, num_actions);

    normalized
}

#[inline]
pub(crate) fn normalized_strategy_compressed(strategy: &[u16], num_actions: usize) -> Vec<f32> {
    let mut normalized = Vec::with_capacity(strategy.len());
    let uninit = normalized.spare_capacity_mut();

    uninit.iter_mut().zip(strategy).for_each(|(n, s)| {
        n.write(*s as f32);
    });
    unsafe { normalized.set_len(strategy.len()) };

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity(row_size);
    sum_slices_uninit(denom.spare_capacity_mut(), &normalized);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    normalized.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });
    apply_strategy_evaluation_threshold(&mut normalized, num_actions);

    normalized
}

#[inline]
pub(crate) fn apply_locking_strategy(dst: &mut [f32], locking: &[f32]) {
    if !locking.is_empty() {
        dst.iter_mut().zip(locking).for_each(|(d, s)| {
            if s.is_sign_positive() {
                *d = *s;
            }
        });
    }
}
