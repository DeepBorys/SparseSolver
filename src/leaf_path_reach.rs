//! Opt-in FP32 decision-path experiment. No intermediate reach vectors between
//! terminal/chance boundaries. Chance/NN handling stays on the established path.
//! Parent policy buffers outlive child tasks; each node has one updating owner.
use super::*;
use std::borrow::Cow;

pub(super) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("PFS_LEAF_PATH_REACH_EXPERIMENT").as_deref() == Ok("1"))
}

pub(super) enum Reach<'a> {
    Initial(&'a [f32]),
    Product { parent: &'a Reach<'a>, probabilities: &'a [f32] },
}

impl<'a> Reach<'a> {
    fn materialize(&self) -> Cow<'a, [f32]> {
        // Paths normally have fewer than eight decisions by either player.
        // Fixed-length kernels let LLVM vectorize across hands, retaining the
        // original FP32 multiplication order. No log-domain or reassociation.
        let mut factors: [&[f32]; 32] = [&[]; 32];
        let mut count = 0;
        let mut current = self;
        let initial = loop {
            match current {
                Self::Initial(values) => break *values,
                Self::Product { parent, probabilities } => {
                    if count == factors.len() {
                        let mut values = parent.materialize().into_owned();
                        mul_slice(&mut values, probabilities);
                        // Apply the already collected suffix, root-to-leaf.
                        for factor in factors[..count].iter().rev() { mul_slice(&mut values, factor); }
                        return Cow::Owned(values);
                    }
                    factors[count] = probabilities;
                    count += 1;
                    current = parent;
                }
            }
        };
        if count == 0 { return Cow::Borrowed(initial); }
        factors[..count].reverse();
        let factors = &factors[..count];
        for factor in factors { assert_eq!(factor.len(), initial.len()); }
        let mut result = Vec::<f32>::with_capacity(initial.len());
        fn kernel<const N: usize>(initial: &[f32], factors: &[&[f32]], out: &mut [MaybeUninit<f32>]) {
            for (i, output) in out.iter_mut().take(initial.len()).enumerate() {
                let mut value = initial[i];
                for factor in &factors[..N] { value *= factor[i]; }
                output.write(value);
            }
        }
        match count {
            1 => kernel::<1>(initial, factors, result.spare_capacity_mut()),
            2 => kernel::<2>(initial, factors, result.spare_capacity_mut()),
            3 => kernel::<3>(initial, factors, result.spare_capacity_mut()),
            4 => kernel::<4>(initial, factors, result.spare_capacity_mut()),
            5 => kernel::<5>(initial, factors, result.spare_capacity_mut()),
            6 => kernel::<6>(initial, factors, result.spare_capacity_mut()),
            7 => kernel::<7>(initial, factors, result.spare_capacity_mut()),
            8 => kernel::<8>(initial, factors, result.spare_capacity_mut()),
            _ => {
                for (i, output) in result.spare_capacity_mut().iter_mut().take(initial.len()).enumerate() {
                    let mut value = initial[i];
                    for factor in factors { value *= factor[i]; }
                    output.write(value);
                }
            }
        }
        // Every requested entry was initialized by the selected kernel.
        unsafe { result.set_len(initial.len()); }
        Cow::Owned(result)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn solve<T: Game>(
    result: &mut [MaybeUninit<f32>], game: &T, node: &mut T::Node, player: usize,
    opponent: &Reach<'_>, own: Option<&Reach<'_>>, params: &DiscountParams,
    outer_iteration: u32, update: bool, below_batch_frontier: bool, use_cached_predictions: bool,
) {
    if node.is_terminal() || node.is_chance() {
        let opponent = opponent.materialize();
        // Exact terminals consume only counterfactual opponent reach. Own
        // reach is needed by neural/capture chance boundaries, not payouts.
        let own = if node.is_terminal() { None } else { own.map(Reach::materialize) };
        solve_recursive_standard(result, game, node, player, &opponent, own.as_deref(),
            params, outer_iteration, update, below_batch_frontier, use_cached_predictions);
        return;
    }
    let _profile = operation_profile::enter(0);
    // Retain the baseline's unconditional decision counter. Removing its
    // atomics would confound a reach-propagation performance comparison.
    RECURSION_COUNTS[1].fetch_add(1, Ordering::Relaxed);
    let num_actions = node.num_actions();
    if num_actions == 1 {
        solve(result, game, &mut node.play(0), player, opponent, own, params,
            outer_iteration, update, below_batch_frontier, use_cached_predictions);
        return;
    }
    let num_hands = result.len();
    let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));
    if node.player() == player {
        let own_actions = own.map(|_| {
            let mut strategy = regret_matching(node.regrets(), num_actions, params.qre_temperature);
            apply_locking_strategy(&mut strategy, game.locking_strategy(node));
            strategy
        });
        for_each_child(node, |action| {
            let own_child = own.zip(own_actions.as_ref()).map(|(parent, probabilities)| Reach::Product {
                parent, probabilities: row(probabilities, action, game.num_private_hands(player)),
            });
            solve(row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands), game,
                &mut node.play(action), player, opponent, own_child.as_ref(), params,
                outer_iteration, update, below_batch_frontier, use_cached_predictions);
        });
        let mut strategy = regret_matching(node.regrets(), num_actions, params.qre_temperature);
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut strategy, locking);
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands); }
        let result = fma_slices_uninit(result, &strategy, &cfv_actions);
        if !update && !below_batch_frontier { return; }
        for (x, y) in node.strategy_mut().iter_mut().zip(&strategy) { *x = *x * params.gamma_t + *y; }
        if fused_regret_updates_enabled() {
            for (regrets, values) in node.regrets_mut().chunks_exact_mut(num_hands).zip(cfv_actions.chunks_exact(num_hands)) {
                for ((x, y), expected) in regrets.iter_mut().zip(values).zip(result.iter()) {
                    let discount = if x.is_sign_positive() { params.alpha_t } else { params.beta_t };
                    let discounted = *x * discount + *y;
                    *x = discounted - *expected;
                }
            }
        } else {
            for (x, y) in node.regrets_mut().iter_mut().zip(&*cfv_actions) {
                *x = *x * if x.is_sign_positive() { params.alpha_t } else { params.beta_t } + *y;
            }
            for values in node.regrets_mut().chunks_exact_mut(num_hands) { sub_slice(values, result); }
        }
    } else {
        let mut probabilities = regret_matching(node.regrets(), num_actions, params.qre_temperature);
        apply_locking_strategy(&mut probabilities, game.locking_strategy(node));
        let opponent_hands = game.num_private_hands(player ^ 1);
        for_each_child(node, |action| {
            let child = Reach::Product { parent: opponent, probabilities: row(&probabilities, action, opponent_hands) };
            solve(row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands), game,
                &mut node.play(action), player, &child, own, params, outer_iteration,
                update, below_batch_frontier, use_cached_predictions);
        });
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands); }
        sum_slices_uninit(result, &cfv_actions);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn leaf_path_initial_is_borrowed() {
        let values = [0., 0.25, 1.];
        assert!(matches!(Reach::Initial(&values).materialize(), Cow::Borrowed(_)));
    }
    #[test]
    fn leaf_path_products_keep_fp32_order_and_zeroes() {
        let values: Vec<f32> = (0..517).map(|i| if i%7==0 { 0. } else { (i as f32+1.)/519. }).collect();
        let a: Vec<f32> = (0..517).map(|i| (i%19) as f32/19.).collect();
        let b: Vec<f32> = (0..517).map(|i| (i%23) as f32/23.).collect();
        let c: Vec<f32> = (0..517).map(|i| (i%11) as f32/11.).collect();
        let initial=Reach::Initial(&values);
        let first=Reach::Product { parent:&initial, probabilities:&a };
        let second=Reach::Product { parent:&first, probabilities:&b };
        let third=Reach::Product { parent:&second, probabilities:&c };
        let mut expected=values.clone();
        for factor in [&a,&b,&c] { mul_slice(&mut expected,factor); }
        let actual=third.materialize();
        assert_eq!(actual.iter().map(|v|v.to_bits()).collect::<Vec<_>>(), expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
    }
}
