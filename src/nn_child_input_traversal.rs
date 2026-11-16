//! Compatible FP32 per-card input preparation at turn and river frontiers.
//! Exact/ALL keeps native CFV returns.
use super::*;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

pub(super) fn enabled() -> Option<bool> {
    static MODE: OnceLock<Option<bool>> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("PFS_NN_CHILD_INPUTS").as_deref() {
        Ok("1") | Ok("parallel") => Some(true),
        Ok("serial") => Some(false),
        Err(_) => crate::sparse_nn::cpu_defaults::available().then_some(true),
        Ok("0") | Ok("off") => None,
        Ok(other) => panic!("Unknown PFS_NN_CHILD_INPUTS={other}"),
    })
}

fn river_children_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = match std::env::var("PFS_NN_RIVER_CHILD_INPUTS_EXPERIMENT").as_deref() {
            Ok("0") | Ok("off") => false,
            Err(_) | Ok("1") | Ok("on") => true,
            Ok(other) => panic!("Unknown PFS_NN_RIVER_CHILD_INPUTS_EXPERIMENT={other}"),
        };
        if enabled { eprintln!("NN_RIVER_CHILD_INPUTS enabled=true"); }
        enabled
    })
}

fn river_children_parallel(default: bool) -> bool {
    static MODE: OnceLock<Option<bool>> = OnceLock::new();
    MODE.get_or_init(|| match std::env::var("PFS_NN_RIVER_CHILD_SCHEDULING_EXPERIMENT").as_deref() {
        Ok("serial") => { eprintln!("NN_RIVER_CHILD_SCHEDULING serial"); Some(false) },
        Ok("parallel") => { eprintln!("NN_RIVER_CHILD_SCHEDULING parallel"); Some(true) },
        Err(_) => None,
        Ok(other) => panic!("Unknown PFS_NN_RIVER_CHILD_SCHEDULING_EXPERIMENT={other}"),
    }).unwrap_or(default)
}

pub(super) fn try_solve<T: Game>(
    result: &mut [MaybeUninit<f32>],
    game: &T,
    node: &mut T::Node,
    conversion: &crate::pio_order::TurnCfvConversion,
    basis_actions: &[usize],
    player: usize,
    cfreach: &[f32],
    player_reach: Option<&[f32]>,
    child_cfreach: &[f32],
    child_own: Option<&[f32]>,
    params: &DiscountParams,
    outer_iteration: u32,
    update: bool,
    below_batch_frontier: bool,
    use_cached_predictions: bool,
) -> bool {
    let Some(parallel) = enabled() else {
        return false;
    };
    if !update || (node.board_turn() != NOT_DEALT && !river_children_enabled())
        || conversion.child_input_model(player).is_none() {
        return false;
    }
    let parallel = if node.board_turn() == NOT_DEALT { parallel }
        else { river_children_parallel(parallel) };
    let own = player_reach.expect("Child NN inputs require same-pass own reach");
    let hands = result.len();
    let prepare = |basis: usize| {
        let mut raw = crate::sparse_nn::buffer_pool::take_raw(hands);
        solve_recursive(
            &mut raw.spare_capacity_mut()[..hands],
            game,
            &mut node.play(basis_actions[basis]),
            player,
            child_cfreach,
            child_own,
            params,
            outer_iteration,
            update,
            below_batch_frontier,
            use_cached_predictions,
        );
        unsafe {
            raw.set_len(hands);
        }
        // The task returns a prepared feature block, square terms and baseline
        // products. Its raw row is retained only to preserve diagnostic capture.
        conversion.prepare_child_input(player, basis, raw)
    };
    let shared =
        || conversion.prepare_common_input(node.id(), player, outer_iteration, cfreach, own);
    #[cfg(feature = "rayon")]
    let (blocks, common) = if parallel && basis_actions.len() > 1 {
        rayon::join(
            || {
                (0..basis_actions.len())
                    .into_par_iter()
                    .map(prepare)
                    .collect::<Vec<_>>()
            },
            shared,
        )
    } else {
        let common = shared();
        let blocks = (0..basis_actions.len()).map(prepare).collect::<Vec<_>>();
        (blocks, common)
    };
    #[cfg(not(feature = "rayon"))]
    let (blocks, common) = {
        let _ = parallel;
        let common = shared();
        let blocks = (0..basis_actions.len()).map(prepare).collect::<Vec<_>>();
        (blocks, common)
    };
    if game.capture_turn_cfvs(node) {
        let raw: Vec<_> = blocks
            .iter()
            .flat_map(|b| b.raw_cfvs.iter())
            .copied()
            .collect();
        let cards = node.chance_cards();
        capture_cfv_actions(
            node.id(),
            outer_iteration,
            player,
            &raw,
            cfreach,
            player_reach,
            node.board_turn(),
            basis_actions.iter().map(|&i| cards[i]).collect(),
            None,
            Some(basis_actions),
        );
    }
    assert!(game.isomorphic_chances(node).is_empty());
    let estimated = conversion.estimate_child_inputs(player, &blocks, common);
    for (out, value) in result.iter_mut().zip(estimated) {
        out.write(value as f32);
    }
    true
}
