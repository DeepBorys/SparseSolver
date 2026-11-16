//! Optional read-only current-strategy evaluation. Both values share one state.
use super::*;

#[derive(Debug)]
pub struct PairedCfvState {
    pub node_id: u64,
    /// 0 decision, 1 terminal, 2 physical chance, 3 neural chance, 4 exact all-in.
    pub kind: u8,
    pub acting_player: u8,
    pub path: Vec<u32>,
    pub runout: Vec<u8>,
    pub reaches: [Vec<f32>; 2],
    pub values: [Vec<f32>; 2],
}

/// Evaluate every represented node after a completed CFR iteration. This never
/// changes regrets, cumulative strategies, or the normal solver traversal.
/// Neural outputs retain the configured approximation; matrix leaves stay exact.
pub fn paired_cfv_snapshot<T: Game>(game: &T, outer_iteration: u32) -> Vec<PairedCfvState> {
    assert!(outer_iteration > 0);
    assert!(game.is_ready() && !game.is_solved());
    assert!(!game.is_raked(), "Zero-sum diagnostic requires no rake");
    assert!(
        !game.has_auxiliary_subgame(),
        "Auxiliary continuations are not supported by this diagnostic"
    );
    let reaches = [
        game.initial_weights(0).to_vec(),
        game.initial_weights(1).to_vec(),
    ];
    let mut captures = Vec::new();
    paired_visit(
        game,
        &mut game.root(),
        &reaches,
        &DiscountParams::new(outer_iteration - 1),
        outer_iteration,
        &mut Vec::new(),
        &mut Vec::new(),
        &mut captures,
        true,
    );
    captures
}

fn paired_visit<T: Game>(
    game: &T,
    node: &mut T::Node,
    reaches: &[Vec<f32>; 2],
    params: &DiscountParams,
    iteration: u32,
    path: &mut Vec<u32>,
    runout: &mut Vec<u8>,
    captures: &mut Vec<PairedCfvState>,
    capture: bool,
) -> [Vec<f32>; 2] {
    let mut values = [
        vec![0.; game.num_private_hands(0)],
        vec![0.; game.num_private_hands(1)],
    ];
    let mut kind = if node.is_terminal() {
        1
    } else if node.is_chance() {
        2
    } else {
        0
    };
    let exact = EXACT_CALLED_ALLINS.load(Ordering::Relaxed)
        && node.is_chance()
        && game.should_traverse_chance_exactly(node);
    let mut matrix_leaf = false;
    if node.is_terminal() || exact {
        let mut handled = [false; 2];
        for player in 0..2 {
            let mut result = vec![MaybeUninit::<f32>::uninit(); values[player].len()];
            if node.is_terminal() {
                game.evaluate(&mut result, node, player, &reaches[player ^ 1]);
                handled[player] = true;
            } else {
                handled[player] =
                    game.evaluate_called_allin(&mut result, node, player, &reaches[player ^ 1]);
            }
            if handled[player] {
                for (out, value) in values[player].iter_mut().zip(result) {
                    *out = unsafe { value.assume_init() };
                }
            }
        }
        assert_eq!(handled[0], handled[1], "One-sided exact all-in handling");
        matrix_leaf = handled[0];
        if exact && matrix_leaf {
            kind = 4;
        }
    }
    if !matrix_leaf {
        if node.is_chance() {
            let conversion = (!exact).then(|| game.turn_cfv_conversion(node)).flatten();
            let basis = conversion
                .filter(|c| PRUNE_CHANCE_BRANCHES.load(Ordering::Relaxed) && c.use_in_solving())
                .map(|c| c.basis_actions());
            // Capturing represented nodes is unambiguous only without omitted
            // isomorphic subtrees. The requested pruned tree has no isomorphism.
            assert!(
                game.isomorphic_chances(node).is_empty(),
                "Paired diagnostic requires no chance isomorphism"
            );
            let actions: Vec<usize> =
                basis.map_or_else(|| (0..node.num_actions()).collect(), |a| a.to_vec());
            let cards = node.chance_cards();
            let factor = 1. / game.chance_factor(node) as f32;
            let child_reaches = [
                reaches[0].iter().map(|r| r * factor).collect(),
                reaches[1].iter().map(|r| r * factor).collect(),
            ];
            let mut samples = [Vec::new(), Vec::new()];
            for &action in &actions {
                path.push(action as u32);
                runout.push(cards[action]);
                let child = paired_visit(
                    game,
                    &mut node.play(action),
                    &child_reaches,
                    params,
                    iteration,
                    path,
                    runout,
                    captures,
                    capture,
                );
                path.pop();
                runout.pop();
                for player in 0..2 {
                    samples[player].extend_from_slice(&child[player]);
                }
            }
            for player in 0..2 {
                let mut total = vec![MaybeUninit::<f64>::uninit(); values[player].len()];
                if let Some(basis) = basis {
                    kind = 3;
                    let estimate = conversion.unwrap().estimate_sum_from_basis_at_iteration(
                        node.id(),
                        player,
                        &samples[player],
                        basis,
                        iteration,
                        &reaches[player ^ 1],
                        Some(&reaches[player]),
                    );
                    for (out, value) in total.iter_mut().zip(estimate) {
                        out.write(value);
                    }
                } else {
                    if conversion.is_some_and(|c| c.use_in_solving()) {
                        kind = 3;
                    }
                    sum_chance_cfvs(
                        &mut total,
                        &samples[player],
                        node.id(),
                        player,
                        iteration,
                        conversion,
                        &reaches[player ^ 1],
                        Some(&reaches[player]),
                    );
                }
                for (out, value) in values[player].iter_mut().zip(total) {
                    *out = unsafe { value.assume_init() } as f32;
                }
            }
        } else {
            let actor = node.player();
            let actions = node.num_actions();
            let mut strategy = if actions == 1 {
                // Ordinary solving bypasses storage at forced single actions.
                vec![1.; reaches[actor].len()]
            } else if game.is_compression_enabled() {
                regret_matching_compressed(
                    node.regrets_compressed(),
                    actions,
                    params.qre_temperature,
                )
                .to_vec()
            } else {
                regret_matching(node.regrets(), actions, params.qre_temperature).to_vec()
            };
            apply_locking_strategy(&mut strategy, game.locking_strategy(node));
            let mut sums = [vec![0f64; values[0].len()], vec![0f64; values[1].len()]];
            for action in 0..actions {
                let probabilities = row(&strategy, action, reaches[actor].len());
                let mut child_reaches = [reaches[0].clone(), reaches[1].clone()];
                for (r, p) in child_reaches[actor].iter_mut().zip(probabilities) {
                    *r *= p;
                }
                path.push(action as u32);
                let child = paired_visit(
                    game,
                    &mut node.play(action),
                    &child_reaches,
                    params,
                    iteration,
                    path,
                    runout,
                    captures,
                    capture,
                );
                path.pop();
                for player in 0..2 {
                    for (hand, &value) in child[player].iter().enumerate() {
                        sums[player][hand] += value as f64
                            * if actor == player {
                                probabilities[hand] as f64
                            } else {
                                1.
                            };
                    }
                }
            }
            for player in 0..2 {
                for (out, value) in values[player].iter_mut().zip(&sums[player]) {
                    *out = *value as f32;
                }
            }
        }
    }
    if capture { captures.push(PairedCfvState {
        node_id: node.id(),
        kind,
        acting_player: if kind == 0 { node.player() as u8 } else { 255 },
        path: path.clone(),
        runout: runout.clone(),
        reaches: reaches.clone(),
        values: values.clone(),
    }); }
    values
}

/// Same-current-strategy sampled CFVs for both heads, without collecting a dump.
#[cfg(feature="neural")]
pub(super) fn paired_frontier_samples<T: Game>(game: &T,node: &T::Node,reaches: &[Vec<f32>;2],
    params: &DiscountParams,iteration: u32) -> [Vec<f32>;2] {
    let conversion=game.turn_cfv_conversion(node).expect("Missing joint turn conversion");
    let factor=1. / game.chance_factor(node) as f32;
    let child_reaches=[reaches[0].iter().map(|r|r*factor).collect(),reaches[1].iter().map(|r|r*factor).collect()];
    let mut output=[Vec::new(),Vec::new()];
    for &action in conversion.basis_actions() {
        let child=paired_visit(game,&mut node.play(action),&child_reaches,params,iteration,
            &mut Vec::new(),&mut Vec::new(),&mut Vec::new(),false);
        for p in 0..2 {output[p].extend_from_slice(&child[p]);}
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    fn fingerprint<T: Game>(node: &T::Node, result: &mut Vec<u32>) {
        if node.is_terminal() {
            return;
        }
        if !node.is_chance() {
            result.extend(
                node.strategy()
                    .iter()
                    .chain(node.regrets())
                    .map(|v| v.to_bits()),
            );
        }
        for action in 0..node.num_actions() {
            fingerprint::<T>(&node.play(action), result);
        }
    }

    #[test]
    fn paired_exact_values_are_zero_sum_and_do_not_change_solver_state() {
        for (street, betting) in [(BoardState::River, true), (BoardState::Turn, true), (BoardState::River, false)] {
            let cards = CardConfig {
                range: [
                    "AA,AKs,QTs,88:0.5".parse().unwrap(),
                    "KK,AQs,JTs,99:0.4".parse().unwrap(),
                ],
                flop: flop_from_str("QsJh2h").unwrap(),
                turn: card_from_str("3d").unwrap(),
                river: if street == BoardState::River {
                    card_from_str("5d").unwrap()
                } else {
                    NOT_DEALT
                },
            };
            let sizes = BetSizeOptions::try_from((if betting { "50%" } else { "" }, "")).unwrap();
            let config = TreeConfig {
                initial_state: street,
                starting_pot: 180,
                effective_stack: 910,
                flop_bet_sizes: [sizes.clone(), sizes.clone()],
                turn_bet_sizes: [sizes.clone(), sizes.clone()],
                river_bet_sizes: [sizes.clone(), sizes],
                add_allin_threshold: 0.,
                force_allin_threshold: 0.,
                ..Default::default()
            };
            let mut game =
                PostFlopGame::with_config(cards, ActionTree::new(config).unwrap()).unwrap();
            game.allocate_memory(false);
            for i in 0..5 {
                solve_step(&game, i);
            }
            let mut before = Vec::new();
            fingerprint::<PostFlopGame>(&game.root(), &mut before);
            let captures = paired_cfv_snapshot(&game, 5);
            assert!(captures.len() > 2);
            for c in &captures {
                let sum: f64 = (0..2)
                    .map(|p| {
                        c.values[p]
                            .iter()
                            .zip(&c.reaches[p])
                            .map(|(&v, &r)| v as f64 * r as f64)
                            .sum::<f64>()
                    })
                    .sum();
                assert!(
                    sum.abs() < 2e-5,
                    "kind={} path={:?} sum={}",
                    c.kind,
                    c.path,
                    sum
                );
            }
            let mut after = Vec::new();
            fingerprint::<PostFlopGame>(&game.root(), &mut after);
            assert_eq!(before, after);
        }
    }
}
