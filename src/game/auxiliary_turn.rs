use super::*;
use crate::interface::{Game, GameNode};

pub(super) struct AuxiliaryExactTurn {
    pub flop_actions: Vec<Action>,
    pub turn: Card,
    pub chance_node_id: u64,
    pub action: usize,
    pub extra_nodes: [u64; 3],
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "neural")]
    use crate::pio_order::{NeuralTurnCfvModel, TurnCfvConversion};
    use crate::{card_from_str, flop_from_str, set_prune_chance_branches};
    #[cfg(feature = "neural")]
    use crate::{finalize, solve_step};

    fn cards() -> CardConfig {
        CardConfig {
            range: ["AA,AK,87s".parse().unwrap(), "KK,KQ,T9s".parse().unwrap()],
            flop: flop_from_str("QsJh2h").unwrap(),
            ..Default::default()
        }
    }

    fn config() -> TreeConfig {
        let flop: crate::BetSizeOptions = ("52%", "").try_into().unwrap();
        let turn: crate::BetSizeOptions = ("55%", "").try_into().unwrap();
        let river: crate::BetSizeOptions = ("70%", "").try_into().unwrap();
        TreeConfig {
            starting_pot: 180,
            effective_stack: 910,
            flop_bet_sizes: [flop.clone(), flop],
            turn_bet_sizes: [turn.clone(), turn],
            river_bet_sizes: [river.clone(), river],
            add_allin_threshold: 0.,
            force_allin_threshold: 0.,
            ..Default::default()
        }
    }

    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            set_turn_basis_count(0);
            set_river_basis_count(0);
            set_custom_turn_basis_cards(&[]);
            set_custom_river_basis_cards(&[]);
            set_exact_allin_matrix_tree(false);
            set_prune_chance_branches(false);
        }
    }
    fn setup() -> Reset {
        set_exact_allin_matrix_tree(true);
        set_turn_basis_count(6);
        set_river_basis_count(6);
        set_custom_turn_basis_cards(
            &["3d", "4h", "6d", "9c", "Tc", "Ks"].map(|c| card_from_str(c).unwrap()),
        );
        set_custom_river_basis_cards(
            &["2d", "5d", "7s", "8c", "Th", "Qc"].map(|c| card_from_str(c).unwrap()),
        );
        set_prune_chance_branches(true);
        Reset
    }
    fn new_aux(turn: &str) -> PostFlopGame {
        PostFlopGame::with_auxiliary_exact_turn(
            cards(),
            ActionTree::new(config()).unwrap(),
            &[Action::Check, Action::Bet(94), Action::Call],
            card_from_str(turn).unwrap(),
        )
        .unwrap()
    }
    #[cfg(feature = "neural")]
    fn install_sum(game: &mut PostFlopGame) {
        install_scaled_sum(game, 1.0);
    }
    #[cfg(feature = "neural")]
    fn install_scaled_sum(game: &mut PostFlopGame, multiplier: f64) {
        let id = game.turn_cfv_capture_contexts()[0].0;
        let modules = std::array::from_fn(|_| {
            let inputs = [
                tch::Tensor::zeros([1, 1326, 6], (tch::Kind::Float, tch::Device::Cpu)),
                tch::Tensor::zeros([1, 1], (tch::Kind::Float, tch::Device::Cpu)),
            ];
            let module =
                tch::CModule::create_by_tracing("TestTurnSum", "forward", &inputs, &mut |xs| {
                    vec![xs[0].sum_dim_intlist(&[2i64][..], false, tch::Kind::Float) * multiplier]
                })
                .unwrap();
            std::sync::Mutex::new(module)
        });
        game.turn_cfv_conversion = Some(TurnCfvConversion {
            node_id: id,
            action_count: 6,
            use_in_solving: true,
            all_turn_nodes: true,
            basis_actions: (0..6).collect(),
            weights: std::array::from_fn(|p| vec![49.0 / 6.0; 6 * game.private_cards[p].len()]),
            bucket_model: None,
            neural_model: Some(NeuralTurnCfvModel {
                modules: Some(modules),
                sparse_models: None,
                direct_contexts: None,
                contexts: game
                    .turn_cfv_capture_contexts()
                    .iter()
                    .map(|row| (row.0, vec![0.]))
                    .collect(),
                mappings: std::array::from_fn(|p| (0..game.private_cards[p].len()).collect()),
                range_features: None,
                current_reaches: false,
                sum_cfvs: false,
                iteration_scale: None,
                deepstack_root_combinations: None,
                deepstack_chance_count: 45.,
                pots: Default::default(),
                device: tch::Device::Cpu,
                batch_state: Default::default(),
            }),
        });
        game.set_weighted_all_turn_sums(true).unwrap();
        game.allocate_memory(false);
    }
    #[cfg(feature = "neural")]
    fn compare(a: &PostFlopGame, ai: usize, b: &PostFlopGame, bi: usize) {
        let an = a.node_arena[ai].lock();
        let bn = b.node_arena[bi].lock();
        assert_eq!(an.prev_action, bn.prev_action);
        if !an.is_terminal() && !an.is_chance() {
            assert_eq!(
                an.strategy(),
                bn.strategy(),
                "strategy differs at {ai}/{bi}"
            );
            if !a.is_solved() {
                assert_eq!(an.regrets(), bn.regrets());
            }
        }
        for action in 0..an.num_actions() {
            compare(
                a,
                an.play(action).id as usize,
                b,
                bn.play(action).id as usize,
            );
        }
    }

    #[test]
    fn auxiliary_exact_turn_preserves_six_inputs_and_flop_updates() {
        let _reset = setup();
        let baseline =
            PostFlopGame::with_config(cards(), ActionTree::new(config()).unwrap()).unwrap();
        let aux = new_aux("As");
        assert_eq!(aux.prepared_showdown_runouts_experiment(), 36 + 48);
        let (id, turn, action) = aux.auxiliary_exact_turn_info().unwrap();
        assert_eq!(action, 6);
        for entry in &aux.node_arena {
            let node = entry.lock();
            if node.is_chance() && node.turn == NOT_DEALT && node.num_actions() > 0 {
                assert_eq!(node.num_actions(), if node.id == id { 7 } else { 6 });
            } else if node.is_chance() && node.turn == turn {
                assert_eq!(node.num_actions(), 48);
                assert_eq!(aux.chance_factor(&node), 44);
                assert!(aux.turn_cfv_conversion(&node).is_none());
            }
        }
        for river in 0..52 {
            if river == turn || cards().flop.contains(&river) {
                continue;
            }
            let index = card_pair_to_index(turn, river);
            assert!(aux.hand_strength[index] == baseline.hand_strength[index]);
            assert_eq!(
                aux.valid_indices_river[index],
                baseline.valid_indices_river[index]
            );
        }
        #[cfg(feature = "neural")]
        {
            let mut baseline = baseline;
            let mut aux = aux;
            install_sum(&mut baseline);
            install_sum(&mut aux);
            for t in 0..20 {
                solve_step(&baseline, t);
                solve_step(&aux, t);
            }
            compare(&baseline, 0, &aux, 0);
            let extra_root = aux.node_arena[id as usize].lock().play(action).id as usize;
            assert!(aux.node_arena[extra_root]
                .lock()
                .regrets()
                .iter()
                .any(|&r| r != 0.));
            let mut batched = new_aux("As");
            install_sum(&mut batched);
            crate::set_batch_neural_turn_frontiers(true);
            for t in 0..20 {
                solve_step(&batched, t);
            }
            crate::set_batch_neural_turn_frontiers(false);
            compare(&aux, 0, &batched, 0);
            finalize(&mut baseline);
            finalize(&mut aux);
            compare(&baseline, 0, &aux, 0);
            assert_eq!(
                crate::compute_current_ev(&baseline),
                crate::compute_current_ev(&aux)
            );
        }
    }

    #[cfg(feature = "neural")]
    #[test]
    fn neural_display_finalization_preserves_policy_and_excludes_auxiliary_row() {
        let _reset = setup();
        let mut physical_only = PostFlopGame::with_config(cards(), ActionTree::new(config()).unwrap()).unwrap();
        let mut corrected = PostFlopGame::with_config(cards(), ActionTree::new(config()).unwrap()).unwrap();
        let mut auxiliary = new_aux("7h");
        for game in [&mut physical_only, &mut corrected, &mut auxiliary] {
            install_scaled_sum(game, 2.0);
            for iteration in 0..20 { solve_step(game, iteration); }
        }
        finalize(&mut physical_only);
        crate::finalize_with_neural_frontiers(&mut corrected, 20);
        crate::finalize_with_neural_frontiers(&mut auxiliary, 20);
        compare(&physical_only, 0, &corrected, 0);
        compare(&corrected, 0, &auxiliary, 0);
        // Exact evaluation ignores display caches and remains bit-identical.
        assert_eq!(crate::compute_current_ev(&physical_only), crate::compute_current_ev(&corrected));
        assert_eq!(crate::compute_mes_ev(&physical_only), crate::compute_mes_ev(&corrected));
        physical_only.cache_normalized_weights();
        corrected.cache_normalized_weights();
        auxiliary.cache_normalized_weights();
        let old = physical_only.expected_values_detail(0);
        let fixed = corrected.expected_values_detail(0);
        assert!(old.iter().zip(&fixed).any(|(a,b)| (a-b).abs() > 0.01),
            "Regression fixture must expose the omitted frontier conversion");
        assert_eq!(fixed, auxiliary.expected_values_detail(0),
            "The extra exact turn must not feed back into the neural aggregate");
    }

    #[test]
    fn auxiliary_exact_turn_rejects_invalid_lines_and_basis_collisions() {
        let _reset = setup();
        for (path, turn) in [
            (vec![Action::Check, Action::Bet(95), Action::Call], "As"),
            (vec![Action::Check], "As"),
            (vec![Action::Check, Action::Bet(94), Action::Call], "3d"),
            (vec![Action::Check, Action::Bet(94), Action::Call], "Qs"),
        ] {
            assert!(PostFlopGame::with_auxiliary_exact_turn(
                cards(),
                ActionTree::new(config()).unwrap(),
                &path,
                card_from_str(turn).unwrap()
            )
            .is_err());
        }
        // A low card is still appended, not inserted before the model rows.
        let aux = new_aux("2c");
        assert_eq!(aux.auxiliary_exact_turn_info().unwrap().2, 6);
    }

    #[cfg(feature = "neural")]
    #[test]
    fn frozen_subgame_step_preserves_trunk_and_matches_fixed_reach_full_updates() {
        let _reset = setup();
        let turn = card_from_str("7h").unwrap();
        let mut full = new_aux("7h");
        let mut local = new_aux("7h");
        install_sum(&mut full);
        install_sum(&mut local);
        for game in [&mut full, &mut local] {
            for (index, entry) in game.node_arena.iter().enumerate() {
                let mut node = entry.lock();
                if node.turn == NOT_DEALT && !node.is_chance() && !node.is_terminal() {
                    node.is_locked = true;
                    game.locking_strategy.insert(index,
                        vec![1. / node.num_actions() as f32; node.num_elements as usize]);
                }
            }
        }
        for t in 0..20 { solve_step(&full, t); solve_step(&local, t); }
        let mut path = Vec::new();
        let mut id = 0;
        let mut reaches = [local.initial_weights(0).to_vec(), local.initial_weights(1).to_vec()];
        for action in [Action::Check, Action::Bet(94), Action::Call, Action::Chance(turn)] {
            let node = local.node_arena[id].lock();
            let selected = (0..node.num_actions()).find(|&i| node.play(i).prev_action == action).unwrap();
            if node.is_chance() {
                let factor = 1. / local.chance_factor(&node) as f32;
                for range in &mut reaches { for v in range { *v *= factor; } }
            } else {
                let factor = 1. / node.num_actions() as f32;
                for v in &mut reaches[node.player()] { *v *= factor; }
            }
            id = node.play(selected).id as usize;
            path.push(selected);
        }
        let outside = |game: &PostFlopGame| -> Vec<_> {
            game.node_arena.iter().filter_map(|entry| {
                let node = entry.lock();
                (node.turn != turn && !node.is_chance() && !node.is_terminal()).then(||
                    (node.id, node.strategy().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                              node.regrets().iter().map(|v| v.to_bits()).collect::<Vec<_>>()))
            }).collect()
        };
        let before = outside(&local);
        let inside_before = local.node_arena[id].lock().regrets().to_vec();
        for t in 20..40 {
            solve_step(&full, t);
            crate::solve_subgame_step(&local, &path, [&reaches[0], &reaches[1]], t);
        }
        assert_eq!(before, outside(&local));
        assert_ne!(inside_before, local.node_arena[id].lock().regrets());
        compare(&full, id, &local, id);
    }

    #[cfg(feature = "neural")]
    #[test]
    fn auxiliary_exact_turn_matches_ordinary_exact_subtree_under_identical_reaches() {
        let _reset = setup();
        let mut aux = new_aux("As");
        install_sum(&mut aux);
        let (id, turn, action) = aux.auxiliary_exact_turn_info().unwrap();
        let extra_root = aux.node_arena[id as usize].lock().play(action).id as usize;

        // Independent ordinary physical traversal: only As is represented at
        // the flop, but every river is built, with no neural conversions.
        // Its parent value is intentionally not an exploitability benchmark.
        set_turn_basis_count(1);
        set_custom_turn_basis_cards(&[turn]);
        set_river_basis_count(0);
        set_custom_river_basis_cards(&[]);
        let mut physical =
            PostFlopGame::with_config(cards(), ActionTree::new(config()).unwrap()).unwrap();
        physical.allocate_memory(false);

        // Fix the same flop policy in both games so their exact subgames see
        // precisely the same reach history, independent of parent feedback.
        for game in [&mut aux, &mut physical] {
            for (index, entry) in game.node_arena.iter().enumerate() {
                let mut node = entry.lock();
                if node.turn == NOT_DEALT && !node.is_chance() && !node.is_terminal() {
                    node.is_locked = true;
                    game.locking_strategy.insert(
                        index,
                        vec![1.0 / node.num_actions() as f32; node.num_elements as usize],
                    );
                }
            }
        }
        let mut physical_id = 0;
        for action in [
            Action::Check,
            Action::Bet(94),
            Action::Call,
            Action::Chance(turn),
        ] {
            let node = physical.node_arena[physical_id].lock();
            physical_id = (0..node.num_actions())
                .find_map(|i| {
                    let child = node.play(i);
                    (child.prev_action == action).then_some(child.id as usize)
                })
                .unwrap();
        }
        for iteration in 0..20 {
            solve_step(&aux, iteration);
            solve_step(&physical, iteration);
        }
        compare(&physical, physical_id, &aux, extra_root);
    }
}

impl PostFlopGame {
    /// Adds one exact, all-river continuation at a particular flop betting line.
    /// The extra card is NOT a seventh neural input and its value is NOT added
    /// to the predicted 49-turn sum. Its regrets use the live flop reaches on
    /// every alternating CFR update. This is not a standalone safe resolve or
    /// a complete-tree exploitability benchmark.
    pub fn with_auxiliary_exact_turn(
        card_config: CardConfig,
        action_tree: ActionTree,
        flop_actions: &[Action],
        turn: Card,
    ) -> Result<Self, String> {
        Self::check_pruned_card_table_config(&card_config)?;
        let mut game = Self::new();
        game.update_config_with_card_tables(
            card_config,
            action_tree,
            true,
            Some(AuxiliaryExactTurn {
                flop_actions: flop_actions.to_vec(),
                turn,
                chance_node_id: u64::MAX,
                action: usize::MAX,
                extra_nodes: [0; 3],
            }),
        )?;
        Ok(game)
    }

    pub(super) fn validate_auxiliary_exact_turn(&mut self, basis: &[Card]) -> Result<(), String> {
        let Some(aux) = &mut self.auxiliary_exact_turn else {
            return Ok(());
        };
        if aux.turn >= 52 || self.card_config.flop.contains(&aux.turn) || basis.contains(&aux.turn)
        {
            return Err(
                "Auxiliary turn must be legal and distinct from every neural basis turn".into(),
            );
        }
        fn find(node: &ActionTreeNode, path: &[Action], stack: i32) -> Result<[u64; 3], String> {
            if path.is_empty() {
                if node.player & PLAYER_CHANCE_FLAG == 0 || node.amount == stack {
                    return Err(
                        "Auxiliary line must end at a non-all-in flop-to-turn chance node".into(),
                    );
                }
                let counts =
                    count_num_action_nodes_exact_allin_matrix(&node.children[0].lock(), stack);
                return Ok([0, counts[1], counts[2] * 48]);
            }
            if node.player & (PLAYER_TERMINAL_FLAG | PLAYER_CHANCE_FLAG) != 0 {
                return Err("Auxiliary betting line must contain flop actions only".into());
            }
            let index = node
                .actions
                .iter()
                .position(|action| *action == path[0])
                .ok_or_else(|| format!("Auxiliary flop action {:?} is unavailable", path[0]))?;
            find(&node.children[index].lock(), &path[1..], stack)
        }
        aux.extra_nodes = find(
            &self.action_root.lock(),
            &aux.flop_actions,
            self.tree_config.effective_stack,
        )?;
        Ok(())
    }

    pub(super) fn auxiliary_at_path(&self, path: &[Action]) -> Option<Card> {
        self.auxiliary_exact_turn
            .as_ref()
            .filter(|aux| aux.flop_actions == path)
            .map(|aux| aux.turn)
    }

    pub(super) fn is_auxiliary_turn(&self, turn: Card) -> bool {
        self.auxiliary_exact_turn
            .as_ref()
            .is_some_and(|aux| aux.turn == turn)
    }

    pub(super) fn resolve_auxiliary_exact_turn(&mut self) {
        let Some(aux) = &self.auxiliary_exact_turn else {
            return;
        };
        let mut id = 0;
        for action in &aux.flop_actions {
            let node = self.node_arena[id].lock();
            id = (0..node.num_actions())
                .find_map(|index| {
                    let child = node.play(index);
                    (child.prev_action == *action).then_some(child.id as usize)
                })
                .expect("validated auxiliary flop line");
        }
        let action = self.node_arena[id].lock().num_actions() - 1;
        assert_eq!(
            self.node_arena[id].lock().play(action).prev_action,
            Action::Chance(aux.turn)
        );
        let aux = self.auxiliary_exact_turn.as_mut().unwrap();
        aux.chance_node_id = id as u64;
        aux.action = action;
    }

    /// (flop chance node ID, extra turn card, physical action index).
    pub fn auxiliary_exact_turn_info(&self) -> Option<(u64, Card, usize)> {
        self.auxiliary_exact_turn
            .as_ref()
            .map(|aux| (aux.chance_node_id, aux.turn, aux.action))
    }

    pub(super) fn neural_turn_action_count(&self, node: &PostFlopNode) -> usize {
        node.num_actions() - usize::from(self.auxiliary_chance_action(node).is_some())
    }
}
