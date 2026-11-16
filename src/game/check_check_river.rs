//! Exact, opt-in turn showdown leaves for strictly check/check-only rivers.
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Jerzy Slaby (DeepBorys)
use super::*;
use crate::interface::{Game, GameNode};

impl PostFlopGame {
    /// Replace proved forced-check river continuations by the existing exact
    /// turn equity matrix, scaled to starting_pot + 2 * node.amount. No stack
    /// is committed and no physical nodes, actions, regrets or averages change.
    /// Requires no rake/bunching and exact matrices for the represented turns.
    /// Missing matrices or a truncated runout are errors, never approximations.
    pub fn enable_check_check_river_matrix(&mut self) -> Result<usize, String> {
        if self.state < State::MemoryAllocated || !self.use_flop_allin_payout_matrix {
            return Err("Check/check river leaves require an allocated game with exact all-in matrices".into());
        }
        if self.tree_config.rake_rate != 0. || self.tree_config.rake_cap != 0.
            || self.bunching_num_dead_cards != 0 || self.partial_card_tables_experiment
            || self.has_auxiliary_subgame() {
            return Err("Check/check river matrix rescaling requires a complete no-rake, no-bunching ordinary game".into());
        }
        fn forced_checks(node: &PostFlopNode, amount: i32) -> bool {
            if node.amount != amount || node.river == NOT_DEALT { return false; }
            if node.is_terminal() {
                return node.player & PLAYER_FOLD_FLAG != PLAYER_FOLD_FLAG;
            }
            !node.is_chance() && node.num_actions() == 1
                && node.play(0).prev_action == Action::Check
                && forced_checks(&node.play(0), amount)
        }
        let mut eligible = std::collections::BTreeSet::new();
        for entry in &self.node_arena {
            let node = entry.lock();
            if !node.is_chance() || node.turn == NOT_DEALT || node.river != NOT_DEALT
                || node.amount == self.tree_config.effective_stack || node.num_actions() == 0 {
                continue;
            }
            if node.children().iter().all(|child| forced_checks(&child.lock(), node.amount)) {
                // Suit-isomorphic river representatives integrate the same
                // physical distribution; all-in matrices always cover all48.
                if node.num_actions() + self.isomorphic_chances(&node).len() != 48 {
                    return Err(format!("Incomplete river runout at node {}", node.id));
                }
                if !self.turn_allin_payout_matrices.contains_key(&node.turn) {
                    return Err(format!("Missing exact turn matrix for card {}", node.turn));
                }
                eligible.insert(node.id);
            }
        }
        self.check_check_river_nodes = eligible;
        Ok(self.check_check_river_nodes.len())
    }

    /// Restore physical river traversal without changing any solver state.
    pub fn disable_check_check_river_matrix(&mut self) {
        self.check_check_river_nodes.clear();
    }

    pub(super) fn showdown_matrix_denominator(&self, node: &PostFlopNode) -> f64 {
        if node.amount == self.tree_config.effective_stack {
            // Preserve historical all-in arithmetic exactly.
            self.num_combinations
        } else {
            debug_assert!(self.check_check_river_nodes.contains(&node.id));
            let allin_pot = f64::from(self.tree_config.starting_pot)
                + 2. * f64::from(self.tree_config.effective_stack);
            let actual_pot = f64::from(self.tree_config.starting_pot) + 2. * f64::from(node.amount);
            self.num_combinations * (allin_pot / actual_pot)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BetSizeOptions;
    use crate::utility::{compute_current_ev, compute_mes_ev};
    use std::mem::MaybeUninit;

    fn game(bets: &str) -> PostFlopGame {
        game_with_band(bets, None)
    }

    fn game_with_band(bets: &str, band: Option<(i32,i32)>) -> PostFlopGame {
        let sizes: BetSizeOptions = ("55%,a", "45%").try_into().unwrap();
        let river: BetSizeOptions = (bets, "45%").try_into().unwrap();
        let cfg = TreeConfig {
            initial_state: BoardState::Turn, starting_pot: 180, effective_stack: 910,
            turn_bet_sizes: [sizes.clone(), sizes], river_bet_sizes: [river.clone(), river],
            add_allin_threshold: 0., force_allin_threshold: 0., ..Default::default()
        };
        let mut tree = ActionTree::new(cfg).unwrap();
        if let Some((min,max)) = band { tree.force_check_only_rivers_in_pot_band(min,max).unwrap(); }
        let mut g = PostFlopGame::with_config(CardConfig {
            range: ["AA,KK,33,AKs,A3s".parse().unwrap(), "QQ,JJ,33,AQs,A3s".parse().unwrap()],
            flop: crate::flop_from_str("Td9d6h").unwrap(), turn: crate::card_from_str("3c").unwrap(),
            river: NOT_DEALT,
        }, tree).unwrap();
        g.allocate_memory(false);
        let turn = g.card_config.turn;
        let canonical = g.build_turn_allin_payout_matrix(turn).unwrap();
        g.install_turn_allin_payout_matrix(turn, &canonical);
        g.use_flop_allin_payout_matrix = true;
        g
    }

    #[test]
    fn check_check_matrix_matches_physical_showdown_at_each_actual_pot() {
        let mut g = game("");
        let expected_ev = compute_current_ev(&g);
        let expected_br = compute_mes_ev(&g);
        let amount_before: Vec<_> = g.node_arena.iter().map(|n| n.lock().amount).collect();
        let eligible = g.enable_check_check_river_matrix().unwrap();
        assert!(eligible >= 2);
        let saved = g.check_check_river_nodes.clone();
        let mut distinct_pots = std::collections::BTreeSet::new();
        fn forced_terminal(g: &PostFlopGame, node: &PostFlopNode, p: usize,
                           reach: &[f32]) -> Vec<MaybeUninit<f32>> {
            if node.is_terminal() {
                let mut out = vec![MaybeUninit::uninit(); g.num_private_hands(p)];
                g.evaluate(&mut out, node, p, reach);
                out
            } else {
                assert_eq!(node.num_actions(), 1);
                forced_terminal(g, &node.play(0), p, reach)
            }
        }
        for id in saved {
            distinct_pots.insert(g.tree_config.starting_pot + 2 * g.node_arena[id as usize].lock().amount);
            for player in 0..2 {
                for fixture in 0..3 {
                    let reach: Vec<_> = g.initial_weights(player ^ 1).iter().enumerate().map(|(i, &w)|
                        if fixture == 2 && i % 3 != 0 { 0. } else { w * ((i*7+fixture*3)%11) as f32 / 10. }).collect();
                    let mut actual = vec![MaybeUninit::uninit(); g.num_private_hands(player)];
                    assert!(g.evaluate_called_allin(&mut actual, &g.node_arena[id as usize].lock(), player, &reach));
                    g.disable_check_check_river_matrix();
                    let mut sums = vec![0f64; actual.len()];
                    let divided: Vec<_> = reach.iter().map(|x| x / 44.).collect();
                    let node = g.node_arena[id as usize].lock();
                    for action in node.action_indices() {
                        let expected = forced_terminal(&g, &node.play(action), player, &divided);
                        for (sum, x) in sums.iter_mut().zip(expected) { *sum += unsafe { x.assume_init() } as f64; }
                    }
                    drop(node);
                    for (x, expected) in actual.into_iter().zip(sums) {
                        let x = unsafe { x.assume_init() } as f64;
                        assert!((x-expected).abs() < 2e-6, "pot {:?}, player {player}: {x} != {expected}", distinct_pots);
                    }
                    g.enable_check_check_river_matrix().unwrap();
                }
            }
        }
        assert!(distinct_pots.len() > 1);
        let actual_ev = compute_current_ev(&g);
        let actual_br = compute_mes_ev(&g);
        for p in 0..2 {
            assert!((expected_ev[p]-actual_ev[p]).abs() < 2e-5);
            assert!((expected_br[p]-actual_br[p]).abs() < 2e-5);
        }
        assert_eq!(amount_before, g.node_arena.iter().map(|n| n.lock().amount).collect::<Vec<_>>());
        g.disable_check_check_river_matrix();
        assert_eq!(expected_ev, compute_current_ev(&g));
        assert_eq!(expected_br, compute_mes_ev(&g));
    }

    #[test]
    fn check_check_matrix_does_not_skip_river_bets_and_rejects_rake() {
        let mut g = game("70%");
        assert_eq!(g.enable_check_check_river_matrix().unwrap(), 0);
        g.tree_config.rake_rate = 0.05;
        assert!(g.enable_check_check_river_matrix().is_err());
    }

    #[test]
    fn hybrid_river_band_keeps_large_pot_betting_and_matches_physical_ev_br() {
        let mut g = game_with_band("70%", Some((50,200)));
        let ev = compute_current_ev(&g); let br = compute_mes_ev(&g);
        assert!(g.enable_check_check_river_matrix().unwrap() > 0);
        for &id in &g.check_check_river_nodes {
            let node = g.node_arena[id as usize].lock();
            assert!((50..=200).contains(&(g.tree_config.starting_pot + 2*node.amount)));
        }
        assert!(g.node_arena.iter().any(|n| {
            let node=n.lock();
            node.river != NOT_DEALT && !node.is_terminal() && !node.is_chance()
                && g.tree_config.starting_pot + 2*node.amount > 200
                && node.children().iter().any(|child| matches!(child.lock().prev_action, Action::Bet(_) | Action::AllIn(_)))
        }));
        let actual_ev=compute_current_ev(&g); let actual_br=compute_mes_ev(&g);
        for p in 0..2 { assert!((ev[p]-actual_ev[p]).abs()<2e-5); assert!((br[p]-actual_br[p]).abs()<2e-5); }
    }

    #[test]
    fn river_pot_band_is_inclusive_and_excludes_uncommitted_stack() {
        for pot in [49,50,200,201] {
            let sizes: BetSizeOptions=("50%,a","3.5x").try_into().unwrap();
            let mut tree=ActionTree::new(TreeConfig {
                initial_state:BoardState::Turn,starting_pot:pot,effective_stack:975,
                turn_bet_sizes:[sizes.clone(),sizes.clone()],river_bet_sizes:[sizes.clone(),sizes],
                ..Default::default()
            }).unwrap();
            tree.force_check_only_rivers_in_pot_band(50,200).unwrap();
            tree.apply_history(&[Action::Check,Action::Check]).unwrap();
            if (50..=200).contains(&pot) {
                assert_eq!(tree.available_actions(), &[Action::Check]);
                tree.play(Action::Check).unwrap(); assert_eq!(tree.available_actions(), &[Action::Check]);
            } else { assert!(tree.available_actions().iter().any(|a| matches!(a,Action::Bet(_) | Action::AllIn(_)))); }
        }
    }
}
