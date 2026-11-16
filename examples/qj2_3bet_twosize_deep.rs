// Based on the postflop-solver example by Wataru Inariba (b-inary).
// Original example: Copyright (C) 2022 Wataru Inariba.
// SparseSolver modifications: Copyright (C) 2026 Jerzy Slaby (DeepBorys).
// SPDX-License-Identifier: AGPL-3.0-or-later
// Modified for SparseSolver inference and benchmarks; see NOTICE and LICENSE.

use postflop_solver::*;
#[path = "support/nn_model.rs"]
mod nn_model;

fn main() {
    // Edit these ranges just as in Wataru's original example.
    let oop_range = "AA,KK,QQ,JJ,TT,99,88,AK,AQ,AJ,ATs,ATo:0.75,A9s,A8s,A5s:0.5,A4s:0.5,A3s:0.5,A2s:0.5,KQ,KJs,KJo:0.75,KTs,KTo:0.25,K9s,QJs,QJo:0.5,QTs,Q9s,JTs,JTo:0.25,J9s,J8s,T9s,T8s,T7s:0.45,98s,97s,96s:0.45,87s,86s:0.75,85s:0.45,76s:0.75,75s:0.75,74s:0.45,65s:0.75,64s:0.5,63s:0.45,54s:0.75,53s:0.5,52s:0.45,43s:0.5,42s:0.45,32s:0.45";
    let ip_range = "AA:0.25,99,88,77,66,55,44,33,22,AQo,AJ,AT,A9,A8,A7s,A6s,A5s,A4s,A3s,A2s,KQ,KJ,KT,K9,K8s,K7s,K6s,K5s,K4s,K3s,K2s,QJ,QT,Q9,Q8s,Q7s,Q6s,Q5s,Q4s,Q3s,Q2s,JT,J9,J8s,J7s,J6s,T9,T8s,T7s,T6s,98,97s,96s,95s:0.5,87s,86s,85s:0.5,76s,75s,74s:0.5,65s,64s,63s:0.5,54s,53s:0.5,43s";

    let card_config = CardConfig {
        range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
        flop: flop_from_str("QsJh2h").unwrap(), // Change this line to study another flop.
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };

    // The entire betting configuration is visible here, not loaded from TSV.
    let flop_sizes = BetSizeOptions::try_from(("25%,75%", "45%")).unwrap();
    let turn_sizes = BetSizeOptions::try_from(("33%,100%", "45%")).unwrap();
    let river_sizes = BetSizeOptions::try_from(("50%,150%", "45%")).unwrap();
    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: 180,
        effective_stack: 910,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [flop_sizes.clone(), flop_sizes],
        turn_bet_sizes: [turn_sizes.clone(), turn_sizes],
        river_bet_sizes: [river_sizes.clone(), river_sizes],
        turn_donk_sizes: None,
        river_donk_sizes: Some(DonkSizeOptions::try_from("50%").unwrap()),
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
    };

    // Change these two settings to choose a schedule and requested continuations.
    let stages = &[("2x2", 100), ("6x6", 100)];
    let frontiers = &[(
        vec![
            Action::Check,
            Action::Bet(45),
            Action::Call,
            Action::Chance(card_from_str("3d").unwrap()),
        ],
        300,
    )];
    let mut plan = nn_model::solve_plan::Plan::prepare(&card_config, stages, frontiers);

    // The familiar game-construction API is unchanged.
    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    let (mem_usage, mem_usage_compressed) = game.memory_usage();
    println!(
        "Memory usage without compression (32-bit float): {:.2}GB",
        mem_usage as f64 / 1073741824.0
    );
    println!(
        "Memory usage with compression (16-bit integer): {:.2}GB",
        mem_usage_compressed as f64 / 1073741824.0
    );
    game.allocate_memory(false);
    plan.run_strategies(&mut game);
}
