// SparseSolver — Copyright (C) 2026 Jerzy Slaby (DeepBorys).
// Uses the postflop-solver engine by Wataru Inariba; see NOTICE.
// SPDX-License-Identifier: AGPL-3.0-or-later
// Modified for SparseSolver inference and benchmarks; see NOTICE and LICENSE.

use postflop_solver::*;
use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
};

pub fn export_flop(game: &mut PostFlopGame, file: &Path, model: &str, iterations: u32) {
    let mut out = BufWriter::new(fs::File::create(file).unwrap());
    writeln!(
        out,
        "model,iterations,path,player,action_index,action,hand,probability,compatible_weight"
    )
    .unwrap();
    fn walk(
        g: &mut PostFlopGame,
        history: &mut Vec<usize>,
        out: &mut impl Write,
        model: &str,
        iterations: u32,
    ) {
        g.apply_history(history);
        if g.is_terminal_node() || g.is_chance_node() {
            return;
        }
        g.cache_normalized_weights();
        let p = g.current_player();
        let actions = g.available_actions();
        let strategy = g.strategy();
        let weights = g.compatible_weights_for_analysis(p);
        let hands = g.private_cards(p);
        let path = history
            .iter()
            .map(|a| format!("a{a}"))
            .collect::<Vec<_>>()
            .join("/");
        for (a, action) in actions.iter().enumerate() {
            for (h, &cards) in hands.iter().enumerate() {
                let q = strategy[a * hands.len() + h];
                assert!(q.is_finite() && (0.0..=1.000001).contains(&q));
                writeln!(
                    out,
                    "{model},{iterations},{path},{p},{a},{action:?},{},{q:.12},{:.12}",
                    hole_to_string(cards).unwrap(),
                    weights[h]
                )
                .unwrap();
            }
        }
        let count = actions.len();
        for a in 0..count {
            history.push(a);
            walk(g, history, out, model, iterations);
            history.pop();
        }
    }
    walk(game, &mut Vec::new(), &mut out, model, iterations);
    out.flush().unwrap();
    game.back_to_root();
}
