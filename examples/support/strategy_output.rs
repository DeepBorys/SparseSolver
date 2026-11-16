// SparseSolver — Copyright (C) 2026 Jerzy Slaby (DeepBorys).
// Uses the postflop-solver engine by Wataru Inariba; see NOTICE.
// SPDX-License-Identifier: AGPL-3.0-or-later
// Modified for SparseSolver inference and benchmarks; see NOTICE and LICENSE.

//! Console hand-class matrices and complete per-combo CSVs for a selected node.
use postflop_solver::*;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

pub fn print_and_save(game: &mut PostFlopGame, output: &Path, label: &str) {
    assert!(!game.is_chance_node() && !game.is_terminal_node());
    game.cache_normalized_weights();
    let player = game.current_player();
    let actions = game.available_actions();
    let strategy = game.strategy();
    let hands = game.private_cards(player);
    let weights = game.normalized_weights(player);
    let n = hands.len();
    let board = game
        .current_board()
        .iter()
        .map(|&c| card_to_string(c).unwrap())
        .collect::<Vec<_>>()
        .join("");
    println!(
        "\nSTRATEGY {label} board={board} player={} actions={actions:?}",
        if player == 0 { "OOP" } else { "IP" }
    );
    let path = output.join(format!("{label}_strategy.csv"));
    let mut file = BufWriter::new(File::create(&path).unwrap());
    writeln!(
        file,
        "label,board,player,hand,action_index,action,probability,normalized_weight"
    )
    .unwrap();
    let mut masses = [[0.0f64; 13]; 13];
    let mut counts = [[0usize; 13]; 13];
    let mut values = vec![[[0.0f64; 13]; 13]; actions.len()];
    let mut raw = values.clone();
    let board_mask = game
        .current_board()
        .iter()
        .fold(0u64, |m, &c| m | (1u64 << c));
    for (h, &(c, d)) in hands.iter().enumerate() {
        if board_mask & ((1u64 << c) | (1u64 << d)) != 0 {
            continue;
        }
        let a = 12 - (c / 4) as usize;
        let b = 12 - (d / 4) as usize;
        let (row, col) = if c % 4 == d % 4 {
            (a.min(b), a.max(b))
        } else {
            (a.max(b), a.min(b))
        };
        masses[row][col] += weights[h] as f64;
        counts[row][col] += 1;
        let hand = format!(
            "{}{}",
            card_to_string(c).unwrap(),
            card_to_string(d).unwrap()
        );
        for (i, action) in actions.iter().enumerate() {
            let p = strategy[i * n + h];
            assert!(p.is_finite());
            values[i][row][col] += p as f64 * weights[h] as f64;
            raw[i][row][col] += p as f64;
            writeln!(
                file,
                "{label},{board},{player},{hand},{i},{action:?},{p:.9},{:.12}",
                weights[h]
            )
            .unwrap();
        }
    }
    let ranks: Vec<_> = "AKQJT98765432".chars().collect();
    for (i, action) in actions.iter().enumerate() {
        println!("{action:?} frequency % (suited above diagonal; offsuit below)");
        print!("    ");
        for rank in &ranks {
            print!("{rank:>4}");
        }
        println!();
        for row in 0..13 {
            print!("{:>3} ", ranks[row]);
            for col in 0..13 {
                if counts[row][col] == 0 {
                    print!("   .");
                } else {
                    let p = if masses[row][col] > 0. {
                        values[i][row][col] / masses[row][col]
                    } else {
                        raw[i][row][col] / counts[row][col] as f64
                    };
                    print!("{:4.0}", 100. * p);
                }
            }
            println!();
        }
    }
    println!("Strategy CSV: {} (unrounded per-combo probabilities; zero-reach cells use stored unweighted means)",path.display());
}
