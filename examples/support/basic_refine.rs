// SparseSolver — Copyright (C) 2026 Jerzy Slaby (DeepBorys).
// Uses the postflop-solver engine by Wataru Inariba; see NOTICE.
// SPDX-License-Identifier: AGPL-3.0-or-later
// Modified for SparseSolver inference and benchmarks; see NOTICE and LICENSE.

//! Disjoint full-river subgames share a fixed averaged flop policy.
use postflop_solver::*;
use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

pub fn capture_flop(game: &PostFlopGame) -> Vec<(u64, Vec<f32>)> {
    fn walk(n: &PostFlopNode, out: &mut Vec<(u64, Vec<f32>)>) {
        if n.is_terminal() || n.is_chance() {
            return;
        }
        out.push((n.id(), n.strategy().to_vec()));
        for a in 0..n.num_actions() {
            walk(&n.play(a), out);
        }
    }
    let mut out = Vec::new();
    walk(&game.root(), &mut out);
    out
}

pub fn run(
    game: &mut PostFlopGame,
    iterations: u32,
    seed_clock: u32,
    output: &Path,
) -> (usize, f64, f64) {
    fn collect(
        n: &PostFlopNode,
        history: Vec<usize>,
        out: &mut Vec<(u64, Vec<usize>, String)>,
        label: String,
    ) {
        if n.is_terminal() {
            return;
        }
        if n.is_chance() {
            if n.chance_cards().is_empty() {
                assert_eq!(n.num_actions(), 0);
                return;
            } // exact all-in payout leaf
            assert_eq!(n.board_turn(), NOT_DEALT);
            assert_eq!(
                n.chance_cards().len(),
                49,
                "Requires full physical turn coverage"
            );
            for (a, c) in n.chance_cards().into_iter().enumerate() {
                let child = n.play(a);
                assert!(!child.is_chance() && !child.is_terminal());
                let mut h = history.clone();
                h.push(a);
                out.push((
                    child.id(),
                    h,
                    format!("{label}/t{}", card_to_string(c).unwrap()),
                ));
            }
            return;
        }
        for a in 0..n.num_actions() {
            let mut h = history.clone();
            h.push(a);
            collect(&n.play(a), h, out, format!("{label}/a{a}"));
        }
    }
    fn reaches(g: &PostFlopGame, n: &PostFlopNode, path: &[usize], r: &mut [Vec<f32>; 2]) {
        if let Some((&a, rest)) = path.split_first() {
            if n.is_chance() {
                for v in r.iter_mut() {
                    for w in v {
                        *w /= g.chance_factor(n) as f32;
                    }
                }
            } else {
                let p = n.player();
                let nh = g.num_private_hands(p);
                let q = n.strategy();
                for (h, w) in r[p].iter_mut().enumerate() {
                    let sum = (0..n.num_actions()).map(|b| q[b * nh + h]).sum::<f32>();
                    *w *= if sum > 0.0 {
                        q[a * nh + h] / sum
                    } else {
                        1.0 / n.num_actions() as f32
                    };
                }
            }
            reaches(g, &n.play(a), rest, r);
        }
    }
    fn full_rivers(n: &PostFlopNode) {
        if n.is_terminal() {
            return;
        }
        if n.is_chance() {
            if n.chance_cards().is_empty() {
                assert_eq!(n.num_actions(), 0);
                return;
            } // exact all-in payout leaf
            assert_eq!(
                n.chance_cards().len(),
                48,
                "Requires all physical river cards"
            );
            return;
        }
        for a in 0..n.num_actions() {
            full_rivers(&n.play(a));
        }
    }
    let flop_ids: Vec<_> = capture_flop(game).iter().map(|x| x.0).collect();
    let flop_avg = game.diagnostic_average_fingerprint_nodes(&flop_ids);
    let flop_regret = game.diagnostic_regret_fingerprint_nodes(&flop_ids);
    let mut inventory = Vec::new();
    collect(&game.root(), Vec::new(), &mut inventory, String::new());
    let mut ids = std::collections::BTreeSet::new();
    let mut log = BufWriter::new(fs::File::create(output.join("refinements.csv")).unwrap());
    writeln!(log, "index,path,iterations,solve_ms").unwrap();
    let mut total_ms = 0.0;
    for (index, (id, path, label)) in inventory.iter().enumerate() {
        assert!(ids.insert(*id), "Overlapping turn roots");
        fn audit(n: &PostFlopNode, path: &[usize]) {
            if let Some((&a, rest)) = path.split_first() {
                audit(&n.play(a), rest);
            } else {
                full_rivers(n);
            }
        }
        audit(&game.root(), path);
        let mut r = [
            game.initial_weights(0).to_vec(),
            game.initial_weights(1).to_vec(),
        ];
        reaches(game, &game.root(), path, &mut r);
        let start = Instant::now();
        for i in 0..iterations {
            solve_subgame_step(game, path, [&r[0], &r[1]], seed_clock + i);
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        total_ms += ms;
        writeln!(log, "{index},{label},{iterations},{ms:.6}").unwrap();
        log.flush().unwrap();
        assert_eq!(
            flop_avg,
            game.diagnostic_average_fingerprint_nodes(&flop_ids)
        );
        assert_eq!(
            flop_regret,
            game.diagnostic_regret_fingerprint_nodes(&flop_ids)
        );
        if (index + 1) % 49 == 0 {
            println!(
                "REFINE completed={}/{} cumulative_solve_ms={total_ms:.6}",
                index + 1,
                inventory.len()
            );
        }
    }
    game.back_to_root();
    let c = compute_current_ev(game);
    let b = compute_mes_ev(game);
    let oop = 100.0 * (b[0] - c[0]) / 180.0;
    let ip = 100.0 * (b[1] - c[1]) / 180.0;
    let full = (oop + ip) / 2.0;
    println!("ASSEMBLED frontiers={} flop_frozen=true full49x48=true refine_solve_ms={total_ms:.6} oop_br_gain_pct_pot={oop:.8} ip_br_gain_pct_pot={ip:.8} full_exploitability_pct_pot={full:.8}",inventory.len());
    (inventory.len(), total_ms, full as f64)
}
