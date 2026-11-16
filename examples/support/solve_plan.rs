// SparseSolver — Copyright (C) 2026 Jerzy Slaby (DeepBorys).
// Uses the postflop-solver engine by Wataru Inariba; see NOTICE.
// SPDX-License-Identifier: AGPL-3.0-or-later
// Modified for SparseSolver inference and benchmarks; see NOTICE and LICENSE.

//! Small declarative schedule adapter; histories use the solver's own Action enum.
use postflop_solver::*;
use std::{
    fs,
    io::{BufWriter, Write},
    time::Instant,
};
#[path = "basic_k1.rs"]
mod k1;

pub type Refinement = (Vec<Action>, u32);
pub struct Plan {
    model: super::Model,
    stages: Vec<(String, u32)>,
    frontiers: Vec<Refinement>,
    pub clock: u32,
    evaluate_full: bool,
}
impl Plan {
    pub fn prepare(cards: &CardConfig, stages: &[(&str, u32)], frontiers: &[Refinement]) -> Self {
        assert!(!stages.is_empty());
        for (i, (name, n)) in stages.iter().enumerate() {
            assert!(*n > 0);
            assert!(matches!(*name, "1x1" | "2x2" | "4x4" | "6x6" | "49x48"));
            assert!(
                *name != "49x48" || i + 1 == stages.len(),
                "Classical49x48 must be the final stage"
            );
        }
        let model = super::Model::prepare_budget(stages[0].0, cards, Some(stages[0].1));
        assert_eq!(
            model.name, stages[0].0,
            "Change the stage list to change its first model"
        );
        let mut stages = stages
            .iter()
            .map(|(m, n)| (m.to_string(), *n))
            .collect::<Vec<_>>();
        stages[0].1 = model.iterations;
        if let Some(n) = model.full_iterations {
            let last = stages.last_mut().unwrap();
            assert_eq!(last.0, "49x48");
            last.1 = n;
        }
        assert!(
            stages.iter().all(|(_, n)| *n > 0),
            "Stage budgets must be positive"
        );
        for (i, (family, _)) in stages.iter().enumerate() {
            let (flop, turns, _) = super::model_samples::select(cards.flop, family);
            super::model_assets::require_flop_models(&model.assets, &flop, family, &turns);
            if i + 1 < stages.len() {
                assert_ne!(
                    family,
                    &stages[i + 1].0,
                    "Combine adjacent stages of the same model"
                );
            }
            if family != "49x48" && (i + 1 < stages.len() || !frontiers.is_empty()) {
                assert_eq!(flop, "QsJh2h", "Bundled K1 maps cover QJ2 only");
                assert!(
                    matches!(family.as_str(), "1x1" | "2x2" | "6x6"),
                    "Bundled K1 sources are 1x1, 2x2 and 6x6"
                );
                assert!(
                    model
                        .assets
                        .join(format!("models/maps/QsJh2h_{family}_k1.tsv.gz"))
                        .is_file(),
                    "Missing K1 transfer map for {family}"
                );
            }
        }
        let mut frontiers = frontiers.to_vec();
        if let Some(n) = model.refine_iterations {
            for (_, budget) in &mut frontiers {
                *budget = n;
            }
        }
        assert!(frontiers.iter().all(|(_, n)| *n > 0));
        // A staged handoff or requested full-river refinement needs a physical target.
        if stages.len() > 1 || !frontiers.is_empty() {
            set_turn_basis_count(0);
            set_river_basis_count(0);
        }
        println!("SCHEDULE {:?}", stages);
        Self {
            model,
            stages,
            frontiers,
            clock: 0,
            evaluate_full: true,
        }
    }
    /// Strategy demonstration: skips final full-game evaluation and EV finalization.
    pub fn run_strategies(&mut self, game: &mut PostFlopGame) {
        self.evaluate_full = false;
        self.run(game);
        self.model.print_strategy(game, "flop_root");
        for (i, (history, _)) in self.frontiers.iter().enumerate() {
            game.back_to_root();
            for action in history {
                let index = game
                    .available_actions()
                    .iter()
                    .position(|a| a == action)
                    .unwrap();
                match action {
                    Action::Chance(card) => game.play(*card as usize),
                    _ => game.play(index),
                }
            }
            self.model.print_strategy(game, &format!("frontier_{i}"));
        }
        game.back_to_root();
    }
    pub fn run(&mut self, game: &mut PostFlopGame) {
        let mut plans = std::collections::BTreeMap::new();
        for (i, (family, _)) in self.stages.iter().enumerate() {
            let transfer = i + 1 < self.stages.len() || !self.frontiers.is_empty();
            if transfer && family != "49x48" && !plans.contains_key(family) {
                plans.insert(
                    family.clone(),
                    k1::prepare_target(
                        game,
                        &self.model.assets,
                        family,
                        self.stages.get(i + 1).map_or("49x48", |s| s.0.as_str()),
                    ),
                );
            }
        }
        // Validate every requested history before changing solver state.
        let histories = self
            .frontiers
            .iter()
            .map(|(h, n)| (resolve(game, h), *n))
            .collect::<Vec<_>>();
        for (i, (a, _)) in histories.iter().enumerate() {
            for (b, _) in histories.iter().skip(i + 1) {
                assert!(
                    !a.starts_with(b) && !b.starts_with(a),
                    "Refinement frontiers must not overlap"
                );
            }
        }
        game.enable_exact_flop_allin_payout_matrix_cached(&self.model.cache)
            .unwrap();
        let mut full_curve =
            BufWriter::new(fs::File::create(self.model.output.join("full_curve.csv")).unwrap());
        writeln!(
            full_curve,
            "full_iteration,cfr_clock,full_solve_ms,evaluation_ms,exploitability_pct_pot"
        )
        .unwrap();
        let mut total_ms = 0.;
        let mut full_ms = 0.;
        let mut transfer_ms = 0.;
        let mut previous = None::<String>;
        for (family, count) in &self.stages {
            if let Some(source) = &previous {
                if source != family {
                    let begin = Instant::now();
                    let plan = &plans[source];
                    game.apply_k1_shared_transfer(plan);
                    game.reset_average_strategy_for_warm_start();
                    let bank = game.take_neural_phase_bank();
                    drop(bank);
                    transfer_ms += begin.elapsed().as_secs_f64() * 1000.;
                    println!(
                        "K1 source={source} target={family} verified_values={}",
                        game.verify_k1_shared_transfer(plan).unwrap()
                    );
                }
            }
            self.model.name = family.clone();
            self.model.sparse = family != "49x48";
            self.model.iterations = *count;
            let (_, turns, rivers) = super::model_samples::select(game.card_config().flop, family);
            self.model.turns = turns;
            self.model.rivers = rivers;
            set_custom_turn_basis_cards(&self.model.turns);
            set_custom_river_basis_cards(&self.model.rivers);
            set_prune_chance_branches(self.model.sparse);
            if self.model.sparse {
                self.model.load(game);
            }
            if !self.model.sparse && self.evaluate_full {
                record_full(game, &mut full_curve, 0, self.clock, full_ms);
            }
            let mut stage_ms = 0.;
            for step in 0..*count {
                let begin = Instant::now();
                solve_step(game, self.clock + step);
                stage_ms += begin.elapsed().as_secs_f64() * 1000.;
                if !self.model.sparse
                    && self.evaluate_full
                    && ((step + 1) % 10 == 0 || step + 1 == *count)
                {
                    record_full(
                        game,
                        &mut full_curve,
                        step + 1,
                        self.clock + step + 1,
                        full_ms + stage_ms,
                    );
                }
            }
            total_ms += stage_ms;
            if !self.model.sparse {
                full_ms += stage_ms;
            }
            self.clock += count;
            println!(
                "STAGE model={family} iterations={count} clock={} solve_ms={stage_ms:.6}",
                self.clock
            );
            if self.model.sparse {
                super::flop_log::export_flop(
                    game,
                    &self.model.output.join("seed_flop_strategy.csv"),
                    family,
                    *count,
                );
            }
            previous = Some(family.clone());
        }
        let seed_ms = total_ms;
        println!("SEED cfr_ms={seed_ms:.6} transfer_ms={transfer_ms:.6}");
        if !histories.is_empty() {
            if self.model.sparse {
                let mut flop = capture_flop(game);
                let family = previous.as_ref().unwrap();
                let plan = &plans[family];
                let begin = Instant::now();
                game.apply_k1_shared_transfer(plan);
                game.reset_average_strategy_for_warm_start();
                drop(game.take_neural_phase_bank());
                set_prune_chance_branches(false);
                transfer_ms += begin.elapsed().as_secs_f64() * 1000.;
                game.diagnostic_swap_average_policies(&mut flop);
                self.model.sparse = false;
                println!(
                    "K1 source={family} target=49x48 verified_values={}",
                    game.verify_k1_shared_transfer(plan).unwrap()
                );
            }
            let ids = capture_flop(game).iter().map(|x| x.0).collect::<Vec<_>>();
            let fingerprint = game.diagnostic_average_fingerprint_nodes(&ids);
            let reaches = histories
                .iter()
                .map(|(p, _)| frozen_reaches(game, p))
                .collect::<Vec<_>>();
            for ((path, count), reach) in histories.iter().zip(&reaches) {
                let begin = Instant::now();
                for i in 0..*count {
                    solve_subgame_step(game, path, [&reach[0], &reach[1]], self.clock + i);
                }
                let ms = begin.elapsed().as_secs_f64() * 1000.;
                total_ms += ms;
                assert_eq!(fingerprint, game.diagnostic_average_fingerprint_nodes(&ids));
                println!("REFINE arena_path={path:?} iterations={count} cfr_ms={ms:.6} seed_plus_this_frontier_cfr_ms={:.6}",seed_ms+ms);
            }
        }
        if self.evaluate_full {
            if self.model.sparse {
                finalize_with_neural_frontiers(game, self.clock);
            } else {
                finalize(game);
            }
        }
        self.model.name = previous.unwrap();
        if self.evaluate_full {
            self.model.write_logs(game, total_ms);
        } else {
            self.model.write_strategy_logs(game, total_ms);
        }
        println!("TIMING seed_cfr_ms={seed_ms:.6} transfers_ms={transfer_ms:.6} refinements_cfr_ms={:.6} total_cfr_ms={total_ms:.6}",total_ms-seed_ms);
        // Retain the regression-check schema for the two-stage warm-start example.
        if histories.is_empty() && self.stages.len() == 2 && self.stages[1].0 == "49x48" {
            fs::write(self.model.output.join("summary.json"),format!("{{\"seed_model\":\"{}\",\"seed_iterations\":{},\"full_iterations\":{},\"threads\":{},\"solve_ms\":{total_ms},\"transfer_ms\":{transfer_ms}}}\n",self.stages[0].0,self.stages[0].1,self.stages[1].1,self.model.threads)).unwrap();
        }
        fs::write(self.model.output.join("plan_summary.json"),format!("{{\"stages\":{:?},\"clock\":{},\"threads\":{},\"transfer_ms\":{transfer_ms},\"refined_frontiers\":{}}}\n",self.stages.iter().map(|(m,n)|format!("{m}/{n}")).collect::<Vec<_>>(),self.clock,self.model.threads,histories.len())).unwrap();
    }
}
fn record_full(g: &PostFlopGame, out: &mut impl Write, step: u32, clock: u32, ms: f64) {
    let start = Instant::now();
    let pct = 100. * compute_exploitability(g) / g.tree_config().starting_pot as f32;
    let eval = start.elapsed().as_secs_f64() * 1000.;
    println!(
        "FULL iteration={step} clock={clock} solve_ms={ms:.6} exploitability_pct_pot={pct:.8}"
    );
    writeln!(out, "{step},{clock},{ms},{eval},{pct}").unwrap();
}
fn resolve(g: &mut PostFlopGame, history: &[Action]) -> Vec<usize> {
    g.back_to_root();
    let mut path = Vec::new();
    for action in history {
        let actions = g.available_actions();
        let index = actions.iter().position(|a| a == action).unwrap_or_else(|| {
            panic!("Invalid refinement action {action:?}; available: {actions:?}")
        });
        path.push(index);
        match action {
            Action::Chance(c) => g.play(*c as usize),
            _ => g.play(index),
        }
    }
    assert!(
        !g.is_terminal_node() && !g.is_chance_node() && g.current_board().len() >= 4,
        "Refinement must end at a turn/river decision node"
    );
    g.back_to_root();
    path
}
fn capture_flop(g: &PostFlopGame) -> Vec<(u64, Vec<f32>)> {
    fn walk(n: &PostFlopNode, out: &mut Vec<(u64, Vec<f32>)>) {
        if n.is_terminal() || n.is_chance() {
            return;
        }
        out.push((n.id(), n.strategy().to_vec()));
        for a in 0..n.num_actions() {
            walk(&n.play(a), out);
        }
    }
    let mut v = Vec::new();
    walk(&g.root(), &mut v);
    v
}
fn frozen_reaches(g: &PostFlopGame, path: &[usize]) -> [Vec<f32>; 2] {
    fn walk(g: &PostFlopGame, n: &PostFlopNode, path: &[usize], r: &mut [Vec<f32>; 2]) {
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
                for (h, w) in r[p].iter_mut().enumerate() {
                    let sum = (0..n.num_actions())
                        .map(|b| n.strategy()[b * nh + h])
                        .sum::<f32>();
                    *w *= if sum > 0. {
                        n.strategy()[a * nh + h] / sum
                    } else {
                        1. / n.num_actions() as f32
                    };
                }
            }
            walk(g, &n.play(a), rest, r);
        }
    }
    let mut r = [g.initial_weights(0).to_vec(), g.initial_weights(1).to_vec()];
    walk(g, &g.root(), path, &mut r);
    r
}
