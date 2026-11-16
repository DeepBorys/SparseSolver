// SparseSolver — Copyright (C) 2026 Jerzy Slaby (DeepBorys).
// Uses the postflop-solver engine by Wataru Inariba; see NOTICE.
// SPDX-License-Identifier: AGPL-3.0-or-later
// Modified for SparseSolver inference and benchmarks; see NOTICE and LICENSE.

//! NN-specific preparation and output helpers. Game inputs and CFR stay in main.
#![allow(dead_code)]
use postflop_solver::*;
use std::{fs, path::PathBuf};
#[path = "flop_log.rs"]
mod flop_log;
#[path = "model_assets.rs"]
mod model_assets;
#[path = "model_samples.rs"]
mod model_samples;
#[cfg(feature = "neural")]
#[path = "solve_plan.rs"]
pub mod solve_plan;

#[path = "strategy_output.rs"]
mod strategy_output;

pub struct Model {
    pub name: String,
    pub iterations: u32,
    pub threads: usize,
    pub sparse: bool,
    full_iterations: Option<u32>,
    refine_iterations: Option<u32>,
    assets: PathBuf,
    output: PathBuf,
    cache: PathBuf,
    turns: Vec<Card>,
    rivers: Vec<Card>,
}
impl Model {
    /// Select the backend and prepare its tree layout before with_config.
    pub fn prepare(default: &str, cards: &CardConfig) -> Self {
        Self::prepare_budget(default, cards, None)
    }
    pub fn prepare_iterations(default: &str, cards: &CardConfig, iterations: u32) -> Self {
        Self::prepare_budget(default, cards, Some(iterations))
    }
    fn prepare_budget(default: &str, cards: &CardConfig, default_budget: Option<u32>) -> Self {
        let available = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let mut name = default.to_owned();
        let mut iterations = None;
        let mut full_iterations = None;
        let mut refine_iterations = None;
        let mut threads = available;
        let mut assets = PathBuf::from(".");
        let mut output = None;
        let mut cache = None;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            if arg == "--help" {
                println!("--model 49x48|6x6|4x4|2x2|1x1 --iterations N --threads N --assets DIR --output DIR --allin-cache FILE\nCopy this example and edit CardConfig/TreeConfig directly. Default workers: {available}");
                std::process::exit(0);
            }
            let value = args
                .next()
                .unwrap_or_else(|| panic!("Missing value for {arg}"));
            match arg.as_str() {
                "--full-iterations" => full_iterations = Some(value.parse::<u32>().unwrap()),
                "--refine-iterations" => refine_iterations = Some(value.parse::<u32>().unwrap()),
                "--model" => name = value,
                "--iterations" => iterations = Some(value.parse::<u32>().unwrap()),
                "--threads" => threads = value.parse().unwrap(),
                "--assets" => assets = value.into(),
                "--output" => output = Some(PathBuf::from(value)),
                "--allin-cache" => cache = Some(PathBuf::from(value)),
                _ => panic!("Unknown argument {arg}; use speed_case for --case-config batch files"),
            }
        }
        let (n, budget) = match name.as_str() {
            "49x48" => (0, 1000),
            "1x1" => (1, 60),
            "2x2" => (2, 120),
            "4x4" => (4, 150),
            "6x6" => (6, 200),
            _ => panic!("Invalid model; choose49x48,1x1,2x2,4x4,6x6"),
        };
        assert!(
            threads > 0 && threads <= available,
            "Requested {threads} workers, available CPUs: {available}"
        );
        #[cfg(not(feature = "neural"))]
        assert_eq!(n, 0, "Sparse models require --features neural");
        if n > 0 {
            assert!(
                cards.turn == NOT_DEALT && cards.river == NOT_DEALT,
                "These NN models start on the flop"
            );
        }
        let (flop, turns, rivers) = model_samples::select(cards.flop, &name);
        model_assets::require_flop_models(&assets, &flop, &name, &turns);
        let output = output.unwrap_or_else(|| PathBuf::from(format!("results/qj2_{name}")));
        assert!(
            !output.join("summary.json").exists(),
            "Choose a fresh --output directory"
        );
        fs::create_dir_all(&output).unwrap();
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
        set_exact_allin_matrix_tree(true);
        set_exact_allin_matrix_sparse_eval(true);
        set_exact_allin_matrix_f32_eval(false);
        set_exact_called_allins(true);
        set_turn_basis_count(n);
        set_river_basis_count(n);
        set_custom_turn_basis_cards(&turns);
        set_custom_river_basis_cards(&rivers);
        set_prune_chance_branches(n > 0);
        set_regret_floor_epsilon(0.);
        set_batch_neural_turn_frontiers(false);
        #[cfg(feature = "neural")]
        {
            pio_order::set_nn_cuda(false);
            pio_order::set_nn_sparse_cpu(true);
            pio_order::set_nn_dtype("fp32");
            pio_order::set_nn_zero_mode(false);
        }
        let iterations = iterations.unwrap_or(default_budget.unwrap_or(budget));
        assert!(iterations > 0);
        println!("Model: {name}\nFlop: {flop}\nIterations: {iterations}\nCPU workers: {threads}");
        Self {
            name,
            full_iterations,
            refine_iterations,
            iterations,
            threads,
            sparse: n > 0,
            assets,
            cache: cache.unwrap_or_else(|| output.join("allin.bin")),
            output,
            turns,
            rivers,
        }
    }
    pub fn load(&self, game: &mut PostFlopGame) {
        game.enable_exact_flop_allin_payout_matrix_cached(&self.cache)
            .unwrap();
        #[cfg(feature = "neural")]
        if self.sparse {
            let (flop, _, _) = model_samples::select(game.card_config().flop, &self.name);
            let bank = self.assets.join("models/banks").join(flop).join(&self.name);
            let order = self.assets.join("piosolver_hand_order.txt");
            game.load_semantic_neural_turn_cfv_conversion_with_ranges(
                game.card_config().flop,
                &self.turns,
                &order,
                bank.join("turn/model_oop.pt"),
                bank.join("turn/model_ip.pt"),
            )
            .unwrap();
            game.set_weighted_all_turn_sums(true).unwrap();
            assert_eq!(
                game.load_neural_river_cfv_conversions(
                    game.card_config().flop,
                    &self.turns,
                    &self.rivers,
                    &order,
                    bank.join("river")
                )
                .unwrap(),
                self.turns.len()
            );
            game.set_neural_river_sums(true).unwrap();
        }
    }
    pub fn finalize(&self, game: &mut PostFlopGame) {
        if self.sparse {
            finalize_with_neural_frontiers(game, self.iterations);
        } else {
            finalize(game);
        }
    }
    pub fn print_strategy(&self, game: &mut PostFlopGame, label: &str) {
        strategy_output::print_and_save(game, &self.output, label);
    }
    pub fn write_strategy_logs(&self, game: &mut PostFlopGame, solve_ms: f64) {
        self.write_logs_evaluate(game, solve_ms, false);
    }
    pub fn write_logs(&self, game: &mut PostFlopGame, solve_ms: f64) {
        self.write_logs_evaluate(game, solve_ms, true);
    }
    fn write_logs_evaluate(&self, game: &mut PostFlopGame, solve_ms: f64, evaluate: bool) {
        flop_log::export_flop(
            game,
            &self.output.join("flop_strategy.csv"),
            &self.name,
            self.iterations,
        );
        println!(
            "Flop strategy log: {}",
            self.output.join("flop_strategy.csv").display()
        );
        let exploitability = if self.sparse || !evaluate {
            println!("Full-tree exploitability: not evaluated");
            "null".to_owned()
        } else {
            let e = 100. * compute_exploitability(game) / game.tree_config().starting_pot as f32;
            println!("Full-tree exploitability: {e:.6}% of pot");
            e.to_string()
        };
        let rss = fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmHWM:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|s| s.parse::<u64>().ok())
            })
            .map(|kb| kb * 1024);
        fs::write(self.output.join("summary.json"),format!("{{\"model\":\"{}\",\"iterations\":{},\"threads\":{},\"solve_ms\":{solve_ms},\"peak_rss_bytes\":{},\"full_tree_exploitability_pct_pot\":{exploitability},\"timing_scope\":\"CFR iterations only; setup, finalization, evaluation and exports excluded\"}}\n",self.name,self.iterations,self.threads,rss.map_or("null".into(),|v|v.to_string()))).unwrap();
    }
}
