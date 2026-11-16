// SparseSolver — Copyright (C) 2026 Jerzy Slaby (DeepBorys).
// Uses the postflop-solver engine by Wataru Inariba; see NOTICE.
// SPDX-License-Identifier: AGPL-3.0-or-later
// Modified for SparseSolver inference and benchmarks; see NOTICE and LICENSE.

#![allow(dead_code)] // Shared by several small example entry points.
//! The same public game/navigation API for classical and sparse flop solving.
use postflop_solver::*;
use std::{
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

#[path = "flop_log.rs"]
mod flop_log;
#[path = "model_assets.rs"]
mod model_assets;
use flop_log::export_flop;
use model_assets::require_flop_models;

// A runtime case file changes public situation inputs, not solver algorithms.
fn force_ip_opening(game: &mut PostFlopGame, check: bool) -> [usize; 2] {
    fn visit(g: &mut PostFlopGame, path: &mut Vec<usize>, counts: &mut [usize; 2], check: bool) {
        g.apply_history(path);
        if g.is_terminal_node() {
            return;
        }
        if g.is_chance_node() {
            let cards = g
                .available_actions()
                .into_iter()
                .map(|a| {
                    if let Action::Chance(c) = a {
                        c as usize
                    } else {
                        unreachable!()
                    }
                })
                .collect::<Vec<_>>();
            for c in cards {
                path.push(c);
                visit(g, path, counts, check);
                path.pop();
            }
            return;
        }
        let actions = g.available_actions();
        let street = g.current_board().len();
        if g.current_player() == 1
            && street >= 4
            && actions.iter().any(|a| matches!(a, Action::Check))
        {
            let selected = if check {
                actions
                    .iter()
                    .position(|a| matches!(a, Action::Check))
                    .unwrap()
            } else {
                actions
                    .iter()
                    .position(|a| matches!(a, Action::Bet(_)))
                    .or_else(|| actions.iter().position(|a| matches!(a, Action::AllIn(_))))
                    .unwrap()
            };
            let hands = g.strategy().len() / actions.len();
            let mut q = vec![0.; hands * actions.len()];
            q[selected * hands..(selected + 1) * hands].fill(1.);
            g.lock_current_strategy(&q);
            counts[street - 4] += 1;
        }
        for a in 0..actions.len() {
            path.push(a);
            visit(g, path, counts, check);
            path.pop();
        }
    }
    let mut counts = [0; 2];
    visit(game, &mut Vec::new(), &mut counts, check);
    game.back_to_root();
    counts
}

pub fn run(default_model: &str) {
    run_inner(default_model, false, false);
}

#[cfg(feature = "neural")]
pub fn run_staged() {
    run_inner("2x2", true, false);
}

#[cfg(feature = "neural")]
pub fn run_refine_all() {
    run_inner("2x2", true, true);
}

fn run_inner(default_model: &str, staged: bool, assemble: bool) {
    let mut model = default_model.to_string();
    let mut iterations = None;
    let mut full_iterations = 100u32;
    let mut refine_iterations = 100u32;
    let available_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut threads = available_threads;
    let mut assets = if Path::new("models/banks").is_dir() {
        PathBuf::from(".")
    } else {
        PathBuf::from("ec2/public-benchmark")
    };
    let mut output = None;
    let mut allin_cache = None;
    let mut case_config = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" {
            println!("--model 49x48|6x6|4x4|2x2|1x1 --iterations N --threads N --assets DIR --output DIR --case-config FILE --full-iterations N (staged) --refine-iterations N (assembly)");
            println!("Defaults: model={default_model}, threads={available_threads} (available CPUs); QJ2 normal flop, pot180/stack910. Models must be hydrated native CPU files. Fixed iteration budgets; exports and final evaluation excluded from solve time.");
            return;
        }
        let value = args
            .next()
            .unwrap_or_else(|| panic!("Missing value for {arg}"));
        match arg.as_str() {
            "--case-config" => case_config = Some(PathBuf::from(value)),
            "--model" => model = value,
            "--refine-iterations" => {
                refine_iterations = value.parse().expect("Invalid refinement iterations")
            }
            "--full-iterations" => {
                full_iterations = value.parse().expect("Invalid full iterations")
            }
            "--iterations" => iterations = Some(value.parse::<u32>().expect("Invalid iterations")),
            "--threads" => threads = value.parse().expect("Invalid threads"),
            "--assets" => assets = value.into(),
            "--output" => output = Some(PathBuf::from(value)),
            "--allin-cache" => allin_cache = Some(PathBuf::from(value)),
            _ => panic!("Unknown argument: {arg}"),
        }
    }
    let config: std::collections::BTreeMap<String, String> = case_config
        .as_ref()
        .map(|path| {
            assert!(!staged, "Custom case files are for sparse speed runs only");
            fs::read_to_string(path)
                .unwrap()
                .lines()
                .filter(|l| !l.is_empty())
                .map(|line| {
                    let (k, v) = line.split_once('\t').expect("Invalid case file");
                    (k.to_string(), v.to_string())
                })
                .collect()
        })
        .unwrap_or_default();
    let value = |k: &str, default: &str| {
        config
            .get(k)
            .cloned()
            .unwrap_or_else(|| default.to_string())
    };
    if case_config.is_some() {
        assert_eq!(value("model", ""), model);
    }
    let flop_name = value("flop", "QsJh2h");
    let flop = flop_from_str(&flop_name).unwrap();
    let pot = value("pot", "180").parse::<i32>().unwrap();
    let stack = value("stack", "910").parse::<i32>().unwrap();
    let oop = value("oop", include_str!("../data/basic_qj2/oop.txt").trim());
    let ip = value("ip", include_str!("../data/basic_qj2/ip.txt").trim());
    let (n, default_iterations) = match model.as_str() {
        "49x48" => (0, 1000),
        "6x6" => (6, 200),
        "4x4" => (4, 150),
        "2x2" => (2, 120),
        "1x1" => (1, 60),
        _ => panic!("Invalid model; choose 49x48,6x6,4x4,2x2,1x1"),
    };
    let iterations = iterations.unwrap_or_else(|| {
        value("iterations", &default_iterations.to_string())
            .parse()
            .unwrap()
    });
    assert!(!staged || model == "2x2", "Staged example seeds with 2x2");
    assert!(!staged || full_iterations > 0);
    assert!(iterations > 0 && threads > 0);
    assert!(threads <= available_threads, "Requested {threads} workers but only {available_threads} CPUs are available; omit --threads for automatic selection");
    #[cfg(not(feature = "neural"))]
    assert_eq!(n, 0, "Sparse models require --features neural");
    let cards = |key: &str, defaults: &str| {
        let text = value(key, defaults);
        let all = text
            .split(',')
            .map(|c| card_from_str(c).unwrap())
            .collect::<Vec<_>>();
        assert!(all.len() >= n);
        all[..n].to_vec()
    };
    let turns = cards("turn_cards", "3d,4h,9c,Ks,6d,Tc");
    let rivers = cards("river_cards", "7s,8c,Th,5d,Qc,2d");
    require_flop_models(&assets, &flop_name, &model, &turns);
    let output = output.unwrap_or_else(|| PathBuf::from(format!("results/basic_{model}")));
    fs::create_dir_all(&output).unwrap();
    assert!(
        !output.join("summary.json").exists(),
        "Output already contains a completed run; choose a new --output"
    );
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    set_exact_allin_matrix_tree(true);
    set_exact_allin_matrix_sparse_eval(true);
    set_exact_allin_matrix_f32_eval(false);
    set_exact_called_allins(true);
    set_turn_basis_count(if staged { 0 } else { n });
    set_river_basis_count(if staged { 0 } else { n });
    set_custom_turn_basis_cards(&turns);
    set_custom_river_basis_cards(&rivers);
    set_prune_chance_branches(n > 0);
    set_regret_floor_epsilon(0.0);
    set_batch_neural_turn_frontiers(false);
    #[cfg(feature = "neural")]
    {
        pio_order::set_nn_cuda(false);
        pio_order::set_nn_sparse_cpu(true);
        pio_order::set_nn_dtype("fp32");
        pio_order::set_nn_zero_mode(false);
    }

    // Wataru's basic example pattern: ranges -> cards -> betting tree -> game.
    let card_config = CardConfig {
        range: [oop.parse().unwrap(), ip.parse().unwrap()],
        flop,
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };
    let raises = value("raises", "45%");
    let bets = |sizes: &str| BetSizeOptions::try_from((sizes, raises.as_str())).unwrap();
    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: pot,
        effective_stack: stack,
        flop_bet_sizes: [
            bets(&value("flop_bets", "52%")),
            bets(&value("flop_bets", "52%")),
        ],
        turn_bet_sizes: [
            bets(&value("turn_bets", "55%")),
            bets(&value("turn_bets", "55%")),
        ],
        river_bet_sizes: [
            bets(&value("river_bets", "70%")),
            bets(&value("river_bets", "70%")),
        ],
        river_donk_sizes: if value("no_river_donk", "false") == "true" {
            None
        } else {
            Some(DonkSizeOptions::try_from("50%").unwrap())
        },
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: if value("disable_merging", "false") == "true" {
            0.
        } else {
            0.1
        },
        ..Default::default()
    };
    let mut game =
        PostFlopGame::with_config(card_config, ActionTree::new(tree_config).unwrap()).unwrap();
    let (memory, compressed) = game.memory_usage();
    println!("Model: {model}\nIterations: {iterations}\nCPU workers: {threads}");
    println!(
        "Memory usage without compression (32-bit float): {:.2}GB",
        memory as f64 / 1073741824.0
    );
    println!(
        "Memory usage with compression (16-bit integer): {:.2}GB",
        compressed as f64 / 1073741824.0
    );
    game.allocate_memory(false);
    game.enable_exact_flop_allin_payout_matrix_cached(
        allin_cache.unwrap_or_else(|| output.join("allin.bin")),
    )
    .unwrap();
    #[cfg(feature = "neural")]
    if n > 0 {
        let bank = assets.join("models/banks").join(&flop_name).join(&model);
        let order = assets.join("piosolver_hand_order.txt");
        game.load_semantic_neural_turn_cfv_conversion_with_ranges(
            flop,
            &turns,
            &order,
            bank.join("turn/model_oop.pt"),
            bank.join("turn/model_ip.pt"),
        )
        .unwrap();
        game.set_weighted_all_turn_sums(true).unwrap();
        assert_eq!(
            game.load_neural_river_cfv_conversions(
                flop,
                &turns,
                &rivers,
                &order,
                bank.join("river")
            )
            .unwrap(),
            n
        );
        game.set_neural_river_sums(true).unwrap();
    }
    let forced = value("force_ip_opening", "none");
    if forced != "none" {
        assert!(forced == "check" || forced == "bet");
        println!(
            "FORCED_IP_OPENING mode={} counts={:?}",
            forced,
            force_ip_opening(&mut game, forced == "check")
        );
    }
    #[cfg(feature = "neural")]
    let plan = if staged {
        Some(super::basic_k1::prepare(&mut game, &assets))
    } else {
        None
    };
    // Exactly the same solve_step API for every model. Time only CFR iterations.
    let start = Instant::now();
    for i in 0..iterations {
        solve_step(&game, i);
    }
    let solve_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!("Solve time: {solve_ms:.6} ms");
    #[cfg(feature = "neural")]
    if staged {
        export_flop(
            &mut game,
            &output.join("seed_flop_strategy.csv"),
            &model,
            iterations,
        );
        let mut flop = super::basic_refine::capture_flop(&game);
        let begin = Instant::now();
        let plan = plan.as_ref().unwrap();
        game.apply_k1_shared_transfer(plan);
        game.reset_average_strategy_for_warm_start();
        let source_models = game.take_neural_phase_bank();
        set_prune_chance_branches(false);
        let transfer_ms = begin.elapsed().as_secs_f64() * 1000.0;
        println!("K=1 transfer + average reset: {transfer_ms:.6} ms (unscaled signed regrets; continuous CFR clock)");
        // Copy verification and file output are outside transfer and CFR timers.
        println!(
            "K=1 verified values: {}",
            game.verify_k1_shared_transfer(plan).unwrap()
        );
        if assemble {
            assert!(refine_iterations > 0);
            game.diagnostic_swap_average_policies(&mut flop);
            let (frontiers, refine_ms, full) =
                super::basic_refine::run(&mut game, refine_iterations, iterations, &output);
            finalize(&mut game);
            export_flop(
                &mut game,
                &output.join("flop_strategy.csv"),
                &model,
                iterations,
            );
            fs::write(output.join("summary.json"), format!("{{\"seed_model\":\"2x2\",\"threads\":{threads},\"seed_iterations\":{iterations},\"refine_iterations\":{refine_iterations},\"seed_solve_ms\":{solve_ms},\"transfer_ms\":{transfer_ms},\"frontiers\":{frontiers},\"refine_all_ms\":{refine_ms},\"full_tree_exploitability_pct_pot\":{full},\"flop_frozen\":true,\"full49x48\":true}}\n")).unwrap();
            return;
        }
        let mut curve = BufWriter::new(fs::File::create(output.join("full_curve.csv")).unwrap());
        writeln!(
            curve,
            "full_iteration,cfr_clock,full_solve_ms,evaluation_ms,exploitability_pct_pot"
        )
        .unwrap();
        let mut full_ms = 0.0;
        for step in 0..=full_iterations {
            if step > 0 {
                let begin = Instant::now();
                solve_step(&game, iterations + step - 1);
                full_ms += begin.elapsed().as_secs_f64() * 1000.0;
            }
            if step % 10 == 0 || step == full_iterations {
                let begin = Instant::now();
                let pct = 100.0 * compute_exploitability(&game) / 180.0;
                let evaluation_ms = begin.elapsed().as_secs_f64() * 1000.0;
                println!("FULL iteration={step} clock={} solve_ms={full_ms:.6} exploitability_pct_pot={pct:.8}", iterations + step);
                writeln!(
                    curve,
                    "{step},{},{full_ms},{evaluation_ms},{pct}",
                    iterations + step
                )
                .unwrap();
                curve.flush().unwrap();
            }
        }
        finalize(&mut game);
        export_flop(
            &mut game,
            &output.join("flop_strategy.csv"),
            "49x48",
            iterations + full_iterations,
        );
        fs::write(output.join("summary.json"), format!("{{\"seed_model\":\"2x2\",\"threads\":{threads},\"seed_iterations\":{iterations},\"full_iterations\":{full_iterations},\"seed_solve_ms\":{solve_ms},\"transfer_ms\":{transfer_ms},\"full_solve_ms\":{full_ms},\"check_interval\":10}}\n")).unwrap();
        drop(source_models);
        return;
    }
    // Sparse values include neural continuations; never label these full-game exploitability.
    if n == 0 {
        finalize(&mut game);
    } else {
        finalize_with_neural_frontiers(&mut game, iterations);
    }
    game.cache_normalized_weights();
    let equity = game.equity(0);
    let ev = game.expected_values(0);
    let weights = game.normalized_weights(0);
    let hands = holes_to_strings(game.private_cards(0)).unwrap();
    println!(
        "Equity of oop_hands[0] ({}): {:.2}%",
        hands[0],
        100.0 * equity[0]
    );
    println!("EV of oop_hands[0]: {:.2}", ev[0]);
    println!(
        "Average equity: {:.2}%",
        100.0 * compute_average(&equity, weights)
    );
    println!("Average EV: {:.2}", compute_average(&ev, weights));
    println!(
        "Value scope: {}",
        if n == 0 {
            "classical full-tree policy"
        } else {
            "sparse policy with neural continuation estimates"
        }
    );
    println!("Available actions (OOP): {:?}", game.available_actions());
    game.play(0);
    println!(
        "Available actions (IP after check): {:?}",
        game.available_actions()
    );
    game.back_to_root();
    export_flop(
        &mut game,
        &output.join("flop_strategy.csv"),
        &model,
        iterations,
    );
    println!(
        "Flop strategy log: {}",
        output.join("flop_strategy.csv").display()
    );
    let exploitability = if n == 0 {
        let x = compute_exploitability(&game);
        println!(
            "Full-tree exploitability: {:.6}% of pot",
            100.0 * x / pot as f32
        );
        format!("{}", 100.0 * x / pot as f32)
    } else {
        println!("Full-tree exploitability: not evaluated (use complete-policy assembly)");
        "null".into()
    };
    let rss = fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<u64>().ok())
        })
        .map(|kb| kb * 1024);
    fs::write(output.join("summary.json"), format!("{{\"model\":\"{model}\",\"iterations\":{iterations},\"threads\":{threads},\"solve_ms\":{solve_ms},\"peak_rss_bytes\":{},\"full_tree_exploitability_pct_pot\":{exploitability},\"timing_scope\":\"CFR iterations only; setup, finalization, evaluation and exports excluded\"}}\n", rss.map_or("null".into(), |v| v.to_string()))).unwrap();
}
