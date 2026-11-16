use super::*;
use crate::interface::{Game, GameNode};
use crate::range::*;
use crate::solver::*;
use crate::utility::*;
use crate::BunchingData;
use std::mem::MaybeUninit;

#[test]
fn all_check_all_range() {
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Td9d6h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 30.0).abs() < 1e-4);
    assert!((ev_ip - 30.0).abs() < 1e-4);

    game.play(0);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 30.0).abs() < 1e-4);
    assert!((ev_ip - 30.0).abs() < 1e-4);

    game.play(0);
    assert!(game.is_chance_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 30.0).abs() < 1e-4);
    assert!((ev_ip - 30.0).abs() < 1e-4);

    game.play(usize::MAX);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 30.0).abs() < 1e-4);
    assert!((ev_ip - 30.0).abs() < 1e-4);

    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(0);
    game.play(0);
    assert!(game.is_terminal_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 30.0).abs() < 1e-4);
    assert!((ev_ip - 30.0).abs() < 1e-4);
}

#[test]
fn one_raise_all_range() {
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Td9d6h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 37.5).abs() < 1e-4);
    assert!((ev_ip - 22.5).abs() < 1e-4);

    game.play(0);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 37.5).abs() < 1e-4);
    assert!((ev_ip - 22.5).abs() < 1e-4);

    game.play(0);
    assert!(game.is_chance_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 37.5).abs() < 1e-4);
    assert!((ev_ip - 22.5).abs() < 1e-4);

    game.play(usize::MAX);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 37.5).abs() < 1e-4);
    assert!((ev_ip - 22.5).abs() < 1e-4);

    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(1);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 75.0).abs() < 1e-4);
    assert!((ev_ip - 15.0).abs() < 1e-4);

    game.play(1);
    assert!(game.is_terminal_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 60.0).abs() < 1e-4);
    assert!((ev_ip - 60.0).abs() < 1e-4);
}

#[test]
fn one_raise_all_range_compressed() {
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Td9d6h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(true);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-4);
    assert!((equity_ip - 0.5).abs() < 1e-4);
    assert!((ev_oop - 37.5).abs() < 1e-2);
    assert!((ev_ip - 22.5).abs() < 1e-2);

    game.play(0);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-4);
    assert!((equity_ip - 0.5).abs() < 1e-4);
    assert!((ev_oop - 37.5).abs() < 1e-2);
    assert!((ev_ip - 22.5).abs() < 1e-2);

    game.play(0);
    assert!(game.is_chance_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-4);
    assert!((equity_ip - 0.5).abs() < 1e-4);
    assert!((ev_oop - 37.5).abs() < 1e-2);
    assert!((ev_ip - 22.5).abs() < 1e-2);

    game.play(usize::MAX);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-4);
    assert!((equity_ip - 0.5).abs() < 1e-4);
    assert!((ev_oop - 37.5).abs() < 1e-2);
    assert!((ev_ip - 22.5).abs() < 1e-2);

    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(1);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-4);
    assert!((equity_ip - 0.5).abs() < 1e-4);
    assert!((ev_oop - 75.0).abs() < 1e-2);
    assert!((ev_ip - 15.0).abs() < 1e-2);

    game.play(1);
    assert!(game.is_terminal_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-4);
    assert!((equity_ip - 0.5).abs() < 1e-4);
    assert!((ev_oop - 60.0).abs() < 1e-2);
    assert!((ev_ip - 60.0).abs() < 1e-2);
}

#[test]
fn one_raise_all_range_with_turn() {
    let card_config = CardConfig {
        flop: flop_from_str("Td9d6h").unwrap(),
        range: [Range::ones(); 2],
        turn: card_from_str("Qc").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: 60,
        effective_stack: 970,
        river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_equity_oop = compute_average(&game.equity(0), weights_oop);
    let root_equity_ip = compute_average(&game.equity(1), weights_ip);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    assert!((root_equity_oop - 0.5).abs() < 1e-5);
    assert!((root_equity_ip - 0.5).abs() < 1e-5);
    assert!((root_ev_oop - 37.5).abs() < 1e-4);
    assert!((root_ev_ip - 22.5).abs() < 1e-4);
}

#[test]
fn one_raise_all_range_with_river() {
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: card_from_str("Qc").unwrap(),
        river: card_from_str("7s").unwrap(),
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 60,
        effective_stack: 970,
        river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 37.5).abs() < 1e-4);
    assert!((ev_ip - 22.5).abs() < 1e-4);

    game.play(0);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 30.0).abs() < 1e-4);
    assert!((ev_ip - 30.0).abs() < 1e-4);

    game.play(0);
    assert!(game.is_terminal_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 30.0).abs() < 1e-4);
    assert!((ev_ip - 30.0).abs() < 1e-4);

    game.back_to_root();
    game.play(1);
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 75.0).abs() < 1e-4);
    assert!((ev_ip - 15.0).abs() < 1e-4);

    game.play(0);
    assert!(game.is_terminal_node());
    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!(game.is_terminal_node());
    assert!((equity_oop - 0.5).abs() < 1e-5);
    assert!((equity_ip - 0.5).abs() < 1e-5);
    assert!((ev_oop - 90.0).abs() < 1e-4);
    assert!((ev_ip - 0.0).abs() < 1e-4);
}

#[test]
fn always_win() {
    // be careful for straight flushes
    let lose_range_str = "KK-22,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";
    let card_config = CardConfig {
        range: ["AA".parse().unwrap(), lose_range_str.parse().unwrap()],
        flop: flop_from_str("AcAdKh").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 1.0).abs() < 1e-5);
    assert!((equity_ip - 0.0).abs() < 1e-5);
    assert!((ev_oop - 60.0).abs() < 1e-4);
    assert!((ev_ip - 0.0).abs() < 1e-4);

    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(0);
    game.play(0);
    assert!(game.is_terminal_node());

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 1.0).abs() < 1e-5);
    assert!((equity_ip - 0.0).abs() < 1e-5);
    assert!((ev_oop - 60.0).abs() < 1e-4);
    assert!((ev_ip - 0.0).abs() < 1e-4);
}

#[test]
fn always_win_raked() {
    // be careful for straight flushes
    let lose_range_str = "KK-22,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";
    let card_config = CardConfig {
        range: ["AA".parse().unwrap(), lose_range_str.parse().unwrap()],
        flop: flop_from_str("AcAdKh").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        rake_rate: 0.05,
        rake_cap: 10.0,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((ev_oop - 57.0).abs() < 1e-4);
    assert!((ev_ip - 0.0).abs() < 1e-4);

    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(0);
    game.play(0);
    assert!(game.is_terminal_node());

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((ev_oop - 57.0).abs() < 1e-4);
    assert!((ev_ip - 0.0).abs() < 1e-4);
}

#[test]
fn always_lose() {
    // be careful for straight flushes
    let lose_range_str = "KK-22,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";
    let card_config = CardConfig {
        range: [lose_range_str.parse().unwrap(), "AA".parse().unwrap()],
        flop: flop_from_str("AcAdKh").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_equity_oop = compute_average(&game.equity(0), weights_oop);
    let root_equity_ip = compute_average(&game.equity(1), weights_ip);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    assert!((root_equity_oop - 0.0).abs() < 1e-5);
    assert!((root_equity_ip - 1.0).abs() < 1e-5);
    assert!((root_ev_oop - 0.0).abs() < 1e-4);
    assert!((root_ev_ip - 60.0).abs() < 1e-4);
}

#[test]
fn always_lose_raked() {
    // be careful for straight flushes
    let lose_range_str = "KK-22,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";
    let card_config = CardConfig {
        range: [lose_range_str.parse().unwrap(), "AA".parse().unwrap()],
        flop: flop_from_str("AcAdKh").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        rake_rate: 0.05,
        rake_cap: 10.0,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    assert!((root_ev_oop - 0.0).abs() < 1e-4);
    assert!((root_ev_ip - 57.0).abs() < 1e-4);
}

#[test]
fn always_tie() {
    let card_config = CardConfig {
        range: ["AA".parse().unwrap(), "AA".parse().unwrap()],
        flop: flop_from_str("2c6dTh").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_equity_oop = compute_average(&game.equity(0), weights_oop);
    let root_equity_ip = compute_average(&game.equity(1), weights_ip);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    assert!((root_equity_oop - 0.5).abs() < 1e-5);
    assert!((root_equity_ip - 0.5).abs() < 1e-5);
    assert!((root_ev_oop - 30.0).abs() < 1e-4);
    assert!((root_ev_ip - 30.0).abs() < 1e-4);
}

#[test]
fn always_tie_raked() {
    let card_config = CardConfig {
        range: ["AA".parse().unwrap(), "AA".parse().unwrap()],
        flop: flop_from_str("2c6dTh").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        rake_rate: 0.05,
        rake_cap: 10.0,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    assert!((root_ev_oop - 28.5).abs() < 1e-4);
    assert!((root_ev_ip - 28.5).abs() < 1e-4);
}

#[test]
fn no_assignment() {
    let card_config = CardConfig {
        range: ["TT".parse().unwrap(), "TT".parse().unwrap()],
        flop: flop_from_str("Td9d6h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let game = PostFlopGame::with_config(card_config, action_tree);
    assert!(game.is_err());
}

#[test]
fn remove_lines() {
    use crate::bet_size::BetSizeOptions;
    let card_config = CardConfig {
        range: ["TT+,AKo,AQs+".parse().unwrap(), "AA".parse().unwrap()],
        flop: flop_from_str("2c6dTh").unwrap(),
        ..Default::default()
    };

    // simple tree: force checks on flop, and only use 1/2 pot bets on turn and river
    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        turn_bet_sizes: [
            BetSizeOptions::try_from(("50%", "")).unwrap(),
            Default::default(),
        ],
        river_bet_sizes: [
            BetSizeOptions::try_from(("50%", "")).unwrap(),
            Default::default(),
        ],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    let lines = vec![
        vec![
            Action::Check,
            Action::Check,
            Action::Chance(2),
            Action::Check,
        ],
        vec![
            Action::Check,
            Action::Check,
            Action::Chance(2),
            Action::Bet(30),
            Action::Call,
            Action::Chance(3),
            Action::Bet(60),
        ],
    ];

    let res = game.remove_lines(&lines);
    assert!(res.is_ok());

    game.allocate_memory(false);

    // check that the turn line is removed
    game.apply_history(&[0, 0, 2]);
    assert_eq!(game.available_actions(), vec![Action::Bet(30)]);

    // check that other turn lines are correct
    game.apply_history(&[0, 0, 3]);
    assert_eq!(
        game.available_actions(),
        vec![Action::Check, Action::Bet(30)]
    );

    // check that the river line is removed
    game.apply_history(&[0, 0, 2, 0, 1, 3]);
    assert_eq!(game.available_actions(), vec![Action::Check]);

    // check that other river lines are correct
    game.apply_history(&[0, 0, 2, 0, 1, 4]);
    assert_eq!(
        game.available_actions(),
        vec![Action::Check, Action::Bet(60)]
    );

    game.apply_history(&[0, 0, 3, 1, 1, 4]);
    assert_eq!(
        game.available_actions(),
        vec![Action::Check, Action::Bet(60)]
    );

    // check that `solve()` does not crash
    solve(&mut game, 10, 0.01, false);
}

#[test]
fn isomorphism_monotone() {
    let oop_range = "88+,A8s+,A5s-A2s:0.5,AJo+,ATo:0.75,K9s+,KQo,KJo:0.75,KTo:0.25,Q9s+,QJo:0.5,J8s+,JTo:0.25,T8s+,T7s:0.45,97s+,96s:0.45,87s,86s:0.75,85s:0.45,75s+:0.75,74s:0.45,65s:0.75,64s:0.5,63s:0.45,54s:0.75,53s:0.5,52s:0.45,43s:0.5,42s:0.45,32s:0.45";
    let ip_range = "AA:0.25,99-22,AJs-A2s,AQo-A8o,K2s+,K9o+,Q2s+,Q9o+,J6s+,J9o+,T6s+,T9o,96s+,95s:0.5,98o,86s+,85s:0.5,75s+,74s:0.5,64s+,63s:0.5,54s,53s:0.5,43s";

    let card_config = CardConfig {
        range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
        flop: flop_from_str("QhJh2h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 100,
        effective_stack: 100,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    let mut check = |history: &[usize],
                     expected_turn_swap: Option<u8>,
                     expected_river_swap: Option<(u8, u8)>| {
        game.apply_history(history);
        game.cache_normalized_weights();
        let weights = game.normalized_weights(0);
        let ev = game.expected_values(0);
        weights.iter().zip(ev.iter()).for_each(|(&w, &v)| {
            assert!(!(w > 0.0 && v == 50.0));
        });
        assert_eq!(game.turn_swap, expected_turn_swap);
        assert_eq!(game.river_swap, expected_river_swap);
    };

    check(&[0, 0, 4], None, None);
    check(&[0, 0, 5], Some(1), None);
    check(&[0, 0, 6], None, None);
    check(&[0, 0, 7], Some(3), None);

    check(&[0, 0, 4, 0, 0, 8], None, None);
    check(&[0, 0, 4, 0, 0, 9], None, None);
    check(&[0, 0, 4, 0, 0, 10], None, None);
    check(&[0, 0, 4, 0, 0, 11], None, Some((0, 3)));

    check(&[0, 0, 5, 0, 0, 8], Some(1), None);
    check(&[0, 0, 5, 0, 0, 9], Some(1), None);
    check(&[0, 0, 5, 0, 0, 10], Some(1), None);
    check(&[0, 0, 5, 0, 0, 11], Some(1), Some((1, 3)));

    check(&[0, 0, 6, 0, 0, 8], None, None);
    check(&[0, 0, 6, 0, 0, 9], None, Some((2, 1)));
    check(&[0, 0, 6, 0, 0, 10], None, None);
    check(&[0, 0, 6, 0, 0, 11], None, Some((2, 3)));

    check(&[0, 0, 7, 0, 0, 8], Some(3), Some((3, 1)));
    check(&[0, 0, 7, 0, 0, 9], Some(3), None);
    check(&[0, 0, 7, 0, 0, 10], Some(3), None);
    check(&[0, 0, 7, 0, 0, 11], Some(3), None);
}

#[test]
fn node_locking() {
    let card_config = CardConfig {
        range: ["AsAh,QsQh".parse().unwrap(), "KsKh".parse().unwrap()],
        flop: flop_from_str("2s3h4d").unwrap(),
        turn: card_from_str("6c").unwrap(),
        river: card_from_str("7c").unwrap(),
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 20,
        effective_stack: 10,
        river_bet_sizes: [("a", "").try_into().unwrap(), ("a", "").try_into().unwrap()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    game.play(1); // all-in
    game.lock_current_strategy(&[0.25, 0.75]); // 25% fold, 75% call
    game.back_to_root();

    solve(&mut game, 1000, 0.0, false);
    game.cache_normalized_weights();

    let ev_oop = game.expected_values(0);
    let ev_ip = game.expected_values(1);
    assert!((ev_oop[0] - 0.0).abs() < 1e-2);
    assert!((ev_oop[1] - 27.5).abs() < 5e-2);
    assert!((ev_ip[0] - 6.25).abs() < 1e-2);

    let strategy_oop = game.strategy();
    assert!((strategy_oop[0] - 1.0).abs() < 1e-3); // QQ check
    assert!((strategy_oop[1] - 0.0).abs() < 1e-3); // AA check
    assert!((strategy_oop[2] - 0.0).abs() < 1e-3); // QQ bet
    assert!((strategy_oop[3] - 1.0).abs() < 1e-3); // AA bet

    game.allocate_memory(false);
    game.play(1); // all-in
    game.lock_current_strategy(&[0.5, 0.5]); // 50% fold, 50% call
    game.back_to_root();

    solve(&mut game, 1000, 0.0, false);
    game.cache_normalized_weights();

    let ev_oop = game.expected_values(0);
    let ev_ip = game.expected_values(1);
    assert!((ev_oop[0] - 5.0).abs() < 1e-2);
    assert!((ev_oop[1] - 25.0).abs() < 5e-2);
    assert!((ev_ip[0] - 5.0).abs() < 1e-2);

    let strategy_oop = game.strategy();
    assert!((strategy_oop[0] - 0.0).abs() < 1e-3); // QQ check
    assert!((strategy_oop[1] - 0.0).abs() < 1e-3); // AA check
    assert!((strategy_oop[2] - 1.0).abs() < 1e-3); // QQ bet
    assert!((strategy_oop[3] - 1.0).abs() < 1e-3); // AA bet
}

#[test]
fn called_allin_chance_nodes_require_exact_traversal() {
    let card_config = CardConfig {
        range: ["AsAh".parse().unwrap(), "KsKh".parse().unwrap()],
        flop: flop_from_str("2s3h4d").unwrap(),
        ..Default::default()
    };
    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: 20,
        effective_stack: 10,
        flop_bet_sizes: [("a", "").try_into().unwrap(), ("a", "").try_into().unwrap()],
        ..Default::default()
    };
    let action_tree = ActionTree::new(tree_config).unwrap();
    let game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    let root = game.root();
    let checked = root.play(0);
    let checked_back = checked.play(0);
    assert!(checked_back.is_chance());
    assert!(!game.should_traverse_chance_exactly(&checked_back));

    let allin = root.play(1);
    let called = allin.play(1);
    assert!(called.is_chance());
    assert_eq!(called.amount, game.tree_config.effective_stack);
    assert!(game.should_traverse_chance_exactly(&called));
}

#[test]
#[ignore = "expensive release-level 1326x1326 matrix diagnostic; covered by the WASM benchmark"]
fn flop_allin_matrix_matches_ordered_exact_evaluator() {
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("QsJh2h").unwrap(),
        ..Default::default()
    };
    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: 180,
        effective_stack: 910,
        ..Default::default()
    };
    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    game.enable_flop_allin_payout_matrix().unwrap();

    let mut node = *game.root();
    node.amount = game.tree_config.effective_stack;
    let player = 0;
    let cfreach = game.weights(player ^ 1).to_vec();
    let num_hands = game.num_private_hands(player);

    let mut matrix_result: Vec<MaybeUninit<f32>> =
        (0..num_hands).map(|_| MaybeUninit::uninit()).collect();
    assert!(game.evaluate_called_allin_matrix(&mut matrix_result, &node, player, &cfreach,));
    let matrix_result: Vec<f32> = matrix_result
        .into_iter()
        .map(|value| unsafe { value.assume_init() })
        .collect();

    let mut exact_sum = vec![0.0f64; num_hands];
    for turn in 0..52u8 {
        if game.card_config.flop.contains(&turn) {
            continue;
        }
        for river in 0..52u8 {
            if turn == river || game.card_config.flop.contains(&river) {
                continue;
            }
            let mut board_node = node;
            board_node.turn = turn;
            board_node.river = river;
            let scaled_reach: Vec<f32> = cfreach.iter().map(|value| value / 1980.0).collect();
            let mut result: Vec<MaybeUninit<f32>> =
                (0..num_hands).map(|_| MaybeUninit::uninit()).collect();
            game.evaluate_internal(&mut result, &board_node, player, &scaled_reach);
            for (sum, value) in exact_sum.iter_mut().zip(result) {
                *sum += unsafe { value.assume_init() } as f64;
            }
        }
    }

    let mut max_diff = 0.0f64;
    for (matrix_value, exact_value) in matrix_result.iter().zip(exact_sum) {
        let exact_value = exact_value as f32;
        max_diff = max_diff.max((*matrix_value as f64 - exact_value as f64).abs());
    }
    assert!(max_diff < 1e-5, "matrix/exact max diff = {max_diff}");
}

#[test]
#[ignore = "expensive release-level 1326x1326 matrix diagnostic"]
fn turn_allin_matrix_matches_exact_river_evaluator() {
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("QsJh2h").unwrap(),
        ..Default::default()
    };
    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: 180,
        effective_stack: 910,
        ..Default::default()
    };
    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    let turn = card_from_str("3d").unwrap();
    let canonical = game.build_turn_allin_payout_matrix(turn).unwrap();
    game.install_turn_allin_payout_matrix(turn, &canonical);

    let mut node = *game.root();
    node.amount = game.tree_config.effective_stack;
    node.turn = turn;
    node.river = NOT_DEALT;
    let player = 0;
    let cfreach = game.weights(player ^ 1).to_vec();
    let num_hands = game.num_private_hands(player);

    let mut matrix_result: Vec<MaybeUninit<f32>> =
        (0..num_hands).map(|_| MaybeUninit::uninit()).collect();
    assert!(game.evaluate_called_allin_matrix(&mut matrix_result, &node, player, &cfreach));
    let matrix_result: Vec<f32> = matrix_result
        .into_iter()
        .map(|value| unsafe { value.assume_init() })
        .collect();

    let mut exact_sum = vec![0.0f64; num_hands];
    for river in 0..52u8 {
        if river == turn || game.card_config.flop.contains(&river) {
            continue;
        }
        let mut board_node = node;
        board_node.river = river;
        let scaled_reach: Vec<f32> = cfreach.iter().map(|value| value / 44.0).collect();
        let mut result: Vec<MaybeUninit<f32>> =
            (0..num_hands).map(|_| MaybeUninit::uninit()).collect();
        game.evaluate_internal(&mut result, &board_node, player, &scaled_reach);
        for (sum, value) in exact_sum.iter_mut().zip(result) {
            *sum += unsafe { value.assume_init() } as f64;
        }
    }

    let max_diff = matrix_result
        .iter()
        .zip(exact_sum)
        .map(|(matrix, exact)| (*matrix as f64 - exact as f32 as f64).abs())
        .fold(0.0f64, f64::max);
    assert!(max_diff < 1e-5, "turn matrix/exact max diff = {max_diff}");
}

#[test]
fn node_locking_partial() {
    let card_config = CardConfig {
        range: ["AsAh,QsQh,JsJh".parse().unwrap(), "KsKh".parse().unwrap()],
        flop: flop_from_str("2s3h4d").unwrap(),
        turn: card_from_str("6c").unwrap(),
        river: card_from_str("7c").unwrap(),
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 10,
        effective_stack: 10,
        river_bet_sizes: [("a", "").try_into().unwrap(), ("a", "").try_into().unwrap()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    game.lock_current_strategy(&[0.8, 0.0, 0.0, 0.2, 0.0, 0.0]); // JJ -> 80% check, 20% all-in

    solve(&mut game, 1000, 0.0, false);
    game.cache_normalized_weights();

    let ev_oop = game.expected_values(0);
    let ev_ip = game.expected_values(1);
    assert!((ev_oop[0] - 0.0).abs() < 1e-2);
    assert!((ev_oop[1] - 0.0).abs() < 1e-2);
    assert!((ev_oop[2] - 15.0).abs() < 5e-2);
    assert!((ev_ip[0] - 5.0).abs() < 1e-2);

    let strategy_oop = game.strategy();
    assert!((strategy_oop[0] - 0.8).abs() < 1e-3); // JJ check
    assert!((strategy_oop[1] - 0.7).abs() < 1e-3); // QQ check
    assert!((strategy_oop[2] - 0.0).abs() < 1e-3); // AA check
    assert!((strategy_oop[3] - 0.2).abs() < 1e-3); // JJ bet
    assert!((strategy_oop[4] - 0.3).abs() < 1e-3); // QQ bet
    assert!((strategy_oop[5] - 1.0).abs() < 1e-3); // AA bet
}

#[test]
fn node_locking_isomorphism() {
    let card_config = CardConfig {
        range: ["AKs".parse().unwrap(), "AKs".parse().unwrap()],
        flop: flop_from_str("2c3c4c").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 10,
        effective_stack: 10,
        river_bet_sizes: [("a", "").try_into().unwrap(), Default::default()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    game.allocate_memory(false);
    game.apply_history(&[0, 0, 15, 0, 0, 14]); // Turn: Spades, River: Hearts
    game.lock_current_strategy(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]); // AhKh -> check

    finalize(&mut game);

    game.apply_history(&[0, 0, 13, 0, 0, 14]);
    assert_eq!(
        game.strategy(),
        vec![0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5]
    );

    game.apply_history(&[0, 0, 13, 0, 0, 15]);
    assert_eq!(
        game.strategy(),
        vec![0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0]
    );

    game.apply_history(&[0, 0, 14, 0, 0, 13]);
    assert_eq!(
        game.strategy(),
        vec![0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5]
    );

    game.apply_history(&[0, 0, 14, 0, 0, 15]);
    assert_eq!(
        game.strategy(),
        vec![0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0]
    );

    game.apply_history(&[0, 0, 15, 0, 0, 13]);
    assert_eq!(
        game.strategy(),
        vec![0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5]
    );

    game.apply_history(&[0, 0, 15, 0, 0, 14]);
    assert_eq!(
        game.strategy(),
        vec![0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5]
    );
}

#[test]
fn set_bunching_effect() {
    let flop = flop_from_str("Td9d6h").unwrap();
    let card_config = CardConfig {
        flop,
        range: [Range::ones(); 2],
        turn: card_from_str("Qc").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: 60,
        effective_stack: 970,
        river_bet_sizes: [("50%", "").try_into().unwrap(), Default::default()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    let co_range = "33:0.59,22:0.635,A8o:0.265,A7o-A6o,A5o:0.445,A4o-A2o,K2s,K9o:0.905,K8o-K2o,Q4s-Q2s,Q9o-Q2o,J6s-J2s,J9o:0.88,J8o-J2o,T7s:0.405,T6s-T2s,T9o:0.96,T8o-T2o,96s-92s,92o+,86s:0.57,85s-82s,82o+,76s:0.37,75s-72s,72o+,65s:0.475,64s-62s,62o+,54s:0.68,53s-52s,52o+,42+,32";
    let sb_range = "66:0.46,55:0.821,44:0.92,33:0.93,22:0.925,A6s:0.73,A3s:0.47,A2s,ATo:0.105,A9o-A2o,K8s:0.795,K7s,K6s:0.85,K5s:0.965,K4s-K2s,KJo:0.085,KTo:0.645,K9o-K2o,Q8s-Q2s,QJo:0.765,QTo-Q2o,J8s-J2s,J2o+,T8s:0.69,T7s-T2s,T2o+,98s:0.905,97s-92s,92o+,87s:0.78,86s-82s,82o+,76s:0.77,75s-72s,72o+,65s:0.845,64s-62s,62o+,54s:0.735,53s-52s,52o+,42+,32";

    let mut bunching_data = BunchingData::new(
        &[co_range.parse().unwrap(), sb_range.parse().unwrap()],
        flop,
    )
    .unwrap();

    bunching_data.process(false);
    game.set_bunching_effect(&bunching_data).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    let current_ev = compute_current_ev(&game);
    assert!((current_ev[0] - 7.5).abs() < 1e-4);
    assert!((current_ev[1] - -7.5).abs() < 1e-4);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_equity_oop = compute_average(&game.equity(0), weights_oop);
    let root_equity_ip = compute_average(&game.equity(1), weights_ip);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    assert!((root_equity_oop - 0.5).abs() < 1e-5);
    assert!((root_equity_ip - 0.5).abs() < 1e-5);
    assert!((root_ev_oop - 37.5).abs() < 1e-4);
    assert!((root_ev_ip - 22.5).abs() < 1e-4);
}

#[test]
fn set_bunching_effect_always_win() {
    let flop = flop_from_str("AcAdKh").unwrap();
    let lose_range_str = "KK-22,K9-K2,Q8-Q2,J8-J2,T8-T2,92+,82+,72+,62+";

    let card_config = CardConfig {
        range: ["AA".parse().unwrap(), lose_range_str.parse().unwrap()],
        flop,
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 60,
        effective_stack: 970,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    let co_range = "33:0.59,22:0.635,A8o:0.265,A7o-A6o,A5o:0.445,A4o-A2o,K2s,K9o:0.905,K8o-K2o,Q4s-Q2s,Q9o-Q2o,J6s-J2s,J9o:0.88,J8o-J2o,T7s:0.405,T6s-T2s,T9o:0.96,T8o-T2o,96s-92s,92o+,86s:0.57,85s-82s,82o+,76s:0.37,75s-72s,72o+,65s:0.475,64s-62s,62o+,54s:0.68,53s-52s,52o+,42+,32";
    let sb_range = "66:0.46,55:0.821,44:0.92,33:0.93,22:0.925,A6s:0.73,A3s:0.47,A2s,ATo:0.105,A9o-A2o,K8s:0.795,K7s,K6s:0.85,K5s:0.965,K4s-K2s,KJo:0.085,KTo:0.645,K9o-K2o,Q8s-Q2s,QJo:0.765,QTo-Q2o,J8s-J2s,J2o+,T8s:0.69,T7s-T2s,T2o+,98s:0.905,97s-92s,92o+,87s:0.78,86s-82s,82o+,76s:0.77,75s-72s,72o+,65s:0.845,64s-62s,62o+,54s:0.735,53s-52s,52o+,42+,32";

    let mut bunching_data = BunchingData::new(
        &[co_range.parse().unwrap(), sb_range.parse().unwrap()],
        flop,
    )
    .unwrap();

    bunching_data.process(false);
    game.set_bunching_effect(&bunching_data).unwrap();

    game.allocate_memory(false);
    finalize(&mut game);

    let current_ev = compute_current_ev(&game);
    assert!((current_ev[0] - 30.0).abs() < 1e-4);
    assert!((current_ev[1] - -30.0).abs() < 1e-4);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 1.0).abs() < 1e-5);
    assert!((equity_ip - 0.0).abs() < 1e-5);
    assert!((ev_oop - 60.0).abs() < 1e-4);
    assert!((ev_ip - 0.0).abs() < 1e-4);

    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(0);
    game.play(0);
    assert!(game.is_chance_node());
    game.play(usize::MAX);
    game.play(0);
    game.play(0);
    assert!(game.is_terminal_node());

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let equity_oop = compute_average(&game.equity(0), weights_oop);
    let equity_ip = compute_average(&game.equity(1), weights_ip);
    let ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let ev_ip = compute_average(&game.expected_values(1), weights_ip);
    assert!((equity_oop - 1.0).abs() < 1e-5);
    assert!((equity_ip - 0.0).abs() < 1e-5);
    assert!((ev_oop - 60.0).abs() < 1e-4);
    assert!((ev_ip - 0.0).abs() < 1e-4);
}

#[test]
#[ignore]
fn solve_pio_preset_normal() {
    let oop_range = "88+,A8s+,A5s-A2s:0.5,AJo+,ATo:0.75,K9s+,KQo,KJo:0.75,KTo:0.25,Q9s+,QJo:0.5,J8s+,JTo:0.25,T8s+,T7s:0.45,97s+,96s:0.45,87s,86s:0.75,85s:0.45,75s+:0.75,74s:0.45,65s:0.75,64s:0.5,63s:0.45,54s:0.75,53s:0.5,52s:0.45,43s:0.5,42s:0.45,32s:0.45";
    let ip_range = "AA:0.25,99-22,AJs-A2s,AQo-A8o,K2s+,K9o+,Q2s+,Q9o+,J6s+,J9o+,T6s+,T9o,96s+,95s:0.5,98o,86s+,85s:0.5,75s+,74s:0.5,64s+,63s:0.5,54s,53s:0.5,43s";

    let card_config = CardConfig {
        range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
        flop: flop_from_str("QsJh2h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 180,
        effective_stack: 910,
        flop_bet_sizes: [
            ("52%", "45%").try_into().unwrap(),
            ("52%", "45%").try_into().unwrap(),
        ],
        turn_bet_sizes: [
            ("55%", "45%").try_into().unwrap(),
            ("55%", "45%").try_into().unwrap(),
        ],
        river_bet_sizes: [
            ("70%", "45%").try_into().unwrap(),
            ("70%", "45%").try_into().unwrap(),
        ],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    println!(
        "memory usage: {:.2}GB",
        game.memory_usage().0 as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    game.allocate_memory(false);

    solve(&mut game, 1000, 180.0 * 0.001, true);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_equity_oop = compute_average(&game.equity(0), weights_oop);
    let root_equity_ip = compute_average(&game.equity(1), weights_ip);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    // verified by PioSOLVER Free
    assert!((root_equity_oop - 0.55347).abs() < 1e-5);
    assert!((root_equity_ip - 0.44653).abs() < 1e-5);
    assert!((root_ev_oop - 105.11).abs() < 0.2);
    assert!((root_ev_ip - 74.89).abs() < 0.2);
}

#[test]
#[ignore]
fn solve_pio_preset_raked() {
    let oop_range = "88+,A8s+,A5s-A2s:0.5,AJo+,ATo:0.75,K9s+,KQo,KJo:0.75,KTo:0.25,Q9s+,QJo:0.5,J8s+,JTo:0.25,T8s+,T7s:0.45,97s+,96s:0.45,87s,86s:0.75,85s:0.45,75s+:0.75,74s:0.45,65s:0.75,64s:0.5,63s:0.45,54s:0.75,53s:0.5,52s:0.45,43s:0.5,42s:0.45,32s:0.45";
    let ip_range = "AA:0.25,99-22,AJs-A2s,AQo-A8o,K2s+,K9o+,Q2s+,Q9o+,J6s+,J9o+,T6s+,T9o,96s+,95s:0.5,98o,86s+,85s:0.5,75s+,74s:0.5,64s+,63s:0.5,54s,53s:0.5,43s";

    let card_config = CardConfig {
        range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
        flop: flop_from_str("QsJh2h").unwrap(),
        ..Default::default()
    };

    let tree_config = TreeConfig {
        starting_pot: 180,
        effective_stack: 910,
        rake_rate: 0.05,
        rake_cap: 30.0,
        flop_bet_sizes: [
            ("52%", "45%").try_into().unwrap(),
            ("52%", "45%").try_into().unwrap(),
        ],
        turn_bet_sizes: [
            ("55%", "45%").try_into().unwrap(),
            ("55%", "45%").try_into().unwrap(),
        ],
        river_bet_sizes: [
            ("70%", "45%").try_into().unwrap(),
            ("70%", "45%").try_into().unwrap(),
        ],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    println!(
        "memory usage: {:.2}GB",
        game.memory_usage().0 as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    game.allocate_memory(false);

    solve(&mut game, 1000, 180.0 * 0.001, true);

    game.cache_normalized_weights();
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);
    let root_ev_oop = compute_average(&game.expected_values(0), weights_oop);
    let root_ev_ip = compute_average(&game.expected_values(1), weights_ip);

    // verified by PioSOLVER Free (but not theoretically guaranteed to be the same)
    assert!((root_ev_oop - 95.57).abs() < 0.2);
    assert!((root_ev_ip - 66.98).abs() < 0.2);
}

#[test]
fn turn_conversion_is_available_during_solving() {
    use crate::interface::{Game, GameNode};
    let cards = CardConfig {
        range: ["AcAd".parse().unwrap(), "KcKd".parse().unwrap()],
        flop: flop_from_str("Td9d6h").unwrap(),
        ..Default::default()
    };
    let tree = TreeConfig {
        starting_pot: 200,
        effective_stack: 900,
        ..Default::default()
    };
    let mut game = PostFlopGame::with_config(cards, ActionTree::new(tree).unwrap()).unwrap();
    let basis: Vec<_> = [
        "2c", "3d", "4h", "5s", "6c", "7d", "8h", "9s", "Tc", "Jd", "Qh", "Ks", "Ac",
    ]
    .iter()
    .map(|card| card_from_str(card).unwrap())
    .collect();
    let node_id = game
        .node_arena
        .iter()
        .position(|node| node.lock().is_chance())
        .unwrap() as u64;
    game.load_turn_cfv_conversion(
        node_id,
        flop_from_str("Td9d6h").unwrap(),
        &basis,
        "piosolver_hand_order.txt",
        "benchmarks/range_batch_weight_fits_diagonal13_f32le.bin",
    )
    .unwrap();
    {
        let root = game.node_arena[node_id as usize].lock();
        let conversion = Game::turn_cfv_conversion(&game, &root).unwrap();
        assert_eq!(conversion.weights(0).len(), 13);
        assert_eq!(conversion.basis_actions().len(), 13);
        let canonical = crate::pio_order::load_canonical_weights_with_basis(
            "benchmarks/range_batch_weight_fits_diagonal13_f32le.bin",
            13,
        )
        .unwrap();
        let order = crate::pio_order::load_pio_hand_order("piosolver_hand_order.txt").unwrap();
        let mapping = crate::pio_order::local_to_pio(&order, game.private_cards(0)).unwrap();
        for basis in 0..13 {
            assert_eq!(
                conversion.weights(0)[basis],
                canonical[0][basis * 1326 + mapping[0]]
            );
        }
    }
    game.allocate_memory(false);
    clear_cfv_captures();
    solve_step(&game, 4);
    let captures = take_cfv_captures();
    let converted: Vec<_> = captures
        .iter()
        .filter(|capture| capture.converted_sum.is_some())
        .collect();
    assert_eq!(converted.len(), 2);
    for capture in converted {
        let root = game.node_arena[node_id as usize].lock();
        let expected = Game::turn_cfv_conversion(&game, &root)
            .unwrap()
            .estimate_sum_at(capture.node_id, capture.player, &capture.values);
        assert_eq!(capture.converted_sum.as_ref().unwrap(), &expected);
        assert!(expected.iter().all(|value| value.is_finite()));
    }
    // Installing diagnostics must leave the exact traversed CFVs unchanged.
    let mut exact_game = PostFlopGame::with_config(
        game.card_config.clone(),
        ActionTree::new(game.tree_config.clone()).unwrap(),
    )
    .unwrap();
    exact_game.allocate_memory(false);
    clear_cfv_captures();
    solve_step(&exact_game, 4);
    let _ = take_cfv_captures();
    assert_eq!(game.storage1, exact_game.storage1);
    assert_eq!(game.storage2, exact_game.storage2);

    let mut invalid_basis = basis.clone();
    invalid_basis[1] = invalid_basis[0];
    assert!(game
        .load_turn_cfv_conversion(
            node_id,
            flop_from_str("Td9d6h").unwrap(),
            &invalid_basis,
            "piosolver_hand_order.txt",
            "benchmarks/range_batch_weight_fits_diagonal13_f32le.bin"
        )
        .is_err());
    game.clear_turn_cfv_conversion();
    assert!(Game::turn_cfv_conversion(&game, &game.node_arena[node_id as usize].lock()).is_none());
}

#[test]
fn weighted_conversion_covers_all_turn_nodes_but_no_rivers() {
    use crate::interface::{Game, GameNode};
    let cards = CardConfig {
        range: ["AcAd".parse().unwrap(), "KcKd".parse().unwrap()],
        flop: flop_from_str("Td9d6h").unwrap(),
        ..Default::default()
    };
    let sizes = crate::BetSizeOptions::try_from(("60%", "2.5x")).unwrap();
    let tree = TreeConfig {
        starting_pot: 60,
        effective_stack: 900,
        flop_bet_sizes: [sizes.clone(), sizes],
        ..Default::default()
    };
    let mut game = PostFlopGame::with_config(cards, ActionTree::new(tree).unwrap()).unwrap();
    let source = game
        .node_arena
        .iter()
        .position(|node| {
            let node = node.lock();
            node.is_chance() && node.turn == NOT_DEALT
        })
        .unwrap() as u64;
    let basis: Vec<_> = (0..52u8)
        .filter(|card| card & 3 != 3 && !game.card_config.flop.contains(card))
        .collect();
    game.load_turn_cfv_conversion(
        source,
        game.card_config.flop,
        &basis,
        "piosolver_hand_order.txt",
        "benchmarks/range_batch_weight_fits_no_spades36_f32le.bin",
    )
    .unwrap();
    assert_eq!(Game::cfv_capture_node_id(&game), source);
    let count = game.set_weighted_all_turn_sums(true).unwrap();
    assert!(count > 1);
    let mut matched = 0;
    let mut rivers = 0;
    for node in &game.node_arena {
        let node = node.lock();
        let conversion = Game::turn_cfv_conversion(&game, &node);
        if node.is_chance() && node.turn == NOT_DEALT {
            assert!(conversion.unwrap().use_in_solving());
            matched += 1;
        } else {
            assert!(conversion.is_none());
            if node.is_chance() {
                rivers += 1;
            }
        }
    }
    assert_eq!(matched, count);
    assert!(rivers > 0);
    game.set_weighted_all_turn_sums(false).unwrap();
    for node in &game.node_arena {
        let node = node.lock();
        if let Some(matrix) = Game::turn_cfv_conversion(&game, &node) {
            assert_eq!(node.id, source);
            assert!(!matrix.use_in_solving());
        }
    }
}

#[test]
fn bucketed_conversion_loads_and_runs() {
    use crate::interface::{Game, GameNode};
    let cards = CardConfig {
        range: ["AcAd".parse().unwrap(), "KcKd".parse().unwrap()],
        flop: flop_from_str("Td9d6h").unwrap(),
        ..Default::default()
    };
    let tree = TreeConfig {
        starting_pot: 200,
        effective_stack: 900,
        ..Default::default()
    };
    let mut game = PostFlopGame::with_config(cards, ActionTree::new(tree).unwrap()).unwrap();
    let basis: Vec<_> = [
        "2c", "3d", "4h", "5s", "6c", "7d", "8h", "9s", "Tc", "Jd", "Qh", "Ks", "Ac",
    ]
    .iter()
    .map(|card| card_from_str(card).unwrap())
    .collect();
    let node_id = game
        .node_arena
        .iter()
        .position(|node| node.lock().is_chance())
        .unwrap() as u64;
    game.load_bucketed_turn_cfv_conversion(
        node_id,
        flop_from_str("Td9d6h").unwrap(),
        &basis,
        "piosolver_hand_order.txt",
        "benchmarks/range_batch_bucketed13_assignments_u8.bin",
        "benchmarks/range_batch_bucketed13_f32le.bin",
        32,
    )
    .unwrap();
    {
        let node = game.node_arena[node_id as usize].lock();
        let conversion = Game::turn_cfv_conversion(&game, &node).unwrap();
        assert_eq!(conversion.basis_actions().len(), 13);
        assert_eq!(conversion.bucket_count(), 32);
    }
    game.allocate_memory(false);
    clear_cfv_captures();
    solve_step(&game, 4);
    let captures = take_cfv_captures();
    assert_eq!(captures.len(), 2);
    assert!(captures.iter().all(|capture| capture
        .converted_sum
        .as_ref()
        .is_some_and(|values| values.iter().all(|value| value.is_finite()))));
}

#[test]
fn all_chance_node_captures_preserve_exact_cfvs() {
    use crate::interface::{Game, GameNode};
    let cards = CardConfig {
        range: ["AcAd".parse().unwrap(), "KcKd".parse().unwrap()],
        flop: flop_from_str("Td9d6h").unwrap(),
        ..Default::default()
    };
    let sizes = crate::BetSizeOptions::try_from(("60%, e, a", "2.5x")).unwrap();
    let tree = TreeConfig {
        starting_pot: 200,
        effective_stack: 900,
        flop_bet_sizes: [sizes.clone(), sizes.clone()],
        turn_bet_sizes: [sizes.clone(), sizes.clone()],
        river_bet_sizes: [sizes.clone(), sizes],
        river_donk_sizes: Some(crate::DonkSizeOptions::try_from("50%").unwrap()),
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
        ..Default::default()
    };
    let mut exact =
        PostFlopGame::with_config(cards.clone(), ActionTree::new(tree.clone()).unwrap()).unwrap();
    let mut all = PostFlopGame::with_config(cards, ActionTree::new(tree).unwrap()).unwrap();
    exact.allocate_memory(false);
    all.allocate_memory(false);
    all.set_capture_all_turn_nodes(true);
    all.set_capture_all_river_nodes(true);
    assert_eq!(all.turn_cfv_capture_contexts().len(), 9);
    assert_eq!(all.river_cfv_capture_node_count(), 2401);
    assert_eq!(all.river_cfv_capture_contexts().len(), 2401);
    let semantic = all.turn_cfv_semantic_contexts().unwrap();
    assert_eq!(semantic.len(), 9);
    assert!(semantic
        .iter()
        .all(|(_, row)| row.len() == 95 && row.iter().all(|v| v.is_finite())));
    let check_check = semantic
        .iter()
        .find(|(_, row)| row[7] == 1.0 && row[9] == 1.0 && row[19] == 1.0 && row[20] == 1.0)
        .unwrap();
    assert!((check_check.1[0] - 200.0 / 900.0).abs() < 1e-6);
    assert_eq!(&check_check.1[1..5], &[1.0, 1.0, 0.0, 0.0]);
    set_cfv_capture_player_reach(true);
    clear_cfv_captures();
    solve_step(&exact, 4);
    let original = take_cfv_captures();
    assert_eq!(original.len(), 2);
    solve_step(&all, 4);
    let expanded = take_cfv_captures();
    assert_eq!(expanded.len(), 18 + 2401 * 2);
    assert!(expanded.iter().all(|capture| {
        capture.player_reach.len() == all.num_private_hands(capture.player)
            && capture
                .player_reach
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
    }));
    for capture in expanded
        .iter()
        .filter(|capture| capture.turn_card == NOT_DEALT)
    {
        let node = all.node_arena[capture.node_id as usize].lock();
        assert!(node.is_chance() && node.turn == NOT_DEALT);
        assert_eq!(capture.values.len(), 49);
        assert_eq!(capture.chance_cards.len(), 49);
    }
    for capture in expanded
        .iter()
        .filter(|capture| capture.turn_card != NOT_DEALT)
    {
        let node = all.node_arena[capture.node_id as usize].lock();
        assert!(node.is_chance() && node.turn == capture.turn_card);
        assert_eq!(capture.values.len(), 48);
        assert_eq!(capture.chance_cards.len(), 48);
        assert!(!capture.chance_cards.contains(&capture.turn_card));
    }
    for capture in original {
        let other = expanded
            .iter()
            .find(|c| c.node_id == capture.node_id && c.player == capture.player)
            .unwrap();
        assert_eq!(capture.values, other.values);
    }
    assert_eq!(compute_exploitability(&exact), compute_exploitability(&all));
    all.set_capture_all_turn_nodes(false);
    all.set_capture_all_river_nodes(false);
    assert!(all.node_arena.iter().all(|n| {
        let n = n.lock();
        Game::capture_turn_cfvs(&all, &n) == (n.id() == Game::cfv_capture_node_id(&all))
    }));
    let selected_turn = card_from_str("2c").unwrap();
    all.set_capture_river_turns(&[selected_turn]);
    assert!(all.node_arena.iter().all(|n| {
        let n = n.lock();
        let expected =
            n.id() == Game::cfv_capture_node_id(&all) || (n.is_chance() && n.turn == selected_turn);
        Game::capture_turn_cfvs(&all, &n) == expected
    }));
    all.set_capture_river_turns(&[]);
    set_cfv_capture_player_reach(false);
}

#[test]
fn analysis_equity_blocker_dominated_reach_is_bounded_and_exact() {
    for tiny in [1e-6f32, 1e-9, 1e-12] {
        let cards = CardConfig {
            range: ["AsAh".parse().unwrap(), "AsKd,KsKh".parse().unwrap()],
            flop: flop_from_str("2c3d4h").unwrap(),
            turn: card_from_str("7s").unwrap(),
            river: card_from_str("9c").unwrap(),
        };
        let config = TreeConfig { starting_pot: 60, effective_stack: 970, initial_state: BoardState::River, ..Default::default() };
        let mut game = PostFlopGame::with_config(cards, ActionTree::new(config).unwrap()).unwrap();
        game.allocate_memory(false);
        finalize(&mut game);
        let blocked_card = card_from_str("As").unwrap();
        for (i, &(a,b)) in game.private_cards[1].iter().enumerate() {
            game.weights[1][i] = if a == blocked_card || b == blocked_card { 1.0 } else { tiny };
        }
        game.is_normalized_weight_cached = false;
        game.cache_normalized_weights();
        let weights_before = game.weights.clone();
        let normalized_before = game.normalized_weights.clone();
        let old_equity = game.equity(0);
        assert!(game.normalized_weights(0)[0] > 0.0);
        assert_eq!(game.equity_for_analysis(0), vec![1.0], "tiny={tiny}");
        assert_eq!(game.weights, weights_before);
        assert_eq!(game.normalized_weights, normalized_before);
        assert_eq!(game.equity(0), old_equity);
    }
}
