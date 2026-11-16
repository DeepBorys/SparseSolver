mod base;
mod auxiliary_turn;
pub use base::{
    set_basis13_river_tree, set_basis13_turn_tree, set_basis6_river_tree, set_basis6_turn_tree,
    set_custom_river_basis_cards, set_custom_turn_basis_cards, set_exact_allin_matrix_f32_eval,
    set_exact_allin_matrix_sparse_eval, set_exact_allin_matrix_tree, set_river_basis_count,
    set_turn_basis_count, set_physical_chance_isomorphism,
};
mod evaluation;
mod interpreter;
mod frozen_river_selection;
mod node;
mod regret_transfer;
mod check_check_river;
mod shared_regret_transfer;
pub use shared_regret_transfer::{K1SharedHandTable,K1SharedNodeCopy,K1SharedTransferPlan};
pub use regret_transfer::{K1RegretNodeCopy,K1RegretTransferPlan};
#[cfg(feature="neural")]
pub use regret_transfer::NeuralPhaseBank;
#[cfg(feature = "bincode")]
mod serialization;

#[cfg(test)]
mod tests;

use crate::action_tree::*;
use crate::card::*;
use crate::mutex_like::*;
use std::collections::BTreeMap;

#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};

#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
enum State {
    ConfigError = 0,
    #[default]
    Uninitialized = 1,
    TreeBuilt = 2,
    MemoryAllocated = 3,
    Solved = 4,
}

/// A struct representing a postflop game.
#[derive(Default)]
pub struct PostFlopGame {
    // state
    state: State,
    // Opt-in pruned builder: omitted runout tables must not serve full-board equities.
    partial_card_tables_experiment: bool,
    // An exact continuation trained alongside, but excluded from, the neural turn sum.
    auxiliary_exact_turn: Option<auxiliary_turn::AuxiliaryExactTurn>,
    // Runtime-only metadata. Owns no pointers; cleared when geometry changes.
    direct_terminal_plans: std::sync::Mutex<BTreeMap<u64, std::sync::Arc<crate::CompiledTerminalPlan>>>,

    // postflop game configurations
    card_config: CardConfig,
    tree_config: TreeConfig,
    added_lines: Vec<Vec<Action>>,
    removed_lines: Vec<Vec<Action>>,
    action_root: Box<MutexLike<ActionTreeNode>>,

    // computed from configurations
    num_combinations: f64,
    initial_weights: [Vec<f32>; 2],
    private_cards: [Vec<(Card, Card)>; 2],
    same_hand_index: [Vec<u16>; 2],

    // indices in `private_cards` that do not conflict with the specified board cards
    valid_indices_flop: [Vec<u16>; 2],
    valid_indices_turn: Vec<[Vec<u16>; 2]>,
    valid_indices_river: Vec<[Vec<u16>; 2]>,

    // hand strength information: indices are stored in ascending strength order
    hand_strength: Vec<[Vec<StrengthItem>; 2]>,

    // Optional exact called-all-in payout matrices. Canonical caches use
    // 1326x1326 hand order; these runtime matrices contain only local range
    // rows and columns. Turn matrices are keyed by the dealt turn card.
    flop_allin_payout_matrix: Option<FlopAllinPayoutMatrix>,
    turn_allin_payout_matrices: BTreeMap<Card, FlopAllinPayoutMatrix>,
    use_flop_allin_payout_matrix: bool,
    // Opt-in exact showdown leaves at proved check/check-only river runouts.
    // Runtime-only; the physical tree and serialized format stay unchanged.
    check_check_river_nodes: std::collections::BTreeSet<u64>,

    // isomorphism information
    // - `isomorphism_ref_*`: indices to which the eliminated events should refer
    // - `isomorphism_card_*`: list of cards eliminated by the isomorphism
    // - `isomorphism_swap_*`: list of hand index pairs that should be swapped when applying the
    //                         isomorphism with the specified suit
    isomorphism_ref_turn: Vec<u8>,
    isomorphism_card_turn: Vec<Card>,
    isomorphism_swap_turn: [SwapList; 4],
    isomorphism_ref_river: Vec<Vec<u8>>,
    isomorphism_card_river: [Vec<Card>; 4],
    isomorphism_swap_river: [[SwapList; 4]; 4],

    // bunching effect
    bunching_num_dead_cards: usize,
    bunching_num_combinations: f64,
    bunching_arena: Vec<f32>,
    bunching_strength: Vec<[Vec<u16>; 2]>,
    bunching_num_flop: [Vec<usize>; 2],
    bunching_num_turn: [Vec<Vec<usize>>; 2],
    bunching_num_river: [Vec<Vec<usize>>; 2],
    bunching_coef_flop: [Vec<usize>; 2],
    bunching_coef_turn: [Vec<Vec<usize>>; 2],

    // store options
    storage_mode: BoardState,
    target_storage_mode: BoardState,
    num_nodes: [u64; 3],
    is_compression_enabled: bool,
    num_storage: u64,
    num_storage_ip: u64,
    num_storage_chance: u64,
    misc_memory_usage: u64,

    // global storage
    // `storage*` are used as a global storage and are referenced by `PostFlopNode::storage*`.
    // Methods like `PostFlopNode::strategy` define how the storage is used.
    node_arena: Vec<MutexLike<PostFlopNode>>,
    storage1: Vec<u8>,
    storage2: Vec<u8>,
    storage_ip: Vec<u8>,
    storage_chance: Vec<u8>,
    locking_strategy: BTreeMap<usize, Vec<f32>>,

    // Optional runtime diagnostic matrix, excluded from serialization.
    turn_cfv_conversion: Option<crate::pio_order::TurnCfvConversion>,
    secondary_turn_cfv_conversion: Option<crate::pio_order::TurnCfvConversion>,
    secondary_turn_cfv_nodes: Vec<u64>,
    tertiary_turn_cfv_conversion: Option<crate::pio_order::TurnCfvConversion>,
    tertiary_turn_cfv_nodes: Vec<u64>,
    river_cfv_conversions: BTreeMap<Card, crate::pio_order::TurnCfvConversion>,
    secondary_river_cfv_conversions: BTreeMap<Card, crate::pio_order::TurnCfvConversion>,
    secondary_river_cfv_nodes: Vec<u64>,
    tertiary_river_cfv_conversions: BTreeMap<Card, crate::pio_order::TurnCfvConversion>,
    tertiary_river_cfv_nodes: Vec<u64>,
    #[cfg(feature = "neural")]
    glued_river_modules: Option<[std::sync::Mutex<tch::CModule>; 2]>,

    // Optional runtime capture scope, excluded from serialization.
    capture_all_turn_nodes: bool,
    capture_all_river_nodes: bool,
    capture_river_turns: Vec<Card>,

    // result interpreter
    action_history: Vec<usize>,
    node_history: Vec<usize>,
    is_normalized_weight_cached: bool,
    turn: Card,
    river: Card,
    turn_swapped_suit: Option<(u8, u8)>,
    turn_swap: Option<u8>,
    river_swap: Option<(u8, u8)>,
    total_bet_amount: [i32; 2],
    weights: [Vec<f32>; 2],
    normalized_weights: [Vec<f32>; 2],
    cfvalues_cache: [Vec<f32>; 2],
}

struct FlopAllinPayoutMatrix {
    // Directional, contiguous local-hand matrices. `values[player]` is laid
    // out as local hero rows by local opponent columns. Keep payouts in f64
    // so this exact path rounds only at the solver's normal f32 CFV boundary.
    values: [Vec<f64>; 2],
    // Optional opponent-major layout used by the experimental sparse evaluator.
    values_transposed: [Vec<f64>; 2],
    // Optional all-f32 row-major and opponent-major experimental layouts.
    values_f32: [Vec<f32>; 2],
    values_transposed_f32: [Vec<f32>; 2],
    local_hands: [usize; 2],
}

/// A struct representing a node in a postflop game tree.
///
/// The nodes must be stored as `Vec<MutexLike<PostFlopNode>>`.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PostFlopNode {
    id: u64,
    prev_action: Action,
    player: u8,
    turn: Card,
    river: Card,
    is_locked: bool,
    amount: i32,
    children_offset: u32,
    num_children: u16,
    num_elements_ip: u16,
    num_elements: u32,
    scale1: f32,
    scale2: f32,
    scale3: f32,
    storage1: *mut u8, // strategy
    storage2: *mut u8, // regrets or cfvalues
    storage3: *mut u8, // IP cfvalues
}

unsafe impl Send for PostFlopNode {}
unsafe impl Sync for PostFlopNode {}
