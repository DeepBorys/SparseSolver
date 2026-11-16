use super::*;
use crate::bunching::*;
use crate::card_from_str;
#[cfg(feature = "neural")]
use crate::card_to_string;
use crate::interface::*;
use crate::utility::*;
use std::mem::{self, MaybeUninit};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

// Exact capture experiment: retain requested cards plus canonical suit representatives.
static PHYSICAL_CHANCE_ISOMORPHISM: Mutex<Option<(Vec<Card>,Vec<Card>)>> = Mutex::new(None);
pub fn set_physical_chance_isomorphism(enabled: bool, turns: &[Card], rivers: &[Card]) {
    *PHYSICAL_CHANCE_ISOMORPHISM.lock().unwrap()=enabled.then(||(turns.to_vec(),rivers.to_vec()));
}
fn retain_required_chances(flop:[Card;3],turn:Card,refs:&mut Vec<u8>,omitted:&mut Vec<Card>,required:&[Card]) {
    let legal:Vec<Card>=(0..52).filter(|c|!flop.contains(c)&&*c!=turn).collect();
    let old:Vec<Card>=legal.iter().copied().filter(|c|!omitted.contains(c)).collect();
    let pairs:Vec<(Card,Card)>=omitted.iter().copied().zip(refs.iter().map(|i|old[*i as usize])).filter(|(c,_)|!required.contains(c)).collect();
    *omitted=pairs.iter().map(|x|x.0).collect();
    let kept:Vec<Card>=legal.into_iter().filter(|c|!omitted.contains(c)).collect();
    *refs=pairs.iter().map(|(_,source)|kept.iter().position(|c|c==source).unwrap() as u8).collect();
}

static BASIS13_TURN_TREE: AtomicBool = AtomicBool::new(false);
static BASIS13_RIVER_TREE: AtomicBool = AtomicBool::new(false);
static BASIS6_TURN_TREE: AtomicBool = AtomicBool::new(false);
static BASIS6_RIVER_TREE: AtomicBool = AtomicBool::new(false);
static TURN_BASIS_COUNT: AtomicUsize = AtomicUsize::new(0);
static RIVER_BASIS_COUNT: AtomicUsize = AtomicUsize::new(0);
static CUSTOM_TURN_BASIS_CARDS: Mutex<Vec<Card>> = Mutex::new(Vec::new());
static CUSTOM_RIVER_BASIS_CARDS: Mutex<Vec<Card>> = Mutex::new(Vec::new());
static EXACT_ALLIN_MATRIX_TREE: AtomicBool = AtomicBool::new(false);
pub(super) static EXACT_ALLIN_MATRIX_SPARSE_EVAL: AtomicBool = AtomicBool::new(false);
pub(super) static EXACT_ALLIN_MATRIX_F32_EVAL: AtomicBool = AtomicBool::new(false);

/// Replaces called flop and turn all-in runout subtrees with exact payout-matrix leaves.
/// Must be set before constructing a `PostFlopGame`.
pub fn set_exact_allin_matrix_tree(enabled: bool) {
    EXACT_ALLIN_MATRIX_TREE.store(enabled, Ordering::Relaxed);
}

/// Selects the experimental zero-skipping f64 evaluator for exact all-in matrices.
/// Must be set before installing the matrices so their transposed layout is built.
pub fn set_exact_allin_matrix_sparse_eval(enabled: bool) {
    EXACT_ALLIN_MATRIX_SPARSE_EVAL.store(enabled, Ordering::Relaxed);
}

/// Selects the experimental all-f32 dense/sparse evaluator.
/// Must be set before installing matrices so their f32 layouts are built.
pub fn set_exact_allin_matrix_f32_eval(enabled: bool) {
    EXACT_ALLIN_MATRIX_F32_EVAL.store(enabled, Ordering::Relaxed);
}

/// Experimental physical tree containing only the six deployed aggregate-model turns.
/// Must be set before constructing a `PostFlopGame`.
pub fn set_basis6_turn_tree(enabled: bool) {
    BASIS6_TURN_TREE.store(enabled, Ordering::Relaxed);
}

/// Experimental physical tree containing only the six deployed aggregate-model rivers.
/// Must be set before constructing a `PostFlopGame`.
pub fn set_basis6_river_tree(enabled: bool) {
    BASIS6_RIVER_TREE.store(enabled, Ordering::Relaxed);
}

/// Experimental physical tree containing only the 13 aggregate-model turns.
/// Must be set before constructing a `PostFlopGame`.
pub fn set_basis13_turn_tree(enabled: bool) {
    BASIS13_TURN_TREE.store(enabled, Ordering::Relaxed);
}

#[cfg(feature = "neural")]
fn river_uses_sparse_backend(requested: bool, metadata: &str, bucketed: bool, native_exists: bool) -> bool {
    requested
        && !metadata.contains("\"torchscript_only\": true")
        && (!bucketed || native_exists)
}

#[cfg(feature = "neural")]
fn turn_torchscript_options(metadata: &str) -> (bool, bool) {
    (
        metadata.contains("\"torchscript_only\": true"),
        metadata.contains("\"fourth_input\": \"initial_ranges\""),
    )
}

#[cfg(all(test, feature = "neural"))]
mod neural_interface_tests {
    use super::PostFlopGame;

    #[test]
    fn turn_torchscript_support_is_explicit_and_opt_in() {
        use super::turn_torchscript_options;
        assert_eq!(turn_torchscript_options(""), (false, false));
        assert_eq!(turn_torchscript_options(r#"{"torchscript_only": false}"#), (false, false));
        assert_eq!(turn_torchscript_options(r#"{"torchscript_only": true, "fourth_input": "initial_ranges"}"#), (true, true));
    }

    #[test]
    fn explicit_torchscript_river_preserves_legacy_backend_defaults() {
        use super::river_uses_sparse_backend;
        assert!(river_uses_sparse_backend(true, "", false, false));
        assert!(river_uses_sparse_backend(true, "", true, true));
        assert!(!river_uses_sparse_backend(true, "", true, false));
        assert!(!river_uses_sparse_backend(false, "", false, true));
        assert!(!river_uses_sparse_backend(true, r#"{"torchscript_only": true}"#, false, true));
        assert!(river_uses_sparse_backend(true, r#"{"torchscript_only": false}"#, false, true));
    }

    fn detect(
        metadata: Option<&str>,
    ) -> (&'static str, &'static str, usize, bool, bool, bool, bool) {
        let directory = std::env::temp_dir().join(format!(
            "postflop_turn_interface_{}_{}",
            std::process::id(),
            metadata.unwrap_or("legacy").len()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        if let Some(contents) = metadata {
            std::fs::write(directory.join("model_interface.json"), contents).unwrap();
        }
        let result = PostFlopGame::turn_model_interface(&directory.join("model_oop.pt")).unwrap();
        std::fs::remove_dir_all(directory).unwrap();
        result
    }

    #[test]
    fn legacy_turn_checkpoint_stays_two_input() {
        assert_eq!(
            detect(None),
            ("semantic", "none", 0, false, false, false, false)
        );
    }

    #[test]
    fn explicit_turn_interfaces_are_not_cli_inferred() {
        assert_eq!(
            detect(Some(
                r#"{"interface_version": 1, "context": "semantic_v1", "third_input": "initial_ranges", "basis_count": 6}"#
            )),
            ("semantic", "initial_ranges", 6, false, true, false, false)
        );
        assert_eq!(
            detect(Some(
                r#"{"interface_version": 1, "context": "pot_fraction", "third_input": "current_reaches", "basis_count": 6}"#
            )),
            (
                "pot_fraction",
                "current_reaches",
                6,
                false,
                true,
                false,
                false
            )
        );
        assert_eq!(
            detect(Some(
                r#"{"interface_version": 1, "context": "pot_fraction", "third_input": "current_reaches", "basis_count": 0, "output": "deepstack_normalized_turn_sum"}"#
            )),
            (
                "pot_fraction",
                "current_reaches",
                0,
                true,
                true,
                false,
                false
            )
        );
    }

    #[test]
    fn supported_hand_rms_requires_explicit_metadata() {
        assert_eq!(
            detect(Some(
                r#"{"interface_version": 1, "context": "pot_fraction", "third_input": "current_reaches", "basis_count": 6, "rms_denominator": "supported_hands_times_basis_count"}"#
            )),
            (
                "pot_fraction",
                "current_reaches",
                6,
                false,
                true,
                false,
                true
            )
        );
    }
}

pub fn set_basis13_river_tree(enabled: bool) {
    BASIS13_RIVER_TREE.store(enabled, Ordering::Relaxed);
}

/// Experimental speed-only tree limit using the deployed model's turn cards.
/// A value of zero preserves the normal physical turn tree.
pub fn set_turn_basis_count(count: usize) {
    TURN_BASIS_COUNT.store(count, Ordering::Relaxed);
}

/// Overrides the historical count-based physical turn selection with the
/// deployed model's explicit basis. Must be set before constructing a game.
pub fn set_custom_turn_basis_cards(cards: &[Card]) {
    *CUSTOM_TURN_BASIS_CARDS.lock().unwrap() = cards.to_vec();
}

/// Overrides the historical count-based physical river selection with the
/// deployed model's explicit basis. Must be set before constructing a game.
pub fn set_custom_river_basis_cards(cards: &[Card]) {
    *CUSTOM_RIVER_BASIS_CARDS.lock().unwrap() = cards.to_vec();
}

/// Experimental speed-only tree limit for deterministic river representatives.
/// A value of zero preserves the normal physical river tree.
pub fn set_river_basis_count(count: usize) {
    RIVER_BASIS_COUNT.store(count, Ordering::Relaxed);
}

fn basis13_turn_cards() -> [Card; 13] {
    [
        "2c", "3d", "4h", "5s", "6c", "7d", "8h", "9s", "Tc", "Jd", "Qh", "Ks", "Ac",
    ]
    .map(|card| card_from_str(card).unwrap())
}

fn basis6_turn_cards() -> [Card; 6] {
    ["3c", "5d", "7h", "9s", "Jc", "Kd"].map(|card| card_from_str(card).unwrap())
}

fn selected_turn_cards(count: usize) -> Vec<Card> {
    let custom = CUSTOM_TURN_BASIS_CARDS.lock().unwrap();
    if !custom.is_empty() {
        assert_eq!(
            custom.len(),
            count,
            "custom physical turn basis must match --turn-basis-count"
        );
        return custom.clone();
    }
    let names: &[&str] = match count {
        1 => &["3c"],
        2 => &["3c", "Kd"],
        4 => &["3c", "7h", "9s", "Kd"],
        6 => &["3c", "5d", "7h", "9s", "Jc", "Kd"],
        13 => &[
            "2c", "3d", "4h", "5s", "6c", "7d", "8h", "9s", "Tc", "Jd", "Qh", "Ks", "Ac",
        ],
        _ => panic!("turn basis count must be 1, 2, 4, 6, or 13"),
    };
    names
        .iter()
        .map(|card| card_from_str(card).unwrap())
        .collect()
}

fn basis6_river_cards() -> [Card; 6] {
    ["2s", "4c", "6s", "8d", "Th", "Qc"].map(|card| card_from_str(card).unwrap())
}

fn selected_river_cards(count: usize) -> Vec<Card> {
    let custom = CUSTOM_RIVER_BASIS_CARDS.lock().unwrap();
    if !custom.is_empty() {
        assert_eq!(
            custom.len(),
            count,
            "custom physical river basis must match --river-basis-count"
        );
        return custom.clone();
    }
    let names: &[&str] = match count {
        1 => &["Qc"],
        2 => &["2s", "Qc"],
        4 => &["2s", "5c", "8d", "Qc"],
        6 => &["2s", "4c", "6s", "8d", "Th", "Qc"],
        _ => panic!("river basis count must be 1, 2, 4, or 6 for the canonical compact tree"),
    };
    names
        .iter()
        .map(|card| card_from_str(card).unwrap())
        .collect()
}

fn basis13_river_cards(flop: [Card; 3], turn: Card) -> [Card; 13] {
    std::array::from_fn(|rank| {
        (0..4)
            .map(|offset| 4 * rank as u8 + ((rank + offset) % 4) as u8)
            .find(|card| !flop.contains(card) && *card != turn)
            .unwrap()
    })
}

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[derive(Default)]
struct BuildTreeInfo {
    flop_path: Vec<Action>,
    flop_index: usize,
    turn_index: usize,
    river_index: usize,
    num_storage: u64,
    num_storage_ip: u64,
    num_storage_chance: u64,
}

impl Game for PostFlopGame {
    type Node = PostFlopNode;

    fn direct_terminal_plan(&self, node: &Self::Node) -> Option<std::sync::Arc<crate::CompiledTerminalPlan>> {
        if let Some(plan) = self.direct_terminal_plans.lock().unwrap().get(&node.id).cloned() { return Some(plan); }
        let plan = std::sync::Arc::new(crate::CompiledTerminalPlan::build(node));
        Some(self.direct_terminal_plans.lock().unwrap().entry(node.id).or_insert(plan).clone())
    }

    fn direct_terminal_node(&self, id: u64) -> Option<MutexGuardLike<Self::Node>> {
        self.node_arena.get(id as usize).map(MutexLike::lock)
    }

    fn auxiliary_chance_action(&self, node: &Self::Node) -> Option<usize> {
        self.auxiliary_exact_turn.as_ref()
            .filter(|aux| aux.chance_node_id == node.id)
            .map(|aux| aux.action)
    }

    fn has_auxiliary_subgame(&self) -> bool {
        self.auxiliary_exact_turn.is_some()
    }

    #[inline]
    fn root(&self) -> MutexGuardLike<Self::Node> {
        self.node_arena[0].lock()
    }

    #[inline]
    fn num_private_hands(&self, player: usize) -> usize {
        self.private_cards[player].len()
    }

    #[inline]
    fn initial_weights(&self, player: usize) -> &[f32] {
        &self.initial_weights[player]
    }

    #[inline]
    fn evaluate(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &Self::Node,
        player: usize,
        cfreach: &[f32],
    ) {
        if self.bunching_num_dead_cards == 0 {
            self.evaluate_internal(result, node, player, cfreach);
        } else {
            self.evaluate_internal_bunching(result, node, player, cfreach);
        }
    }

    #[inline]
    fn evaluate_called_allin(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &Self::Node,
        player: usize,
        cfreach: &[f32],
    ) -> bool {
        self.use_flop_allin_payout_matrix
            && self.evaluate_called_allin_matrix(result, node, player, cfreach)
    }

    #[inline]
    fn compare_called_allin(
        &self,
        result: &[f64],
        node: &Self::Node,
        player: usize,
        cfreach: &[f32],
    ) -> Option<f64> {
        self.compare_called_allin_matrix(result, node, player, cfreach)
    }

    #[inline]
    fn chance_factor(&self, node: &Self::Node) -> usize {
        if node.turn == NOT_DEALT {
            45 - self.bunching_num_dead_cards
        } else {
            44 - self.bunching_num_dead_cards
        }
    }

    #[inline]
    fn should_traverse_chance_exactly(&self, node: &Self::Node) -> bool {
        node.is_chance() && (node.amount == self.tree_config.effective_stack
            || self.check_check_river_nodes.contains(&node.id))
    }

    #[inline]
    fn is_solved(&self) -> bool {
        self.state == State::Solved
    }

    #[inline]
    fn set_solved(&mut self) {
        self.state = State::Solved;
        let history = self.action_history.clone();
        self.apply_history(&history);
    }

    #[inline]
    fn is_ready(&self) -> bool {
        self.state == State::MemoryAllocated && self.storage_mode == BoardState::River
    }

    #[inline]
    fn is_raked(&self) -> bool {
        self.tree_config.rake_rate > 0.0 && self.tree_config.rake_cap > 0.0
    }

    #[inline]
    fn isomorphic_chances(&self, node: &Self::Node) -> &[u8] {
        if node.turn == NOT_DEALT {
            &self.isomorphism_ref_turn
        } else {
            &self.isomorphism_ref_river[node.turn as usize]
        }
    }

    #[inline]
    fn isomorphic_cards(&self,node:&Self::Node)->&[u8]{
        if node.turn==NOT_DEALT{&self.isomorphism_card_turn}else{&self.isomorphism_card_river[node.turn as usize&3]}
    }

    #[inline]
    fn isomorphic_swap(&self, node: &Self::Node, index: usize) -> &[Vec<(u16, u16)>; 2] {
        if node.turn == NOT_DEALT {
            &self.isomorphism_swap_turn[self.isomorphism_card_turn[index] as usize & 3]
        } else {
            &self.isomorphism_swap_river[node.turn as usize & 3]
                [self.isomorphism_card_river[node.turn as usize & 3][index] as usize & 3]
        }
    }

    fn cfv_capture_node_id(&self) -> u64 {
        self.turn_cfv_conversion
            .as_ref()
            .map_or(4, |matrix| matrix.node_id)
    }

    fn capture_turn_cfvs(&self, node: &Self::Node) -> bool {
        node.id() == self.cfv_capture_node_id()
            || (self.capture_all_turn_nodes && node.turn == NOT_DEALT && node.is_chance())
            || (self.capture_all_river_nodes && node.turn != NOT_DEALT && node.is_chance())
            || (node.is_chance()
                && node.turn != NOT_DEALT
                && self.capture_river_turns.contains(&node.turn))
    }

    fn turn_cfv_conversion(
        &self,
        node: &Self::Node,
    ) -> Option<&crate::pio_order::TurnCfvConversion> {
        if node.turn != NOT_DEALT {
            if self.is_auxiliary_turn(node.turn) {
                return None;
            }
            if self.tertiary_river_cfv_nodes.contains(&node.id()) {
                return self.tertiary_river_cfv_conversions.get(&node.turn);
            }
            if self.secondary_river_cfv_nodes.contains(&node.id()) {
                return self.secondary_river_cfv_conversions.get(&node.turn);
            }
            return self.river_cfv_conversions.get(&node.turn);
        }
        if self.tertiary_turn_cfv_nodes.contains(&node.id()) {
            return self.tertiary_turn_cfv_conversion.as_ref();
        }
        if self.secondary_turn_cfv_nodes.contains(&node.id()) {
            return self.secondary_turn_cfv_conversion.as_ref();
        }
        self.turn_cfv_conversion.as_ref().filter(|conversion| {
            node.is_chance()
                && node.turn == NOT_DEALT
                && (conversion.all_turn_nodes || conversion.node_id == node.id)
        })
    }

    fn begin_neural_river_batch(&self) -> bool {
        #[cfg(feature = "neural")]
        {
            let mut river_batches = 0;
            for conversion in self
                .river_cfv_conversions
                .values()
                .chain(self.secondary_river_cfv_conversions.values())
                .chain(self.tertiary_river_cfv_conversions.values())
            {
                if conversion.use_in_solving() && conversion.begin_neural_batch_collection() {
                    river_batches += 1;
                }
            }
            return river_batches > 0;
        }
        #[cfg(not(feature = "neural"))]
        false
    }

    fn requires_player_reach(&self) -> bool {
        self.turn_cfv_conversion
            .as_ref()
            .is_some_and(|conversion| conversion.uses_current_reaches())
            || self
                .secondary_turn_cfv_conversion
                .as_ref()
                .is_some_and(|conversion| conversion.uses_current_reaches())
            || self
                .tertiary_turn_cfv_conversion
                .as_ref()
                .is_some_and(|conversion| conversion.uses_current_reaches())
            || self
                .river_cfv_conversions
                .values()
                .any(|conversion| conversion.uses_current_reaches())
            || self
                .secondary_river_cfv_conversions
                .values()
                .any(|conversion| conversion.uses_current_reaches())
            || self
                .tertiary_river_cfv_conversions
                .values()
                .any(|conversion| conversion.uses_current_reaches())
    }

    fn begin_neural_turn_batch(&self) -> bool {
        #[cfg(feature = "neural")]
        {
            let primary = self.turn_cfv_conversion.as_ref().is_some_and(|conversion| {
                conversion.use_in_solving() && conversion.begin_neural_batch_collection()
            });
            let secondary = self
                .secondary_turn_cfv_conversion
                .as_ref()
                .is_some_and(|conversion| {
                    conversion.use_in_solving() && conversion.begin_neural_batch_collection()
                });
            let tertiary = self
                .tertiary_turn_cfv_conversion
                .as_ref()
                .is_some_and(|conversion| {
                    conversion.use_in_solving() && conversion.begin_neural_batch_collection()
                });
            return primary || secondary || tertiary;
        }
        #[cfg(not(feature = "neural"))]
        false
    }

    fn finish_neural_turn_batch(&self, player: usize) {
        #[cfg(feature = "neural")]
        {
            if let Some(modules) = &self.glued_river_modules {
                let conversions: Vec<_> = self
                    .river_cfv_conversions
                    .values()
                    .chain(self.secondary_river_cfv_conversions.values())
                    .chain(self.tertiary_river_cfv_conversions.values())
                    .filter(|conversion| conversion.neural_batch_is_collecting())
                    .collect();
                if !conversions.is_empty() {
                    crate::pio_order::finish_glued_neural_batches(
                        &conversions,
                        player,
                        &modules[player],
                    );
                    return;
                }
            }
            let mut finished = 0;
            for conversion in self
                .river_cfv_conversions
                .values()
                .chain(self.secondary_river_cfv_conversions.values())
                .chain(self.tertiary_river_cfv_conversions.values())
            {
                if conversion.neural_batch_is_collecting() {
                    conversion.finish_neural_batch(player);
                    finished += 1;
                }
            }
            if finished == 0 {
                let mut turn_finished = 0;
                for conversion in [
                    self.turn_cfv_conversion.as_ref(),
                    self.secondary_turn_cfv_conversion.as_ref(),
                    self.tertiary_turn_cfv_conversion.as_ref(),
                ]
                .into_iter()
                .flatten()
                {
                    if conversion.neural_batch_is_collecting() {
                        conversion.finish_neural_batch(player);
                        turn_finished += 1;
                    }
                }
                assert!(turn_finished > 0, "neural turn batch without conversion");
            }
        }
    }

    fn end_neural_turn_batch(&self) {
        #[cfg(feature = "neural")]
        {
            for conversion in self
                .river_cfv_conversions
                .values()
                .chain(self.secondary_river_cfv_conversions.values())
                .chain(self.tertiary_river_cfv_conversions.values())
            {
                if conversion.neural_batch_is_consuming() {
                    conversion.end_neural_batch();
                }
            }
            if let Some(conversion) = &self.turn_cfv_conversion {
                if conversion.neural_batch_is_consuming() {
                    conversion.end_neural_batch();
                }
            }
            if let Some(conversion) = &self.secondary_turn_cfv_conversion {
                if conversion.neural_batch_is_consuming() {
                    conversion.end_neural_batch();
                }
            }
            if let Some(conversion) = &self.tertiary_turn_cfv_conversion {
                if conversion.neural_batch_is_consuming() {
                    conversion.end_neural_batch();
                }
            }
        }
    }

    #[inline]
    fn locking_strategy(&self, node: &Self::Node) -> &[f32] {
        if !node.is_locked {
            &[]
        } else {
            let index = self.node_index(node);
            self.locking_strategy.get(&index).unwrap()
        }
    }

    #[inline]
    fn is_compression_enabled(&self) -> bool {
        self.is_compression_enabled
    }
}

impl PostFlopGame {
    /// Initial OOP/IP combination mass after exposing one additional public card.
    ///
    /// DeepStack-normalized river targets are scaled independently for each turn,
    /// so their denominator must exclude private hands containing that turn card.
    #[cfg(feature = "neural")]
    fn num_combinations_after_public_card(&self, card: Card) -> f64 {
        let card_mask = 1u64 << card;
        let mut combinations = 0.0;
        for (&(c1, c2), &w1) in self.private_cards[0]
            .iter()
            .zip(self.initial_weights[0].iter())
        {
            let oop_mask = (1u64 << c1) | (1u64 << c2);
            if oop_mask & card_mask != 0 {
                continue;
            }
            for (&(c3, c4), &w2) in self.private_cards[1]
                .iter()
                .zip(self.initial_weights[1].iter())
            {
                let ip_mask = (1u64 << c3) | (1u64 << c4);
                if ip_mask & card_mask == 0 && oop_mask & ip_mask == 0 {
                    combinations += w1 as f64 * w2 as f64;
                }
            }
        }
        combinations
    }

    #[cfg(feature = "neural")]
    fn turn_model_interface(
        model_path: &std::path::Path,
    ) -> Result<(&'static str, &'static str, usize, bool, bool, bool, bool), String> {
        let Some(directory) = model_path.parent() else {
            return Ok(("semantic", "none", 0, false, false, false, false));
        };
        let metadata_path = directory.join("model_interface.json");
        if !metadata_path.exists() {
            // Checkpoints exported before interface metadata were all two-input
            // semantic models.  Preserve that interface; never infer a third
            // tensor from a CLI flag.
            return Ok(("semantic", "none", 0, false, false, false, false));
        }
        let metadata = std::fs::read_to_string(&metadata_path).map_err(|e| e.to_string())?;
        if !metadata.contains("\"interface_version\": 1") {
            return Err(format!(
                "Unsupported turn-model interface: {}",
                metadata_path.display()
            ));
        }
        let context = if metadata.contains("\"context\": \"pot_fraction\"") {
            "pot_fraction"
        } else if metadata.contains("\"context\": \"semantic_v1\"") {
            "semantic"
        } else {
            return Err(format!(
                "Unknown turn-model context: {}",
                metadata_path.display()
            ));
        };
        let third = if metadata.contains("\"third_input\": \"current_reaches\"") {
            "current_reaches"
        } else if metadata.contains("\"third_input\": \"initial_ranges\"") {
            "initial_ranges"
        } else if metadata.contains("\"third_input\": \"none\"") {
            "none"
        } else {
            return Err(format!(
                "Unknown turn-model third input: {}",
                metadata_path.display()
            ));
        };
        let basis_marker = "\"basis_count\":";
        let basis_tail = metadata
            .split_once(basis_marker)
            .map(|(_, tail)| tail.trim_start())
            .ok_or_else(|| {
                format!(
                    "Missing turn-model basis count: {}",
                    metadata_path.display()
                )
            })?;
        let basis_digits: String = basis_tail
            .chars()
            .take_while(|ch| ch.is_ascii_digit())
            .collect();
        let basis_count = basis_digits.parse::<usize>().map_err(|_| {
            format!(
                "Invalid turn-model basis count: {}",
                metadata_path.display()
            )
        })?;
        if basis_count > 49 {
            return Err(format!(
                "Invalid turn-model basis count: {}",
                metadata_path.display()
            ));
        }
        let deepstack_normalized =
            metadata.contains("\"output\": \"deepstack_normalized_turn_sum\"");
        let uses_current_reaches = !metadata.contains("\"uses_current_reaches\": false");
        let sum_cfvs = metadata.contains("\"sum_cfvs\": true");
        let supported_hand_rms =
            metadata.contains("\"rms_denominator\": \"supported_hands_times_basis_count\"");
        Ok((
            context,
            third,
            basis_count,
            deepstack_normalized,
            uses_current_reaches,
            sum_cfvs,
            supported_hand_rms,
        ))
    }

    /// Topology-independent numeric contexts for flop-to-turn chance nodes.
    ///
    /// The vector contains pot/stack/investment state followed by up to eight
    /// ordered flop actions.  It intentionally contains no node ID and no
    /// acting/target-player field: the node is a chance deal and the predicted
    /// CFV player is selected by the model head.
    pub fn turn_cfv_semantic_contexts(&self) -> Result<Vec<(u64, Vec<f32>)>, String> {
        const MAX_ACTIONS: usize = 8;
        const ACTION_WIDTH: usize = 11;

        fn action_kind(action: Action) -> Option<usize> {
            match action {
                Action::Check => Some(0),
                Action::Bet(_) => Some(1),
                Action::Call => Some(2),
                Action::Raise(_) => Some(3),
                Action::Fold => Some(4),
                Action::AllIn(_) => Some(5),
                _ => None,
            }
        }

        fn walk(
            game: &PostFlopGame,
            node_index: usize,
            invested: [i32; 2],
            history: &mut Vec<Vec<f32>>,
            raises: usize,
            allin: [bool; 2],
            output: &mut Vec<(u64, Vec<f32>)>,
        ) -> Result<(), String> {
            let node = game.node_arena[node_index].lock();
            if node.is_chance() && node.turn == NOT_DEALT {
                if history.len() > MAX_ACTIONS {
                    return Err(format!(
                        "flop action history has {} actions; semantic NN supports at most {MAX_ACTIONS}",
                        history.len()
                    ));
                }
                let stack = game.tree_config.effective_stack as f32;
                let pot = game.tree_config.starting_pot + invested[0] + invested[1];
                let mut context = Vec::with_capacity(7 + MAX_ACTIONS * ACTION_WIDTH);
                context.extend_from_slice(&[
                    pot as f32 / stack,
                    (game.tree_config.effective_stack - invested[0]) as f32 / stack,
                    (game.tree_config.effective_stack - invested[1]) as f32 / stack,
                    invested[0] as f32 / stack,
                    invested[1] as f32 / stack,
                    raises as f32 / MAX_ACTIONS as f32,
                    f32::from(allin[0] || allin[1]),
                ]);
                for action in history.iter() {
                    context.extend_from_slice(action);
                }
                context.resize(7 + MAX_ACTIONS * ACTION_WIDTH, 0.0);
                output.push((node.id(), context));
                return Ok(());
            }
            if node.is_terminal() || node.is_chance() {
                return Ok(());
            }
            let actor = node.player();
            let parent_prev_action = node.prev_action;
            let children: Vec<_> = (0..node.num_actions())
                .map(|index| {
                    let child = node.play(index);
                    (child.id() as usize, child.prev_action)
                })
                .collect();
            drop(node);
            for (child_index, action) in children {
                let Some(kind) = action_kind(action) else {
                    continue;
                };
                let pot_before = game.tree_config.starting_pot + invested[0] + invested[1];
                let remaining_before = game.tree_config.effective_stack - invested[actor];
                let amount = match action {
                    Action::Bet(value) | Action::Raise(value) | Action::AllIn(value) => value,
                    Action::Call => (invested[actor ^ 1] - invested[actor]).max(0),
                    _ => 0,
                };
                let mut token = vec![0.0; ACTION_WIDTH];
                token[actor] = 1.0;
                token[2 + kind] = 1.0;
                token[8] = amount as f32 / pot_before.max(1) as f32;
                token[9] = amount as f32 / remaining_before.max(1) as f32;
                token[10] = 1.0;
                let mut next_invested = invested;
                match action {
                    Action::Call => next_invested[actor] = next_invested[actor ^ 1],
                    Action::Bet(value) | Action::Raise(value) | Action::AllIn(value) => {
                        let previous = match parent_prev_action {
                            Action::Bet(value) | Action::Raise(value) | Action::AllIn(value) => {
                                value
                            }
                            _ => 0,
                        };
                        let to_call = next_invested[actor ^ 1] - next_invested[actor];
                        next_invested[actor] += value - previous + to_call;
                    }
                    _ => {}
                }
                let mut next_allin = allin;
                next_allin[actor] |= matches!(action, Action::AllIn(_));
                history.push(token);
                walk(
                    game,
                    child_index,
                    next_invested,
                    history,
                    raises
                        + usize::from(matches!(
                            action,
                            Action::Bet(_) | Action::Raise(_) | Action::AllIn(_)
                        )),
                    next_allin,
                    output,
                )?;
                history.pop();
            }
            Ok(())
        }

        if self.tree_config.initial_state != BoardState::Flop {
            return Err("semantic turn contexts require a flop-starting tree".into());
        }
        let mut output = Vec::new();
        walk(
            self,
            0,
            [0, 0],
            &mut Vec::new(),
            0,
            [false, false],
            &mut output,
        )?;
        output.sort_unstable_by_key(|row| row.0);
        Ok(output)
    }

    /// Loads two player-specific TorchScript turn-sum models exported by
    /// `python_scripts/export_turn_sum_torchscript.py`.
    #[cfg(feature = "neural")]
    pub fn load_neural_turn_cfv_conversion(
        &mut self,
        trained_flop: [Card; 3],
        basis_cards: &[Card],
        pio_order_path: impl AsRef<std::path::Path>,
        oop_model_path: impl AsRef<std::path::Path>,
        ip_model_path: impl AsRef<std::path::Path>,
    ) -> Result<(), String> {
        use crate::pio_order::*;
        if self.state <= State::Uninitialized {
            return Err("Conversion requires an initialized game".into());
        }
        let mut expected = trained_flop;
        let mut actual = self.card_config.flop;
        expected.sort_unstable();
        actual.sort_unstable();
        if expected != actual {
            return Err("NN conversion requires its trained flop".into());
        }
        let mut nodes = self.turn_cfv_capture_contexts();
        nodes.sort_unstable_by_key(|row| row.0);
        let node_ids: Vec<_> = nodes.iter().map(|row| row.0).collect();
        if self.tree_config.effective_stack != 900 || node_ids != [4, 8, 11, 14, 16, 18, 21, 24, 26]
        {
            return Err(format!(
                "NN requires the trained stack-900 nine-node tree topology; got node IDs {node_ids:?}"
            ));
        }
        let source_id = nodes[0].0;
        let source = self.node_arena[source_id as usize].lock();
        if !(self.neural_turn_action_count(&source) >= basis_cards.len()
            && matches!(self.neural_turn_action_count(&source), 1 | 2 | 4 | 6 | 13 | 49))
            || !self.isomorphic_chances(&source).is_empty()
        {
            return Err("NN requires a containing 1, 2, 4, 6, 13, or 49-turn physical basis with isomorphism disabled".into());
        }
        let action_count = self.neural_turn_action_count(&source);
        let mut basis_actions = Vec::with_capacity(basis_cards.len());
        for &card in basis_cards {
            let action = (0..source.num_actions())
                .find(|&index| source.play(index).prev_action == Action::Chance(card))
                .ok_or("Basis card is not a legal physical turn")?;
            if basis_actions.contains(&action) {
                return Err("Duplicate basis turn".into());
            }
            basis_actions.push(action);
        }
        drop(source);
        let order = load_pio_hand_order(pio_order_path)?;
        let mappings = [
            local_to_pio(&order, &self.private_cards[0])?,
            local_to_pio(&order, &self.private_cards[1])?,
        ];
        let mut contexts = std::collections::HashMap::new();
        for (index, &(id, pot, stack)) in nodes.iter().enumerate() {
            let mut context = vec![0.0; 11];
            context[index] = 1.0;
            context[9] = pot as f32 / 200.0;
            context[10] = stack as f32 / 900.0;
            contexts.insert(id, context);
        }
        let device = nn_device();
        let modules = Some([
            std::sync::Mutex::new(
                load_nn_module(oop_model_path, device).map_err(|e| e.to_string())?,
            ),
            std::sync::Mutex::new(
                load_nn_module(ip_model_path, device).map_err(|e| e.to_string())?,
            ),
        ]);
        self.turn_cfv_conversion = Some(TurnCfvConversion {
            node_id: source_id,
            action_count,
            use_in_solving: false,
            all_turn_nodes: false,
            basis_actions,
            weights: [Vec::new(), Vec::new()],
            bucket_model: None,
            neural_model: Some(NeuralTurnCfvModel {
                modules,
                sparse_models: None,
                direct_contexts: NeuralTurnCfvModel::build_direct_contexts(&contexts),
                contexts,
                mappings,
                range_features: None,
                current_reaches: false,
                sum_cfvs: false,
                iteration_scale: None,
                deepstack_root_combinations: None,
                deepstack_chance_count: 45.0,
                pots: std::collections::HashMap::new(),
                device,
                batch_state: Default::default(),
            }),
        });
        Ok(())
    }

    /// Loads a topology-independent turn-sum NN whose context follows
    /// [`PostFlopGame::turn_cfv_semantic_contexts`].
    #[cfg(feature = "neural")]
    pub fn load_semantic_neural_turn_cfv_conversion(
        &mut self,
        trained_flop: [Card; 3],
        basis_cards: &[Card],
        pio_order_path: impl AsRef<std::path::Path>,
        oop_model_path: impl AsRef<std::path::Path>,
        ip_model_path: impl AsRef<std::path::Path>,
    ) -> Result<(), String> {
        use crate::pio_order::*;
        if self.state <= State::Uninitialized || self.state == State::Solved {
            return Err("Conversion requires an initialized, unsolved game".into());
        }
        let mut expected = trained_flop;
        let mut actual = self.card_config.flop;
        expected.sort_unstable();
        actual.sort_unstable();
        if expected != actual {
            return Err("NN conversion requires its trained flop".into());
        }
        let oop_interface = Self::turn_model_interface(oop_model_path.as_ref())?;
        let ip_interface = Self::turn_model_interface(ip_model_path.as_ref())?;
        if oop_interface != ip_interface {
            return Err("OOP/IP turn-model interfaces differ".into());
        }
        let torchscript_options = |path: &std::path::Path| -> Result<(bool, bool), String> {
            let metadata = path.parent().unwrap_or_else(|| std::path::Path::new("."))
                .join("model_interface.json");
            if !metadata.exists() { return Ok((false, false)); }
            Ok(turn_torchscript_options(&std::fs::read_to_string(metadata).map_err(|e| e.to_string())?))
        };
        let turn_options = torchscript_options(oop_model_path.as_ref())?;
        if turn_options != torchscript_options(ip_model_path.as_ref())? {
            return Err("OOP/IP turn TorchScript support interfaces differ".into());
        }
        let use_sparse_turn = nn_sparse_cpu() && !turn_options.0;
        let uses_initial_support = turn_options.1;
        if uses_initial_support && (use_sparse_turn || oop_interface.1 != "current_reaches" || !oop_interface.4) {
            return Err("Four-input turn models require TorchScript with current reaches and explicit initial ranges".into());
        }
        if oop_interface.2 != 0 && oop_interface.2 != basis_cards.len() {
            return Err(format!(
                "Turn-model basis mismatch: checkpoint requires {}, tree supplied {}",
                oop_interface.2,
                basis_cards.len()
            ));
        }
        if oop_interface.6 && !use_sparse_turn && !uses_initial_support {
            return Err(
                "Supported-hand RMS turn checkpoints currently require sparse CPU inference; the TorchScript wrapper has no initial-support-mask input"
                    .into(),
            );
        }
        let pot_fraction_only = oop_interface.0 == "pot_fraction";
        let capture_contexts = self.turn_cfv_capture_contexts();
        let pots: std::collections::HashMap<_, _> = capture_contexts
            .iter()
            .map(|&(id, pot, _)| (id, pot as f32))
            .collect();
        let contexts: std::collections::HashMap<_, _> = if pot_fraction_only {
            capture_contexts
                .into_iter()
                .map(|(id, pot, stack)| {
                    let denominator = pot + stack;
                    (
                        id,
                        vec![if denominator == 0 {
                            0.0
                        } else {
                            pot as f32 / denominator as f32
                        }],
                    )
                })
                .collect()
        } else {
            self.turn_cfv_semantic_contexts()?.into_iter().collect()
        };
        let source_id = *contexts
            .keys()
            .min()
            .ok_or("Tree has no flop-to-turn chance nodes")?;
        let source = self.node_arena[source_id as usize].lock();
        if !(self.neural_turn_action_count(&source) >= basis_cards.len()
            && matches!(self.neural_turn_action_count(&source), 1 | 2 | 4 | 6 | 13 | 49))
            || !self.isomorphic_chances(&source).is_empty()
        {
            return Err("NN requires a containing 1, 2, 4, 6, 13, or 49-turn physical basis with isomorphism disabled".into());
        }
        let action_count = self.neural_turn_action_count(&source);
        let mut basis_actions = Vec::with_capacity(basis_cards.len());
        for &card in basis_cards {
            let action = (0..source.num_actions())
                .find(|&index| source.play(index).prev_action == Action::Chance(card))
                .ok_or("Basis card is not a legal physical turn")?;
            if basis_actions.contains(&action) {
                return Err("Duplicate basis turn".into());
            }
            basis_actions.push(action);
        }
        drop(source);
        let order = load_pio_hand_order(pio_order_path)?;
        let mappings = [
            local_to_pio(&order, &self.private_cards[0])?,
            local_to_pio(&order, &self.private_cards[1])?,
        ];
        let device = nn_device();
        let (modules, sparse_models) = if nn_zero_mode() {
            (None, None)
        } else if use_sparse_turn {
            if oop_interface.0 != "pot_fraction"
                || !matches!(oop_interface.1, "none" | "current_reaches")
                || oop_interface.5
            {
                return Err("sparse CPU inference currently requires a direct, unsummed pot-fraction turn model, optionally with current reaches or DeepStack-normalized output".into());
            }
            let cpu_path = |path: &std::path::Path| {
                let stem = path
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .unwrap_or("model");
                path.with_file_name(format!("{stem}.cpu.bin"))
            };
            let mut sparse = [
                crate::sparse_nn::SparseCpuModel::load(cpu_path(oop_model_path.as_ref()))?,
                crate::sparse_nn::SparseCpuModel::load(cpu_path(ip_model_path.as_ref()))?,
            ];
            for model in &mut sparse {
                model.set_supported_hand_rms(oop_interface.6);
            }
            if sparse
                .iter()
                .any(|model| model.basis_count() != basis_cards.len())
            {
                return Err(
                    "sparse CPU checkpoint basis count does not match supplied cards".into(),
                );
            }
            if sparse.iter().any(|model| {
                model
                    .basis_cards()
                    .is_some_and(|cards| cards != basis_cards)
            }) {
                return Err(
                    "sparse CPU turn checkpoint card order does not match supplied cards".into(),
                );
            }
            if std::env::var_os("SPARSE_NN_DISABLE_RANGE_SPECIALIZATION").is_none() {
                for player in 0..2 {
                    let (inputs, outputs) =
                        sparse[player].specialize_for_ranges(&mappings, player)?;
                    println!(
                        "SPARSE_RANGE_SPECIALIZATION street=turn player={} input_rows={} output_rows={}",
                        if player == 0 { "OOP" } else { "IP" },
                        inputs,
                        outputs
                    );
                }
            }
            (None, Some(sparse))
        } else {
            (
                Some([
                    std::sync::Mutex::new(
                        load_nn_module(oop_model_path, device).map_err(|e| e.to_string())?,
                    ),
                    std::sync::Mutex::new(
                        load_nn_module(ip_model_path, device).map_err(|e| e.to_string())?,
                    ),
                ]),
                None,
            )
        };
        let range_features = if uses_initial_support {
            Some(canonical_range_features(&mappings, &self.initial_weights)?)
        } else { None };
        if !use_sparse_turn && !nn_zero_mode() {
            println!("TURN_TORCHSCRIPT_BACKEND device={device:?} initial_support={uses_initial_support}");
        }
        self.turn_cfv_conversion = Some(TurnCfvConversion {
            node_id: source_id,
            action_count,
            use_in_solving: false,
            all_turn_nodes: false,
            basis_actions,
            weights: [Vec::new(), Vec::new()],
            bucket_model: None,
            neural_model: Some(NeuralTurnCfvModel {
                modules,
                sparse_models,
                direct_contexts: NeuralTurnCfvModel::build_direct_contexts(&contexts),
                contexts,
                mappings,
                range_features,
                current_reaches: oop_interface.1 == "current_reaches" && oop_interface.4,
                sum_cfvs: oop_interface.5,
                iteration_scale: None,
                deepstack_root_combinations: oop_interface.3.then_some(self.num_combinations),
                deepstack_chance_count: 45.0,
                pots,
                device,
                batch_state: Default::default(),
            }),
        });
        Ok(())
    }

    /// Loads a semantic turn-sum model whose third input is the two players'
    /// raw initial range weights in canonical `[1, 1326, 2]` hand order.
    #[cfg(feature = "neural")]
    pub fn load_semantic_neural_turn_cfv_conversion_with_ranges(
        &mut self,
        trained_flop: [Card; 3],
        basis_cards: &[Card],
        pio_order_path: impl AsRef<std::path::Path>,
        oop_model_path: impl AsRef<std::path::Path>,
        ip_model_path: impl AsRef<std::path::Path>,
    ) -> Result<(), String> {
        use crate::pio_order::canonical_range_features;
        let interface = Self::turn_model_interface(oop_model_path.as_ref())?;
        self.load_semantic_neural_turn_cfv_conversion(
            trained_flop,
            basis_cards,
            pio_order_path,
            oop_model_path,
            ip_model_path,
        )?;
        if interface.1 != "initial_ranges" {
            // Two-input legacy models and current-reach models must not receive
            // the static initial-range tensor merely because --range-inputs was
            // present on the command line.
            return Ok(());
        }
        let mappings = self
            .turn_cfv_conversion
            .as_ref()
            .and_then(|conversion| conversion.neural_model.as_ref())
            .ok_or("Range-aware NN conversion was not installed")?
            .mappings
            .clone();
        let range_features = canonical_range_features(&mappings, &self.initial_weights)?;
        self.turn_cfv_conversion
            .as_mut()
            .and_then(|conversion| conversion.neural_model.as_mut())
            .ok_or("Range-aware NN conversion was not installed")?
            .range_features = Some(range_features);
        Ok(())
    }

    /// Installs two semantic turn models and routes `secondary_nodes` to the
    /// secondary model. Intended for experiments with node-dependent basis sizes.
    #[cfg(feature = "neural")]
    pub fn load_hybrid_semantic_neural_turn_cfv_conversions_with_ranges(
        &mut self,
        trained_flop: [Card; 3],
        primary_basis_cards: &[Card],
        primary_model_dir: impl AsRef<std::path::Path>,
        secondary_basis_cards: &[Card],
        secondary_model_dir: impl AsRef<std::path::Path>,
        secondary_nodes: &[u64],
        pio_order_path: impl AsRef<std::path::Path>,
    ) -> Result<(), String> {
        if secondary_nodes.is_empty() {
            return Err("Hybrid turn routing requires secondary nodes".into());
        }
        let available: std::collections::HashSet<_> = self
            .turn_cfv_capture_contexts()
            .into_iter()
            .map(|row| row.0)
            .collect();
        if secondary_nodes.iter().any(|node| !available.contains(node)) {
            return Err("Hybrid turn routing contains an unknown node".into());
        }
        let primary_model_dir = primary_model_dir.as_ref();
        let secondary_model_dir = secondary_model_dir.as_ref();
        let pio_order_path = pio_order_path.as_ref();
        self.load_semantic_neural_turn_cfv_conversion_with_ranges(
            trained_flop,
            primary_basis_cards,
            pio_order_path,
            primary_model_dir.join("model_oop.pt"),
            primary_model_dir.join("model_ip.pt"),
        )?;
        let primary = self.turn_cfv_conversion.take().unwrap();
        self.load_semantic_neural_turn_cfv_conversion_with_ranges(
            trained_flop,
            secondary_basis_cards,
            pio_order_path,
            secondary_model_dir.join("model_oop.pt"),
            secondary_model_dir.join("model_ip.pt"),
        )?;
        self.secondary_turn_cfv_conversion = self.turn_cfv_conversion.take();
        self.turn_cfv_conversion = Some(primary);
        self.secondary_turn_cfv_nodes = secondary_nodes.to_vec();
        Ok(())
    }

    /// Adds a third turn-model routing tier after the primary/secondary hybrid
    /// has been installed.
    #[cfg(feature = "neural")]
    pub fn load_tertiary_semantic_neural_turn_cfv_conversion_with_ranges(
        &mut self,
        trained_flop: [Card; 3],
        basis_cards: &[Card],
        model_dir: impl AsRef<std::path::Path>,
        nodes: &[u64],
        pio_order_path: impl AsRef<std::path::Path>,
    ) -> Result<(), String> {
        let available: std::collections::HashSet<_> = self
            .turn_cfv_capture_contexts()
            .into_iter()
            .map(|row| row.0)
            .collect();
        if nodes.is_empty() || nodes.iter().any(|node| !available.contains(node)) {
            return Err("Tertiary turn routing requires valid flop frontier nodes".into());
        }
        let primary = self.turn_cfv_conversion.take();
        let secondary = self.secondary_turn_cfv_conversion.take();
        let model_dir = model_dir.as_ref();
        self.load_semantic_neural_turn_cfv_conversion_with_ranges(
            trained_flop,
            basis_cards,
            pio_order_path,
            model_dir.join("model_oop.pt"),
            model_dir.join("model_ip.pt"),
        )?;
        self.tertiary_turn_cfv_conversion = self.turn_cfv_conversion.take();
        self.turn_cfv_conversion = primary;
        self.secondary_turn_cfv_conversion = secondary;
        self.tertiary_turn_cfv_nodes = nodes.to_vec();
        Ok(())
    }

    /// Enables both sides of an installed hybrid turn-model routing table.
    pub fn set_hybrid_all_turn_sums(&mut self, enabled: bool) -> Result<usize, String> {
        if self.secondary_turn_cfv_conversion.is_none() {
            return Err("Load hybrid turn conversions first".into());
        }
        let count = self.turn_cfv_capture_contexts().len();
        for conversion in [
            self.turn_cfv_conversion.as_mut(),
            self.secondary_turn_cfv_conversion.as_mut(),
            self.tertiary_turn_cfv_conversion.as_mut(),
        ]
        .into_iter()
        .flatten()
        {
            conversion.all_turn_nodes = enabled;
            conversion.use_in_solving = enabled;
        }
        Ok(count)
    }

    /// Loads one direct missing-river-sum model for each selected turn.
    /// Legacy models use `model_{oop,ip}.pt`; sparse CPU models use
    /// `model_{oop,ip}.cpu.bin` in the same per-turn directories.
    #[cfg(feature = "neural")]
    pub fn load_neural_river_cfv_conversions(
        &mut self,
        trained_flop: [Card; 3],
        turns: &[Card],
        basis_rivers: &[Card],
        pio_order_path: impl AsRef<std::path::Path>,
        model_root: impl AsRef<std::path::Path>,
    ) -> Result<usize, String> {
        use crate::pio_order::*;
        if self.state <= State::Uninitialized || self.state == State::Solved {
            return Err("River conversion requires an initialized, unsolved game".into());
        }
        let mut expected = trained_flop;
        let mut actual = self.card_config.flop;
        expected.sort_unstable();
        actual.sort_unstable();
        if expected != actual || turns.is_empty() {
            return Err("River NN requires its trained flop and nonempty turns".into());
        }
        let order = load_pio_hand_order(pio_order_path)?;
        let mappings = [
            local_to_pio(&order, &self.private_cards[0])?,
            local_to_pio(&order, &self.private_cards[1])?,
        ];
        // Exact turn-all-in matrix leaves have no physical river children and
        // must not be registered as neural river frontiers.
        let all_contexts: Vec<_> = self
            .river_cfv_capture_contexts()
            .into_iter()
            .filter(|&(id, _, _, _)| self.node_arena[id as usize].lock().num_actions() != 0)
            .collect();
        let pot_fraction_only = model_root.as_ref().join("pot_fraction_only.json").exists();
        // The deployed classic per-hand bundle has a manifest and its scripted
        // modules accept only (CFVs, context).  New aggregate bundles consume
        // actual current-node reaches as a third tensor.
        let classic_per_hand =
            model_root.as_ref().join("manifest.json").exists() && !pot_fraction_only;
        let river_interface = model_root.as_ref().join("model_interface.json");
        let river_interface = if river_interface.exists() {
            std::fs::read_to_string(&river_interface).map_err(|error| error.to_string())?
        } else {
            String::new()
        };
        let sum_cfvs = river_interface.contains("\"sum_cfvs_before_transfer\": true");
        // Universal bucket adapters are TorchScript modules with a dedicated
        // four-input interface.  They may coexist with sparse CPU turn models
        // in the same solve, so river backend selection cannot rely solely on
        // the process-wide sparse flag.
        let bucketed_model = river_interface.contains("\"bucket_model\": \"universal_qsjh2h_500\"");
        let deepstack_normalized =
            river_interface.contains("\"output\": \"deepstack_normalized_river_sum\"");
        let zero_river_basis = river_interface.contains("\"basis_count\": 0");
        let supported_hand_rms =
            river_interface.contains("\"rms_denominator\": \"supported_hands_times_basis_count\"");
        if supported_hand_rms && !nn_sparse_cpu() {
            return Err(
                "Supported-hand RMS river checkpoints currently require sparse CPU inference; the TorchScript wrapper has no initial-support-mask input"
                    .into(),
            );
        }
        let omits_current_reaches = river_interface.contains("\"uses_current_reaches\": false");
        // Bucketed river models form CFV bucket means over the player's
        // *initial* supported hands, while their range-mass channels use the
        // live reaches.  Keep this opt-in metadata marker separate from the
        // older three-input current-reach interface.
        let uses_initial_ranges = river_interface.contains("\"uses_initial_ranges\": true");
        let uses_current_reaches = if sum_cfvs || omits_current_reaches {
            false
        } else {
            !classic_per_hand
        };
        if uses_initial_ranges && !uses_current_reaches {
            return Err("River model requests initial ranges but not current reaches".into());
        }
        let initial_range_features = if uses_initial_ranges {
            Some(canonical_range_features(&mappings, &self.initial_weights)?)
        } else {
            None
        };
        let mut conversions = BTreeMap::new();
        for &turn in turns {
            if conversions.contains_key(&turn) {
                return Err("Duplicate river-model turn".into());
            }
            let mut nodes: Vec<_> = all_contexts
                .iter()
                .copied()
                .filter(|row| row.1 == turn)
                .collect();
            nodes.sort_unstable_by_key(|row| row.0);
            if nodes.is_empty() {
                return Err(format!(
                    "River NN found no chance nodes below turn {}",
                    card_to_string(turn)?
                ));
            }
            let source_id = nodes[0].0;
            let source = self.node_arena[source_id as usize].lock();
            if !matches!(source.num_actions(), 1 | 2 | 4 | 6 | 13 | 48)
                || !self.isomorphic_chances(&source).is_empty()
            {
                return Err(
                    "River NN requires 1, 2, 4, 6, 13, or 48 physical rivers with isomorphism disabled"
                        .into(),
                );
            }
            let action_count = source.num_actions();
            let basis_actions =
                if matches!(action_count, 1 | 2 | 4 | 6 | 13) && basis_rivers.is_empty() {
                    // A compact river tree already contains exactly the trained
                    // physical rivers, in canonical action order.
                    (0..action_count).collect()
                } else {
                    if basis_rivers.is_empty() && !zero_river_basis {
                        return Err("Full river tree requires a nonempty river basis".into());
                    }
                    let mut actions = Vec::with_capacity(basis_rivers.len());
                    for &card in basis_rivers {
                        let action = (0..source.num_actions())
                            .find(|&index| source.play(index).prev_action == Action::Chance(card))
                            .ok_or_else(|| {
                                format!(
                                    "River basis card {} is not legal after turn {}",
                                    card_to_string(card).unwrap_or_else(|_| format!("#{card}")),
                                    card_to_string(turn).unwrap_or_else(|_| format!("#{turn}")),
                                )
                            })?;
                        actions.push(action);
                    }
                    actions
                };
            let physical_basis_cards: Vec<_> = basis_actions
                .iter()
                .map(|&action| match source.play(action).prev_action {
                    Action::Chance(card) => card,
                    _ => unreachable!("river basis action is not a chance card"),
                })
                .collect();
            drop(source);
            let mut contexts = std::collections::HashMap::new();
            for &(id, _, pot, stack) in &nodes {
                let context = if pot_fraction_only {
                    let denominator = pot + stack;
                    vec![if denominator == 0 {
                        0.0
                    } else {
                        pot as f32 / denominator as f32
                    }]
                } else {
                    let mut legacy = vec![0.0; 95 + 2];
                    let stack_scale = (stack as f32).max(1.0);
                    legacy[0] = pot as f32 / stack_scale;
                    legacy[1] = 1.0;
                    legacy[2] = (turn / 4) as f32 / 12.0;
                    legacy[3] = (turn % 4) as f32 / 3.0;
                    legacy[95] = pot as f32 / 200.0;
                    legacy[96] = stack as f32 / 900.0;
                    legacy
                };
                contexts.insert(id, context);
            }
            let directory = model_root.as_ref().join(card_to_string(turn)?);
            let device = nn_device();
            let (modules, sparse_models) = if nn_zero_mode() {
                (None, None)
            } else if river_uses_sparse_backend(
                nn_sparse_cpu(), &river_interface, bucketed_model,
                directory.join("model_oop.cpu.bin").is_file(),
            ) {
                if sum_cfvs {
                    return Err(
                        "sparse CPU river inference requires direct, unsummed CFV inputs".into(),
                    );
                }
                let mut sparse = [
                    crate::sparse_nn::SparseCpuModel::load(directory.join("model_oop.cpu.bin"))?,
                    crate::sparse_nn::SparseCpuModel::load(directory.join("model_ip.cpu.bin"))?,
                ];
                for model in &mut sparse {
                    model.set_supported_hand_rms(supported_hand_rms);
                }
                if sparse
                    .iter()
                    .any(|model| model.basis_count() != basis_actions.len())
                {
                    return Err(format!(
                        "sparse CPU river checkpoint basis count does not match tree below turn {}",
                        card_to_string(turn)?
                    ));
                }
                if sparse.iter().any(|model| {
                    model
                        .basis_cards()
                        .is_some_and(|cards| cards != physical_basis_cards.as_slice())
                }) {
                    return Err(format!(
                        "sparse CPU river checkpoint card order does not match tree below turn {}",
                        card_to_string(turn)?
                    ));
                }
                if bucketed_model || std::env::var_os("SPARSE_NN_DISABLE_RANGE_SPECIALIZATION").is_none() {
                    for player in 0..2 {
                        let (inputs, outputs) =
                            sparse[player].specialize_for_ranges(&mappings, player)?;
                        println!(
                            "SPARSE_RANGE_SPECIALIZATION street=river turn={} player={} input_rows={} output_rows={}",
                            card_to_string(turn)?,
                            if player == 0 { "OOP" } else { "IP" },
                            inputs,
                            outputs
                        );
                    }
                }
                (None, Some(sparse))
            } else {
                (
                    Some([
                        std::sync::Mutex::new(
                            load_nn_module(directory.join("model_oop.pt"), device)
                                .map_err(|e| e.to_string())?,
                        ),
                        std::sync::Mutex::new(
                            load_nn_module(directory.join("model_ip.pt"), device)
                                .map_err(|e| e.to_string())?,
                        ),
                    ]),
                    None,
                )
            };
            let deepstack_root_combinations = if deepstack_normalized {
                let combinations = self.num_combinations_after_public_card(turn);
                if combinations == 0.0 {
                    return Err(format!(
                        "No compatible initial combinations remain after turn {}",
                        card_to_string(turn)?
                    ));
                }
                println!(
                    "DEEPSTACK_RIVER_NORMALIZATION turn={} root_combinations={combinations:.9} chance_count=44",
                    card_to_string(turn)?
                );
                Some(combinations)
            } else {
                None
            };
            conversions.insert(
                turn,
                TurnCfvConversion {
                    node_id: source_id,
                    action_count,
                    use_in_solving: false,
                    all_turn_nodes: false,
                    basis_actions,
                    weights: [Vec::new(), Vec::new()],
                    bucket_model: None,
                    neural_model: Some(NeuralTurnCfvModel {
                        modules,
                        sparse_models,
                        direct_contexts: NeuralTurnCfvModel::build_direct_contexts(&contexts),
                        contexts,
                        mappings: mappings.clone(),
                        range_features: initial_range_features.clone(),
                        current_reaches: uses_current_reaches,
                        sum_cfvs,
                        iteration_scale: if pot_fraction_only { None } else { Some(50.0) },
                        deepstack_root_combinations,
                        deepstack_chance_count: 44.0,
                        pots: if deepstack_normalized {
                            nodes
                                .iter()
                                .map(|&(id, _, pot, _)| (id, pot as f32))
                                .collect()
                        } else {
                            std::collections::HashMap::new()
                        },
                        device,
                        batch_state: Default::default(),
                    }),
                },
            );
        }
        let count = conversions.len();
        self.river_cfv_conversions = conversions;
        if nn_glued_rivers() {
            let prefix = if crate::pio_order::nn_grouped_rivers() {
                "grouped_model"
            } else {
                "glued_model"
            };
            self.glued_river_modules = Some([
                std::sync::Mutex::new(
                    load_nn_module(
                        model_root.as_ref().join(format!("{prefix}_oop.pt")),
                        nn_device(),
                    )
                    .map_err(|error| error.to_string())?,
                ),
                std::sync::Mutex::new(
                    load_nn_module(
                        model_root.as_ref().join(format!("{prefix}_ip.pt")),
                        nn_device(),
                    )
                    .map_err(|error| error.to_string())?,
                ),
            ]);
        }
        Ok(count)
    }

    /// Loads two river-model bundles and routes the complete river frontier
    /// below `secondary_flop_nodes` to the secondary bundle. The selected IDs
    /// must be physical flop-to-turn chance nodes.
    #[cfg(feature = "neural")]
    pub fn load_hybrid_neural_river_cfv_conversions(
        &mut self,
        trained_flop: [Card; 3],
        primary_turns: &[Card],
        primary_rivers: &[Card],
        primary_model_root: impl AsRef<std::path::Path>,
        secondary_turns: &[Card],
        secondary_rivers: &[Card],
        secondary_model_root: impl AsRef<std::path::Path>,
        secondary_flop_nodes: &[u64],
        pio_order_path: impl AsRef<std::path::Path>,
    ) -> Result<(usize, usize), String> {
        let available: std::collections::HashSet<_> = self
            .turn_cfv_capture_contexts()
            .into_iter()
            .map(|row| row.0)
            .collect();
        if secondary_flop_nodes.is_empty()
            || secondary_flop_nodes
                .iter()
                .any(|node| !available.contains(node))
        {
            return Err("Hybrid river routing requires valid flop frontier nodes".into());
        }

        let pio_order_path = pio_order_path.as_ref();
        let primary_count = self.load_neural_river_cfv_conversions(
            trained_flop,
            primary_turns,
            primary_rivers,
            pio_order_path,
            primary_model_root,
        )?;
        let primary = std::mem::take(&mut self.river_cfv_conversions);
        let secondary_count = self.load_neural_river_cfv_conversions(
            trained_flop,
            secondary_turns,
            secondary_rivers,
            pio_order_path,
            secondary_model_root,
        )?;
        let secondary = std::mem::take(&mut self.river_cfv_conversions);

        fn collect_river_frontiers(game: &PostFlopGame, node_id: u64, output: &mut Vec<u64>) {
            let node = game.node_arena[node_id as usize].lock();
            if node.is_terminal() {
                return;
            }
            if node.is_chance() && node.turn != NOT_DEALT {
                output.push(node.id());
                return;
            }
            let children: Vec<_> = (0..node.num_actions())
                .map(|action| node.play(action).id())
                .collect();
            drop(node);
            for child in children {
                collect_river_frontiers(game, child, output);
            }
        }

        let mut secondary_river_nodes = Vec::new();
        for &node in secondary_flop_nodes {
            collect_river_frontiers(self, node, &mut secondary_river_nodes);
        }
        secondary_river_nodes.sort_unstable();
        secondary_river_nodes.dedup();
        self.river_cfv_conversions = primary;
        self.secondary_river_cfv_conversions = secondary;
        self.secondary_river_cfv_nodes = secondary_river_nodes;
        Ok((primary_count, secondary_count))
    }

    /// Adds a third river-model routing tier below selected flop frontiers.
    #[cfg(feature = "neural")]
    pub fn load_tertiary_neural_river_cfv_conversions(
        &mut self,
        trained_flop: [Card; 3],
        turns: &[Card],
        rivers: &[Card],
        model_root: impl AsRef<std::path::Path>,
        flop_nodes: &[u64],
        pio_order_path: impl AsRef<std::path::Path>,
    ) -> Result<usize, String> {
        let available: std::collections::HashSet<_> = self
            .turn_cfv_capture_contexts()
            .into_iter()
            .map(|row| row.0)
            .collect();
        if flop_nodes.is_empty() || flop_nodes.iter().any(|node| !available.contains(node)) {
            return Err("Tertiary river routing requires valid flop frontier nodes".into());
        }
        let primary = std::mem::take(&mut self.river_cfv_conversions);
        let secondary = std::mem::take(&mut self.secondary_river_cfv_conversions);
        let count = self.load_neural_river_cfv_conversions(
            trained_flop,
            turns,
            rivers,
            pio_order_path,
            model_root,
        )?;
        let tertiary = std::mem::take(&mut self.river_cfv_conversions);

        fn collect(game: &PostFlopGame, node_id: u64, output: &mut Vec<u64>) {
            let node = game.node_arena[node_id as usize].lock();
            if node.is_terminal() {
                return;
            }
            if node.is_chance() && node.turn != NOT_DEALT {
                output.push(node.id());
                return;
            }
            let children: Vec<_> = (0..node.num_actions())
                .map(|action| node.play(action).id())
                .collect();
            drop(node);
            for child in children {
                collect(game, child, output);
            }
        }

        let mut river_nodes = Vec::new();
        for &node in flop_nodes {
            collect(self, node, &mut river_nodes);
        }
        river_nodes.sort_unstable();
        river_nodes.dedup();
        self.river_cfv_conversions = primary;
        self.secondary_river_cfv_conversions = secondary;
        self.tertiary_river_cfv_conversions = tertiary;
        self.tertiary_river_cfv_nodes = river_nodes;
        Ok(count)
    }

    /// Enables or disables all installed turn-specific river-sum models.
    pub fn set_neural_river_sums(&mut self, enabled: bool) -> Result<usize, String> {
        if self.state == State::Solved {
            return Err("Cannot change solving mode on a solved game".into());
        }
        let mut node_count = 0;
        for (&turn, conversion) in self
            .river_cfv_conversions
            .iter()
            .chain(self.secondary_river_cfv_conversions.iter())
            .chain(self.tertiary_river_cfv_conversions.iter())
        {
            let source = self.node_arena[conversion.node_id as usize].lock();
            let cards: Vec<_> = conversion
                .basis_actions
                .iter()
                .map(|&action| source.play(action).prev_action)
                .collect();
            drop(source);
            for node in &self.node_arena {
                let node = node.lock();
                if !node.is_chance() || node.turn != turn {
                    continue;
                }
                if node.num_actions() == 0
                    && node.amount == self.tree_config.effective_stack
                    && self.turn_allin_payout_matrices.contains_key(&turn)
                {
                    continue;
                }
                if node.num_actions() != conversion.action_count
                    || !self.isomorphic_chances(&node).is_empty()
                {
                    return Err("River conversion action count differs between nodes".into());
                }
                for (&action, &card) in conversion.basis_actions.iter().zip(&cards) {
                    if node.play(action).prev_action != card {
                        return Err("River basis row ordering differs between nodes".into());
                    }
                }
                node_count += 1;
            }
        }
        if self.river_cfv_conversions.is_empty() {
            return Err("Load river models first".into());
        }
        for conversion in self
            .river_cfv_conversions
            .values_mut()
            .chain(self.secondary_river_cfv_conversions.values_mut())
            .chain(self.tertiary_river_cfv_conversions.values_mut())
        {
            conversion.use_in_solving = enabled;
        }
        Ok(node_count)
    }

    /// Loads Pio-ordered 36-turn weights once, validating board, node and basis identities.
    /// This enables diagnostics during solving; exact CFV sums still drive CFR updates.
    /// The caller must select weights trained for the same tree and node semantics.
    pub fn load_turn_cfv_conversion(
        &mut self,
        node_id: u64,
        trained_flop: [Card; 3],
        basis_cards: &[Card],
        pio_order_path: impl AsRef<std::path::Path>,
        weights_path: impl AsRef<std::path::Path>,
    ) -> Result<(), String> {
        use crate::pio_order::*;
        if self.state <= State::Uninitialized || self.state == State::Solved {
            return Err("Conversion requires an initialized, unsolved game".into());
        }
        let mut expected = trained_flop;
        let mut actual = self.card_config.flop;
        expected.sort_unstable();
        actual.sort_unstable();
        if expected != actual || basis_cards.is_empty() {
            return Err("Conversion flop or basis size does not match".into());
        }
        let node = self
            .node_arena
            .get(node_id as usize)
            .ok_or("Invalid conversion node ID")?
            .lock();
        if !node.is_chance()
            || node.turn != NOT_DEALT
            || node.num_actions() != 49
            || !self.isomorphic_chances(&node).is_empty()
        {
            return Err("Conversion requires 49 physical turns with isomorphism disabled".into());
        }
        let mut basis_actions = Vec::with_capacity(basis_cards.len());
        for &card in basis_cards {
            let action = (0..node.num_actions())
                .find(|&index| node.play(index).prev_action == Action::Chance(card))
                .ok_or("Basis card is not a legal physical turn")?;
            if basis_actions.contains(&action) {
                return Err("Duplicate basis turn".into());
            }
            basis_actions.push(action);
        }
        drop(node);
        let order = load_pio_hand_order(pio_order_path)?;
        let canonical = load_canonical_weights_with_basis(weights_path, basis_cards.len())?;
        if canonical.iter().flatten().any(|value| !value.is_finite()) {
            return Err("Nonfinite conversion weight".into());
        }
        let mappings = [
            local_to_pio(&order, &self.private_cards[0])?,
            local_to_pio(&order, &self.private_cards[1])?,
        ];
        let weights = convert_weights_to_local(&canonical, [&mappings[0], &mappings[1]])?;
        self.turn_cfv_conversion = Some(TurnCfvConversion {
            node_id,
            action_count: 49,
            basis_actions,
            weights,
            use_in_solving: false,
            all_turn_nodes: false,
            bucket_model: None,
            neural_model: None,
        });
        Ok(())
    }

    /// Loads a Pio-ordered cross-hand bucket model for physical turn sums.
    pub fn load_bucketed_turn_cfv_conversion(
        &mut self,
        node_id: u64,
        trained_flop: [Card; 3],
        basis_cards: &[Card],
        pio_order_path: impl AsRef<std::path::Path>,
        assignments_path: impl AsRef<std::path::Path>,
        weights_path: impl AsRef<std::path::Path>,
        bucket_count: usize,
    ) -> Result<(), String> {
        use crate::pio_order::*;
        if self.state <= State::Uninitialized || self.state == State::Solved {
            return Err("Conversion requires an initialized, unsolved game".into());
        }
        let mut expected = trained_flop;
        let mut actual = self.card_config.flop;
        expected.sort_unstable();
        actual.sort_unstable();
        if expected != actual || basis_cards.is_empty() {
            return Err("Conversion flop or basis size does not match".into());
        }
        let node = self
            .node_arena
            .get(node_id as usize)
            .ok_or("Invalid conversion node ID")?
            .lock();
        if !node.is_chance()
            || node.turn != NOT_DEALT
            || node.num_actions() != 49
            || !self.isomorphic_chances(&node).is_empty()
        {
            return Err("Conversion requires 49 physical turns with isomorphism disabled".into());
        }
        let mut basis_actions = Vec::with_capacity(basis_cards.len());
        for &card in basis_cards {
            let action = (0..node.num_actions())
                .find(|&index| node.play(index).prev_action == Action::Chance(card))
                .ok_or("Basis card is not a legal physical turn")?;
            if basis_actions.contains(&action) {
                return Err("Duplicate basis turn".into());
            }
            basis_actions.push(action);
        }
        drop(node);

        let order = load_pio_hand_order(pio_order_path)?;
        let mappings = [
            local_to_pio(&order, &self.private_cards[0])?,
            local_to_pio(&order, &self.private_cards[1])?,
        ];
        let canonical = load_canonical_bucket_weights(
            assignments_path,
            weights_path,
            basis_cards.len(),
            bucket_count,
        )?;
        let weights =
            convert_weights_to_local(&canonical.self_weights, [&mappings[0], &mappings[1]])?;
        let bucket_weights =
            convert_weights_to_local(&canonical.bucket_weights, [&mappings[0], &mappings[1]])?;
        let assignments = [0, 1].map(|player| {
            mappings[player]
                .iter()
                .map(|&pio| canonical.assignments[pio])
                .collect()
        });
        self.turn_cfv_conversion = Some(TurnCfvConversion {
            node_id,
            action_count: 49,
            basis_actions,
            weights,
            use_in_solving: false,
            all_turn_nodes: false,
            bucket_model: Some(LocalBucketWeights {
                bucket_sizes: canonical.bucket_sizes,
                assignments,
                weights: bucket_weights,
            }),
            neural_model: None,
        });
        Ok(())
    }

    /// Opts into replacing the configured node's exact sum during CFR traversal.
    /// Exploitability and finalization continue to evaluate the full physical tree.
    pub fn set_weighted_turn_sum(&mut self, enabled: bool) -> Result<(), String> {
        if self.state == State::Solved {
            return Err("Cannot change solving mode on a solved game".into());
        }
        let conversion = self
            .turn_cfv_conversion
            .as_mut()
            .ok_or("Load turn conversion weights first")?;
        conversion.use_in_solving = enabled;
        Ok(())
    }

    /// Apply the installed matrix at every flop-to-turn chance node.
    /// Reusing weights fitted at one node is experimental: pots and reaches can differ.
    /// All selected physical cards must occupy the same action rows at every target node.
    pub fn set_weighted_all_turn_sums(&mut self, enabled: bool) -> Result<usize, String> {
        if self.state == State::Solved {
            return Err("Cannot change solving mode on a solved game".into());
        }
        let conversion = self
            .turn_cfv_conversion
            .as_ref()
            .ok_or("Load turn conversion weights first")?;
        let source = self.node_arena[conversion.node_id as usize].lock();
        let cards: Vec<_> = conversion
            .basis_actions
            .iter()
            .map(|&action| source.play(action).prev_action)
            .collect();
        drop(source);
        let mut count = 0;
        for node in &self.node_arena {
            let node = node.lock();
            if !node.is_chance() || node.turn != NOT_DEALT {
                continue;
            }
            // Exact flop-all-in matrix leaves retain the chance-player marker
            // but deliberately have no physical turn children. They are exact
            // terminal evaluations, not neural turn frontiers.
            if node.num_actions() == 0 {
                continue;
            }
            if self.neural_turn_action_count(&node) != conversion.action_count
                || !self.isomorphic_chances(&node).is_empty()
            {
                return Err("All-turn conversion action count differs between target nodes".into());
            }
            for (&action, &card) in conversion.basis_actions.iter().zip(&cards) {
                if node.play(action).prev_action != card {
                    return Err("Turn basis row ordering differs between nodes".into());
                }
            }
            count += 1;
        }
        let conversion = self.turn_cfv_conversion.as_mut().unwrap();
        conversion.all_turn_nodes = enabled;
        conversion.use_in_solving = enabled;
        Ok(count)
    }

    /// Physical flop-to-turn diagnostic contexts: node ID, pot, remaining stack.
    pub fn turn_cfv_capture_contexts(&self) -> Vec<(u64, i32, i32)> {
        self.node_arena
            .iter()
            .filter_map(|entry| {
                let node = entry.lock();
                (node.is_chance() && node.turn == NOT_DEALT && node.num_actions() > 0).then_some((
                    node.id(),
                    self.tree_config.starting_pot + 2 * node.amount,
                    self.tree_config.effective_stack - node.amount,
                ))
            })
            .collect()
    }

    /// Enables exact CFV diagnostics at every physical flop-to-turn node.
    /// Does not change CFR aggregation; disabled on game reconfiguration.
    pub fn set_capture_all_turn_nodes(&mut self, enabled: bool) {
        self.capture_all_turn_nodes = enabled;
    }

    /// Enables exact CFV diagnostics at every physical turn-to-river node.
    /// Does not change CFR aggregation; disabled on game reconfiguration.
    pub fn set_capture_all_river_nodes(&mut self, enabled: bool) {
        self.capture_all_river_nodes = enabled;
    }

    /// Enables exact river CFV diagnostics only below the specified turns.
    /// Does not change CFR aggregation; disabled on game reconfiguration.
    pub fn set_capture_river_turns(&mut self, turns: &[Card]) {
        assert!(turns.iter().all(|&turn| turn < 52));
        self.capture_river_turns.clear();
        self.capture_river_turns.extend_from_slice(turns);
        self.capture_river_turns.sort_unstable();
        self.capture_river_turns.dedup();
    }

    /// Number of physical turn-to-river chance nodes in this tree.
    pub fn river_cfv_capture_node_count(&self) -> usize {
        self.node_arena
            .iter()
            .filter(|entry| {
                let node = entry.lock();
                node.is_chance() && node.turn != NOT_DEALT
            })
            .count()
    }

    /// River capture identity and chip state: node ID, turn, pot, remaining stack.
    pub fn river_cfv_capture_contexts(&self) -> Vec<(u64, Card, i32, i32)> {
        self.node_arena
            .iter()
            .filter_map(|entry| {
                let node = entry.lock();
                (node.is_chance() && node.turn != NOT_DEALT).then_some((
                    node.id(),
                    node.turn,
                    self.tree_config.starting_pot + 2 * node.amount,
                    self.tree_config.effective_stack - node.amount,
                ))
            })
            .collect()
    }

    /// Removes runtime conversion diagnostics.
    pub fn clear_turn_cfv_conversion(&mut self) {
        self.turn_cfv_conversion = None;
        self.secondary_turn_cfv_conversion = None;
        self.secondary_turn_cfv_nodes.clear();
        self.tertiary_turn_cfv_conversion = None;
        self.tertiary_turn_cfv_nodes.clear();
        self.river_cfv_conversions.clear();
        self.secondary_river_cfv_conversions.clear();
        self.secondary_river_cfv_nodes.clear();
        self.tertiary_river_cfv_conversions.clear();
        self.tertiary_river_cfv_nodes.clear();
    }

    /// Evaluates the installed primary turn or river model on captured rows at
    /// the same chance node. This is diagnostic-only: every required
    /// basis card must be present in `captured_cards`.
    #[cfg(feature = "neural")]
    pub fn diagnostic_estimate_turn_capture(
        &self,
        node_id: u64,
        player: usize,
        captured_values: &[f32],
        captured_cards: &[Card],
        outer_iteration: u32,
        opponent_reach: &[f32],
        player_reach: &[f32],
    ) -> Result<Vec<f64>, String> {
        let node = self
            .node_arena
            .get(node_id as usize)
            .ok_or("Captured turn node ID is outside the tree")?
            .lock();
        let conversion = if node.turn == NOT_DEALT {
            self.turn_cfv_conversion.as_ref()
        } else {
            self.river_cfv_conversions.get(&node.turn)
        }.ok_or("No primary conversion is installed for the captured chance node")?;
        let physical_cards = node.chance_cards();
        let hands = captured_values
            .len()
            .checked_div(captured_cards.len())
            .ok_or("Captured turn basis is empty")?;
        if hands * captured_cards.len() != captured_values.len() {
            return Err("Captured turn matrix has inconsistent dimensions".into());
        }
        let mut compact = Vec::with_capacity(conversion.basis_actions.len() * hands);
        for &action in &conversion.basis_actions {
            let card = *physical_cards
                .get(action)
                .ok_or("Turn model action is outside the physical chance node")?;
            let row = captured_cards
                .iter()
                .position(|&candidate| candidate == card)
                .ok_or_else(|| {
                    format!(
                        "Turn model card {} is absent from captured basis",
                        card_to_string(card).unwrap_or_else(|_| format!("#{card}"))
                    )
                })?;
            compact.extend_from_slice(&captured_values[row * hands..(row + 1) * hands]);
        }
        drop(node);
        Ok(conversion.estimate_sum_from_basis_at_iteration(
            node_id,
            player,
            &compact,
            &conversion.basis_actions,
            outer_iteration,
            opponent_reach,
            (!player_reach.is_empty()).then_some(player_reach),
        ))
    }

    /// Creates a new empty [`PostFlopGame`].
    ///
    /// Use of this method is strongly discouraged because an instance created by this method is
    /// invalid until [`update_config`] is called.
    /// Please use [`with_config`] instead whenever possible.
    ///
    /// [`update_config`]: #method.update_config
    /// [`with_config`]: #method.with_config
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new [`PostFlopGame`] with the specified configuration.
    #[inline]
    pub fn with_config(card_config: CardConfig, action_tree: ActionTree) -> Result<Self, String> {
        let mut game = Self::new();
        game.update_config(card_config, action_tree)?;
        Ok(game)
    }

    /// Prepares runtime-only decision plans for the opt-in direct-dispatch
    /// experiment. No strategies, ranges, regrets or numerical values change.
    #[doc(hidden)]
    pub fn prepare_direct_terminal_dispatch(&self) -> usize {
        for node in &self.node_arena {
            let node=node.lock();
            if !node.is_terminal() && !node.is_chance()
                && (node.id==0 || matches!(node.prev_action,Action::Chance(_))) {
                let _=self.direct_terminal_plan(&node);
            }
        }
        self.direct_terminal_plans.lock().unwrap().len()
    }

    /// Updates the game configuration. The solved result will be lost.
    #[inline]
    pub fn update_config(
        &mut self,
        card_config: CardConfig,
        action_tree: ActionTree,
    ) -> Result<(), String> {
        self.update_config_with_card_tables(card_config, action_tree, false, None)
    }

    /// Opt-in builder for reduced physical trees with exact matrix all-in leaves.
    /// Does not prepare omitted-runout equities. Full-tree/BR builders are unchanged.
    pub fn with_pruned_card_tables_experiment(
        card_config: CardConfig,
        action_tree: ActionTree,
    ) -> Result<Self, String> {
        Self::check_pruned_card_table_config(&card_config)?;
        let mut game = Self::new();
        game.update_config_with_card_tables(card_config, action_tree, true, None)?;
        Ok(game)
    }

    pub(super) fn check_pruned_card_table_config(card_config: &CardConfig) -> Result<(), String> {
        if !EXACT_ALLIN_MATRIX_TREE.load(Ordering::Relaxed)
            || TURN_BASIS_COUNT.load(Ordering::Relaxed) == 0
            || RIVER_BASIS_COUNT.load(Ordering::Relaxed) == 0
            || card_config.turn != NOT_DEALT || card_config.river != NOT_DEALT
            || BASIS13_TURN_TREE.load(Ordering::Relaxed) || BASIS13_RIVER_TREE.load(Ordering::Relaxed)
            || BASIS6_TURN_TREE.load(Ordering::Relaxed) || BASIS6_RIVER_TREE.load(Ordering::Relaxed)
        {
            return Err("Pruned card initialization requires explicit reduced turn/river bases, flop start and exact matrix all-ins".into());
        }
        Ok(())
    }

    pub(super) fn update_config_with_card_tables(
        &mut self,
        card_config: CardConfig,
        action_tree: ActionTree,
        partial: bool,
        auxiliary: Option<super::auxiliary_turn::AuxiliaryExactTurn>,
    ) -> Result<(), String> {
        self.direct_terminal_plans.get_mut().unwrap().clear();
        self.auxiliary_exact_turn = auxiliary;
        self.partial_card_tables_experiment = partial;
        self.turn_cfv_conversion = None;
        self.secondary_turn_cfv_conversion = None;
        self.secondary_turn_cfv_nodes.clear();
        self.tertiary_turn_cfv_conversion = None;
        self.tertiary_turn_cfv_nodes.clear();
        self.secondary_river_cfv_conversions.clear();
        self.secondary_river_cfv_nodes.clear();
        self.tertiary_river_cfv_conversions.clear();
        self.tertiary_river_cfv_nodes.clear();
        self.capture_all_turn_nodes = false;
        self.capture_all_river_nodes = false;
        self.capture_river_turns.clear();
        self.flop_allin_payout_matrix = None;
        self.turn_allin_payout_matrices.clear();
        self.use_flop_allin_payout_matrix = false;
        self.state = State::ConfigError;
        self.check_check_river_nodes.clear();

        if !action_tree.invalid_terminals().is_empty() {
            return Err("Invalid terminal is found in action tree".to_string());
        }

        self.card_config = card_config;
        (
            self.tree_config,
            self.added_lines,
            self.removed_lines,
            self.action_root,
        ) = action_tree.eject();

        self.check_card_config()?;
        if self.auxiliary_exact_turn.is_some() {
            self.validate_auxiliary_exact_turn(&selected_turn_cards(TURN_BASIS_COUNT.load(Ordering::Relaxed)))?;
        }
        if partial {
            self.init_pruned_card_fields_experiment()?;
        } else {
            self.init_card_fields();
        }
        self.init_root()?;
        self.resolve_auxiliary_exact_turn();

        self.state = State::TreeBuilt;

        self.init_interpreter();
        self.reset_bunching_effect();

        Ok(())
    }

    /// Builds the exact blocker-aware 1326x1326 flop payout matrix used by the
    /// opt-in called-all-in evaluator. The matrix is independent of ranges;
    /// this game retains only compact directional matrices for the active
    /// local OOP and IP hands.
    ///
    /// The matrix averages every unordered turn/river runout. This is exactly
    /// equivalent to the ordinary ordered 45-by-44 chance traversal because
    /// both orderings have identical showdown payouts.
    pub fn enable_flop_allin_payout_matrix(&mut self) -> Result<(), String> {
        if self.card_config.turn != NOT_DEALT || self.card_config.river != NOT_DEALT {
            return Err("flop all-in payout matrix requires a flop-start game".to_string());
        }
        if self.bunching_num_dead_cards != 0 {
            return Err("flop all-in payout matrix does not support bunching".to_string());
        }

        let values = self.build_flop_allin_payout_matrix();
        self.install_flop_allin_payout_matrix(&values);
        self.misc_memory_usage = self.memory_usage_internal();
        Ok(())
    }

    fn compact_allin_payout_matrix(&self, canonical: &[f64]) -> FlopAllinPayoutMatrix {
        const HANDS: usize = 52 * 51 / 2;
        assert_eq!(canonical.len(), HANDS * HANDS);
        let local_to_canonical: [Vec<usize>; 2] = std::array::from_fn(|player| {
            self.private_cards[player]
                .iter()
                .map(|&(c1, c2)| card_pair_to_index(c1, c2))
                .collect()
        });
        let local_hands = std::array::from_fn(|player| local_to_canonical[player].len());
        let values = std::array::from_fn(|player| {
            let opponent = player ^ 1;
            let mut compact = Vec::with_capacity(local_hands[player] * local_hands[opponent]);
            for &hero in &local_to_canonical[player] {
                let source = &canonical[hero * HANDS..(hero + 1) * HANDS];
                compact.extend(
                    local_to_canonical[opponent]
                        .iter()
                        .map(|&opponent| source[opponent]),
                );
            }
            compact
        });
        let values_transposed = if EXACT_ALLIN_MATRIX_SPARSE_EVAL.load(Ordering::Relaxed) {
            std::array::from_fn(|player| {
                let rows = local_hands[player];
                let cols = local_hands[player ^ 1];
                let mut transposed = vec![0.0; rows * cols];
                for row in 0..rows {
                    for col in 0..cols {
                        transposed[col * rows + row] = values[player][row * cols + col];
                    }
                }
                transposed
            })
        } else {
            std::array::from_fn(|_| Vec::new())
        };
        let values_f32 = if EXACT_ALLIN_MATRIX_F32_EVAL.load(Ordering::Relaxed) {
            std::array::from_fn(|player| values[player].iter().map(|&value| value as f32).collect())
        } else {
            std::array::from_fn(|_| Vec::new())
        };
        let values_transposed_f32 = if EXACT_ALLIN_MATRIX_F32_EVAL.load(Ordering::Relaxed) {
            std::array::from_fn(|player| {
                let rows = local_hands[player];
                let cols = local_hands[player ^ 1];
                let mut transposed = vec![0.0; rows * cols];
                for row in 0..rows {
                    for col in 0..cols {
                        transposed[col * rows + row] = values_f32[player][row * cols + col];
                    }
                }
                transposed
            })
        } else {
            std::array::from_fn(|_| Vec::new())
        };
        FlopAllinPayoutMatrix {
            values,
            values_transposed,
            values_f32,
            values_transposed_f32,
            local_hands,
        }
    }

    pub(super) fn install_flop_allin_payout_matrix(&mut self, canonical: &[f64]) {
        self.flop_allin_payout_matrix = Some(self.compact_allin_payout_matrix(canonical));
    }

    pub(super) fn install_turn_allin_payout_matrix(&mut self, turn: Card, canonical: &[f64]) {
        let compact = self.compact_allin_payout_matrix(canonical);
        self.turn_allin_payout_matrices.insert(turn, compact);
    }

    pub(super) fn exact_allin_matrix_turns(&self) -> Vec<Card> {
        let mut turns = Vec::new();
        for node in &self.node_arena {
            let node = node.lock();
            if node.is_chance()
                && node.amount == self.tree_config.effective_stack
                && node.turn != NOT_DEALT
                && node.river == NOT_DEALT
            {
                turns.push(node.turn);
            }
        }
        turns.sort_unstable();
        turns.dedup();
        turns
    }

    fn turn_allin_cache_path(path: &std::path::Path, turn: Card) -> std::path::PathBuf {
        let stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("allin");
        path.with_file_name(format!("{stem}_turn_{turn:02}.bin"))
    }

    /// Loads a reusable canonical matrix cache, or builds and atomically saves
    /// it when absent, then installs compact directional matrices for this
    /// game's active OOP and IP hands. Returns true when the cache was loaded.
    pub fn enable_flop_allin_payout_matrix_cached(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<bool, String> {
        if self.card_config.turn != NOT_DEALT || self.card_config.river != NOT_DEALT {
            return Err("flop all-in payout matrix requires a flop-start game".to_string());
        }
        if self.bunching_num_dead_cards != 0 {
            return Err("flop all-in payout matrix does not support bunching".to_string());
        }
        let path = path.as_ref();
        let (values, loaded) = if path.exists() {
            (self.load_flop_allin_payout_matrix_cache(path)?, true)
        } else {
            let values = self.build_flop_allin_payout_matrix();
            self.save_flop_allin_payout_matrix_cache(path, &values)?;
            (values, false)
        };
        self.install_flop_allin_payout_matrix(&values);
        self.misc_memory_usage = self.memory_usage_internal();
        Ok(loaded)
    }

    /// Enables exact matrices as solver leaves for called flop and turn all-ins.
    /// The game must have been constructed with `set_exact_allin_matrix_tree(true)`
    /// so those chance descendants are absent from the physical tree.
    pub fn enable_exact_flop_allin_payout_matrix(&mut self) -> Result<(), String> {
        if !EXACT_ALLIN_MATRIX_TREE.load(Ordering::Relaxed) {
            return Err(
                "exact flop all-in matrix requires matrix tree mode during construction"
                    .to_string(),
            );
        }
        self.enable_flop_allin_payout_matrix()?;
        for turn in self.exact_allin_matrix_turns() {
            let values = self.build_turn_allin_payout_matrix(turn)?;
            self.install_turn_allin_payout_matrix(turn, &values);
        }
        self.misc_memory_usage = self.memory_usage_internal();
        self.use_flop_allin_payout_matrix = true;
        Ok(())
    }

    /// Cached variant of [`Self::enable_exact_flop_allin_payout_matrix`].
    pub fn enable_exact_flop_allin_payout_matrix_cached(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<bool, String> {
        if !EXACT_ALLIN_MATRIX_TREE.load(Ordering::Relaxed) {
            return Err(
                "exact flop all-in matrix requires matrix tree mode during construction"
                    .to_string(),
            );
        }
        let path = path.as_ref();
        let loaded = self.enable_flop_allin_payout_matrix_cached(path)?;
        for turn in self.exact_allin_matrix_turns() {
            let turn_path = Self::turn_allin_cache_path(path, turn);
            let values = if turn_path.exists() {
                self.load_turn_allin_payout_matrix_cache(&turn_path, turn)?
            } else {
                let values = self.build_turn_allin_payout_matrix(turn)?;
                self.save_turn_allin_payout_matrix_cache(&turn_path, turn, &values)?;
                values
            };
            self.install_turn_allin_payout_matrix(turn, &values);
        }
        self.misc_memory_usage = self.memory_usage_internal();
        self.use_flop_allin_payout_matrix = true;
        Ok(loaded)
    }

    /// Number of compact exact called-all-in matrices currently installed:
    /// one optional flop matrix plus one matrix per represented turn card.
    pub fn exact_allin_payout_matrix_counts(&self) -> (usize, usize) {
        (
            usize::from(self.flop_allin_payout_matrix.is_some()),
            self.turn_allin_payout_matrices.len(),
        )
    }

    /// Sets the bunching effect configuration.
    ///
    /// **Warning**: Enabling the bunching effect will significantly slow down the solving process.
    /// Specifically, the computational complexity of the terminal evaluation will increase from
    /// *O*(#(OOP private hands) + #(IP private hands)) to *O*(#(OOP private hands) * #(IP private
    /// hands)).
    #[inline]
    pub fn set_bunching_effect(&mut self, bunching_data: &BunchingData) -> Result<(), String> {
        if self.partial_card_tables_experiment {
            return Err("Bunching requires complete card tables; materialize analysis tables first".into());
        }
        if self.state <= State::Uninitialized {
            return Err("Game is not successfully initialized".to_string());
        }

        if !bunching_data.is_ready() {
            return Err("Bunching configuration is not ready".to_string());
        }

        if self.flop_allin_payout_matrix.is_some() || !self.turn_allin_payout_matrices.is_empty() {
            return Err("flop all-in payout matrix does not support bunching".to_string());
        }

        let mut flop_sorted = self.card_config.flop;
        flop_sorted.sort_unstable();
        if flop_sorted != bunching_data.flop() {
            return Err("Flop cards do not match".to_string());
        }

        self.reset_bunching_effect();
        self.set_bunching_effect_internal(bunching_data)?;

        Ok(())
    }

    /// Resets the bunching effect configuration. The current node will also be reset to the root.
    #[inline]
    pub fn reset_bunching_effect(&mut self) {
        self.bunching_num_dead_cards = 0;
        self.bunching_num_combinations = 0.0;
        self.bunching_arena = Vec::new();
        self.bunching_strength = Vec::new();
        self.bunching_num_flop = Default::default();
        self.bunching_num_turn = Default::default();
        self.bunching_num_river = Default::default();
        self.bunching_coef_flop = Default::default();
        self.bunching_coef_turn = Default::default();
        self.back_to_root();
    }

    /// Obtains the card configuration.
    #[inline]
    pub fn card_config(&self) -> &CardConfig {
        &self.card_config
    }

    /// Obtains the tree configuration.
    #[inline]
    pub fn tree_config(&self) -> &TreeConfig {
        &self.tree_config
    }

    /// Obtains the added lines.
    #[inline]
    pub fn added_lines(&self) -> &[Vec<Action>] {
        &self.added_lines
    }

    /// Obtains the removed lines.
    #[inline]
    pub fn removed_lines(&self) -> &[Vec<Action>] {
        &self.removed_lines
    }

    /// Returns the card list of private hands of the given player.
    ///
    /// The returned list contains only card pairs with positive weight, i.e., card pairs with zero
    /// weight are excluded. The returned list is sorted as follows:
    ///
    /// - Each card pair has IDs in `(low_id, high_id)` order.
    /// - Card pairs are sorted in the lexicographic order.
    #[inline]
    pub fn private_cards(&self, player: usize) -> &[(Card, Card)] {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        &self.private_cards[player]
    }

    /// Returns the estimated memory usage in bytes (uncompressed, compressed).
    #[inline]
    pub fn memory_usage(&self) -> (u64, u64) {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        let num_elements = 2 * self.num_storage + self.num_storage_ip + self.num_storage_chance;
        let uncompressed = 4 * num_elements + self.misc_memory_usage;
        let compressed = 2 * num_elements + self.misc_memory_usage;

        (uncompressed, compressed)
    }

    /// Passive numeric-storage report for diagnostic examples; no solver state changes.
    #[cfg(feature="neural")]
    pub fn diagnostic_numeric_storage(&self)->Vec<(String,u64,u64)>{
        let mut result=Vec::new();
        for(name,v)in [("cumulative_strategy",&self.storage1),("cumulative_regrets",&self.storage2),("IP_CFVs",&self.storage_ip),("chance_CFVs",&self.storage_chance)]{
            result.push((name.to_string(),v.len() as u64,v.capacity() as u64));
        }
        let mut matrices=0u64;
        for m in self.flop_allin_payout_matrix.iter().chain(self.turn_allin_payout_matrices.values()){
            for v in m.values.iter().chain(m.values_transposed.iter()){matrices+=v.capacity() as u64*8;}
            for v in m.values_f32.iter().chain(m.values_transposed_f32.iter()){matrices+=v.capacity() as u64*4;}
        }
        result.push(("packed_allin_matrices".into(),matrices,matrices));
        let misc=self.memory_usage_internal();result.push(("game_metadata_cards_matrices_inclusive".into(),misc,misc));
        let mut models=(0u64,0u64);let mut conversion=0u64;
        for cv in self.turn_cfv_conversion.iter().chain(self.river_cfv_conversions.values()){
            conversion+=cv.basis_actions.capacity() as u64*8;for v in &cv.weights{conversion+=v.capacity() as u64*4;}
            if let Some(n)=&cv.neural_model{
                conversion+=n.diagnostic_extra_vector_bytes();
                if let Some(ms)=&n.sparse_models{for m in ms{let(a,b)=m.diagnostic_numeric_bytes();models.0+=a;models.1+=b;}}
                for v in n.contexts.values(){conversion+=v.capacity() as u64*4;}for v in &n.mappings{conversion+=v.capacity() as u64*8;}
                if let Some(v)=&n.range_features{conversion+=v.capacity() as u64*4;}
            }
        }
        result.push(("NN_parameter_vectors_including_FP32_and_BF16".into(),models.0,models.1));
        result.push(("NN_context_mapping_range_vectors".into(),conversion,conversion));result
    }

    /// Returns the estimated additional memory usage in bytes when the bunching effect is enabled.
    #[inline]
    pub fn memory_usage_bunching(&self) -> u64 {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        self.memory_usage_bunching_internal()
    }

    /// Remove lines after building the `PostFlopGame` but before allocating memory.
    ///
    /// This allows the removal of chance-specific lines (e.g., remove overbets on board-pairing
    /// turns) which we cannot do while building an action tree.
    pub fn remove_lines(&mut self, lines: &[Vec<Action>]) -> Result<(), String> {
        if self.state <= State::Uninitialized {
            return Err("Game is not successfully initialized".to_string());
        } else if self.state >= State::MemoryAllocated {
            return Err("Game has already been allocated".to_string());
        }

        for line in lines {
            let mut root = self.root();
            let info = self.remove_line_recursive(&mut root, line)?;
            self.num_storage -= info.num_storage;
            self.num_storage_ip -= info.num_storage_ip;
            self.num_storage_chance -= info.num_storage_chance;
        }

        Ok(())
    }

    /// Returns whether the memory is allocated.
    ///
    /// If the memory is allocated, returns `Some(is_compression_enabled)`;
    /// otherwise, returns `None`.
    #[inline]
    pub fn is_memory_allocated(&self) -> Option<bool> {
        if self.state <= State::TreeBuilt {
            None
        } else {
            Some(self.is_compression_enabled)
        }
    }

    /// Allocates the memory.
    pub fn allocate_memory(&mut self, enable_compression: bool) {
        if self.state <= State::Uninitialized {
            panic!("Game is not successfully initialized");
        }

        if self.state == State::MemoryAllocated
            && self.storage_mode == BoardState::River
            && self.is_compression_enabled == enable_compression
        {
            return;
        }

        let num_bytes = if enable_compression { 2 } else { 4 };
        if num_bytes * self.num_storage > isize::MAX as u64
            || num_bytes * self.num_storage_chance > isize::MAX as u64
        {
            panic!("Memory usage exceeds maximum size");
        }

        self.state = State::MemoryAllocated;
        self.is_compression_enabled = enable_compression;

        self.clear_storage();

        let storage_bytes = (num_bytes * self.num_storage) as usize;
        let storage_ip_bytes = (num_bytes * self.num_storage_ip) as usize;
        let storage_chance_bytes = (num_bytes * self.num_storage_chance) as usize;

        self.storage1 = vec![0; storage_bytes];
        self.storage2 = vec![0; storage_bytes];
        self.storage_ip = vec![0; storage_ip_bytes];
        self.storage_chance = vec![0; storage_chance_bytes];

        self.allocate_memory_nodes();

        self.storage_mode = BoardState::River;
        self.target_storage_mode = BoardState::River;
    }

    /// Checks the card configuration.
    pub(crate) fn check_card_config(&mut self) -> Result<(), String> {
        let config = &self.card_config;
        let (flop, turn, river) = (config.flop, config.turn, config.river);
        let range = &config.range;

        if flop.contains(&NOT_DEALT) {
            return Err("Flop cards not initialized".to_string());
        }

        if flop.iter().any(|&c| 52 <= c) {
            return Err(format!("Flop cards must be in [0, 52): flop = {flop:?}"));
        }

        if flop[0] == flop[1] || flop[0] == flop[2] || flop[1] == flop[2] {
            return Err(format!("Flop cards must be unique: flop = {flop:?}"));
        }

        if turn != NOT_DEALT {
            if 52 <= turn {
                return Err(format!("Turn card must be in [0, 52): turn = {turn}"));
            }

            if flop.contains(&turn) {
                return Err(format!(
                    "Turn card must be different from flop cards: turn = {turn}"
                ));
            }
        }

        if river != NOT_DEALT {
            if 52 <= river {
                return Err(format!("River card must be in [0, 52): river = {river}"));
            }

            if flop.contains(&river) {
                return Err(format!(
                    "River card must be different from flop cards: river = {river}"
                ));
            }

            if turn == river {
                return Err(format!(
                    "River card must be different from turn card: river = {river}"
                ));
            }

            if turn == NOT_DEALT {
                return Err(format!(
                    "River card specified without turn card: river = {river}"
                ));
            }
        }

        let expected_state = match (turn != NOT_DEALT, river != NOT_DEALT) {
            (false, _) => BoardState::Flop,
            (true, false) => BoardState::Turn,
            (true, true) => BoardState::River,
        };

        if self.tree_config.initial_state != expected_state {
            return Err(format!(
                "Invalid initial state of `tree_config`: expected = {:?}, actual = {:?}",
                expected_state, self.tree_config.initial_state
            ));
        }

        if range[0].is_empty() {
            return Err("OOP range is empty".to_string());
        }

        if range[1].is_empty() {
            return Err("IP range is empty".to_string());
        }

        if !range[0].is_valid() {
            return Err("OOP range is invalid (loaded broken data?)".to_string());
        }

        if !range[1].is_valid() {
            return Err("IP range is invalid (loaded broken data?)".to_string());
        }

        self.init_hands();
        self.num_combinations = 0.0;

        for (&(c1, c2), &w1) in self.private_cards[0]
            .iter()
            .zip(self.initial_weights[0].iter())
        {
            let oop_mask: u64 = (1 << c1) | (1 << c2);
            for (&(c3, c4), &w2) in self.private_cards[1]
                .iter()
                .zip(self.initial_weights[1].iter())
            {
                let ip_mask: u64 = (1 << c3) | (1 << c4);
                if oop_mask & ip_mask == 0 {
                    self.num_combinations += w1 as f64 * w2 as f64;
                }
            }
        }

        if self.num_combinations == 0.0 {
            return Err("Valid card assignment does not exist".to_string());
        }

        Ok(())
    }

    /// Initializes fields `initial_weights` and `private_cards`.
    #[inline]
    fn init_hands(&mut self) {
        let config = &self.card_config;
        let (flop, turn, river) = (config.flop, config.turn, config.river);
        let range = &config.range;

        let mut board_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);
        if turn != NOT_DEALT {
            board_mask |= 1 << turn;
        }
        if river != NOT_DEALT {
            board_mask |= 1 << river;
        }

        for player in 0..2 {
            let (hands, weights) = range[player].get_hands_weights(board_mask);
            self.initial_weights[player] = weights;
            self.private_cards[player] = hands;
        }
    }

    /// Initializes fields related to cards.
    pub(super) fn init_card_fields(&mut self) {
        self.partial_card_tables_experiment = false;
        self.init_same_hand_indices();

        (
            self.valid_indices_flop,
            self.valid_indices_turn,
            self.valid_indices_river,
        ) = self.card_config.valid_indices(&self.private_cards);

        self.hand_strength = self.card_config.hand_strength(&self.private_cards);

        (
            self.isomorphism_ref_turn,
            self.isomorphism_card_turn,
            self.isomorphism_swap_turn,
            self.isomorphism_ref_river,
            self.isomorphism_card_river,
            self.isomorphism_swap_river,
        ) = self.card_config.isomorphism(&self.private_cards);

        if let Some((turns,rivers))=PHYSICAL_CHANCE_ISOMORPHISM.lock().unwrap().clone(){
            retain_required_chances(self.card_config.flop,NOT_DEALT,&mut self.isomorphism_ref_turn,&mut self.isomorphism_card_turn,&turns);
            // River omitted-card lists are shared by turn suit. Compute each
            // turn's new source indices from its own legal-card ordering.
            let original=self.isomorphism_card_river.clone();
            for turn in 0..52u8{
                if self.card_config.flop.contains(&turn)||(self.card_config.turn!=NOT_DEALT&&self.card_config.turn!=turn){continue;}
                let mut omitted=original[turn as usize&3].clone();
                retain_required_chances(self.card_config.flop,turn,&mut self.isomorphism_ref_river[turn as usize],&mut omitted,&rivers);
                self.isomorphism_card_river[turn as usize&3]=omitted;
            }
        }else{
            self.isomorphism_card_turn.clear();
            for v in &mut self.isomorphism_card_river {v.clear();}
            self.isomorphism_ref_turn.clear();
            for v in &mut self.isomorphism_ref_river {v.clear();}
        }
    }

    fn init_same_hand_indices(&mut self) {
        for player in 0..2 {
            let same_hand_index = &mut self.same_hand_index[player];
            same_hand_index.clear();

            let player_hands = &self.private_cards[player];
            let opponent_hands = &self.private_cards[player ^ 1];
            for hand in player_hands {
                same_hand_index.push(
                    opponent_hands
                        .binary_search(hand)
                        .map_or(u16::MAX, |i| i as u16),
                );
            }
        }

    }

    fn init_pruned_card_fields_experiment(&mut self) -> Result<(), String> {
        let turns = selected_turn_cards(TURN_BASIS_COUNT.load(Ordering::Relaxed));
        let rivers = selected_river_cards(RIVER_BASIS_COUNT.load(Ordering::Relaxed));
        if turns.iter().chain(&rivers).any(|c| self.card_config.flop.contains(c))
            || turns.is_empty() || rivers.is_empty()
        {
            return Err("Illegal/empty pruned card basis".into());
        }
        let mut turn_mask = [false; 52];
        let mut pairs = vec![false; 1326];
        for turn in turns {
            turn_mask[turn as usize] = true;
            for &river in &rivers {
                if turn != river { pairs[card_pair_to_index(turn, river)] = true; }
            }
        }
        if let Some(aux) = &self.auxiliary_exact_turn {
            turn_mask[aux.turn as usize] = true;
            for river in 0..52 {
                if river != aux.turn && !self.card_config.flop.contains(&river) {
                    pairs[card_pair_to_index(aux.turn, river)] = true;
                }
            }
        }
        self.init_same_hand_indices();
        (self.valid_indices_flop,self.valid_indices_turn,self.valid_indices_river) =
            self.card_config.valid_indices_for_runouts(&self.private_cards,Some((&turn_mask,&pairs)));
        self.hand_strength = self.card_config.hand_strength_for_runouts(&self.private_cards,Some(&pairs));
        self.isomorphism_ref_turn.clear();
        self.isomorphism_card_turn.clear();
        self.isomorphism_swap_turn = Default::default();
        self.isomorphism_ref_river = vec![Vec::new();52];
        self.isomorphism_card_river = Default::default();
        self.isomorphism_swap_river = Default::default();
        Ok(())
    }

    /// Explicitly pay for complete tables before requesting full-board analysis.
    pub fn materialize_analysis_tables_experiment(&mut self) {
        if self.partial_card_tables_experiment { self.init_card_fields(); }
    }

    pub fn prepared_showdown_runouts_experiment(&self) -> usize {
        self.hand_strength.iter().filter(|rows| !rows[0].is_empty()).count()
    }

    /// Initializes the root node of game tree.
    fn init_root(&mut self) -> Result<(), String> {
        let num_nodes = self.count_num_nodes();
        let total_num_nodes = num_nodes[0] + num_nodes[1] + num_nodes[2];

        if total_num_nodes > u32::MAX as u64
            || mem::size_of::<PostFlopNode>() as u64 * total_num_nodes > isize::MAX as u64
        {
            return Err("Too many nodes".to_string());
        }

        self.num_nodes = num_nodes;
        self.node_arena = (0..total_num_nodes)
            .map(|id| {
                let mut node = PostFlopNode::default();
                node.id = id;
                MutexLike::new(node)
            })
            .collect::<Vec<_>>();
        self.clear_storage();

        let mut info = BuildTreeInfo {
            turn_index: num_nodes[0] as usize,
            river_index: (num_nodes[0] + num_nodes[1]) as usize,
            ..Default::default()
        };

        match self.tree_config.initial_state {
            BoardState::Flop => info.flop_index += 1,
            BoardState::Turn => info.turn_index += 1,
            BoardState::River => info.river_index += 1,
        }

        let mut root = self.node_arena[0].lock();
        root.turn = self.card_config.turn;
        root.river = self.card_config.river;

        self.build_tree_recursive(0, &self.action_root.lock(), &mut info);

        self.num_storage = info.num_storage;
        self.num_storage_ip = info.num_storage_ip;
        self.num_storage_chance = info.num_storage_chance;
        self.misc_memory_usage = self.memory_usage_internal();

        Ok(())
    }

    /// Initializes the interpreter.
    #[inline]
    pub(super) fn init_interpreter(&mut self) {
        let vecs = [
            vec![0.0; self.num_private_hands(0)],
            vec![0.0; self.num_private_hands(1)],
        ];

        self.weights = vecs.clone();
        self.normalized_weights = vecs.clone();
        self.cfvalues_cache = vecs;
    }

    /// Clears the storage.
    #[inline]
    fn clear_storage(&mut self) {
        self.storage1 = Vec::new();
        self.storage2 = Vec::new();
        self.storage_ip = Vec::new();
        self.storage_chance = Vec::new();
    }

    /// Counts the number of nodes in the game tree.
    #[inline]
    fn count_num_nodes(&self) -> [u64; 3] {
        let (turn_coef, river_coef) = match (self.card_config.turn, self.card_config.river) {
            (NOT_DEALT, _) => {
                let mut river_coef = 0;
                let flop = self.card_config.flop;
                let skip_cards = &self.isomorphism_card_turn;
                let flop_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);
                let skip_mask: u64 = skip_cards.iter().map(|&card| 1 << card).sum();
                let selected13 = basis13_turn_cards();
                let selected6 = basis6_turn_cards();
                for turn in 0..52 {
                    if (1 << turn) & (flop_mask | skip_mask) == 0 {
                        if (BASIS13_TURN_TREE.load(Ordering::Relaxed)
                            && !selected13.contains(&turn))
                            || (BASIS6_TURN_TREE.load(Ordering::Relaxed)
                                && !selected6.contains(&turn))
                        {
                            continue;
                        }
                        let river_basis_count = RIVER_BASIS_COUNT.load(Ordering::Relaxed);
                        river_coef += if river_basis_count > 0 {
                            river_basis_count
                        } else if BASIS6_RIVER_TREE.load(Ordering::Relaxed) {
                            6
                        } else if BASIS13_RIVER_TREE.load(Ordering::Relaxed) {
                            13
                        } else {
                            48 - self.isomorphism_card_river[turn as usize & 3].len()
                        };
                    }
                }
                let turn_coef = if BASIS6_TURN_TREE.load(Ordering::Relaxed) {
                    selected6
                        .iter()
                        .filter(|&&card| (1 << card) & flop_mask == 0)
                        .count()
                } else if BASIS13_TURN_TREE.load(Ordering::Relaxed) {
                    selected13
                        .iter()
                        .filter(|&&card| (1 << card) & flop_mask == 0)
                        .count()
                } else {
                    49 - self.isomorphism_card_turn.len()
                };
                (turn_coef, river_coef)
            }
            (turn, NOT_DEALT) => (
                1,
                if RIVER_BASIS_COUNT.load(Ordering::Relaxed) > 0 {
                    RIVER_BASIS_COUNT.load(Ordering::Relaxed)
                } else if BASIS6_RIVER_TREE.load(Ordering::Relaxed) {
                    6
                } else if BASIS13_RIVER_TREE.load(Ordering::Relaxed) {
                    13
                } else {
                    48 - self.isomorphism_card_river[turn as usize & 3].len()
                },
            ),
            _ => (0, 1),
        };

        let num_action_nodes = if EXACT_ALLIN_MATRIX_TREE.load(Ordering::Relaxed) {
            count_num_action_nodes_exact_allin_matrix(
                &self.action_root.lock(),
                self.tree_config.effective_stack,
            )
        } else {
            count_num_action_nodes(&self.action_root.lock())
        };

        let extra = self.auxiliary_exact_turn.as_ref().map_or([0; 3], |aux| aux.extra_nodes);
        [
            num_action_nodes[0],
            num_action_nodes[1] * turn_coef as u64 + extra[1],
            num_action_nodes[2] * river_coef as u64 + extra[2],
        ]
    }

    /// Computes the memory usage of this struct.
    #[inline]
    fn memory_usage_internal(&self) -> u64 {
        // untracked: tree_config, action_root

        let mut memory_usage = mem::size_of::<Self>() as u64;

        memory_usage += vec_memory_usage(&self.added_lines);
        memory_usage += vec_memory_usage(&self.removed_lines);
        for line in &self.added_lines {
            memory_usage += vec_memory_usage(line);
        }
        for line in &self.removed_lines {
            memory_usage += vec_memory_usage(line);
        }

        memory_usage += vec_memory_usage(&self.valid_indices_turn);
        memory_usage += vec_memory_usage(&self.valid_indices_river);
        memory_usage += vec_memory_usage(&self.hand_strength);
        if let Some(matrix) = &self.flop_allin_payout_matrix {
            for values in &matrix.values {
                memory_usage += vec_memory_usage(values);
            }
            for values in &matrix.values_transposed {
                memory_usage += vec_memory_usage(values);
            }
            for values in &matrix.values_f32 {
                memory_usage += vec_memory_usage(values);
            }
            for values in &matrix.values_transposed_f32 {
                memory_usage += vec_memory_usage(values);
            }
        }
        for matrix in self.turn_allin_payout_matrices.values() {
            for values in &matrix.values {
                memory_usage += vec_memory_usage(values);
            }
            for values in &matrix.values_transposed {
                memory_usage += vec_memory_usage(values);
            }
            for values in &matrix.values_f32 {
                memory_usage += vec_memory_usage(values);
            }
            for values in &matrix.values_transposed_f32 {
                memory_usage += vec_memory_usage(values);
            }
        }
        memory_usage += vec_memory_usage(&self.isomorphism_ref_turn);
        memory_usage += vec_memory_usage(&self.isomorphism_card_turn);
        memory_usage += vec_memory_usage(&self.isomorphism_ref_river);

        for refs in &self.isomorphism_ref_river {
            memory_usage += vec_memory_usage(refs);
        }

        for cards in &self.isomorphism_card_river {
            memory_usage += vec_memory_usage(cards);
        }

        for player in 0..2 {
            memory_usage += vec_memory_usage(&self.initial_weights[player]);
            memory_usage += vec_memory_usage(&self.private_cards[player]);
            memory_usage += vec_memory_usage(&self.same_hand_index[player]);
            memory_usage += vec_memory_usage(&self.valid_indices_flop[player]);
            for indices in &self.valid_indices_turn {
                memory_usage += vec_memory_usage(&indices[player]);
            }
            for indices in &self.valid_indices_river {
                memory_usage += vec_memory_usage(&indices[player]);
            }
            for strength in &self.hand_strength {
                memory_usage += vec_memory_usage(&strength[player]);
            }
            for swap in &self.isomorphism_swap_turn {
                memory_usage += vec_memory_usage(&swap[player]);
            }
            for swap_list in &self.isomorphism_swap_river {
                for swap in swap_list {
                    memory_usage += vec_memory_usage(&swap[player]);
                }
            }
        }

        memory_usage += vec_memory_usage(&self.node_arena);

        memory_usage
    }

    /// Builds the game tree recursively.
    fn build_tree_recursive(
        &self,
        node_index: usize,
        action_node: &ActionTreeNode,
        info: &mut BuildTreeInfo,
    ) {
        let mut node = self.node_arena[node_index].lock();
        node.player = action_node.player;
        node.amount = action_node.amount;

        if node.is_terminal() {
            return;
        }

        if EXACT_ALLIN_MATRIX_TREE.load(Ordering::Relaxed)
            && node.is_chance()
            && node.amount == self.tree_config.effective_stack
        {
            return;
        }

        if node.is_chance() {
            self.push_chances(node_index, info);
            for action_index in 0..node.num_actions() {
                let child_index = node_index + node.children_offset as usize + action_index;
                self.build_tree_recursive(child_index, &action_node.children[0].lock(), info);
            }
        } else {
            self.push_actions(node_index, action_node, info);
            for action_index in 0..node.num_actions() {
                let child_index = node_index + node.children_offset as usize + action_index;
                let flop_action = self.auxiliary_exact_turn.is_some() && node.turn == NOT_DEALT;
                if flop_action { info.flop_path.push(action_node.actions[action_index]); }
                self.build_tree_recursive(
                    child_index,
                    &action_node.children[action_index].lock(),
                    info,
                );
                if flop_action { info.flop_path.pop(); }
            }
        }
    }

    /// Pushes the chance actions to the `node`.
    fn push_chances(&self, node_index: usize, info: &mut BuildTreeInfo) {
        let mut node = self.node_arena[node_index].lock();
        let flop = self.card_config.flop;
        let flop_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);

        // deal turn
        if node.turn == NOT_DEALT {
            let skip_cards = &self.isomorphism_card_turn;
            let skip_mask: u64 = skip_cards.iter().map(|&card| 1 << card).sum();

            node.children_offset = (info.turn_index - node_index) as u32;
            let selected13 = basis13_turn_cards();
            let selected6 = basis6_turn_cards();
            let turn_basis_count = TURN_BASIS_COUNT.load(Ordering::Relaxed);
            let selected = (turn_basis_count > 0).then(|| selected_turn_cards(turn_basis_count));
            for card in 0..52 {
                if (1 << card) & (flop_mask | skip_mask) == 0 {
                    if (BASIS13_TURN_TREE.load(Ordering::Relaxed) && !selected13.contains(&card))
                        || (BASIS6_TURN_TREE.load(Ordering::Relaxed) && !selected6.contains(&card))
                        || selected
                            .as_ref()
                            .is_some_and(|cards| !cards.contains(&card))
                    {
                        continue;
                    }
                    node.num_children += 1;
                    let mut child = node.children().last().unwrap().lock();
                    child.prev_action = Action::Chance(card);
                    child.turn = card;
                }
            }

            // Append, rather than sort, so the original six model row indices
            // stay identical at every frontier (even for a low extra card).
            if let Some(card) = self.auxiliary_at_path(&info.flop_path) {
                node.num_children += 1;
                let mut child = node.children().last().unwrap().lock();
                child.prev_action = Action::Chance(card);
                child.turn = card;
            }

            info.turn_index += node.num_children as usize;
        }
        // deal river
        else {
            let turn_mask = flop_mask | (1 << node.turn);
            let skip_cards = &self.isomorphism_card_river[node.turn as usize & 3];
            let skip_mask: u64 = skip_cards.iter().map(|&card| 1 << card).sum();

            node.children_offset = (info.river_index - node_index) as u32;
            // A full tree does not need a one-per-rank experimental basis;
            // on a quads turn no card remains for that rank.
            let selected13 = if BASIS13_RIVER_TREE.load(Ordering::Relaxed) {
                basis13_river_cards(flop,node.turn)
            }else{[NOT_DEALT;13]};
            let selected6 = basis6_river_cards();
            let river_basis_count = RIVER_BASIS_COUNT.load(Ordering::Relaxed);
            let selected_reduced =
                (river_basis_count > 0).then(|| selected_river_cards(river_basis_count));
            for card in 0..52 {
                if (1 << card) & (turn_mask | skip_mask) == 0 {
                    if !self.is_auxiliary_turn(node.turn) && ((BASIS13_RIVER_TREE.load(Ordering::Relaxed) && !selected13.contains(&card))
                        || (BASIS6_RIVER_TREE.load(Ordering::Relaxed) && !selected6.contains(&card))
                        || selected_reduced
                            .as_ref()
                            .is_some_and(|selected| !selected.contains(&card)))
                    {
                        continue;
                    }
                    node.num_children += 1;
                    let mut child = node.children().last().unwrap().lock();
                    child.prev_action = Action::Chance(card);
                    child.turn = node.turn;
                    child.river = card;
                }
            }

            info.river_index += node.num_children as usize;
        }

        node.num_elements = node
            .cfvalue_storage_player()
            .map_or(0, |player| self.num_private_hands(player)) as u32;

        info.num_storage_chance += node.num_elements as u64;
    }

    /// Pushes the actions to the `node`.
    fn push_actions(
        &self,
        node_index: usize,
        action_node: &ActionTreeNode,
        info: &mut BuildTreeInfo,
    ) {
        let mut node = self.node_arena[node_index].lock();

        let street = match (node.turn, node.river) {
            (NOT_DEALT, _) => BoardState::Flop,
            (_, NOT_DEALT) => BoardState::Turn,
            _ => BoardState::River,
        };

        let base = match street {
            BoardState::Flop => &mut info.flop_index,
            BoardState::Turn => &mut info.turn_index,
            BoardState::River => &mut info.river_index,
        };

        node.children_offset = (*base - node_index) as u32;
        node.num_children = action_node.children.len() as u16;
        *base += node.num_children as usize;

        for (child, action) in node.children().iter().zip(action_node.actions.iter()) {
            let mut child = child.lock();
            child.prev_action = *action;
            child.turn = node.turn;
            child.river = node.river;
        }

        let num_private_hands = self.num_private_hands(node.player as usize);
        node.num_elements = (node.num_actions() * num_private_hands) as u32;
        node.num_elements_ip = match node.prev_action {
            Action::None | Action::Chance(_) => self.num_private_hands(PLAYER_IP as usize) as u16,
            _ => 0,
        };

        info.num_storage += node.num_elements as u64;
        info.num_storage_ip += node.num_elements_ip as u64;
    }

    /// Sets the bunching effect.
    fn set_bunching_effect_internal(&mut self, bunching_data: &BunchingData) -> Result<(), String> {
        self.bunching_num_dead_cards = bunching_data.fold_ranges().len() * 2;
        let mut arena = vec![0.0]; // store dummy element

        // hand strength
        self.bunching_strength = self
            .hand_strength
            .iter()
            .map(|strength| {
                if strength[0].is_empty() {
                    return [Vec::new(), Vec::new()];
                }

                let mut ret = [
                    vec![0; self.num_private_hands(0)],
                    vec![0; self.num_private_hands(1)],
                ];

                for player in 0..2 {
                    let len = strength[player].len();
                    for &item in &strength[player][1..len - 1] {
                        ret[player][item.index as usize] = item.strength;
                    }
                }

                ret
            })
            .collect();

        // flop num combinations
        if self.card_config.turn == NOT_DEALT {
            for player in 0..2 {
                let player_cards = &self.private_cards[player];
                let opponent_cards = &self.private_cards[player ^ 1];
                let mut indices = Vec::with_capacity(player_cards.len());

                for &(c1, c2) in player_cards {
                    indices.push(arena.len());
                    let player_mask: u64 = (1 << c1) | (1 << c2);

                    for &(c3, c4) in opponent_cards {
                        let opponent_mask: u64 = (1 << c3) | (1 << c4);
                        if player_mask & opponent_mask != 0 {
                            arena.push(0.0);
                        } else {
                            let mask = player_mask | opponent_mask;
                            arena.push(bunching_data.result_4cards(mask));
                        }
                    }
                }

                if player == 0 {
                    self.bunching_num_combinations = arena.iter().fold(0.0, |a, &x| a + x as f64);
                    if self.bunching_num_combinations == 0.0 {
                        self.reset_bunching_effect();
                        return Err("Valid combination not found".to_string());
                    }
                }

                self.bunching_num_flop[player] = indices;
            }
        }

        let flop_mask: u64 = self.card_config.flop.iter().map(|&c| 1 << c).sum();
        let skip_turn_mask: u64 = self.isomorphism_card_turn.iter().map(|&c| 1 << c).sum();

        // turn num combinations
        if self.card_config.river == NOT_DEALT {
            for player in 0..2 {
                let player_cards = &self.private_cards[player];
                let opponent_cards = &self.private_cards[player ^ 1];

                let buf = into_par_iter(0..52)
                    .map(|turn| {
                        let bit_turn: u64 = 1 << turn;
                        if bit_turn & (flop_mask | skip_turn_mask) != 0
                            || (self.card_config.turn != NOT_DEALT
                                && self.card_config.turn != turn as Card)
                        {
                            return Vec::new();
                        }

                        let mut outer = Vec::with_capacity(player_cards.len());

                        for &(c1, c2) in player_cards {
                            let player_mask: u64 = (1 << c1) | (1 << c2);
                            if player_mask & bit_turn != 0 {
                                outer.push(Vec::new());
                                continue;
                            }

                            let mut inner = Vec::with_capacity(opponent_cards.len());
                            for &(c3, c4) in opponent_cards {
                                let opponent_mask: u64 = (1 << c3) | (1 << c4);
                                if (player_mask | bit_turn) & opponent_mask != 0 {
                                    inner.push(0.0);
                                } else {
                                    let mask = player_mask | opponent_mask | bit_turn;
                                    inner.push(bunching_data.result_5cards(mask));
                                }
                            }

                            outer.push(inner);
                        }

                        outer
                    })
                    .collect::<Vec<_>>();

                self.bunching_num_turn[player] = Self::push_vec_to_arena_f32(&mut arena, buf);

                if self.card_config.turn != NOT_DEALT && player == 0 {
                    self.bunching_num_combinations = arena.iter().fold(0.0, |a, &x| a + x as f64);
                    if self.bunching_num_combinations == 0.0 {
                        self.reset_bunching_effect();
                        return Err("Valid combination not found".to_string());
                    }
                }
            }
        }

        let is_board_possible = |turn: Card, river: Card| {
            let bit_turn: u64 = 1 << turn;
            let bit_river: u64 = 1 << river;
            let iso_card = &self.isomorphism_card_river[turn as usize & 3];

            bit_turn & (flop_mask | skip_turn_mask) == 0
                && bit_river & flop_mask == 0
                && !iso_card.contains(&river)
                && (self.card_config.turn == NOT_DEALT || self.card_config.turn == turn)
                && (self.card_config.river == NOT_DEALT || self.card_config.river == river)
        };

        // river num combinations
        for player in 0..2 {
            let player_cards = &self.private_cards[player];
            let opponent_cards = &self.private_cards[player ^ 1];

            let buf = into_par_iter(0..52 * 51 / 2)
                .map(|index| {
                    let (board1, board2) = index_to_card_pair(index);
                    if !is_board_possible(board1, board2) && !is_board_possible(board2, board1) {
                        return Vec::new();
                    }

                    let board_mask: u64 = (1 << board1) | (1 << board2);
                    let mut outer = Vec::with_capacity(player_cards.len());

                    for &(c1, c2) in player_cards {
                        let player_mask: u64 = (1 << c1) | (1 << c2);
                        if player_mask & board_mask != 0 {
                            outer.push(Vec::new());
                            continue;
                        }

                        let mut inner = Vec::with_capacity(opponent_cards.len());
                        for &(c3, c4) in opponent_cards {
                            let opponent_mask: u64 = (1 << c3) | (1 << c4);
                            if (player_mask | board_mask) & opponent_mask != 0 {
                                inner.push(0.0);
                            } else {
                                let mask = player_mask | opponent_mask | board_mask;
                                inner.push(bunching_data.result_6cards(mask));
                            }
                        }

                        outer.push(inner);
                    }

                    outer
                })
                .collect::<Vec<_>>();

            self.bunching_num_river[player] = Self::push_vec_to_arena_f32(&mut arena, buf);

            if self.card_config.river != NOT_DEALT && player == 0 {
                self.bunching_num_combinations = arena.iter().fold(0.0, |a, &x| a + x as f64);
                if self.bunching_num_combinations == 0.0 {
                    self.reset_bunching_effect();
                    return Err("Valid combination not found".to_string());
                }
            }
        }

        if self.card_config.river != NOT_DEALT {
            self.bunching_arena = arena;
            self.assign_zero_weights();
            return Ok(());
        }

        // turn equity coefficients
        for player in 0..2 {
            let player_cards = &self.private_cards[player];
            let opponent_cards = &self.private_cards[player ^ 1];
            let player_len = player_cards.len();
            let opponent_len = opponent_cards.len();

            let buf = into_par_iter(0..52)
                .map(|turn| {
                    let bit_turn: u64 = 1 << turn;
                    if bit_turn & (flop_mask | skip_turn_mask) != 0
                        || (self.card_config.turn != NOT_DEALT
                            && self.card_config.turn != turn as Card)
                    {
                        return Vec::new();
                    }

                    let mut outer = Vec::with_capacity(player_len);

                    for &(c1, c2) in player_cards {
                        let player_mask: u64 = (1 << c1) | (1 << c2);
                        if player_mask & bit_turn != 0 {
                            outer.push(Vec::new());
                        } else {
                            outer.push(vec![0.0; opponent_len]);
                        }
                    }

                    let mut children = Vec::with_capacity(48);
                    let iso_ref = &self.isomorphism_ref_river[turn];
                    let iso_card = &self.isomorphism_card_river[turn & 3];
                    let iso_swap = &self.isomorphism_swap_river[turn & 3];

                    for river in 0..52 {
                        let bit_river: u64 = 1 << river;
                        if bit_river & (flop_mask | bit_turn) != 0 {
                            continue;
                        }

                        let pos = iso_card.iter().position(|&c| c == river);
                        let (river_ref, swap_option) = if let Some(pos) = pos {
                            let child_index = iso_ref[pos] as usize;
                            let swap = &iso_swap[river as usize & 3];
                            (children[child_index], Some(swap))
                        } else {
                            children.push(river);
                            (river, None)
                        };

                        let player_swap = swap_option.map(|swap| {
                            let mut tmp = (0..player_len).collect::<Vec<_>>();
                            apply_swap(&mut tmp, &swap[player]);
                            tmp
                        });

                        let pair_index = card_pair_to_index(turn as Card, river_ref);
                        let arena_indices = &self.bunching_num_river[player][pair_index];
                        let player_strength = &self.bunching_strength[pair_index][player];
                        let opponent_strength = &self.bunching_strength[pair_index][player ^ 1];

                        for (i, inner) in outer.iter_mut().enumerate() {
                            let player_index = player_swap.as_ref().map_or(i, |map| map[i]);
                            let index = arena_indices[player_index];
                            let threshold = player_strength[player_index];

                            if index == 0 {
                                continue;
                            }

                            let mut tmp = (Vec::new(), Vec::new());
                            let slices = if let Some(swap) = swap_option {
                                tmp.0.extend_from_slice(&arena[index..index + opponent_len]);
                                tmp.1.extend_from_slice(opponent_strength);
                                apply_swap(&mut tmp.0, &swap[player ^ 1]);
                                apply_swap(&mut tmp.1, &swap[player ^ 1]);
                                (tmp.0.as_slice(), &tmp.1)
                            } else {
                                (&arena[index..index + opponent_len], opponent_strength)
                            };

                            inner.iter_mut().zip(slices.0).zip(slices.1).for_each(
                                |((dst, num), &strength)| {
                                    #[allow(clippy::comparison_chain)]
                                    if strength < threshold {
                                        *dst += *num as f64;
                                    } else if strength > threshold {
                                        *dst += -*num as f64;
                                    } else {
                                        *dst += 0.0;
                                    }
                                },
                            );
                        }
                    }

                    let num_possible_river = (44 - self.bunching_num_dead_cards) as f64;
                    outer.iter_mut().for_each(|inner| {
                        inner.iter_mut().for_each(|c| {
                            *c /= num_possible_river;
                        });
                    });

                    outer
                })
                .collect::<Vec<_>>();

            self.bunching_coef_turn[player] = Self::push_vec_to_arena_f64(&mut arena, buf);
        }

        if self.card_config.turn != NOT_DEALT {
            self.bunching_arena = arena;
            self.assign_zero_weights();
            return Ok(());
        }

        // flop equity coefficients
        for player in 0..2 {
            let player_cards = &self.private_cards[player];
            let opponent_cards = &self.private_cards[player ^ 1];
            let player_len = player_cards.len();
            let opponent_len = opponent_cards.len();

            let mut outer = vec![vec![0.0; opponent_len]; player_len];
            let mut children = Vec::with_capacity(49);

            for turn in 0..52 {
                let bit_turn: u64 = 1 << turn;
                if bit_turn & flop_mask != 0 {
                    continue;
                }

                let iso_card = &self.isomorphism_card_turn;
                let pos = iso_card.iter().position(|&c| c == turn as Card);

                let (turn_ref, swap_option) = if let Some(pos) = pos {
                    let child_index = self.isomorphism_ref_turn[pos] as usize;
                    let swap = &self.isomorphism_swap_turn[turn & 3];
                    (children[child_index], Some(swap))
                } else {
                    children.push(turn);
                    (turn, None)
                };

                let player_swap = swap_option.map(|swap| {
                    let mut tmp = (0..player_len).collect::<Vec<_>>();
                    apply_swap(&mut tmp, &swap[player]);
                    tmp
                });

                let arena_indices = &self.bunching_coef_turn[player][turn_ref];

                for (i, inner) in outer.iter_mut().enumerate() {
                    let player_index = player_swap.as_ref().map_or(i, |map| map[i]);
                    let index = arena_indices[player_index];
                    if index == 0 {
                        continue;
                    }

                    let mut tmp = Vec::new();
                    let slice = &arena[index..index + opponent_len];
                    let slice = if let Some(swap) = swap_option {
                        tmp.extend_from_slice(slice);
                        apply_swap(&mut tmp, &swap[player ^ 1]);
                        &tmp
                    } else {
                        slice
                    };

                    inner.iter_mut().zip(slice).for_each(|(dst, &num)| {
                        *dst += num as f64;
                    });
                }
            }

            let num_possible_turn = (45 - self.bunching_num_dead_cards) as f64;
            outer.iter_mut().for_each(|inner| {
                inner.iter_mut().for_each(|c| {
                    *c /= num_possible_turn;
                });
            });

            Self::push_vec_to_arena_f64(&mut arena, vec![outer])
                .into_iter()
                .for_each(|v| {
                    self.bunching_coef_flop[player] = v;
                });
        }

        self.bunching_arena = arena;
        self.assign_zero_weights();
        Ok(())
    }

    /// Sets the bunching effect.
    fn memory_usage_bunching_internal(&self) -> u64 {
        let mut ret = 4;

        let oop_len = self.num_private_hands(0);
        let ip_len = self.num_private_hands(1);

        // hand strength
        self.hand_strength.iter().for_each(|strength| {
            ret += mem::size_of::<[Vec<u16>; 2]>() as u64;
            if !strength[0].is_empty() {
                ret += 2 * (oop_len + ip_len) as u64;
            }
        });

        // flop num combinations / equity coefficients
        if self.card_config.turn == NOT_DEALT {
            ret += 2 * (oop_len * ip_len * mem::size_of::<usize>()) as u64;
            ret += 2 * 2 * 4 * (oop_len * ip_len) as u64;
        }

        let flop_mask: u64 = self.card_config.flop.iter().map(|&c| 1 << c).sum();
        let skip_turn_mask: u64 = self.isomorphism_card_turn.iter().map(|&c| 1 << c).sum();

        // turn num combinations / equity coefficients
        if self.card_config.river == NOT_DEALT {
            for player in 0..2 {
                let player_cards = &self.private_cards[player];
                let opponent_cards = &self.private_cards[player ^ 1];
                let player_len = player_cards.len();
                let opponent_len = opponent_cards.len();

                ret += 2 * 52 * mem::size_of::<Vec<usize>>() as u64;

                for turn in 0..52 {
                    let bit_turn: u64 = 1 << turn;
                    if bit_turn & (flop_mask | skip_turn_mask) != 0
                        || (self.card_config.turn != NOT_DEALT
                            && self.card_config.turn != turn as Card)
                    {
                        continue;
                    }

                    ret += 2 * (player_len * mem::size_of::<usize>()) as u64;

                    for &(c1, c2) in player_cards {
                        let player_mask: u64 = (1 << c1) | (1 << c2);
                        if player_mask & bit_turn == 0 {
                            ret += 2 * 4 * opponent_len as u64;
                        }
                    }
                }
            }
        }

        let is_board_possible = |turn: Card, river: Card| {
            let bit_turn: u64 = 1 << turn;
            let bit_river: u64 = 1 << river;
            let iso_card = &self.isomorphism_card_river[turn as usize & 3];

            bit_turn & (flop_mask | skip_turn_mask) == 0
                && bit_river & flop_mask == 0
                && !iso_card.contains(&river)
                && (self.card_config.turn == NOT_DEALT || self.card_config.turn == turn)
                && (self.card_config.river == NOT_DEALT || self.card_config.river == river)
        };

        // river num combinations
        for player in 0..2 {
            let player_cards = &self.private_cards[player];
            let opponent_cards = &self.private_cards[player ^ 1];
            let player_len = player_cards.len();
            let opponent_len = opponent_cards.len();

            ret += 52 * 51 / 2 * mem::size_of::<Vec<usize>>() as u64;

            for index in 0..52 * 51 / 2 {
                let (board1, board2) = index_to_card_pair(index);
                if !is_board_possible(board1, board2) && !is_board_possible(board2, board1) {
                    continue;
                }

                let board_mask: u64 = (1 << board1) | (1 << board2);
                ret += (player_len * mem::size_of::<usize>()) as u64;

                for &(c1, c2) in player_cards {
                    let player_mask: u64 = (1 << c1) | (1 << c2);
                    if player_mask & board_mask == 0 {
                        ret += 4 * opponent_len as u64;
                    }
                }
            }
        }

        ret
    }

    /// Pushes a nested `Vec` to an arena and returns the indices of the pushed elements.
    fn push_vec_to_arena_f32(arena: &mut Vec<f32>, vec: Vec<Vec<Vec<f32>>>) -> Vec<Vec<usize>> {
        let mut ret = Vec::with_capacity(vec.len());

        for outer in vec.into_iter() {
            let mut indices = Vec::with_capacity(outer.len());

            for inner in outer.into_iter() {
                if inner.is_empty() {
                    indices.push(0);
                } else {
                    indices.push(arena.len());
                    arena.extend(inner);
                }
            }

            ret.push(indices);
        }

        ret
    }

    /// Pushes a nested `Vec` to an arena and returns the indices of the pushed elements.
    fn push_vec_to_arena_f64(arena: &mut Vec<f32>, vec: Vec<Vec<Vec<f64>>>) -> Vec<Vec<usize>> {
        let mut ret = Vec::with_capacity(vec.len());

        for outer in vec.into_iter() {
            let mut indices = Vec::with_capacity(outer.len());

            for inner in outer.into_iter() {
                if inner.is_empty() {
                    indices.push(0);
                } else {
                    indices.push(arena.len());
                    arena.extend(inner.into_iter().map(|v| v as f32));
                }
            }

            ret.push(indices);
        }

        ret
    }

    /// Calculates the number of storage elements that will be removed.
    fn calculate_removed_line_info_recursive(node: &mut PostFlopNode, info: &mut BuildTreeInfo) {
        if node.is_terminal() {
            return;
        }

        if node.is_chance() {
            info.num_storage_chance += node.num_elements as u64;
            node.num_elements = 0;
        } else {
            info.num_storage += node.num_elements as u64;
            info.num_storage_ip += node.num_elements_ip as u64;
            node.num_elements = 0;
            node.num_elements_ip = 0;
        }

        for action in node.action_indices() {
            Self::calculate_removed_line_info_recursive(&mut node.play(action), info);
        }
    }

    /// Remove a line from a `PostFlopGame` tree.
    fn remove_line_recursive(
        &self,
        node: &mut PostFlopNode,
        line: &[Action],
    ) -> Result<BuildTreeInfo, String> {
        if line.is_empty() {
            return Err("Empty line".to_string());
        }

        if node.is_terminal() {
            return Err("Unexpected terminal node".to_string());
        }

        let action = line[0];
        let search_result = node
            .children()
            .binary_search_by(|child| child.lock().prev_action.cmp(&action));

        if search_result.is_err() {
            return Err(format!("Action does not exist: {action:?}"));
        }

        let index = search_result.unwrap();
        if line.len() > 1 {
            let result = self.remove_line_recursive(&mut node.children()[index].lock(), &line[1..]);
            return result;
        }

        if node.is_chance() {
            return Err("Cannot remove a line ending in a chance action".to_string());
        }

        if node.num_actions() <= 1 {
            return Err("Cannot remove the last action from a node".to_string());
        }

        // Remove action/children at index. To do this we must
        // 1. compute the storage space required by the tree rooted at action index
        // 2. remove children and actions
        // 3. re-define `num_elements` after we remove children and actions

        // STEP 1
        let mut info = BuildTreeInfo {
            num_storage: self.num_private_hands(node.player as usize) as u64,
            ..Default::default()
        };

        let mut node_to_remove = node.play(index);
        Self::calculate_removed_line_info_recursive(&mut node_to_remove, &mut info);

        // STEP 2
        let children = node.children();
        for i in index..node.num_children as usize - 1 {
            let mut x = children[i].lock();
            let mut y = children[i + 1].lock();
            mem::swap(&mut *x, &mut *y);
            if x.children_offset > 0 {
                x.children_offset += 1;
            }
        }
        node.num_children -= 1;

        // STEP 3
        node.num_elements -= self.num_private_hands(node.player as usize) as u32;

        Ok(info)
    }

    /// Allocates memory recursively.
    fn allocate_memory_nodes(&mut self) {
        let num_bytes = if self.is_compression_enabled { 2 } else { 4 };
        let mut action_counter = 0;
        let mut ip_counter = 0;
        let mut chance_counter = 0;

        for node in &self.node_arena {
            let mut node = node.lock();
            if node.is_terminal() {
                // do nothing
            } else if node.is_chance() {
                unsafe {
                    let ptr = self.storage_chance.as_mut_ptr();
                    node.storage1 = ptr.add(chance_counter);
                }
                chance_counter += num_bytes * node.num_elements as usize;
            } else {
                unsafe {
                    let ptr1 = self.storage1.as_mut_ptr();
                    let ptr2 = self.storage2.as_mut_ptr();
                    let ptr3 = self.storage_ip.as_mut_ptr();
                    node.storage1 = ptr1.add(action_counter);
                    node.storage2 = ptr2.add(action_counter);
                    node.storage3 = ptr3.add(ip_counter);
                }
                action_counter += num_bytes * node.num_elements as usize;
                ip_counter += num_bytes * node.num_elements_ip as usize;
            }
        }
    }
}

#[cfg(test)]
mod physical_basis_tests {
    use super::*;
    use crate::BetSizeOptions;

    #[test]
    fn pruned_card_tables_match_full_selected_runouts() {
        let turns_all = ["3d", "4h", "6d", "9c", "Tc", "Ks"];
        let rivers_all = ["2d", "5d", "7s", "8c", "Th", "Qc"];
        let cards = CardConfig {
            range: ["AA,AK".parse().unwrap(), "KK,KQ".parse().unwrap()],
            flop: crate::flop_from_str("QsJh2h").unwrap(), turn: NOT_DEALT, river: NOT_DEALT,
        };
        let sizes = BetSizeOptions::try_from(("52%", "45%")).unwrap();
        let config = TreeConfig { initial_state: BoardState::Flop, starting_pot:180, effective_stack:910,
            flop_bet_sizes:[sizes.clone(),sizes.clone()], turn_bet_sizes:[sizes.clone(),sizes.clone()],
            river_bet_sizes:[sizes.clone(),sizes], ..Default::default() };
        set_exact_allin_matrix_tree(true);
        for count in [1,2,4,6] {
            let turns:Vec<_>=turns_all[..count].iter().map(|c|card_from_str(c).unwrap()).collect();
            let rivers:Vec<_>=rivers_all[..count].iter().map(|c|card_from_str(c).unwrap()).collect();
            set_turn_basis_count(count); set_river_basis_count(count);
            set_custom_turn_basis_cards(&turns); set_custom_river_basis_cards(&rivers);
            let full=PostFlopGame::with_config(cards.clone(),ActionTree::new(config.clone()).unwrap()).unwrap();
            let mut reduced=PostFlopGame::with_pruned_card_tables_experiment(cards.clone(),ActionTree::new(config.clone()).unwrap()).unwrap();
            assert_eq!(reduced.prepared_showdown_runouts_experiment(),count*count);
            assert_eq!(full.node_arena.len(),reduced.node_arena.len());
            assert_eq!(full.same_hand_index,reduced.same_hand_index);
            assert_eq!(full.valid_indices_flop,reduced.valid_indices_flop);
            for &turn in &turns {
                assert_eq!(full.valid_indices_turn[turn as usize],reduced.valid_indices_turn[turn as usize]);
                for &river in &rivers {
                    let index=card_pair_to_index(turn,river);
                    assert!(full.hand_strength[index]==reduced.hand_strength[index]);
                    assert_eq!(full.valid_indices_river[index],reduced.valid_indices_river[index]);
                }
            }
            assert!(reduced.isomorphism_swap_turn.iter().flatten().all(|v|v.is_empty()));
            assert!(reduced.isomorphism_ref_river.iter().all(|v|v.is_empty()));
            reduced.materialize_analysis_tables_experiment();
            assert_eq!(reduced.prepared_showdown_runouts_experiment(),1176);
            assert!(reduced.hand_strength==full.hand_strength);
        }
        set_turn_basis_count(0); set_river_basis_count(0);
        set_custom_turn_basis_cards(&[]); set_custom_river_basis_cards(&[]);
        set_exact_allin_matrix_tree(false);
        assert!(PostFlopGame::with_pruned_card_tables_experiment(cards,ActionTree::new(config).unwrap()).is_err());
    }

    #[test]
    fn explicit_turn_basis_overrides_historical_count_selection() {
        let expected = ["3c", "Kd"].map(|card| card_from_str(card).unwrap());
        set_custom_turn_basis_cards(&expected);
        assert_eq!(selected_turn_cards(2), expected);
        set_custom_turn_basis_cards(&[]);
    }

    #[test]
    fn protected_chance_cards_reindex_sources_without_losing_multiplicity(){
        let flop=crate::flop_from_str("QsJh2h").unwrap();
        let mut omitted:Vec<Card>=(0..13).map(|r|4*r+1).collect();
        let legal:Vec<Card>=(0..52).filter(|c|!flop.contains(c)).collect();
        let original:Vec<Card>=legal.iter().copied().filter(|c|!omitted.contains(c)).collect();
        let mut refs:Vec<u8>=omitted.iter().map(|c|original.iter().position(|x|*x==*c-1).unwrap() as u8).collect();
        let protected=[crate::card_from_str("3d").unwrap(),crate::card_from_str("6d").unwrap()];
        retain_required_chances(flop,NOT_DEALT,&mut refs,&mut omitted,&protected);
        assert_eq!(omitted.len(),11);assert!(!omitted.iter().any(|c|protected.contains(c)));
        let kept:Vec<Card>=legal.iter().copied().filter(|c|!omitted.contains(c)).collect();
        assert_eq!(kept.len()+omitted.len(),49);
        for (&c,&reference) in omitted.iter().zip(&refs){assert_eq!(kept[reference as usize],c-1);}
    }

    #[test]
    fn explicit_river_basis_overrides_historical_count_selection() {
        let expected = ["2s", "4c", "8d", "Qc"].map(|card| card_from_str(card).unwrap());
        set_custom_river_basis_cards(&expected);
        assert_eq!(selected_river_cards(4), expected);
        set_custom_river_basis_cards(&[]);
        assert_eq!(selected_river_cards(1), vec![card_from_str("Qc").unwrap()]);
    }
}
