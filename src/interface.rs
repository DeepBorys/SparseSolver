use crate::mutex_like::*;
use std::mem::MaybeUninit;
use std::ops::Range;

/// The trait representing a game.
pub trait Game: Send + Sync {
    /// The type representing a node in game tree.
    #[doc(hidden)]
    type Node: GameNode;

    /// Returns the root node of game tree.
    #[doc(hidden)]
    fn root(&self) -> MutexGuardLike<Self::Node>;

    /// Returns the number of private hands of given player.
    #[doc(hidden)]
    fn num_private_hands(&self, player: usize) -> usize;

    /// Returns the initial reach probabilities of given player.
    #[doc(hidden)]
    fn initial_weights(&self, player: usize) -> &[f32];

    /// Opt-in prepared terminal-dispatch plan, owned by this game instance.
    #[doc(hidden)]
    fn direct_terminal_plan(&self, _node: &Self::Node) -> Option<std::sync::Arc<crate::CompiledTerminalPlan>> { None }

    /// Direct arena lookup for the terminal-dispatch experiment.
    #[doc(hidden)]
    fn direct_terminal_node(&self, _id: u64) -> Option<MutexGuardLike<Self::Node>> { None }

    /// Computes the counterfactual values of given node.
    #[doc(hidden)]
    fn evaluate(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &Self::Node,
        player: usize,
        cfreach: &[f32],
    );

    /// Evaluates a called all-in chance node through an optional exact
    /// precomputed payout matrix. Returns `true` when the node was handled and
    /// recursion should stop. The default keeps ordinary physical traversal.
    #[doc(hidden)]
    fn evaluate_called_allin(
        &self,
        _result: &mut [MaybeUninit<f32>],
        _node: &Self::Node,
        _player: usize,
        _cfreach: &[f32],
    ) -> bool {
        false
    }

    /// Compares a physically summed called-all-in CFV against an optional
    /// equivalent matrix representation. The physical result remains
    /// authoritative; this hook is diagnostic only.
    #[doc(hidden)]
    fn compare_called_allin(
        &self,
        _result: &[f64],
        _node: &Self::Node,
        _player: usize,
        _cfreach: &[f32],
    ) -> Option<f64> {
        None
    }

    /// Returns the effective number of chances.
    #[doc(hidden)]
    fn chance_factor(&self, node: &Self::Node) -> usize;

    /// Whether this chance node must always be traversed exactly.
    #[doc(hidden)]
    fn should_traverse_chance_exactly(&self, _node: &Self::Node) -> bool {
        false
    }

    /// Returns whether the instance is solved.
    #[doc(hidden)]
    fn is_solved(&self) -> bool;

    /// Sets the instance to be solved.
    #[doc(hidden)]
    fn set_solved(&mut self);

    /// Returns whether the instance is ready to be solved.
    #[doc(hidden)]
    fn is_ready(&self) -> bool {
        true
    }

    /// Returns whether the game is raked.
    #[doc(hidden)]
    fn is_raked(&self) -> bool {
        false
    }

    /// Returns the list of indices that isomorphic chances refer to.
    #[doc(hidden)]
    fn isomorphic_chances(&self, _node: &Self::Node) -> &[u8] {
        &[]
    }

    /// Physical cards omitted by exact suit isomorphism, parallel to references.
    #[doc(hidden)]
    fn isomorphic_cards(&self, _node: &Self::Node) -> &[u8] { &[] }

    /// Reconstruct all physical chance rows without changing representative storage.
    #[doc(hidden)]
    fn expand_physical_chance_values(&self,node:&Self::Node,player:usize,values:&[f32])->(Vec<u8>,Vec<f32>){
        let cards=node.chance_cards();let n=self.num_private_hands(player);
        assert_eq!(values.len(),cards.len()*n);
        let mut rows:Vec<(u8,Vec<f32>)>=cards.iter().enumerate().map(|(i,&c)|(c,values[i*n..(i+1)*n].to_vec())).collect();
        assert_eq!(self.isomorphic_cards(node).len(),self.isomorphic_chances(node).len());
        for (i,(&card,&source)) in self.isomorphic_cards(node).iter().zip(self.isomorphic_chances(node)).enumerate(){
            let mut row=values[source as usize*n..(source as usize+1)*n].to_vec();
            for &(a,b) in &self.isomorphic_swap(node,i)[player]{row.swap(a as usize,b as usize);}
            rows.push((card,row));
        }
        rows.sort_unstable_by_key(|x|x.0);
        (rows.iter().map(|x|x.0).collect(),rows.into_iter().flat_map(|x|x.1).collect())
    }

    /// Returns the swap list of the given isomorphic chance.
    #[doc(hidden)]
    fn isomorphic_swap(&self, _node: &Self::Node, _index: usize) -> &[Vec<(u16, u16)>; 2] {
        unreachable!()
    }

    /// Returns the configured physical-turn diagnostic node.
    #[doc(hidden)]
    fn cfv_capture_node_id(&self) -> u64 {
        4
    }

    /// Whether diagnostics should capture this physical flop-to-turn node.
    #[doc(hidden)]
    fn capture_turn_cfvs(&self, node: &Self::Node) -> bool {
        node.id() == self.cfv_capture_node_id()
    }

    /// Returns optional conversion weights for this physical-turn chance node.
    #[doc(hidden)]
    fn turn_cfv_conversion(
        &self,
        _node: &Self::Node,
    ) -> Option<&crate::pio_order::TurnCfvConversion> {
        None
    }

    /// Optional exact continuation, solved alongside but excluded from the
    /// neural chance aggregate. Never an extra neural basis row.
    #[doc(hidden)]
    fn auxiliary_chance_action(&self, _node: &Self::Node) -> Option<usize> { None }

    #[doc(hidden)]
    fn has_auxiliary_subgame(&self) -> bool { false }

    /// Starts collection for neural river-frontier inputs.
    #[doc(hidden)]
    fn begin_neural_river_batch(&self) -> bool {
        false
    }

    /// Whether normal solving must propagate the updating player's reach.
    #[doc(hidden)]
    fn requires_player_reach(&self) -> bool {
        false
    }

    /// Starts collection for neural turn-frontier inputs.
    #[doc(hidden)]
    fn begin_neural_turn_batch(&self) -> bool {
        false
    }

    /// Runs the collected requests as one batch and enables cached consumption.
    #[doc(hidden)]
    fn finish_neural_turn_batch(&self, _player: usize) {}

    /// Clears cached predictions after the shallow update pass.
    #[doc(hidden)]
    fn end_neural_turn_batch(&self) {}

    /// Returns the locking strategy.
    #[doc(hidden)]
    fn locking_strategy(&self, _node: &Self::Node) -> &[f32] {
        &[]
    }

    /// Returns whether the compression is enabled.
    #[doc(hidden)]
    fn is_compression_enabled(&self) -> bool {
        false
    }
}

/// The trait representing a node in game tree.
pub trait GameNode: Send + Sync {
    /// Returns whether the node is terminal.
    #[doc(hidden)]
    fn is_terminal(&self) -> bool;

    /// Returns the immutable deterministic tree-construction ID of this node.
    #[doc(hidden)]
    fn id(&self) -> u64 {
        0
    }

    /// Returns whether the node is chance.
    #[doc(hidden)]
    fn is_chance(&self) -> bool;

    /// Returns the current player.
    #[doc(hidden)]
    fn player(&self) -> usize;

    /// Returns the number of actions.
    #[doc(hidden)]
    fn num_actions(&self) -> usize;

    /// Returns the dealt turn card, or `NOT_DEALT` at a flop chance node.
    #[doc(hidden)]
    fn board_turn(&self) -> u8 {
        crate::card::NOT_DEALT
    }

    /// Returns the dealt river card, or `NOT_DEALT` before the river.
    #[doc(hidden)]
    fn board_river(&self) -> u8 {
        crate::card::NOT_DEALT
    }

    /// Returns physical cards in chance-action order.
    #[doc(hidden)]
    fn chance_cards(&self) -> Vec<u8> {
        Vec::new()
    }

    /// Returns the node after taking the given action.
    #[doc(hidden)]
    fn play(&self, action: usize) -> MutexGuardLike<Self>;

    /// Returns the strategy.
    #[doc(hidden)]
    fn strategy(&self) -> &[f32];

    /// Returns the mutable reference to the strategy.
    #[doc(hidden)]
    fn strategy_mut(&mut self) -> &mut [f32];

    /// Returns the cumulative regrets.
    #[doc(hidden)]
    fn regrets(&self) -> &[f32];

    /// Returns the mutable reference to the cumulative regrets.
    #[doc(hidden)]
    fn regrets_mut(&mut self) -> &mut [f32];

    /// Returns the counterfactual values.
    #[doc(hidden)]
    fn cfvalues(&self) -> &[f32];

    /// Returns the mutable reference to the counterfactual values.
    #[doc(hidden)]
    fn cfvalues_mut(&mut self) -> &mut [f32];

    /// Returns whether IP's counterfactual values are stored.
    #[doc(hidden)]
    fn has_cfvalues_ip(&self) -> bool {
        false
    }

    /// Returns IP's counterfactual values.
    #[doc(hidden)]
    fn cfvalues_ip(&self) -> &[f32] {
        unreachable!()
    }

    /// Returns the mutable reference to IP's counterfactual values.
    #[doc(hidden)]
    fn cfvalues_ip_mut(&mut self) -> &mut [f32] {
        unreachable!()
    }

    /// Returns the player whose counterfactual values are stored (for chance node).
    #[doc(hidden)]
    fn cfvalue_storage_player(&self) -> Option<usize> {
        None
    }

    /// Returns the buffer for counterfactual values.
    #[doc(hidden)]
    fn cfvalues_chance(&self) -> &[f32] {
        unreachable!()
    }

    /// Returns the mutable reference to the buffer for counterfactual values.
    #[doc(hidden)]
    fn cfvalues_chance_mut(&mut self) -> &mut [f32] {
        unreachable!()
    }

    /// Returns the [`Range`] struct of actions.
    #[doc(hidden)]
    fn action_indices(&self) -> Range<usize> {
        0..self.num_actions()
    }

    /// Returns the compressed strategy.
    #[doc(hidden)]
    fn strategy_compressed(&self) -> &[u16] {
        unreachable!()
    }

    /// Returns the mutable reference to the compressed strategy.
    #[doc(hidden)]
    fn strategy_compressed_mut(&mut self) -> &mut [u16] {
        unreachable!()
    }

    /// Returns the compressed cumulative regrets.
    #[doc(hidden)]
    fn regrets_compressed(&self) -> &[i16] {
        unreachable!()
    }

    /// Returns the mutable reference to the compressed cumulative regrets.
    #[doc(hidden)]
    fn regrets_compressed_mut(&mut self) -> &mut [i16] {
        unreachable!()
    }

    /// Returns the compressed counterfactual values.
    #[doc(hidden)]
    fn cfvalues_compressed(&self) -> &[i16] {
        unreachable!()
    }

    /// Returns the mutable reference to the compressed counterfactual values.
    #[doc(hidden)]
    fn cfvalues_compressed_mut(&mut self) -> &mut [i16] {
        unreachable!()
    }

    /// Returns IP's compressed counterfactual values.
    #[doc(hidden)]
    fn cfvalues_ip_compressed(&self) -> &[i16] {
        unreachable!()
    }

    /// Returns the mutable reference to IP's compressed counterfactual values.
    #[doc(hidden)]
    fn cfvalues_ip_compressed_mut(&mut self) -> &mut [i16] {
        unreachable!()
    }

    /// Returns the compressed buffer for counterfactual values.
    #[doc(hidden)]
    fn cfvalues_chance_compressed(&self) -> &[i16] {
        unreachable!()
    }

    /// Returns the mutable reference to the compressed buffer for counterfactual values.
    #[doc(hidden)]
    fn cfvalues_chance_compressed_mut(&mut self) -> &mut [i16] {
        unreachable!()
    }

    /// Returns the scale of the compressed strategy.
    #[doc(hidden)]
    fn strategy_scale(&self) -> f32 {
        unreachable!()
    }

    /// Sets the scale of the compressed strategy.
    #[doc(hidden)]
    fn set_strategy_scale(&mut self, _scale: f32) {
        unreachable!()
    }

    /// Returns the scale of the compressed cumulative regrets.
    #[doc(hidden)]
    fn regret_scale(&self) -> f32 {
        unreachable!()
    }

    /// Sets the scale of the compressed cumulative regrets.
    #[doc(hidden)]
    fn set_regret_scale(&mut self, _scale: f32) {
        unreachable!()
    }

    /// Returns the scale of the compressed counterfactual values.
    #[doc(hidden)]
    fn cfvalue_scale(&self) -> f32 {
        unreachable!()
    }

    /// Sets the scale of the compressed counterfactual values.
    #[doc(hidden)]
    fn set_cfvalue_scale(&mut self, _scale: f32) {
        unreachable!()
    }

    /// Returns the scale of the compressed counterfactual values for IP.
    #[doc(hidden)]
    fn cfvalue_ip_scale(&self) -> f32 {
        unreachable!()
    }

    /// Sets the scale of the compressed counterfactual values for IP.
    #[doc(hidden)]
    fn set_cfvalue_ip_scale(&mut self, _scale: f32) {
        unreachable!()
    }

    /// Returns the scale of the compressed buffer for counterfactual values.
    #[doc(hidden)]
    fn cfvalue_chance_scale(&self) -> f32 {
        unreachable!()
    }

    /// Sets the scale of the compressed buffer for counterfactual values.
    #[doc(hidden)]
    fn set_cfvalue_chance_scale(&mut self, _scale: f32) {
        unreachable!()
    }

    /// Hint for parallelization. By default, it is set to `false`.
    #[doc(hidden)]
    fn enable_parallelization(&self) -> bool {
        false
    }
}
