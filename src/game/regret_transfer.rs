//! Prepared, signed-bit-preserving K=1 transfers within one allocated tree.
use super::*;
use crate::interface::*;
#[cfg(feature="rayon")]
use rayon::prelude::*;

/// One destination decision, with one source decision/hand for each legal hand.
/// `None` is required exactly for hands blocked by the destination board.
pub struct K1RegretNodeCopy {
    pub target_node: u64,
    pub source_hands: Vec<Option<(u64, usize)>>,
}

struct Task { offset:usize, indices:Vec<u32> }

/// Bound to the same live, uncompressed tree storage used at preparation.
/// Source decisions are disjoint from every destination and remain read-only.
pub struct K1RegretTransferPlan {
    tasks:Vec<Task>,
    storage:usize,
    arena:usize,
    bytes:usize,
    values:usize,
    legal_values:usize,
}

extern "C" {
    fn pfs_k1_regret_gather(base:*const f32,dst:*mut f32,indices:*const u32,count:usize);
    fn pfs_k1_regret_gather_backend()->*const std::ffi::c_char;
}

impl K1RegretTransferPlan {
    pub fn destination_nodes(&self)->usize{self.tasks.len()}
    pub fn destination_values(&self)->usize{self.values}
    pub fn legal_destination_values(&self)->usize{self.legal_values}
    pub fn index_bytes(&self)->usize{self.values*4}
    pub fn backend(&self)->&'static str{
        // SAFETY: the C++ routine returns a static, NUL-terminated literal.
        unsafe{std::ffi::CStr::from_ptr(pfs_k1_regret_gather_backend()).to_str().unwrap()}
    }
}

impl PostFlopGame {
    /// Commit solver buffer pages during setup, before the smaller seed phase.
    /// Volatile read/write preserves every bit and prevents lazy-zero page
    /// faults from being charged to the later transfer or average reset.
    pub fn prefault_warm_start_storage(&mut self){
        assert!(self.state==State::MemoryAllocated&&!self.is_compression_enabled);
        for buffer in [&mut self.storage1,&mut self.storage2]{
            for index in (0..buffer.len()).step_by(4096){
                // SAFETY: index is in this exclusively borrowed initialized byte buffer.
                unsafe{let p=buffer.as_mut_ptr().add(index);std::ptr::write_volatile(p,std::ptr::read_volatile(p));}
            }
            if let Some(last)=buffer.last_mut(){
                unsafe{std::ptr::write_volatile(last,std::ptr::read_volatile(last));}
            }
        }
    }

    /// Compile and validate a K=1 transfer while traversal is stopped.
    /// Betting-history/action identities must be matched by the caller; this
    /// additionally validates actor, chip amount, source/target legality and
    /// all read/write ranges. No source node may also be a destination.
    pub fn prepare_k1_regret_transfer(&self,copies:Vec<K1RegretNodeCopy>)->Result<K1RegretTransferPlan,String>{
        if self.state!=State::MemoryAllocated||self.is_compression_enabled{return Err("K1 requires an allocated, unsolved FP32 game".into());}
        if !self.locking_strategy.is_empty(){return Err("K1 does not support locked strategies".into());}
        let base=self.storage2.as_ptr() as usize;
        if base%4!=0||self.storage2.len()%4!=0||self.storage2.len()/4>i32::MAX as usize{return Err("K1 storage alignment/size unsupported".into());}
        let targets:std::collections::BTreeSet<_>=copies.iter().map(|c|c.target_node as usize).collect();
        if targets.len()!=copies.len(){return Err("Duplicate K1 destination".into());}
        let mut tasks=Vec::new();let mut values=0;let mut legal_values=0;
        let offset=|node:&PostFlopNode|->Result<usize,String>{
            let p=node.storage2 as usize;
            if p<base||p%4!=0||p+node.num_elements as usize*4>base+self.storage2.len(){return Err("Node outside K1 storage".into());}
            Ok((p-base)/4)
        };
        for copy in copies {
            let target=self.node_arena.get(copy.target_node as usize).ok_or("Unknown K1 target")?.lock();
            if target.is_chance()||target.is_terminal(){return Err("K1 target must be a decision".into());}
            let p=target.player();let n=self.private_cards[p].len();
            if copy.source_hands.len()!=n{return Err("K1 hand count mismatch".into());}
            let mut sources=Vec::with_capacity(n);
            for (h,mapping) in copy.source_hands.iter().enumerate(){
                let (a,b)=self.private_cards[p][h];let blocked=[target.turn,target.river].iter().any(|&c|c==a||c==b);
                if blocked!=mapping.is_none(){return Err("K1 target legality mismatch".into());}
                if let Some((source_id,hand))=*mapping {
                    if targets.contains(&(source_id as usize)){return Err("K1 source is also a destination".into());}
                    let source=self.node_arena.get(source_id as usize).ok_or("Unknown K1 source")?.lock();
                    if source.is_chance()||source.is_terminal()||source.player()!=p||source.num_actions()!=target.num_actions()||source.amount!=target.amount{return Err("K1 node actor/action/chip mismatch".into());}
                    let &(a,b)=self.private_cards[p].get(hand).ok_or("Unknown K1 source hand")?;
                    if [source.turn,source.river].iter().any(|&c|c==a||c==b){return Err("K1 source hand is blocked".into());}
                    sources.push(Some((offset(&source)?,hand)));
                }else{sources.push(None);}
            }
            let mut indices=Vec::with_capacity(target.num_elements as usize);
            for action in 0..target.num_actions(){
                for source in &sources {
                    indices.push(if let Some((start,h))=source{legal_values+=1;(start+action*n+h) as u32}else{u32::MAX});
                }
            }
            if indices.len()!=target.num_elements as usize{return Err("K1 action storage mismatch".into());}
            values+=indices.len();tasks.push(Task{offset:offset(&target)?,indices});
        }
        tasks.sort_unstable_by_key(|t|t.offset);
        if tasks.windows(2).any(|w|w[0].offset+w[0].indices.len()>w[1].offset){return Err("Overlapping K1 destinations".into());}
        Ok(K1RegretTransferPlan{tasks,storage:base,arena:self.node_arena.as_ptr() as usize,bytes:self.storage2.len(),values,legal_values})
    }

    /// Transfer in place. The prepared plan contains every index; the timed
    /// kernel allocates nothing, performs no lookup and changes no source.
    pub fn apply_k1_regret_transfer(&mut self,plan:&K1RegretTransferPlan){
        assert!(self.state==State::MemoryAllocated&&!self.is_compression_enabled);
        assert_eq!((self.storage2.as_ptr() as usize,self.node_arena.as_ptr() as usize,self.storage2.len()),(plan.storage,plan.arena,plan.bytes),"K1 plan belongs to different storage");
        let base=self.storage2.as_mut_ptr() as usize;
        let apply=|task:&Task|{
            // SAFETY: preparation proves every index lies inside initialized
            // FP32 storage; destinations are pairwise disjoint and no source
            // node is written. &mut self excludes concurrent solver updates.
            unsafe{pfs_k1_regret_gather(base as *const f32,(base as *mut f32).add(task.offset),task.indices.as_ptr(),task.indices.len());}
        };
        #[cfg(feature="rayon")]
        plan.tasks.par_iter().with_min_len(16).for_each(apply);
        #[cfg(not(feature="rayon"))]
        plan.tasks.iter().for_each(apply);
    }

    /// Fresh averaged strategies at a warm-start boundary. Regrets, clock and
    /// discount/reset schedule are untouched; chance display caches are reset.
    pub fn reset_average_strategy_for_warm_start(&mut self){
        assert!(self.state==State::MemoryAllocated&&!self.is_compression_enabled);
        self.storage1.fill(0);
        self.back_to_root();
    }

    /// Diagnostic fingerprint of flop/turn signed regrets, excluding rivers.
    pub fn diagnostic_flop_turn_regret_fingerprint(&self)->(u64,usize){
        use std::hash::{Hash,Hasher};
        let mut hash=std::collections::hash_map::DefaultHasher::new();let mut count=0;
        for entry in &self.node_arena{let n=entry.lock();if n.is_terminal()||n.is_chance()||n.river!=NOT_DEALT{continue;}
            n.id().hash(&mut hash);for x in n.regrets(){x.to_bits().hash(&mut hash);count+=1;}
        }
        (hash.finish(),count)
    }

    /// Diagnostic snapshot of the floor-zero current regret-matched policy.
    /// Regrets remain untouched; evaluation normalizes these positive weights.
    pub fn snapshot_regret_policy_for_evaluation(&mut self){
        assert!(self.state==State::MemoryAllocated&&!self.is_compression_enabled);
        for entry in &self.node_arena {
            let mut node=entry.lock();
            if node.is_terminal() || node.is_chance(){continue;}
            let weights:Vec<f32>=node.regrets().iter().map(|r|r.max(0.0)).collect();
            node.strategy_mut().copy_from_slice(&weights);
        }
        self.back_to_root();
    }
    /// Discard every river seed, including retained source runouts.
    pub fn clear_river_state_for_warm_start(&mut self)->usize{
        assert!(self.state==State::MemoryAllocated&&!self.is_compression_enabled);
        let mut count=0;
        for entry in &self.node_arena{let mut node=entry.lock();if !node.is_terminal()&&!node.is_chance()&&node.river!=NOT_DEALT{node.regrets_mut().fill(0.);node.strategy_mut().fill(0.);count+=1;}}
        self.back_to_root();count
    }
    /// Retain only the flop seed; all downstream decisions start from zero.
    pub fn clear_turn_and_river_state_for_warm_start(&mut self)->usize{
        assert!(self.state==State::MemoryAllocated&&!self.is_compression_enabled);
        let mut count=0;
        for entry in &self.node_arena{let mut node=entry.lock();if !node.is_terminal()&&!node.is_chance()&&node.turn!=NOT_DEALT{node.regrets_mut().fill(0.);node.strategy_mut().fill(0.);count+=1;}}
        self.back_to_root();count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flop_from_str;
    #[test]
    fn gather_preserves_bits_masks_and_simd_tails(){
        let source:Vec<f32>=(0..80).map(|i|f32::from_bits(match i%5{0=>0x80000000,1=>0xc0800000,2=>0x7fc01234,_=>i as u32+0x3f000000})).collect();
        for length in 0..=65 {
            let ids:Vec<u32>=(0..length).map(|i|if i%7==0{u32::MAX}else{((i*13)%source.len()) as u32}).collect();
            let mut actual=vec![123.;length];
            unsafe{pfs_k1_regret_gather(source.as_ptr(),actual.as_mut_ptr(),ids.as_ptr(),length);}
            for (i,&id) in ids.iter().enumerate(){assert_eq!(actual[i].to_bits(),if id==u32::MAX{0}else{source[id as usize].to_bits()});}
        }
    }
    #[test]
    fn clearing_rivers_keeps_flop_and_turn_regrets(){
        let tree=TreeConfig{initial_state:BoardState::Flop,starting_pot:100,effective_stack:1,
            flop_bet_sizes:[("a","").try_into().unwrap(),("a","").try_into().unwrap()],..Default::default()};
        let cards=CardConfig{range:["AA".parse().unwrap(),"KK".parse().unwrap()],flop:flop_from_str("2c3d4h").unwrap(),turn:NOT_DEALT,river:NOT_DEALT};
        let mut game=PostFlopGame::with_config(cards,ActionTree::new(tree).unwrap()).unwrap();game.allocate_memory(false);
        for entry in &game.node_arena{let mut node=entry.lock();if !node.is_terminal()&&!node.is_chance(){node.regrets_mut().fill(-3.);node.strategy_mut().fill(2.);}}
        assert!(game.clear_river_state_for_warm_start()>0);let mut retained=0;
        for entry in &game.node_arena{let node=entry.lock();if node.is_terminal()||node.is_chance(){continue;}if node.river==NOT_DEALT{assert!(node.regrets().iter().all(|x|*x == -3.));assert!(node.strategy().iter().all(|x|*x==2.));retained+=1;}else{assert!(node.regrets().iter().chain(node.strategy()).all(|x|*x==0.));}}
        assert!(retained>0);
        assert!(game.clear_turn_and_river_state_for_warm_start()>0);
        for entry in &game.node_arena{let node=entry.lock();if node.is_terminal()||node.is_chance(){continue;}if node.turn==NOT_DEALT{assert!(node.regrets().iter().all(|x|*x == -3.));assert!(node.strategy().iter().all(|x|*x==2.));}else{assert!(node.regrets().iter().chain(node.strategy()).all(|x|*x==0.));}}
    }
    #[test]
    fn holding_flop_policy_preserves_regrets_but_learns_downstream(){
        let tree=TreeConfig{initial_state:BoardState::Flop,starting_pot:100,effective_stack:1,
            flop_bet_sizes:[("a","").try_into().unwrap(),("a","").try_into().unwrap()],turn_bet_sizes:[("a","").try_into().unwrap(),("a","").try_into().unwrap()],river_bet_sizes:[("a","").try_into().unwrap(),("a","").try_into().unwrap()],..Default::default()};
        let cards=CardConfig{range:["AA".parse().unwrap(),"KK".parse().unwrap()],flop:flop_from_str("2c3d4h").unwrap(),turn:NOT_DEALT,river:NOT_DEALT};
        let mut game=PostFlopGame::with_config(cards,ActionTree::new(tree).unwrap()).unwrap();game.allocate_memory(false);
        for entry in &game.node_arena{let mut n=entry.lock();if !n.is_terminal()&&!n.is_chance(){n.regrets_mut().fill(1.);}}
        crate::set_diagnostic_hold_flop_policy(true);
        for i in 60..70{crate::solve_step(&game,i);}
        crate::set_diagnostic_hold_flop_policy(false);
        let mut downstream_changed=false;let mut flop_count=0;
        for entry in &game.node_arena{let n=entry.lock();if n.is_terminal()||n.is_chance(){continue;}if n.turn==NOT_DEALT{assert!(n.regrets().iter().all(|&x|x==1.));assert!(n.strategy().iter().any(|&x|x>0.));flop_count+=1;}else{downstream_changed|=n.regrets().iter().any(|&x|x!=1.);}}
        assert!(flop_count>0&&downstream_changed);
        assert!(crate::compute_mes_ev(&game).iter().all(|v|v.is_finite()));
        crate::solve_step(&game,70);
        assert!(game.root().regrets().iter().any(|&x|x!=1.));
    }
    #[test]
    fn holding_flop_turn_policy_preserves_regrets_but_learns_rivers(){
        let tree=TreeConfig{initial_state:BoardState::Flop,starting_pot:100,effective_stack:1,
            flop_bet_sizes:[("a","").try_into().unwrap(),("a","").try_into().unwrap()],turn_bet_sizes:[("a","").try_into().unwrap(),("a","").try_into().unwrap()],river_bet_sizes:[("a","").try_into().unwrap(),("a","").try_into().unwrap()],..Default::default()};
        let cards=CardConfig{range:["AA".parse().unwrap(),"KK".parse().unwrap()],flop:flop_from_str("2c3d4h").unwrap(),turn:NOT_DEALT,river:NOT_DEALT};
        let mut game=PostFlopGame::with_config(cards,ActionTree::new(tree).unwrap()).unwrap();game.allocate_memory(false);
        for entry in &game.node_arena{let mut n=entry.lock();if !n.is_terminal()&&!n.is_chance(){n.regrets_mut().fill(1.);}}
        crate::set_diagnostic_hold_flop_policy(true);crate::set_diagnostic_hold_turn_policy(true);
        let fingerprint=game.diagnostic_flop_turn_regret_fingerprint();
        for i in 60..70{crate::solve_step(&game,i);}
        crate::set_diagnostic_hold_flop_policy(false);crate::set_diagnostic_hold_turn_policy(false);
        assert_eq!(fingerprint,game.diagnostic_flop_turn_regret_fingerprint());
        let mut downstream_changed=false;let mut flop_count=0;
        for entry in &game.node_arena{let n=entry.lock();if n.is_terminal()||n.is_chance(){continue;}if n.river==NOT_DEALT{assert!(n.regrets().iter().all(|&x|x==1.));assert!(n.strategy().iter().any(|&x|x>0.));flop_count+=1;}else{downstream_changed|=n.regrets().iter().any(|&x|x!=1.);}}
        assert!(flop_count>0&&downstream_changed);
        assert!(crate::compute_mes_ev(&game).iter().all(|v|v.is_finite()));
        crate::solve_step(&game,70);
        assert!(game.root().regrets().iter().any(|&x|x!=1.));
    }
    #[test]
    fn current_policy_snapshot_preserves_signed_regrets(){
        let tree=TreeConfig{initial_state:BoardState::River,starting_pot:100,effective_stack:1,
            river_bet_sizes:[("a","").try_into().unwrap(),("a","").try_into().unwrap()],..Default::default()};
        let cards=CardConfig{range:["AA".parse().unwrap(),"KK".parse().unwrap()],flop:flop_from_str("2c3d4h").unwrap(),turn:crate::card_from_str("5s").unwrap(),river:crate::card_from_str("6c").unwrap()};
        let mut game=PostFlopGame::with_config(cards,ActionTree::new(tree).unwrap()).unwrap();game.allocate_memory(false);
        let original={let mut root=game.root();for (i,r) in root.regrets_mut().iter_mut().enumerate(){*r=if i%2==0{-2.0}else{3.0};}root.regrets().to_vec()};
        game.snapshot_regret_policy_for_evaluation();
        {let root=game.root();assert_eq!(root.regrets(),original);assert_eq!(root.strategy(),original.iter().map(|r|r.max(0.0)).collect::<Vec<_>>());}
        assert!(crate::compute_mes_ev(&game).iter().all(|v|v.is_finite()));
    }
    #[test]
    fn reject_a_source_node_that_is_also_a_destination(){
        let tree=TreeConfig{initial_state:BoardState::Flop,starting_pot:100,effective_stack:1,
            flop_bet_sizes:[("a","").try_into().unwrap(),("a","").try_into().unwrap()],..Default::default()};
        let cards=CardConfig{range:["AA".parse().unwrap(),"KK".parse().unwrap()],flop:flop_from_str("2c3d4h").unwrap(),turn:NOT_DEALT,river:NOT_DEALT};
        let mut game=PostFlopGame::with_config(cards,ActionTree::new(tree).unwrap()).unwrap();game.allocate_memory(false);
        let copy=K1RegretNodeCopy{target_node:0,source_hands:(0..game.num_private_hands(0)).map(|h|Some((0,h))).collect()};
        assert!(game.prepare_k1_regret_transfer(vec![copy]).err().unwrap().contains("also a destination"));
    }
}

#[cfg(feature="neural")]
pub struct NeuralPhaseBank {
    turn:Option<crate::pio_order::TurnCfvConversion>,
    rivers:BTreeMap<Card,crate::pio_order::TurnCfvConversion>,
}

#[cfg(feature="neural")]
impl PostFlopGame {
    /// Move the primary model bank without changing game or regret storage.
    /// This pipeline intentionally excludes hybrid/tertiary routing.
    pub fn take_neural_phase_bank(&mut self)->NeuralPhaseBank{
        assert!(self.state==State::MemoryAllocated);
        assert!(self.secondary_turn_cfv_conversion.is_none()&&self.tertiary_turn_cfv_conversion.is_none());
        assert!(self.secondary_river_cfv_conversions.is_empty()&&self.tertiary_river_cfv_conversions.is_empty()&&self.glued_river_modules.is_none());
        NeuralPhaseBank{turn:self.turn_cfv_conversion.take(),rivers:std::mem::take(&mut self.river_cfv_conversions)}
    }
    pub fn swap_neural_phase_bank(&mut self,bank:&mut NeuralPhaseBank){
        assert!(self.state==State::MemoryAllocated);
        std::mem::swap(&mut self.turn_cfv_conversion,&mut bank.turn);
        std::mem::swap(&mut self.river_cfv_conversions,&mut bank.rivers);
    }
}

// 2026-11-10 opt-in corridor diagnostics; leave signed regrets untouched.
impl PostFlopGame {
 pub fn diagnostic_capture_fixed_policies(&self,ids:&[u64])->Vec<(u64,Vec<f32>)>{
  assert!(self.state==State::MemoryAllocated&&!self.is_compression_enabled);
  ids.iter().map(|&id|{let node=self.node_arena[id as usize].lock();assert!(!node.is_terminal()&&!node.is_chance());(id,crate::solver::regret_matching(node.regrets(),node.num_actions(),0.).to_vec())}).collect()
 }
 pub fn diagnostic_accumulate_fixed_averages(&mut self,policies:&[(u64,Vec<f32>)],gamma:f32){
  assert!(self.state==State::MemoryAllocated&&!self.is_compression_enabled);
  for (id,policy) in policies{let mut node=self.node_arena[*id as usize].lock();let average=node.strategy_mut();assert_eq!(average.len(),policy.len());for(x,y)in average.iter_mut().zip(policy){*x=*x*gamma+*y;}}
 }
 pub fn diagnostic_regret_fingerprint_nodes(&self,ids:&[u64])->u64{
  use std::hash::{Hash,Hasher};let mut h=std::collections::hash_map::DefaultHasher::new();
  for &id in ids{let n=self.node_arena[id as usize].lock();id.hash(&mut h);for x in n.regrets(){x.to_bits().hash(&mut h);}}h.finish()
 }
}

// Read-only viewer-hybrid evaluation: swap average buffers, evaluate unrestricted,
// swap back. No regret mutation or node locks (2026-11-10 experiment).
impl PostFlopGame {
 pub fn diagnostic_swap_average_policies(&mut self,policies:&mut[(u64,Vec<f32>)]){
  assert!(self.state==State::MemoryAllocated&&!self.is_compression_enabled);
  for(id,policy)in policies{let mut n=self.node_arena[*id as usize].lock();let a=n.strategy_mut();assert_eq!(a.len(),policy.len());for(x,y)in a.iter_mut().zip(policy.iter_mut()){std::mem::swap(x,y);}}
 }
 pub fn diagnostic_average_fingerprint_nodes(&self,ids:&[u64])->u64{
  use std::hash::{Hash,Hasher};let mut h=std::collections::hash_map::DefaultHasher::new();
  for &id in ids{let n=self.node_arena[id as usize].lock();id.hash(&mut h);for x in n.strategy(){x.to_bits().hash(&mut h);}}h.finish()
 }
}

impl PostFlopGame {
 pub fn diagnostic_all_averages_zero(&self)->bool{self.storage1.iter().all(|&x|x==0)}
}
