//! Opt-in fixed-policy evaluation shortcut for the river-freeze experiment.
use super::*;
use std::collections::BTreeSet;
static ENABLED: AtomicBool = AtomicBool::new(false);
static ROOTS: OnceLock<Mutex<BTreeSet<u64>>> = OnceLock::new();
static CALLS: AtomicU64 = AtomicU64::new(0);
static CURRENT_POLICY: AtomicBool = AtomicBool::new(false);
pub fn set_diagnostic_frozen_river_current_policy(enabled:bool){CURRENT_POLICY.store(enabled,Ordering::Relaxed);}
static CACHE_VALUES: AtomicBool = AtomicBool::new(false);
static CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static VALUES: OnceLock<Mutex<std::collections::BTreeMap<(u64,usize),Vec<f32>>>> = OnceLock::new();
pub fn set_diagnostic_frozen_river_cached_values(enabled: bool) {
    VALUES.get_or_init(Default::default).lock().unwrap().clear();
    CACHE_HITS.store(0,Ordering::Relaxed);
    CACHE_VALUES.store(enabled,Ordering::Relaxed);
}
pub fn diagnostic_frozen_river_value_cache_stats() -> (usize,u64) {
    (VALUES.get_or_init(Default::default).lock().unwrap().len(),CACHE_HITS.load(Ordering::Relaxed))
}
pub fn set_diagnostic_frozen_river_roots(roots: &[u64], direct: bool) {
    *ROOTS.get_or_init(Default::default).lock().unwrap() = roots.iter().copied().collect();
    ENABLED.store(direct && !roots.is_empty(), Ordering::Relaxed);
}
pub fn diagnostic_frozen_river_calls() -> u64 { CALLS.load(Ordering::Relaxed) }
pub(super) fn try_evaluate<T: Game>(out: &mut [MaybeUninit<f32>], game: &T,
    node: &T::Node, player: usize, reach: &[f32]) -> bool {
    if !ENABLED.load(Ordering::Relaxed) || node.is_chance() { return false; }
    if !ROOTS.get().unwrap().lock().unwrap().contains(&node.id()) { return false; }
    if CACHE_VALUES.load(Ordering::Relaxed) {
        let cache=VALUES.get().unwrap().lock().unwrap();
        if let Some(values)=cache.get(&(node.id(),player)) {
            for (out,&value) in out.iter_mut().zip(values) {out.write(value);}
            CACHE_HITS.fetch_add(1,Ordering::Relaxed);
            CALLS.fetch_add(1,Ordering::Relaxed);
            return true;
        }
    }
    fixed(out, game, node, player, reach);
    if CACHE_VALUES.load(Ordering::Relaxed) {
        let values=out.iter().map(|v|unsafe{v.assume_init()}).collect();
        VALUES.get().unwrap().lock().unwrap().insert((node.id(),player),values);
    }
    CALLS.fetch_add(1, Ordering::Relaxed);
    true
}
fn fixed<T: Game>(out: &mut [MaybeUninit<f32>], game: &T, node: &T::Node,
    player: usize, reach: &[f32]) {
    if node.is_terminal() { game.evaluate(out,node,player,reach); return; }
    assert!(!node.is_chance(), "Frozen river subtree must have no chance branches");
    let actions=node.num_actions(); let hands=out.len();
    // With node-only freezing, cumulative averages below the selected root are
    // never updated. Read those policies directly; no per-hand locks are needed.
    let average;
    let policy=if game.locking_strategy(node).is_empty() {
        average=if CURRENT_POLICY.load(Ordering::Relaxed){assert!(!game.is_compression_enabled());regret_matching(node.regrets(),actions,0.).to_vec()}else if game.is_compression_enabled() {
            normalized_strategy_compressed(node.strategy_compressed(),actions)
        }else{normalized_strategy(node.strategy(),actions)};
        average.as_slice()
    }else{game.locking_strategy(node)};
    assert_eq!(policy.len(),actions*game.num_private_hands(node.player()));
    assert!(policy.iter().all(|v| *v>=0.), "Every descendant must be fully locked");
    let mut values=Vec::<f32>::with_capacity(actions*hands);
    for action in 0..actions {
        let target=row_mut(values.spare_capacity_mut(),action,hands);
        if node.player()==player {
            fixed(target,game,&node.play(action),player,reach);
        } else {
            let next:Vec<_>=reach.iter().enumerate().map(|(i,v)|v*policy[action*reach.len()+i]).collect();
            fixed(target,game,&node.play(action),player,&next);
        }
    }
    unsafe { values.set_len(actions*hands); }
    if node.player()==player { fma_slices_uninit(out,policy,&values); }
    else { sum_slices_uninit(out,&values); }
}
