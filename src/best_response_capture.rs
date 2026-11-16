//! Optional exact fixed-opponent BR frontier capture; never changes solver state.
use crate::interface::*;
use crate::sliceop::*;
use crate::utility::*;
use std::{collections::BTreeMap, mem::MaybeUninit, sync::Mutex};

/// One chance frontier under a single, coherent fixed-opponent BR profile.
/// Hand vectors use the game's private-hand ordering. Reaches are before the
/// next chance-factor division; sampled CFVs include that division, just as CFR.
#[derive(Debug, Clone)]
pub struct BestResponseFrontier {
    pub responder: usize,
    pub node_id: u64,
    pub path: Vec<u32>,
    pub turn_card: u8,
    pub chance_cards: Vec<u8>,
    pub reaches: [Vec<f32>; 2],
    /// All physical child rows, action-major, for each output player.
    pub sampled_cfvs: [Vec<f32>; 2],
    /// FP64 sum of the physical FP32 child vectors; not another sampled card.
    pub aggregate_cfvs: [Vec<f64>; 2],
}

/// Find a BR to the frozen *average* opponent strategy using the existing exact
/// evaluator, then stream both-player frontier values/reaches under that profile.
/// Responder node locks remain constraints; ties select the first action.
/// Covers every physical chance card; exact suit-isomorphic representatives
/// are expanded back to physical card/hand coordinates before capture.
/// Neural frontier predictions are never consulted. Exact all-ins remain exact.
/// The callback can filter/sample rows and write them without retaining a dataset.
pub fn capture_best_response_frontiers<T: Game, F: FnMut(&BestResponseFrontier)>(
    game: &T,
    responder: usize,
    mut capture: F,
) -> [Vec<f32>; 2] {
    assert!(responder < 2);
    assert!(game.is_ready() || game.is_solved());
    assert!(
        !game.has_auxiliary_subgame(),
        "BR capture requires a complete physical tree"
    );
    let policies = Mutex::new(BTreeMap::new());
    let timing = std::env::var("PFS_BR_CAPTURE_TIMINGS").as_deref() == Ok("1");
    let calculation_started = std::time::Instant::now();
    let mut reference = vec![MaybeUninit::uninit(); game.num_private_hands(responder)];
    compute_best_cfv_with_policy_recursive(
        &mut reference,
        game,
        &game.root(),
        responder,
        game.initial_weights(responder ^ 1),
        Some(&policies),
    );
    let calculation_seconds = calculation_started.elapsed().as_secs_f64();
    let policies = policies.into_inner().unwrap();
    let reaches = [
        game.initial_weights(0).to_vec(),
        game.initial_weights(1).to_vec(),
    ];
    let traversal_started = std::time::Instant::now();
    let mut callback_seconds = 0.0;
    let mut timed_capture = |record: &BestResponseFrontier| {
        if timing {
            let started = std::time::Instant::now();
            capture(record);
            callback_seconds += started.elapsed().as_secs_f64();
        } else {
            capture(record);
        }
    };
    let values = visit(
        game,
        &game.root(),
        responder,
        &policies,
        &reaches,
        &mut Vec::new(),
        &mut timed_capture,
    );
    let traversal_seconds = traversal_started.elapsed().as_secs_f64();
    if timing {
        println!("BR_FRONTIER_TIMINGS responder={responder} calculation_seconds={calculation_seconds:.9} profile_traversal_seconds={:.9} dump_callback_seconds={callback_seconds:.9} traversal_and_dump_seconds={traversal_seconds:.9}",
            traversal_seconds - callback_seconds);
    }
    for (&got, expected) in values[responder].iter().zip(reference) {
        let expected = unsafe { expected.assume_init() };
        assert!(
            (got - expected).abs() <= 2e-5 * expected.abs().max(1.),
            "Recorded BR profile disagrees with exact evaluator: {got} vs {expected}"
        );
    }
    values
}

// Called during the BR pass for all-in chance frontiers whose legacy
// capture expands showdown cards even when the native evaluator short-circuits.
pub(crate) fn prepare_turn_allin_cache<T:Game>(game:&T,node:&T::Node,player:usize,cfreach:&[f32],cache:&Mutex<BTreeMap<u64,TurnBrCache>>)->Vec<f32>{
 let mut reaches=[game.initial_weights(0).to_vec(),game.initial_weights(1).to_vec()];reaches[player^1]=cfreach.to_vec();
 let values=visit(game,node,player,&BTreeMap::new(),&reaches,&mut Vec::new(),&mut |r:&BestResponseFrontier|{
  if r.turn_card==crate::NOT_DEALT{assert!(cache.lock().unwrap().insert(r.node_id,TurnBrCache{cards:r.chance_cards.clone(),samples:r.sampled_cfvs[player].clone(),aggregate:r.aggregate_cfvs[player].clone()}).is_none());}
 });values[player].clone()
}

/// Turn-only BR capture. The full BR pass retains exact frontier vectors;
/// the following reach/dump pass stops at those frontiers. Only the responder's
/// CFVs/root vector are provided; the other player's CFV arrays are empty.
/// All reaches retain both players. Historical general capture is unchanged.
pub fn capture_best_response_turn_frontiers<T:Game,F:FnMut(&BestResponseFrontier)>(game:&T,responder:usize,mut capture:F)->Vec<f32>{
    assert!(responder<2);assert!(!game.has_auxiliary_subgame());
    let policies=Mutex::new(BTreeMap::new());let cache=Mutex::new(BTreeMap::new());
    let mut native=vec![MaybeUninit::uninit();game.num_private_hands(responder)];let mut shadow=native.clone();
    let start=std::time::Instant::now();
    compute_best_cfv_with_turn_cache_recursive(&mut native,game,&game.root(),responder,game.initial_weights(responder^1),Some(&policies),&mut shadow,&cache);
    let seconds=start.elapsed().as_secs_f64();let policies=policies.into_inner().unwrap();let cache=cache.into_inner().unwrap();
    let reaches=[game.initial_weights(0).to_vec(),game.initial_weights(1).to_vec()];let start=std::time::Instant::now();
    let values=visit_turn_cache(game,&game.root(),responder,&policies,&cache,&reaches,&mut Vec::new(),&mut capture);
    for ((&v,n),s) in values.iter().zip(native).zip(shadow){let n=unsafe{n.assume_init()};let s=unsafe{s.assume_init()};assert_eq!(v.to_bits(),s.to_bits(),"flop-only cache traversal changed CFVs");assert!((v-n).abs()<=2e-5*n.abs().max(1.));}
    if std::env::var("PFS_BR_CAPTURE_TIMINGS").as_deref()==Ok("1"){println!("BR_TURN_CACHE_TIMINGS responder={responder} full_BR_and_cache_seconds={seconds:.9} flop_reach_dump_seconds={:.9} cached_frontiers={}",start.elapsed().as_secs_f64(),cache.len());}
    values
}
fn visit_turn_cache<T:Game,F:FnMut(&BestResponseFrontier)>(game:&T,node:&T::Node,responder:usize,policies:&BTreeMap<u64,Vec<u32>>,cache:&BTreeMap<u64,TurnBrCache>,reaches:&[Vec<f32>;2],path:&mut Vec<u32>,capture:&mut F)->Vec<f32>{
    assert_eq!(node.board_turn(),crate::NOT_DEALT,"dump traversal entered a turn child");
    let n=game.num_private_hands(responder);
    if node.is_terminal()||node.is_chance(){
        let mut out=vec![MaybeUninit::uninit();n];
        let handled=if node.is_terminal(){game.evaluate(&mut out,node,responder,&reaches[responder^1]);true}else{game.evaluate_called_allin(&mut out,node,responder,&reaches[responder^1])};
        if handled && (node.is_terminal()||node.num_actions()==0){return out.into_iter().map(|x|unsafe{x.assume_init()}).collect();}
    }
    if node.is_chance(){
        let saved=&cache[&node.id()];let mut samples=[Vec::new(),Vec::new()];let mut aggregate=[Vec::new(),Vec::new()];samples[responder]=saved.samples.clone();aggregate[responder]=saved.aggregate.clone();
        capture(&BestResponseFrontier{responder,node_id:node.id(),path:path.clone(),turn_card:crate::NOT_DEALT,chance_cards:saved.cards.clone(),reaches:reaches.clone(),sampled_cfvs:samples,aggregate_cfvs:aggregate});
        return saved.aggregate.iter().map(|&x|x as f32).collect();
    }
    let count=node.num_actions();let actor=node.player();let hands=game.num_private_hands(actor);
    let mut strategy=if count==1{vec![1.;hands]}else if actor==responder{
        let mut s=vec![0.;count*hands];for (h,&a) in policies[&node.id()].iter().enumerate(){s[a as usize*hands+h]=1.;}s
    }else if game.is_compression_enabled(){normalized_strategy_compressed(node.strategy_compressed(),count)}else{normalized_strategy(node.strategy(),count)};
    apply_locking_strategy(&mut strategy,game.locking_strategy(node));let mut sums=vec![0f64;n];
    for action in 0..count{
        let probabilities=row(&strategy,action,hands);let mut child_reaches=reaches.clone();for (r,&p) in child_reaches[actor].iter_mut().zip(probabilities){*r*=p;}
        path.push(action as u32);let child=visit_turn_cache(game,&node.play(action),responder,policies,cache,&child_reaches,path,capture);path.pop();
        for (h,&v) in child.iter().enumerate(){sums[h]+=v as f64*if actor==responder{probabilities[h] as f64}else{1.};}
    }
    sums.into_iter().map(|x|x as f32).collect()
}

fn visit<T: Game, F: FnMut(&BestResponseFrontier)>(
    game: &T,
    node: &T::Node,
    responder: usize,
    policies: &BTreeMap<u64, Vec<u32>>,
    reaches: &[Vec<f32>; 2],
    path: &mut Vec<u32>,
    capture: &mut F,
) -> [Vec<f32>; 2] {
    let mut values = [
        vec![0.; game.num_private_hands(0)],
        vec![0.; game.num_private_hands(1)],
    ];
    if node.is_terminal() || node.is_chance() {
        let mut handled = [false; 2];
        for player in 0..2 {
            let mut out = vec![MaybeUninit::uninit(); values[player].len()];
            if node.is_terminal() {
                game.evaluate(&mut out, node, player, &reaches[player ^ 1]);
                handled[player] = true;
            } else {
                handled[player] =
                    game.evaluate_called_allin(&mut out, node, player, &reaches[player ^ 1]);
            }
            if handled[player] {
                for (dst, src) in values[player].iter_mut().zip(out) {
                    *dst = unsafe { src.assume_init() };
                }
            }
        }
        assert_eq!(handled[0], handled[1]);
        if handled[0] && (node.is_terminal() || node.num_actions() == 0) {
            return values;
        }
    }
    let count = node.num_actions();
    let mut sums = [vec![0f64; values[0].len()], vec![0f64; values[1].len()]];
    if node.is_chance() {
        let cards = node.chance_cards();
        assert_eq!(cards.len(), count);
        assert_eq!(
            count+game.isomorphic_chances(node).len(),
            if node.board_turn() == crate::NOT_DEALT {
                49
            } else {
                48
            },
            "BR capture refuses a physically pruned chance tree"
        );
        let factor = 1. / game.chance_factor(node) as f32;
        let child_reaches = [
            reaches[0].iter().map(|r| r * factor).collect(),
            reaches[1].iter().map(|r| r * factor).collect(),
        ];
        let mut samples = [Vec::new(), Vec::new()];
        for action in 0..count {
            path.push(action as u32);
            let child = visit(
                game,
                &node.play(action),
                responder,
                policies,
                &child_reaches,
                path,
                capture,
            );
            path.pop();
            for player in 0..2 {
                for (sum, &v) in sums[player].iter_mut().zip(&child[player]) {
                    *sum += v as f64;
                }
                samples[player].extend_from_slice(&child[player]);
            }
        }
        let mut physical_cards=cards;
        if !game.isomorphic_chances(node).is_empty(){
            for player in 0..2{
                let (cards,expanded)=game.expand_physical_chance_values(node,player,&samples[player]);
                physical_cards=cards;sums[player].fill(0.);
                let n=values[player].len();
                for row in expanded.chunks_exact(n){for (sum,&v) in sums[player].iter_mut().zip(row){*sum+=v as f64;}}
                samples[player]=expanded;
            }
        }
        capture(&BestResponseFrontier {
            responder,
            node_id: node.id(),
            path: path.clone(),
            turn_card: node.board_turn(),
            chance_cards: physical_cards,
            reaches: reaches.clone(),
            sampled_cfvs: samples,
            aggregate_cfvs: sums.clone(),
        });
    } else {
        let actor = node.player();
        let hands = values[actor].len();
        let mut strategy = if count == 1 {
            vec![1.; hands]
        } else if actor == responder {
            let choices = &policies[&node.id()];
            assert_eq!(choices.len(), hands);
            let mut out = vec![0.; hands * count];
            for (hand, &action) in choices.iter().enumerate() {
                out[action as usize * hands + hand] = 1.;
            }
            out
        } else if game.is_compression_enabled() {
            normalized_strategy_compressed(node.strategy_compressed(), count)
        } else {
            normalized_strategy(node.strategy(), count)
        };
        apply_locking_strategy(&mut strategy, game.locking_strategy(node));
        for action in 0..count {
            let probabilities = row(&strategy, action, hands);
            let mut child_reaches = reaches.clone();
            for (r, &p) in child_reaches[actor].iter_mut().zip(probabilities) {
                *r *= p;
            }
            path.push(action as u32);
            let child = visit(
                game,
                &node.play(action),
                responder,
                policies,
                &child_reaches,
                path,
                capture,
            );
            path.pop();
            for player in 0..2 {
                for (hand, &v) in child[player].iter().enumerate() {
                    sums[player][hand] += v as f64
                        * if actor == player {
                            probabilities[hand] as f64
                        } else {
                            1.
                        };
                }
            }
        }
    }
    for player in 0..2 {
        for (out, &v) in values[player].iter_mut().zip(&sums[player]) {
            *out = v as f32;
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    fn fingerprint(node: &PostFlopNode, out: &mut Vec<u32>) {
        if node.is_terminal() {
            return;
        }
        if !node.is_chance() {
            out.extend(
                node.strategy()
                    .iter()
                    .chain(node.regrets())
                    .map(|v| v.to_bits()),
            );
        }
        for a in 0..node.num_actions() {
            fingerprint(&node.play(a), out);
        }
    }
    #[test]
    fn best_response_capture_matches_mes_and_preserves_solver_state() {
        for compressed in [false, true] {
            let cards = CardConfig {
                range: [
                    "AA,AKs,QTs,88:0.5".parse().unwrap(),
                    "KK,AQs,JTs,99:0.4".parse().unwrap(),
                ],
                flop: flop_from_str("QsJh2h").unwrap(),
                turn: card_from_str("3d").unwrap(),
                river: NOT_DEALT,
            };
            let sizes = BetSizeOptions::try_from(("50%", "")).unwrap();
            let cfg = TreeConfig {
                initial_state: BoardState::Turn,
                starting_pot: 180,
                effective_stack: 150,
                turn_bet_sizes: [sizes.clone(), sizes.clone()],
                river_bet_sizes: [sizes.clone(), sizes],
                add_allin_threshold: 0.,
                force_allin_threshold: 0.,
                ..Default::default()
            };
            let mut game = PostFlopGame::with_config(cards, ActionTree::new(cfg).unwrap()).unwrap();
            game.allocate_memory(compressed);
            for i in 0..5 {
                solve_step(&game, i);
            }
            // Full locks must remain constraints even on the responder's nodes.
            let n = game.num_private_hands(0);
            let actions = game.root().num_actions();
            let mut locked = vec![0.; n * actions];
            locked[..n].fill(1.);
            game.lock_current_strategy(&locked);
            let expected = compute_mes_ev(&game);
            let mut before = Vec::new();
            if !compressed {
                fingerprint(&game.root(), &mut before);
            }
            for responder in 0..2 {
                let mut count = 0;
                let root = capture_best_response_frontiers(&game, responder, |c| {
                    count += 1;
                    assert_eq!(c.responder, responder);
                    assert_eq!(c.chance_cards.len(), 48);
                    for p in 0..2 {
                        let n = game.num_private_hands(p);
                        assert_eq!(c.reaches[p].len(), n);
                        assert_eq!(c.sampled_cfvs[p].len(), 48 * n);
                        for h in 0..n {
                            let total = (0..48)
                                .map(|a| c.sampled_cfvs[p][a * n + h] as f64)
                                .sum::<f64>();
                            assert_eq!(total, c.aggregate_cfvs[p][h]);
                        }
                    }
                    let zero_sum = (0..2)
                        .map(|p| {
                            c.aggregate_cfvs[p]
                                .iter()
                                .zip(&c.reaches[p])
                                .map(|(v, r)| v * (*r as f64))
                                .sum::<f64>()
                        })
                        .sum::<f64>();
                    assert!(zero_sum.abs() < 1e-4, "profile not zero-sum: {zero_sum}");
                });
                assert!(count > 0);
                let weighted = root[responder]
                    .iter()
                    .zip(game.initial_weights(responder))
                    .map(|(v, r)| v * r)
                    .sum::<f32>();
                assert!((weighted - expected[responder]).abs() < 1e-4);
                let zero_sum = (0..2)
                    .map(|p| {
                        root[p]
                            .iter()
                            .zip(game.initial_weights(p))
                            .map(|(v, r)| (*v as f64) * (*r as f64))
                            .sum::<f64>()
                    })
                    .sum::<f64>();
                assert!(zero_sum.abs() < 1e-4);
            }
            if !compressed {
                let mut after = Vec::new();
                fingerprint(&game.root(), &mut after);
                assert_eq!(before, after);
            }
        }
    }
    #[test]
    fn turn_cache_matches_legacy_frontiers_with_locks_and_preserves_state(){
        let flop=flop_from_str("Td9d6h").unwrap();let turns:Vec<_>=(0..52).filter(|c|!flop.contains(c)).collect();
        set_physical_chance_isomorphism(true,&turns,&(0..52).collect::<Vec<_>>());set_prune_chance_branches(false);set_turn_basis_count(0);set_river_basis_count(0);
        for (compressed,stack) in [(false,80),(true,80),(false,600)]{
            let cards=CardConfig{range:["AA,KK,AKs:0.3".parse().unwrap(),"QQ,JJ,AQs:0.7".parse().unwrap()],flop,turn:NOT_DEALT,river:NOT_DEALT};
            let sizes:BetSizeOptions=("50%,a","").try_into().unwrap();
            let tree=TreeConfig{initial_state:BoardState::Flop,starting_pot:300,effective_stack:stack,flop_bet_sizes:[sizes.clone(),sizes.clone()],turn_bet_sizes:[sizes.clone(),sizes.clone()],river_bet_sizes:[sizes.clone(),sizes],..Default::default()};
            let mut game=PostFlopGame::with_config(cards,ActionTree::new(tree).unwrap()).unwrap();game.allocate_memory(compressed);for i in 0..4{solve_step(&game,i);}
            let n=game.num_private_hands(0);let count=game.root().num_actions();let mut lock=vec![0.;n*count];lock[..n].fill(1.);game.lock_current_strategy(&lock);
            let mut before=Vec::new();if !compressed{fingerprint(&game.root(),&mut before);}
            for responder in 0..2{
                let mut original=BTreeMap::new();let roots=capture_best_response_frontiers(&game,responder,|r|{if r.turn_card==NOT_DEALT{original.insert(r.node_id,r.clone());}});
                let mut seen=0;let cached=capture_best_response_turn_frontiers(&game,responder,|r|{
                    let old=&original[&r.node_id];assert_eq!(r.path,old.path);assert_eq!(r.chance_cards,old.chance_cards);assert_eq!(r.reaches,old.reaches);assert_eq!(r.sampled_cfvs[responder],old.sampled_cfvs[responder]);assert_eq!(r.aggregate_cfvs[responder],old.aggregate_cfvs[responder]);assert!(r.sampled_cfvs[responder^1].is_empty());seen+=1;
                });assert_eq!(seen,original.len());assert_eq!(cached,roots[responder]);
            }
            if !compressed{let mut after=Vec::new();fingerprint(&game.root(),&mut after);assert_eq!(before,after);}
        }
    }

}
