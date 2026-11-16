//! Experimental prepared per-street terminal dispatch. Exact/NN boundaries are
//! evaluated directly by arena ID, followed by a flat backward decision pass.
//! Policies are frozen separately for each alternating player update. Plans
//! belong to the game, contain IDs rather than pointers, and are not serialized.
use super::*;
#[cfg(feature="rayon")]
use rayon::prelude::*;

pub(super) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("PFS_DIRECT_TERMINAL_DISPATCH_EXPERIMENT").as_deref() == Ok("1"))
}

struct Entry {
    id: u64,
    player: usize,
    children: Vec<usize>,
    path: Vec<(usize,usize)>,
    boundary: bool,
}

/// Opaque metadata for an explicitly enabled traversal experiment.
#[doc(hidden)]
pub struct CompiledTerminalPlan {
    entries: Vec<Entry>,
    boundaries: Vec<usize>,
}

impl CompiledTerminalPlan {
    pub fn build<N: GameNode>(root: &N) -> Self {
        fn visit<N: GameNode>(node: &N, path: &mut Vec<(usize,usize)>, plan: &mut CompiledTerminalPlan) -> usize {
            let index=plan.entries.len();
            let boundary=node.is_terminal() || node.is_chance();
            plan.entries.push(Entry { id:node.id(),player:if boundary { 2 } else { node.player() },
                children:Vec::new(),path:if boundary { path.clone() } else { Vec::new() },boundary });
            if boundary { plan.boundaries.push(index); }
            else {
                for action in 0..node.num_actions() {
                    path.push((index,action));
                    let child=visit(&*node.play(action),path,plan);
                    path.pop();plan.entries[index].children.push(child);
                }
            }
            index
        }
        let mut plan=Self { entries:Vec::new(),boundaries:Vec::new() };
        visit(root,&mut Vec::new(),&mut plan);plan
    }
}

// Scalar-index outer loop preserves exact root-to-leaf FP32 order. The fixed
// count cases permit vectorization across hands without intermediate ranges.
fn path_product(initial: &[f32], factors: &[&[f32]]) -> Vec<f32> {
    fn kernel<const N: usize>(initial:&[f32],factors:&[&[f32]],out:&mut [MaybeUninit<f32>]) {
        for (i,result) in out.iter_mut().take(initial.len()).enumerate() {
            let mut value=initial[i];
            for factor in &factors[..N] { value*=factor[i]; }
            result.write(value);
        }
    }
    for f in factors { assert_eq!(f.len(),initial.len()); }
    let mut result=Vec::<f32>::with_capacity(initial.len());
    match factors.len() {
        0 => return initial.to_vec(),
        1 => kernel::<1>(initial,factors,result.spare_capacity_mut()),
        2 => kernel::<2>(initial,factors,result.spare_capacity_mut()),
        3 => kernel::<3>(initial,factors,result.spare_capacity_mut()),
        4 => kernel::<4>(initial,factors,result.spare_capacity_mut()),
        5 => kernel::<5>(initial,factors,result.spare_capacity_mut()),
        6 => kernel::<6>(initial,factors,result.spare_capacity_mut()),
        7 => kernel::<7>(initial,factors,result.spare_capacity_mut()),
        8 => kernel::<8>(initial,factors,result.spare_capacity_mut()),
        _ => for (i,out) in result.spare_capacity_mut().iter_mut().take(initial.len()).enumerate() {
            let mut value=initial[i];for f in factors { value*=f[i]; }out.write(value);
        },
    }
    unsafe { result.set_len(initial.len()); }result
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_solve<T:Game>(result:&mut [MaybeUninit<f32>],game:&T,root:&T::Node,
    player:usize,cfreach:&[f32],own_reach:Option<&[f32]>,params:&DiscountParams,
    iteration:u32,update:bool,below_batch:bool,cached:bool) -> bool {
    let Some(plan)=game.direct_terminal_plan(root) else { return false; };
    let hands=game.num_private_hands(player);
    let opponent_hands=game.num_private_hands(player^1);
    let policies: Vec<Vec<f32>>=plan.entries.iter().map(|entry| {
        if entry.boundary || entry.children.len()==1 { return Vec::new(); }
        let node=game.direct_terminal_node(entry.id).expect("Plan arena ID");
        let mut policy=regret_matching(node.regrets(),entry.children.len(),params.qre_temperature);
        apply_locking_strategy(&mut policy,game.locking_strategy(&node));policy
    }).collect();
    let evaluate = |&index:&usize| {
        let entry=&plan.entries[index];
        let mut node=game.direct_terminal_node(entry.id).expect("Boundary arena ID");
        let factors = |side:usize,n:usize| -> Vec<&[f32]> {
            entry.path.iter().filter_map(|&(parent,action)| {
                let e=&plan.entries[parent];
                (e.player==side && e.children.len()>1).then(|| row(&policies[parent],action,n))
            }).collect()
        };
        let opponent=path_product(cfreach,&factors(player^1,opponent_hands));
        let own=if node.is_terminal() { None } else {
            own_reach.map(|reach| path_product(reach,&factors(player,hands)))
        };
        let mut values=Vec::with_capacity(hands);
        solve_recursive_standard(values.spare_capacity_mut(),game,&mut node,player,&opponent,
            own.as_deref(),params,iteration,update,below_batch,cached);
        unsafe { values.set_len(hands); }(index,values)
    };
    #[cfg(feature="rayon")]
    let boundary_values:Vec<_>=plan.boundaries.par_iter().map(evaluate).collect();
    #[cfg(not(feature="rayon"))]
    let boundary_values:Vec<_>=plan.boundaries.iter().map(evaluate).collect();
    let mut values:Vec<Vec<f32>>=(0..plan.entries.len()).map(|_|Vec::new()).collect();
    for (index,value) in boundary_values { values[index]=value; }
    // The only decision traversal is backward. Parent policies were captured
    // before any updates; sibling subtrees own disjoint nodes and storage.
    for index in (0..plan.entries.len()).rev() {
        let entry=&plan.entries[index];if entry.boundary { continue; }
        RECURSION_COUNTS[1].fetch_add(1,Ordering::Relaxed);
        let actions=entry.children.len();
        if actions==1 { values[index]=std::mem::take(&mut values[entry.children[0]]);continue; }
        let mut action_values=Vec::with_capacity(actions*hands);
        for &child in &entry.children { action_values.extend_from_slice(&values[child]); }
        let mut output=Vec::with_capacity(hands);
        if entry.player==player {
            fma_slices_uninit(output.spare_capacity_mut(),&policies[index],&action_values);
        } else { sum_slices_uninit(output.spare_capacity_mut(),&action_values); }
        unsafe { output.set_len(hands); }
        if entry.player==player && (update || below_batch) {
            let mut node=game.direct_terminal_node(entry.id).expect("Decision arena ID");
            for (x,y) in node.strategy_mut().iter_mut().zip(&policies[index]) { *x=*x*params.gamma_t+*y; }
            if fused_regret_updates_enabled() {
                for (regrets,cfvs) in node.regrets_mut().chunks_exact_mut(hands).zip(action_values.chunks_exact(hands)) {
                    for ((x,y),expected) in regrets.iter_mut().zip(cfvs).zip(&output) {
                        let d=if x.is_sign_positive() {params.alpha_t}else{params.beta_t};
                        let discounted=*x*d+*y;*x=discounted-*expected;
                    }
                }
            } else {
                for (x,y) in node.regrets_mut().iter_mut().zip(&action_values) {
                    *x=*x*if x.is_sign_positive(){params.alpha_t}else{params.beta_t}+*y;
                }
                for chunk in node.regrets_mut().chunks_exact_mut(hands) { sub_slice(chunk,&output); }
            }
        }
        values[index]=output;
    }
    for (out,value) in result.iter_mut().zip(&values[0]) { out.write(*value); }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn direct_terminal_product_preserves_fp32_sequence() {
        let a=[0.,0.3,1.,f32::MIN_POSITIVE];let b=[0.5,0.7,0.125,0.2];let c=[0.8,0.,0.7,0.3];
        let result=path_product(&a,&[&b,&c]);
        for i in 0..a.len() { assert_eq!(result[i].to_bits(),((a[i]*b[i])*c[i]).to_bits()); }
    }
    #[test]
    fn direct_terminal_plan_is_owned_by_game_and_invalidated_on_rebuild() {
        let cards=crate::CardConfig { range:["AsAh,QsQh".parse().unwrap(),"KsKh".parse().unwrap()],
            flop:crate::flop_from_str("2s3h4d").unwrap(),turn:crate::card_from_str("6c").unwrap(),
            river:crate::card_from_str("7c").unwrap() };
        let config=crate::TreeConfig { initial_state:crate::BoardState::River,starting_pot:20,effective_stack:10,
            river_bet_sizes:[("a","").try_into().unwrap(),("a","").try_into().unwrap()],..Default::default() };
        let mut game=crate::PostFlopGame::with_config(cards.clone(),crate::ActionTree::new(config.clone()).unwrap()).unwrap();
        let old=game.direct_terminal_plan(&game.root()).unwrap();
        let reused=game.direct_terminal_plan(&game.root()).unwrap();
        assert!(std::sync::Arc::ptr_eq(&old,&reused));
        assert!(!old.boundaries.is_empty());
        game.update_config(cards,crate::ActionTree::new(config).unwrap()).unwrap();
        let rebuilt=game.direct_terminal_plan(&game.root()).unwrap();
        assert!(!std::sync::Arc::ptr_eq(&old,&rebuilt));
        assert_eq!(old.entries.len(),rebuilt.entries.len());
    }
}
