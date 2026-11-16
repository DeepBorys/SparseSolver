//! Experimental single-pass selection of whole low-reach river subtrees.
use super::*;
use crate::interface::{Game,GameNode};
use crate::utility::{normalized_strategy,normalized_strategy_compressed,apply_locking_strategy};

impl PostFlopGame {
    /// Select subtree roots without modifying per-hand strategies or installing locks.
    /// Root mass is constant; reaches use the current average profile, without chance factors.
    pub fn diagnostic_select_frozen_river_nodes(&self, turns:&[Card], rivers:&[Card], threshold:f64,
        root_mass:f64) -> (Vec<u64>,usize) {
        assert_eq!(self.bunching_num_dead_cards,0);
        assert!(root_mass>0. && (0.0..1.0).contains(&threshold));
        fn descendants(node:&PostFlopNode)->usize {
            if node.is_terminal() || node.is_chance(){return 0;}
            1+(0..node.num_actions()).map(|a|descendants(&node.play(a))).sum::<usize>()
        }
        fn joint(game:&PostFlopGame,a:&[f32],b:&[f32],board:u64)->f64 {
            let mut total=0.;let mut minus=[0.;52];
            for (i,&(c,d)) in game.private_cards[1].iter().enumerate(){
                if board&((1u64<<c)|(1u64<<d))==0 {let w=b[i] as f64;total+=w;minus[c as usize]+=w;minus[d as usize]+=w;}
            }
            let mut sum=0.;
            for (i,&(c,d)) in game.private_cards[0].iter().enumerate(){
                if board&((1u64<<c)|(1u64<<d))==0 {
                    let same=game.same_hand_index[0][i];
                    let w=if same==u16::MAX{0.}else{b[same as usize] as f64};
                    sum+=(a[i]*(total+w-minus[c as usize]-minus[d as usize]) as f32) as f64;
                }
            }
            sum
        }
        fn walk(game:&PostFlopGame,node:&PostFlopNode,a:&[f32],b:&[f32],board:u64,
            turns:&[Card],rivers:&[Card],threshold:f64,root_mass:f64,roots:&mut Vec<u64>,count:&mut usize){
            if node.is_terminal() || node.amount==game.tree_config.effective_stack{return;}
            if node.is_chance(){
                let wanted=if node.turn==NOT_DEALT{turns}else{rivers};
                for child in node.children(){let child=child.lock();
                    if let Action::Chance(card)=child.prev_action {if wanted.contains(&card){
                        walk(game,&child,a,b,board|(1u64<<card),turns,rivers,threshold,root_mass,roots,count);
                    }}
                }
                return;
            }
            if node.river!=NOT_DEALT && joint(game,a,b,board)/root_mass<threshold {
                roots.push(node.id());*count+=descendants(node);return;
            }
            let mut policy=if game.is_compression_enabled {
                normalized_strategy_compressed(node.strategy_compressed(),node.num_actions())
            }else{normalized_strategy(node.strategy(),node.num_actions())};
            apply_locking_strategy(&mut policy,game.locking_strategy(node));
            let own=if node.player()==0{a}else{b};
            for action in 0..node.num_actions(){
                let next:Vec<_>=own.iter().enumerate().map(|(i,w)|*w*policy[action*own.len()+i]).collect();
                let (x,y)=if node.player()==0{(next.as_slice(),b)}else{(a,next.as_slice())};
                walk(game,&node.play(action),x,y,board,turns,rivers,threshold,root_mass,roots,count);
            }
        }
        let mut roots=Vec::new();let mut count=0;
        let board=self.card_config.flop.iter().fold(0u64,|m,c|m|(1u64<<c));
        walk(self,&self.root(),&self.initial_weights[0],&self.initial_weights[1],board,
            turns,rivers,threshold,root_mass,&mut roots,&mut count);
        (roots,count)
    }
}
