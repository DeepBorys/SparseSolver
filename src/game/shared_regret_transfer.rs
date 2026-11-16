//! One compact board/player hand map reused across all betting histories/actions.
use super::*;
use crate::interface::*;
#[cfg(feature="rayon")]
use rayon::prelude::*;

#[derive(Clone)]
pub struct K1SharedHandTable {
    pub player:usize,
    pub target_board:(Card,Card),
    pub source_boards:Vec<(Card,Card)>,
    pub source_hands:Vec<Option<(usize,usize)>>,
}
#[derive(Clone)]
pub struct K1SharedNodeCopy {
    pub target_node:u64,
    pub table:usize,
    pub source_nodes:Vec<Option<u64>>,
}
struct Table { hands:Vec<u16>,slots:Vec<u8>,used:Vec<usize>,player:usize,board:(Card,Card),source_boards:Vec<(Card,Card)> }
#[repr(C)]
struct Task { offset:u32,source_start:u32,hands:u16,table:u16,actions:u8,source_count:u8 }
fn rebase_sources(offsets:&[u32],elements:usize)->Option<(u32,[u32;96])>{
    let low=*offsets.iter().min()?;let high=*offsets.iter().max()?;
    if (high-low) as u64+elements as u64>i32::MAX as u64{return None;}
    let mut relative=[0;96];for(i,&x)in offsets.iter().enumerate(){relative[i]=x-low;}
    Some((low,relative))
}
pub struct K1SharedTransferPlan { tables:Vec<Table>,tasks:Vec<Task>,source_offsets:Vec<u32>,storage:usize,arena:usize,bytes:usize,values:usize,legal:usize }
extern "C" {
    fn pfs_k1_shared_regret_gather(base:*const f32,dst:*mut f32,sources:*const u32,hands:*const u16,slots:*const u8,n:usize,actions:usize);
    fn pfs_k1_shared_regret_gather_many(base:*const f32,dst:*mut f32,sources:*const u32,hands:*const u16,slots:*const u8,n:usize,actions:usize,source_count:usize);
    fn pfs_k1_regret_gather_backend()->*const std::ffi::c_char;
}
impl K1SharedTransferPlan {
    pub fn destination_nodes(&self)->usize{self.tasks.len()}
    pub fn destination_values(&self)->usize{self.values}
    pub fn legal_destination_values(&self)->usize{self.legal}
    pub fn table_count(&self)->usize{self.tables.len()}
    pub fn table_bytes(&self)->usize{self.tables.iter().map(|t|t.hands.len()*2+t.slots.len()).sum()}
    pub fn node_bytes(&self)->usize{self.tasks.len()*std::mem::size_of::<Task>()+self.source_offsets.len()*4}
    pub fn index_bytes(&self)->usize{self.table_bytes()+self.node_bytes()}
    pub fn resident_bytes(&self)->usize{
        std::mem::size_of::<Self>()+self.source_offsets.capacity()*4+self.tasks.capacity()*std::mem::size_of::<Task>()+self.tables.capacity()*std::mem::size_of::<Table>()+
            self.tables.iter().map(|t|t.hands.capacity()*2+t.slots.capacity()+t.used.capacity()*std::mem::size_of::<usize>()+t.source_boards.capacity()*std::mem::size_of::<(Card,Card)>()).sum::<usize>()
    }
    pub fn backend(&self)->&'static str{unsafe{std::ffi::CStr::from_ptr(pfs_k1_regret_gather_backend()).to_str().unwrap()}}
}
impl PostFlopGame {
    /// Opt-in nonnegative free-sum mixture. Sources stay immutable; no floor.
    pub fn apply_or_verify_fitted_transfer(&mut self,plans:&[K1SharedTransferPlan],weights:&[Vec<Vec<f64>>],verify:bool)->usize{
        assert!(self.state==State::MemoryAllocated&&!self.is_compression_enabled);
        assert!(matches!(plans.len(),2|4));assert_eq!(plans.len(),weights.len());let first=&plans[0];
        for (p,w) in plans.iter().zip(weights){assert_eq!((self.storage2.as_ptr() as usize,self.node_arena.as_ptr() as usize,self.storage2.len()),(p.storage,p.arena,p.bytes));assert_eq!(p.tasks.len(),first.tasks.len());assert_eq!(w.len(),p.tables.len());for(t,row)in p.tables.iter().zip(w){assert_eq!(row.len(),t.hands.len());assert!(row.iter().all(|x|x.is_finite()&&*x>=0.));}}
        let base=self.storage2.as_mut_ptr() as usize;
        let apply=|i:usize|{let dst=&first.tasks[i];let n=dst.hands as usize;
            for action in 0..dst.actions as usize{for h in 0..n{
                let mut value=0f64;
                for(p,w)in plans.iter().zip(weights){let task=&p.tasks[i];assert_eq!((task.offset,task.hands,task.actions),(dst.offset,dst.hands,dst.actions));let table=&p.tables[task.table as usize];if table.hands[h]==u16::MAX{continue;}
                    let slot=if table.slots.is_empty(){0}else{table.slots[h] as usize};let offset=p.source_offsets[task.source_start as usize+slot] as usize+action*n+table.hands[h] as usize;
                    value+=w[task.table as usize][h]*unsafe{*(base as *const f32).add(offset)} as f64;
                }
                let value=value as f32;assert!(value.is_finite());unsafe{let out=(base as *mut f32).add(dst.offset as usize+action*n+h);if verify{assert_eq!((*out).to_bits(),value.to_bits());}else{*out=value;}}
            }}
        };
        #[cfg(feature="rayon")]
        (0..first.tasks.len()).into_par_iter().with_min_len(16).for_each(apply);
        #[cfg(not(feature="rayon"))]
        (0..first.tasks.len()).for_each(apply);
        first.values
    }
    /// Experimental convex K2 fill; prepared K1 plans validate both sources.
    /// Arithmetic uses f64 then one f32 rounding; signed regrets are not clamped.
    pub fn apply_or_verify_k2_shared_transfer(&mut self,a:&K1SharedTransferPlan,b:&K1SharedTransferPlan,weights:&[Vec<f64>],verify:bool)->usize{
        assert!(self.state==State::MemoryAllocated&&!self.is_compression_enabled);
        for p in [a,b]{assert_eq!((self.storage2.as_ptr() as usize,self.node_arena.as_ptr() as usize,self.storage2.len()),(p.storage,p.arena,p.bytes));}
        assert_eq!(a.tasks.len(),b.tasks.len());assert_eq!(weights.len(),a.tables.len());
        for (i,w) in weights.iter().enumerate(){assert_eq!(w.len(),a.tables[i].hands.len());assert!(w.iter().all(|x|x.is_finite()&&*x>=0.&&*x<=1.));}
        let base=self.storage2.as_mut_ptr() as usize;
        let apply=|(x,y):(&Task,&Task)|{
            assert_eq!((x.offset,x.hands,x.actions,x.table),(y.offset,y.hands,y.actions,y.table));
            let ta=&a.tables[x.table as usize];let tb=&b.tables[y.table as usize];let n=x.hands as usize;
            for action in 0..x.actions as usize {for h in 0..n {
                let value=if ta.hands[h]==u16::MAX{assert_eq!(tb.hands[h],u16::MAX);0.}else{
                    let read=|p:&K1SharedTransferPlan,t:&Table,task:&Task|{let slot=if t.slots.is_empty(){0}else{t.slots[h] as usize};unsafe{*((base as *const f32).add(p.source_offsets[task.source_start as usize+slot] as usize+action*n+t.hands[h] as usize))}};
                    let u=read(a,ta,x);let v=read(b,tb,y);let w=weights[x.table as usize][h];
                    if w==1.{u}else if w==0.{v}else{(w*u as f64+(1.-w)*v as f64) as f32}
                };
                unsafe{let dst=(base as *mut f32).add(x.offset as usize+action*n+h);if verify{assert_eq!((*dst).to_bits(),value.to_bits());}else{*dst=value;}}
            }}
        };
        #[cfg(feature="rayon")]
        a.tasks.par_iter().zip(b.tasks.par_iter()).with_min_len(16).for_each(apply);
        #[cfg(not(feature="rayon"))]
        a.tasks.iter().zip(b.tasks.iter()).for_each(apply);
        a.values
    }
    pub fn prepare_k1_shared_transfer(&self,definitions:Vec<K1SharedHandTable>,copies:Vec<K1SharedNodeCopy>)->Result<K1SharedTransferPlan,String>{
        if self.state!=State::MemoryAllocated||self.is_compression_enabled||!self.locking_strategy.is_empty(){return Err("Shared K1 needs an unlocked unsolved FP32 tree".into());}
        if definitions.len()>u16::MAX as usize+1||self.storage2.len()/4>u32::MAX as usize{return Err("Shared K1 dimensions unsupported".into());}
        let base=self.storage2.as_ptr() as usize;
        if base%4!=0||self.storage2.len()%4!=0{return Err("Unaligned shared K1 storage".into());}
        let mut tables=Vec::with_capacity(definitions.len());
        for d in definitions {
            if d.player>1||d.source_hands.len()!=self.private_cards[d.player].len()||d.source_hands.len()>u16::MAX as usize{return Err("Shared K1 table hand count".into());}
            let mut used=Vec::new();
            for &(slot,_) in d.source_hands.iter().flatten(){if !used.contains(&slot){used.push(slot);}}
            if used.is_empty()||used.len()>96{return Err("Shared K1 supports one through96 source boards per table".into());}
            used.shrink_to_fit();
            let mut hands=Vec::with_capacity(d.source_hands.len());let mut slots=Vec::with_capacity(d.source_hands.len());
            for (h,mapping) in d.source_hands.iter().enumerate(){
                let(a,b)=self.private_cards[d.player][h];let blocked=[d.target_board.0,d.target_board.1].iter().any(|&c|c==a||c==b);
                if blocked!=mapping.is_none(){return Err("Shared K1 target legality mismatch".into());}
                if let Some((slot,index))=*mapping{
                    let board=*d.source_boards.get(slot).ok_or("Shared K1 source slot invalid")?;
                    let &(a,b)=self.private_cards[d.player].get(index).ok_or("Shared K1 source hand invalid")?;
                    if [board.0,board.1].iter().any(|&c|c==a||c==b){return Err("Shared K1 source hand is blocked".into());}
                    hands.push(index as u16);slots.push(used.iter().position(|s|*s==slot).unwrap() as u8);
                }else{hands.push(u16::MAX);slots.push(0);}
            }
            if used.len()==1{slots=Vec::new();}
            tables.push(Table{hands,slots,used,player:d.player,board:d.target_board,source_boards:d.source_boards});
        }
        let mut targets=vec![false;self.node_arena.len()];
        for c in &copies {
            let target=targets.get_mut(c.target_node as usize).ok_or("Unknown shared K1 destination")?;
            if std::mem::replace(target,true){return Err("Duplicate shared K1 destination".into());}
        }
        let offset=|node:&PostFlopNode|->Result<u32,String>{
            let p=node.storage2 as usize;
            if p<base||p%4!=0||p+node.num_elements as usize*4>base+self.storage2.len(){return Err("Shared K1 node outside storage".into());}
            Ok(((p-base)/4) as u32)
        };
        let mut tasks=Vec::with_capacity(copies.len());let mut source_offsets=Vec::new();let mut values=0;let mut legal=0;
        for c in copies {
            let table=tables.get(c.table).ok_or("Shared K1 table missing")?;
            let target=self.node_arena[c.target_node as usize].lock();
            if target.is_terminal()||target.is_chance()||target.player()!=table.player||(target.turn,target.river)!=table.board||target.num_actions()>u8::MAX as usize{return Err("Shared K1 target/table mismatch".into());}
            let source_start=u32::try_from(source_offsets.len()).map_err(|_|"Shared K1 source index too large")?;
            let mut sources=vec![0;table.used.len().max(4)];
            for (compact,&global) in table.used.iter().enumerate(){
                let id=c.source_nodes.get(global).copied().flatten().ok_or("Shared K1 source history absent")? as usize;
                if *targets.get(id).ok_or("Unknown shared K1 source")?{return Err("Shared K1 source is also a destination".into());}
                let source=self.node_arena[id].lock();
                if source.is_terminal()||source.is_chance()||source.player()!=table.player||source.num_actions()!=target.num_actions()||source.amount!=target.amount||(source.turn,source.river)!=table.source_boards[global]{return Err("Shared K1 source node mismatch".into());}
                sources[compact]=offset(&source)?;
            }
            let n=table.hands.len();let actions=target.num_actions();
            if target.num_elements as usize!=n*actions{return Err("Shared K1 action layout mismatch".into());}
            values+=n*actions;legal+=table.hands.iter().filter(|&&h|h!=u16::MAX).count()*actions;
            source_offsets.extend(sources);
            tasks.push(Task{offset:offset(&target)?,source_start,hands:n as u16,table:c.table as u16,actions:actions as u8,source_count:table.used.len() as u8});
        }
        tasks.sort_unstable_by_key(|t|t.offset);
        if tasks.windows(2).any(|w|w[0].offset as usize+w[0].hands as usize*w[0].actions as usize>w[1].offset as usize){return Err("Shared K1 destinations overlap".into());}
        Ok(K1SharedTransferPlan{tables,tasks,source_offsets,storage:base,arena:self.node_arena.as_ptr() as usize,bytes:self.storage2.len(),values,legal})
    }
    /// Diagnostic validation outside the timed copy: compare every destination
    /// bit to its retained source, including signed zero and blocked-hand zeros.
    pub fn verify_k1_shared_transfer(&self,plan:&K1SharedTransferPlan)->Result<usize,String>{
        if (self.storage2.as_ptr() as usize,self.node_arena.as_ptr() as usize,self.storage2.len())!=(plan.storage,plan.arena,plan.bytes){return Err("Shared K1 plan storage changed".into());}
        let base=self.storage2.as_ptr() as *const f32;let mut checked=0;
        for task in &plan.tasks {
            let table=&plan.tables[task.table as usize];let n=task.hands as usize;
            for action in 0..task.actions as usize {for h in 0..n {
                let expected=if table.hands[h]==u16::MAX{0}else{
                    let slot=if table.slots.is_empty(){0}else{table.slots[h] as usize};
                    let offset=plan.source_offsets[task.source_start as usize+slot] as usize+action*n+table.hands[h] as usize;
                    unsafe{(*base.add(offset)).to_bits()}
                };
                let actual=unsafe{(*base.add(task.offset as usize+action*n+h)).to_bits()};
                if actual!=expected{return Err(format!("Shared K1 copy mismatch at destination{} action{} hand{}",task.offset,action,h));}
                checked+=1;
            }}
        }
        Ok(checked)
    }
    pub fn apply_k1_shared_average_transfer(&mut self,plan:&K1SharedTransferPlan){
        assert!(self.state==State::MemoryAllocated&&!self.is_compression_enabled);
        assert_eq!((self.storage2.as_ptr() as usize,self.node_arena.as_ptr() as usize,self.storage2.len()),(plan.storage,plan.arena,plan.bytes));
        assert_eq!(self.storage1.len(),self.storage2.len());
        let first=self.storage1.as_ptr() as usize;let second=self.storage2.as_ptr() as usize;
        for entry in &self.node_arena{let node=entry.lock();if !node.is_terminal()&&!node.is_chance(){
            assert_eq!(node.storage1 as usize-first,node.storage2 as usize-second);
        }}
        let base=self.storage1.as_mut_ptr() as usize;
        let apply=|task:&Task|{
            let table=&plan.tables[task.table as usize];
            if plan.bytes/4>i32::MAX as usize{
                let offsets=&plan.source_offsets[task.source_start as usize..task.source_start as usize+task.source_count as usize];
                if let Some((low,relative))=rebase_sources(offsets,task.hands as usize*task.actions as usize){
                    unsafe{pfs_k1_shared_regret_gather_many((base as *const f32).add(low as usize),(base as *mut f32).add(task.offset as usize),relative.as_ptr(),table.hands.as_ptr(),if table.slots.is_empty(){std::ptr::null()}else{table.slots.as_ptr()},task.hands as usize,task.actions as usize,task.source_count as usize);}
                }else{
                    let n=task.hands as usize;
                    for a in 0..task.actions as usize{for h in 0..n{let value=if table.hands[h]==u16::MAX{0.}else{let slot=if table.slots.is_empty(){0}else{table.slots[h] as usize};unsafe{*(base as *const f32).add(offsets[slot] as usize+a*n+table.hands[h] as usize)}};unsafe{*(base as *mut f32).add(task.offset as usize+a*n+h)=value;}}}
                }
                return;
            }
            // SAFETY: preparation validates all row/slot bounds. No source node
            // is written; destination ranges are pairwise disjoint. &mut self
            // excludes concurrent traversal while the shared tables are read.
            unsafe{pfs_k1_shared_regret_gather_many(base as *const f32,(base as *mut f32).add(task.offset as usize),plan.source_offsets.as_ptr().add(task.source_start as usize),table.hands.as_ptr(),
                if table.slots.is_empty(){std::ptr::null()}else{table.slots.as_ptr()},task.hands as usize,task.actions as usize,task.source_count as usize);}
        };
        #[cfg(feature="rayon")]
        plan.tasks.par_iter().with_min_len(16).for_each(apply);
        #[cfg(not(feature="rayon"))]
        plan.tasks.iter().for_each(apply);
    }
    pub fn verify_k1_shared_average_transfer(&self,plan:&K1SharedTransferPlan)->Result<usize,String>{
        if (self.storage2.as_ptr() as usize,self.node_arena.as_ptr() as usize,self.storage2.len())!=(plan.storage,plan.arena,plan.bytes){return Err("Shared K1 plan storage changed".into());}
        let base=self.storage1.as_ptr() as *const f32;let mut checked=0;
        for task in &plan.tasks {
            let table=&plan.tables[task.table as usize];let n=task.hands as usize;
            for action in 0..task.actions as usize {for h in 0..n {
                let expected=if table.hands[h]==u16::MAX{0}else{
                    let slot=if table.slots.is_empty(){0}else{table.slots[h] as usize};
                    let offset=plan.source_offsets[task.source_start as usize+slot] as usize+action*n+table.hands[h] as usize;
                    unsafe{(*base.add(offset)).to_bits()}
                };
                let actual=unsafe{(*base.add(task.offset as usize+action*n+h)).to_bits()};
                if actual!=expected{return Err(format!("Shared K1 copy mismatch at destination{} action{} hand{}",task.offset,action,h));}
                checked+=1;
            }}
        }
        Ok(checked)
    }
    pub fn apply_k1_shared_transfer(&mut self,plan:&K1SharedTransferPlan){
        assert!(self.state==State::MemoryAllocated&&!self.is_compression_enabled);
        assert_eq!((self.storage2.as_ptr() as usize,self.node_arena.as_ptr() as usize,self.storage2.len()),(plan.storage,plan.arena,plan.bytes));
        let base=self.storage2.as_mut_ptr() as usize;
        let apply=|task:&Task|{
            let table=&plan.tables[task.table as usize];
            if plan.bytes/4>i32::MAX as usize{
                let offsets=&plan.source_offsets[task.source_start as usize..task.source_start as usize+task.source_count as usize];
                if let Some((low,relative))=rebase_sources(offsets,task.hands as usize*task.actions as usize){
                    unsafe{pfs_k1_shared_regret_gather_many((base as *const f32).add(low as usize),(base as *mut f32).add(task.offset as usize),relative.as_ptr(),table.hands.as_ptr(),if table.slots.is_empty(){std::ptr::null()}else{table.slots.as_ptr()},task.hands as usize,task.actions as usize,task.source_count as usize);}
                }else{
                    let n=task.hands as usize;
                    for a in 0..task.actions as usize{for h in 0..n{let value=if table.hands[h]==u16::MAX{0.}else{let slot=if table.slots.is_empty(){0}else{table.slots[h] as usize};unsafe{*(base as *const f32).add(offsets[slot] as usize+a*n+table.hands[h] as usize)}};unsafe{*(base as *mut f32).add(task.offset as usize+a*n+h)=value;}}}
                }
                return;
            }
            // SAFETY: preparation validates all row/slot bounds. No source node
            // is written; destination ranges are pairwise disjoint. &mut self
            // excludes concurrent traversal while the shared tables are read.
            unsafe{pfs_k1_shared_regret_gather_many(base as *const f32,(base as *mut f32).add(task.offset as usize),plan.source_offsets.as_ptr().add(task.source_start as usize),table.hands.as_ptr(),
                if table.slots.is_empty(){std::ptr::null()}else{table.slots.as_ptr()},task.hands as usize,task.actions as usize,task.source_count as usize);}
        };
        #[cfg(feature="rayon")]
        plan.tasks.par_iter().with_min_len(16).for_each(apply);
        #[cfg(not(feature="rayon"))]
        plan.tasks.iter().for_each(apply);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn average_transfer_preserves_regrets_sources_and_blocked_zeros(){
        let tree=TreeConfig{initial_state:BoardState::River,starting_pot:100,effective_stack:1,
            river_bet_sizes:[("a","").try_into().unwrap(),("a","").try_into().unwrap()],..Default::default()};
        let cards=CardConfig{range:["AA".parse().unwrap(),"KK".parse().unwrap()],flop:crate::flop_from_str("2c3d4h").unwrap(),turn:crate::card_from_str("5s").unwrap(),river:crate::card_from_str("6c").unwrap()};
        let mut game=PostFlopGame::with_config(cards,ActionTree::new(tree).unwrap()).unwrap();game.allocate_memory(false);
        game.storage2.fill(0x3f);let original=game.storage2.clone();
        let input=[2f32,3.,99.,4.,5.,99.];
        unsafe{std::ptr::copy_nonoverlapping(input.as_ptr(),game.storage1.as_mut_ptr() as *mut f32,6);}
        let plan=K1SharedTransferPlan{tables:vec![Table{hands:vec![1,0,u16::MAX],slots:vec![],used:vec![0],player:0,board:(NOT_DEALT,NOT_DEALT),source_boards:vec![]}],tasks:vec![Task{offset:12,source_start:0,hands:3,table:0,actions:2,source_count:1}],source_offsets:vec![0],storage:game.storage2.as_ptr() as usize,arena:game.node_arena.as_ptr() as usize,bytes:game.storage2.len(),values:6,legal:4};
        game.apply_k1_shared_average_transfer(&plan);
        assert_eq!(game.verify_k1_shared_average_transfer(&plan).unwrap(),6);
        assert_eq!(game.storage2,original);
        let data=unsafe{std::slice::from_raw_parts(game.storage1.as_ptr() as *const f32,18)};
        assert_eq!(&data[..6],&input);assert_eq!(&data[12..18],&[3.,2.,0.,5.,4.,0.]);
    }

    #[test]
    fn large_arena_sources_rebase_without_signed_index_overflow(){
        let(low,relative)=rebase_sources(&[2_500_000_000,3_000_000_000,2_600_000_000],1800).unwrap();
        assert_eq!(low,2_500_000_000);assert_eq!(&relative[..3],&[0,500_000_000,100_000_000]);
        assert!(rebase_sources(&[0,3_000_000_000],1800).is_none());
        assert!(rebase_sources(&[u32::MAX-100,u32::MAX-50],10).is_some());
        assert!(rebase_sources(&[0,i32::MAX as u32-5],10).is_none());
    }
    #[test]
    fn convex_k2_preserves_sources_signed_values_endpoints_and_blocked_zeros(){
        let tree=TreeConfig{initial_state:BoardState::River,starting_pot:100,effective_stack:1,
            river_bet_sizes:[("a","").try_into().unwrap(),("a","").try_into().unwrap()],..Default::default()};
        let cards=CardConfig{range:["AA".parse().unwrap(),"KK".parse().unwrap()],flop:crate::flop_from_str("2c3d4h").unwrap(),turn:crate::card_from_str("5s").unwrap(),river:crate::card_from_str("6c").unwrap()};
        let mut game=PostFlopGame::with_config(cards,ActionTree::new(tree).unwrap()).unwrap();game.allocate_memory(false);
        assert!(game.storage2.len()>=18*4);let base=game.storage2.as_mut_ptr() as *mut f32;
        let original=[-4.,-0.,666.,8.,-3.,666.,4.,5.,666.,-8.,7.,666.];
        unsafe{std::ptr::copy_nonoverlapping(original.as_ptr(),base,12);}
        let plan=|offset|K1SharedTransferPlan{tables:vec![Table{hands:vec![0,1,u16::MAX],slots:vec![],used:vec![0],player:0,board:(NOT_DEALT,NOT_DEALT),source_boards:vec![]}],tasks:vec![Task{offset:12,source_start:0,hands:3,table:0,actions:2,source_count:1}],source_offsets:vec![offset],storage:base as usize,arena:game.node_arena.as_ptr() as usize,bytes:game.storage2.len(),values:6,legal:4};
        let a=plan(0);let b=plan(6);let four=vec![plan(0),plan(6),plan(0),plan(6)];let weights=vec![vec![0.25,1.,1.]];
        assert_eq!(game.apply_or_verify_k2_shared_transfer(&a,&b,&weights,false),6);
        unsafe{for(i,v)in [2f32,-0.,0.,-4.,-3.,0.].iter().enumerate(){assert_eq!((*base.add(12+i)).to_bits(),v.to_bits());}for(i,v)in original.iter().enumerate(){assert_eq!((*base.add(i)).to_bits(),v.to_bits());}}
        assert_eq!(game.apply_or_verify_k2_shared_transfer(&a,&b,&weights,true),6);
        let plans=vec![a,b];let free=vec![vec![vec![0.72,0.,0.]],vec![vec![0.34,0.,0.]]];
        game.apply_or_verify_fitted_transfer(&plans,&free,false);
        unsafe{assert!((*base.add(12)+1.52).abs()<1e-6);assert!((*base.add(15)-3.04).abs()<1e-6);assert_eq!(*base.add(14),0.);}
        game.apply_or_verify_fitted_transfer(&plans,&free,true);
        let free4=vec![vec![vec![0.2,0.,0.]],vec![vec![0.3,0.,0.]],vec![vec![0.4,0.,0.]],vec![vec![0.5,0.,0.]]];
        game.apply_or_verify_fitted_transfer(&four,&free4,false);
        unsafe{assert!((*base.add(12)-0.8).abs()<1e-6);assert!((*base.add(15)+1.6).abs()<1e-6);for(i,v)in original.iter().enumerate(){assert_eq!((*base.add(i)).to_bits(),v.to_bits());}}
        game.apply_or_verify_fitted_transfer(&four,&free4,true);
    }
    #[test]
    fn shared_regret_transfer_reuses_maps_across_actions_and_preserves_bits(){
        let n=35;let actions=3;let mut source=vec![0f32;1000];
        for (i,v) in source.iter_mut().enumerate(){*v=if i%7==0{-0.}else{i as f32-300.};}
        let offsets=[0,200,400,600];let hands:Vec<u16>=(0..n).map(|i|if i%6==0{u16::MAX}else{(n-1-i) as u16}).collect();
        let slots:Vec<u8>=(0..n).map(|i|(i%4) as u8).collect();
        for variable in [false,true] {
            let mut out=vec![99.;n*actions];
            unsafe{pfs_k1_shared_regret_gather(source.as_ptr(),out.as_mut_ptr(),offsets.as_ptr(),hands.as_ptr(),if variable{slots.as_ptr()}else{std::ptr::null()},n,actions);}
            for a in 0..actions{for h in 0..n{
                let expected=if hands[h]==u16::MAX{0.}else{source[offsets[if variable{slots[h] as usize}else{0}] as usize+a*n+hands[h] as usize]};
                assert_eq!(out[a*n+h].to_bits(),expected.to_bits());
            }}
        }
    }
    #[test]
    fn shared_large_source_pools_preserve_signed_bits_and_tails(){
        for count in [5usize,16,36,48,96] {for n in [1usize,7,8,15,16,17,35,64,65,129] {
            let actions=3;let mut base=vec![0f32;count*n*actions];
            for (i,v) in base.iter_mut().enumerate(){*v=f32::from_bits(match i%4{0=>0x80000000,1=>0xc0800000,2=>0x7fc01234,_=>0x3f800000});}
            let sources:Vec<u32>=(0..count).map(|s|(s*n*actions) as u32).collect();
            let hands:Vec<u16>=(0..n).map(|h|if h%6==0{u16::MAX}else{(n-1-h) as u16}).collect();
            let slots:Vec<u8>=(0..n).map(|h|(h%count) as u8).collect();let mut out=vec![123.;n*actions];
            unsafe{pfs_k1_shared_regret_gather_many(base.as_ptr(),out.as_mut_ptr(),sources.as_ptr(),hands.as_ptr(),slots.as_ptr(),n,actions,count);}
            for a in 0..actions {for h in 0..n {
                let expected=if hands[h]==u16::MAX{0}else{base[sources[slots[h] as usize] as usize+a*n+hands[h] as usize].to_bits()};
                assert_eq!(out[a*n+h].to_bits(),expected);
            }}
        }}
    }

}
