use super::*;
use crate::interface::*;
use std::ptr;
use std::slice;

// Timing experiment only. Production keeps river action recursion serial.
static PARALLEL_RIVER_EXPERIMENT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

// Opt-in task-granularity experiment. Chance frontiers keep their existing
// ordered reductions; only betting-action task creation is restricted.
static SUBTREE_SCHEDULING_EXPERIMENT: std::sync::OnceLock<u8> = std::sync::OnceLock::new();

impl GameNode for PostFlopNode {
    #[inline]
    fn id(&self) -> u64 {
        self.id
    }

    #[inline]
    fn is_terminal(&self) -> bool {
        self.player & PLAYER_TERMINAL_FLAG != 0
    }

    #[inline]
    fn is_chance(&self) -> bool {
        self.player & PLAYER_CHANCE_FLAG != 0
    }

    #[inline]
    fn cfvalue_storage_player(&self) -> Option<usize> {
        let prev_player = self.player & PLAYER_MASK;
        match prev_player {
            0 => Some(1),
            1 => Some(0),
            _ => None,
        }
    }

    #[inline]
    fn player(&self) -> usize {
        self.player as usize
    }

    #[inline]
    fn num_actions(&self) -> usize {
        self.num_children as usize
    }

    #[inline]
    fn board_turn(&self) -> u8 {
        self.turn
    }

    #[inline]
    fn board_river(&self) -> u8 { self.river }

    fn chance_cards(&self) -> Vec<u8> {
        assert!(self.is_chance());
        self.children()
            .iter()
            .map(|child| match child.lock().prev_action {
                Action::Chance(card) => card,
                _ => unreachable!(),
            })
            .collect()
    }

    #[inline]
    fn play(&self, action: usize) -> MutexGuardLike<Self> {
        self.children()[action].lock()
    }

    #[inline]
    fn strategy(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.storage1 as *const f32, self.num_elements as usize) }
    }

    #[inline]
    fn strategy_mut(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.storage1 as *mut f32, self.num_elements as usize) }
    }

    #[inline]
    fn regrets(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.storage2 as *const f32, self.num_elements as usize) }
    }

    #[inline]
    fn regrets_mut(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut f32, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.storage2 as *const f32, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_mut(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut f32, self.num_elements as usize) }
    }

    #[inline]
    fn has_cfvalues_ip(&self) -> bool {
        self.num_elements_ip != 0
    }

    #[inline]
    fn cfvalues_ip(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.storage3 as *const f32, self.num_elements_ip as usize) }
    }

    #[inline]
    fn cfvalues_ip_mut(&mut self) -> &mut [f32] {
        unsafe {
            slice::from_raw_parts_mut(self.storage3 as *mut f32, self.num_elements_ip as usize)
        }
    }

    #[inline]
    fn cfvalues_chance(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.storage1 as *const f32, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_chance_mut(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.storage1 as *mut f32, self.num_elements as usize) }
    }

    #[inline]
    fn strategy_compressed(&self) -> &[u16] {
        unsafe { slice::from_raw_parts(self.storage1 as *const u16, self.num_elements as usize) }
    }

    #[inline]
    fn strategy_compressed_mut(&mut self) -> &mut [u16] {
        unsafe { slice::from_raw_parts_mut(self.storage1 as *mut u16, self.num_elements as usize) }
    }

    #[inline]
    fn regrets_compressed(&self) -> &[i16] {
        unsafe { slice::from_raw_parts(self.storage2 as *const i16, self.num_elements as usize) }
    }

    #[inline]
    fn regrets_compressed_mut(&mut self) -> &mut [i16] {
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut i16, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_compressed(&self) -> &[i16] {
        unsafe { slice::from_raw_parts(self.storage2 as *const i16, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_compressed_mut(&mut self) -> &mut [i16] {
        unsafe { slice::from_raw_parts_mut(self.storage2 as *mut i16, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_ip_compressed(&self) -> &[i16] {
        unsafe { slice::from_raw_parts(self.storage3 as *const i16, self.num_elements_ip as usize) }
    }

    #[inline]
    fn cfvalues_ip_compressed_mut(&mut self) -> &mut [i16] {
        unsafe {
            slice::from_raw_parts_mut(self.storage3 as *mut i16, self.num_elements_ip as usize)
        }
    }

    #[inline]
    fn cfvalues_chance_compressed(&self) -> &[i16] {
        unsafe { slice::from_raw_parts(self.storage1 as *const i16, self.num_elements as usize) }
    }

    #[inline]
    fn cfvalues_chance_compressed_mut(&mut self) -> &mut [i16] {
        unsafe { slice::from_raw_parts_mut(self.storage1 as *mut i16, self.num_elements as usize) }
    }

    #[inline]
    fn strategy_scale(&self) -> f32 {
        self.scale1
    }

    #[inline]
    fn set_strategy_scale(&mut self, scale: f32) {
        self.scale1 = scale;
    }

    #[inline]
    fn regret_scale(&self) -> f32 {
        self.scale2
    }

    #[inline]
    fn set_regret_scale(&mut self, scale: f32) {
        self.scale2 = scale;
    }

    #[inline]
    fn cfvalue_scale(&self) -> f32 {
        self.scale2
    }

    #[inline]
    fn set_cfvalue_scale(&mut self, scale: f32) {
        self.scale2 = scale;
    }

    #[inline]
    fn cfvalue_ip_scale(&self) -> f32 {
        self.scale3
    }

    #[inline]
    fn set_cfvalue_ip_scale(&mut self, scale: f32) {
        self.scale3 = scale;
    }

    #[inline]
    fn cfvalue_chance_scale(&self) -> f32 {
        self.scale1
    }

    #[inline]
    fn set_cfvalue_chance_scale(&mut self, scale: f32) {
        self.scale1 = scale;
    }

    #[inline]
    fn enable_parallelization(&self) -> bool {
        let scheduling = *SUBTREE_SCHEDULING_EXPERIMENT.get_or_init(|| {
            match std::env::var("PFS_SUBTREE_SCHEDULING_EXPERIMENT").as_deref() {
                Err(_) | Ok("default") => 0,
                Ok("chance_only") => 1,
                Ok("flop_and_chance") => 2,
                Ok(other) => panic!("Unknown PFS_SUBTREE_SCHEDULING_EXPERIMENT={other}"),
            }
        });
        if scheduling != 0 {
            return self.is_chance() || (scheduling == 2 && self.turn == NOT_DEALT);
        }
        self.river == NOT_DEALT || *PARALLEL_RIVER_EXPERIMENT.get_or_init(|| {
            std::env::var("PFS_PARALLEL_RIVER_EXPERIMENT").as_deref() == Ok("1")
        })
    }
}

impl Default for PostFlopNode {
    #[inline]
    fn default() -> Self {
        Self {
            id: 0,
            prev_action: Action::None,
            player: PLAYER_OOP,
            turn: NOT_DEALT,
            river: NOT_DEALT,
            is_locked: false,
            amount: 0,
            children_offset: 0,
            num_children: 0,
            num_elements_ip: 0,
            storage1: ptr::null_mut(),
            storage2: ptr::null_mut(),
            storage3: ptr::null_mut(),
            num_elements: 0,
            scale1: 0.0,
            scale2: 0.0,
            scale3: 0.0,
        }
    }
}

impl PostFlopNode {
    #[inline]
    pub(super) fn children(&self) -> &[MutexLike<Self>] {
        // This is safe because `MutexLike<T>` is a `repr(transparent)` wrapper around `T`.
        let self_ptr = self as *const _ as *const MutexLike<PostFlopNode>;
        unsafe {
            slice::from_raw_parts(
                self_ptr.add(self.children_offset as usize),
                self.num_children as usize,
            )
        }
    }
}
