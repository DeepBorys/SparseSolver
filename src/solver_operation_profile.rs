//! Opt-in exclusive wall-time attribution for single-thread diagnostics.
use std::{cell::RefCell, sync::OnceLock, time::Instant};
const NAMES: [&str; 11] = ["traversal_other", "terminal_evaluation", "exact_allin_matrix",
    "regret_matching", "reach_propagation", "regret_update",
    "chance_aggregation", "frontier_estimate", "action_buffer_allocation", "decision_cfv_sum", "average_strategy_update"];
struct State { active: usize, stamp: Instant, nanos: [u64; 11], calls: [u64; 11] }
thread_local! { static STATE: RefCell<State> = RefCell::new(State {
    active: 11, stamp: Instant::now(), nanos: [0; 11], calls: [0; 11] }); }
#[inline]
pub fn enabled() -> bool {
    if !cfg!(feature = "solver-operation-profile") { return false; }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("QJ2_SOLVER_OPERATION_PROFILE").as_deref() == Ok("1"))
}
pub struct Guard { previous: usize }
#[inline]
pub fn enter(bucket: usize) -> Option<Guard> {
    if !enabled() { return None; }
    Some(STATE.with(|cell| {
        let mut s = cell.borrow_mut(); let now = Instant::now();
        let previous = s.active;
        if previous < 11 { s.nanos[previous] += now.duration_since(s.stamp).as_nanos() as u64; }
        s.active = bucket; s.stamp = now; s.calls[bucket] += 1;
        Guard { previous }
    }))
}
impl Drop for Guard {
    fn drop(&mut self) {
        STATE.with(|cell| {
            let mut s = cell.borrow_mut(); let now = Instant::now();
            let active = s.active;
            if active < 11 { s.nanos[active] += now.duration_since(s.stamp).as_nanos() as u64; }
            s.active = self.previous; s.stamp = now;
        });
    }
}
pub fn reset() {
    if enabled() { STATE.with(|cell| { let mut s = cell.borrow_mut();
        s.nanos = [0; 11]; s.calls = [0; 11]; s.active = 11; s.stamp = Instant::now();
    }); }
}
pub fn report() {
    if !enabled() { return; }
    STATE.with(|cell| { let s = cell.borrow();
        for (i, name) in NAMES.iter().enumerate() {
            println!("SOLVER_OPERATION_PROFILE operation={} exclusive_ns={} calls={}", name, s.nanos[i], s.calls[i]);
        }
    });
}
