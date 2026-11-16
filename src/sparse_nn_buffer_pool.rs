//! Bounded worker-local storage. Buffers remain exclusively owned while live,
//! including across Rayon work stealing; no TLS borrow spans traversal or joins.
use std::cell::RefCell;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    OnceLock,
};
const BINS: usize = 17;
const MAX_BYTES: usize = 1024 * 1024;
#[derive(Default)]
struct Pool {
    bins: [Vec<Vec<f32>>; BINS],
    bytes: usize,
}
thread_local! { static POOL: RefCell<Pool> = RefCell::new(Pool::default()); }
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
fn stats() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("PFS_NN_BUFFER_STATS").as_deref() == Ok("1"))
}
pub(super) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("PFS_NN_REUSE_BUFFERS").as_deref() == Ok("1"))
}
/// Empty, fully owned vector with sufficient capacity for a raw child return.
pub(crate) fn take_raw(len: usize) -> Vec<f32> {
    if !enabled() {
        return Vec::with_capacity(len);
    }
    let cap = len
        .max(1)
        .checked_next_power_of_two()
        .expect("NN buffer too large");
    let bin = cap.trailing_zeros() as usize;
    let got = if bin < BINS {
        POOL.with_borrow_mut(|p| {
            let value = p.bins[bin].pop();
            if let Some(v) = &value {
                p.bytes -= v.capacity() * 4;
            }
            value
        })
    } else {
        None
    };
    if stats() {
        if got.is_some() { &HITS } else { &MISSES }.fetch_add(1, Ordering::Relaxed);
    }
    got.unwrap_or_else(|| Vec::with_capacity(cap))
}
pub(super) fn take(len: usize) -> Vec<f32> {
    let mut v = take_raw(len);
    v.resize(len, 0.);
    v
}
pub(super) fn recycle(mut v: Vec<f32>) {
    if !enabled() || v.capacity() == 0 {
        return;
    }
    let bin = (usize::BITS - 1 - v.capacity().leading_zeros()) as usize;
    let bytes = v.capacity() * 4;
    if bin >= BINS || bytes > MAX_BYTES {
        return;
    }
    v.clear();
    // Dropping during TLS destruction must never re-enter a destroyed pool.
    let _ = POOL.try_with(|p| {
        let mut p = p.borrow_mut();
        if p.bytes + bytes <= MAX_BYTES && p.bins[bin].len() < 8 {
            p.bytes += bytes;
            p.bins[bin].push(v);
        }
    });
}
pub(crate) fn report() {
    if stats() {
        println!(
            "NN_BUFFER_POOL hits={} misses={} max_cached_bytes_per_worker={MAX_BYTES}",
            HITS.load(Ordering::Relaxed),
            MISSES.load(Ordering::Relaxed)
        );
    }
}
