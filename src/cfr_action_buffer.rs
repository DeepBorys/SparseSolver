//! Optional bounded scratch experiments. Every live buffer has one owner.
use crate::mutex_like::MutexLike;
use std::{cell::RefCell, collections::BTreeMap, ops::Deref, sync::OnceLock};
const LIMIT: usize = 8 * 1024 * 1024;
#[derive(Default)]
struct Cache { pool: Vec<Vec<f32>>, nodes: BTreeMap<u64, Vec<f32>>, bytes: usize }
thread_local! { static CACHE: RefCell<Cache> = RefCell::new(Cache::default()); }
fn mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("PFS_CFR_ACTION_BUFFER_MODE").as_deref() {
        Ok("pool") => 1, Ok("node") => 2, _ => 0,
    })
}
pub struct ActionBuffer { inner: MutexLike<Vec<f32>>, mode: u8, node: u64 }
impl ActionBuffer {
    pub fn new(node: u64, len: usize) -> Self {
        let mode = mode();
        let value = if mode == 0 { None } else { CACHE.with(|cell| {
            let mut cache = cell.borrow_mut();
            let value = if mode == 2 { cache.nodes.remove(&node) }
                else { cache.pool.iter().position(|v| v.capacity() >= len).map(|i| cache.pool.swap_remove(i)) };
            if let Some(v) = &value { cache.bytes -= v.capacity() * 4; }
            value.filter(|v| v.capacity() >= len)
        }) };
        Self { inner: MutexLike::new(value.unwrap_or_else(|| Vec::with_capacity(len))), mode, node }
    }
}
impl Deref for ActionBuffer {
    type Target = MutexLike<Vec<f32>>;
    fn deref(&self) -> &Self::Target { &self.inner }
}
impl Drop for ActionBuffer {
    fn drop(&mut self) {
        if self.mode == 0 { return; }
        let mut value = std::mem::take(&mut *self.inner.lock()); value.clear();
        let bytes = value.capacity() * 4;
        let _ = CACHE.try_with(|cell| {
            let mut cache = cell.borrow_mut();
            if bytes == 0 || cache.bytes + bytes > LIMIT { return; }
            if self.mode == 2 {
                if let Some(old) = cache.nodes.insert(self.node, value) { cache.bytes -= old.capacity() * 4; }
            } else { cache.pool.push(value); }
            cache.bytes += bytes;
        });
    }
}
