//! Validated FP32 defaults. Existing environment overrides remain authoritative.
//! Hardware/model guards preserve older CPU and checkpoint paths.
pub(crate) fn available() -> bool {
    if matches!(
        std::env::var("SPARSE_NN_F32_KERNEL").as_deref(),
        Ok("avx2") | Ok("scalar")
    ) {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
pub(crate) fn flag(name: &str) -> bool {
    std::env::var(name).map_or_else(|_| available(), |value| value == "1")
}
