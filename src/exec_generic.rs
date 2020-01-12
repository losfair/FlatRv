#[inline(always)]
#[cfg(feature = "ext-a")]
pub fn atomic_is_supported() -> bool {
    false
}

#[inline(always)]
#[cfg(feature = "ext-a")]
pub fn atomic_lr(_p: &core::sync::atomic::AtomicU32) -> (u32, bool) {
    unreachable!()
}

#[inline(always)]
#[cfg(feature = "ext-a")]
pub fn atomic_sc(_p: &core::sync::atomic::AtomicU32, _val: u32) -> bool {
    unreachable!()
}
