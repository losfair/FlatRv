#[cfg(feature = "ext-a")]
pub fn atomic_is_supported() -> bool {
    use core::arch::x86_64::__cpuid_count;
    let ebx = unsafe { __cpuid_count(7, 0) }.ebx;

    // CPUID.(EAX=7, ECX=0):EBX.RTM[bit 11]
    (ebx >> 11) & 1 != 0
}

#[inline(always)]
#[cfg(feature = "ext-a")]
pub fn atomic_lr(p: &core::sync::atomic::AtomicU32) -> (u32, bool) {
    use core::arch::x86_64::{_xbegin, _XBEGIN_STARTED};
    let status = unsafe { _xbegin() };
    let value = p.load(core::sync::atomic::Ordering::Relaxed);
    if status == _XBEGIN_STARTED {
        (value, true)
    } else {
        (value, false)
    }
}

#[inline(always)]
#[cfg(feature = "ext-a")]
pub fn atomic_sc(p: &core::sync::atomic::AtomicU32, val: u32) {
    use core::arch::x86_64::_xend;
    p.store(val, core::sync::atomic::Ordering::Relaxed);
    unsafe {
        _xend();
    }
}
