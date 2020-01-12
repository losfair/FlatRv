use core::sync::atomic::{AtomicU32, Ordering};
use core::arch::x86_64::{__cpuid_count, _xbegin, _xend, _XBEGIN_STARTED};

#[cfg(feature = "ext-a")]
pub fn atomic_is_supported() -> bool {
    let ebx = unsafe {
        __cpuid_count(7, 0)
    }.ebx;

    // CPUID.(EAX=7, ECX=0):EBX.RTM[bit 11]
    (ebx >> 11) & 1 != 0
}

#[inline(always)]
#[cfg(feature = "ext-a")]
pub fn atomic_lr(p: &AtomicU32) -> (u32, bool) {
    let status = unsafe {
        _xbegin()
    };
    let value = p.load(Ordering::Relaxed);
    if status == _XBEGIN_STARTED {
        (value, true)
    } else {
        (value, false)
    }
}

#[inline(always)]
#[cfg(feature = "ext-a")]
pub fn atomic_sc(p: &AtomicU32, val: u32) {
    p.store(val, Ordering::Relaxed);
    unsafe {
        _xend();
    }
}
