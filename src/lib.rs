#![no_std]
#![feature(core_intrinsics, const_fn, atomic_min_max, stdsimd)]

#[cfg(feature = "elf")]
pub mod elf;
pub mod exec;

#[cfg(target_arch = "x86_64")]
#[path = "exec_x86_64.rs"]
mod exec_arch;

#[cfg(not(target_arch = "x86_64"))]
#[path = "exec_generic.rs"]
mod exec_arch;

#[cfg(debug_assertions)]
fn no_debug() {
    compile_error!("flatrv depends on release mode for correctness");
}
