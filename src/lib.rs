#![no_std]
#![feature(core_intrinsics, const_fn, atomic_min_max, stdsimd)]

#[macro_use]
extern crate bitflags;

pub mod exec;
pub mod elf;

#[cfg(target_arch = "x86_64")]
#[path = "exec_x86_64.rs"]
mod exec_arch;

#[cfg(debug_assertions)]
fn no_debug() {
    compile_error!("flatrv depends on release mode for correctness");
}
