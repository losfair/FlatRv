#![no_std]
#![feature(core_intrinsics, const_fn)]

#[macro_use]
extern crate bitflags;

pub mod exec;
pub mod elf;

#[cfg(debug_assertions)]
fn no_debug() {
    compile_error!("flatrv depends on release mode for correctness");
}
