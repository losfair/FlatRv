# FlatRv

A cross-platform [RISC-V](https://riscv.org/) interpreter that implements the RV32IMA instruction set.

Inspired by [wasm3](https://github.com/wasm3/wasm3), FlatRv depends on compiler optimizations to convert tail calls into one single
indirect branch, therefore preserving Rust's safety guarantee without giving up performance. FlatRv as an interpreter
is very efficient. It takes ~3.5 seconds on an Intel i9-9900K CPU to calculate `fib(40)` with on-the-fly instruction decoding, compared
to ~0.8 seconds on QEMU (RV32, TCG JIT) and ~2 seconds on wasm3 (WebAssembly, interpreter with a transforming pass).

Supports `no_std`.
