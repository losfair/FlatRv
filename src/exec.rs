//! Core decode and execution logic for RV32I instructions.

use core::hint::unreachable_unchecked;
use core::intrinsics::{likely, unlikely};
use core::marker::PhantomData;
use core::sync::atomic::Ordering;
use spin::Once;

const OPCODE_LUT_SIZE: usize = 1 << 10;

pub struct GlobalContext<H: Host> {
    lut: Once<OpcodeLut<H>>,
}

/// Machine state.
pub struct Machine<H: Host> {
    pub gregs: [u32; 32],
    #[cfg(feature = "ext-a")]
    lr_sc: bool,
    #[cfg(feature = "ext-a")]
    lr_sc_failed: bool,
    pub host: H,
    lut: &'static OpcodeLut<H>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Extension {
    #[cfg(feature = "ext-a")]
    A,
}

/// Host bindings.
pub trait Host: Sized + 'static {
    fn raise_exception(m: &mut Machine<Self>, pc: u32, exc: Exception) -> !;
    fn ecall(m: &mut Machine<Self>, pc: u32) -> EcallOutput;
    fn global_context() -> &'static GlobalContext<Self>;

    #[inline(always)]
    fn cycle_will_run(_m: &mut Machine<Self>, _pc: u32) {}

    #[inline(always)]
    fn mem_address_offset(_m: &mut Machine<Self>) -> usize {
        0
    }

    #[inline(always)]
    fn mem_address_limit(_m: &mut Machine<Self>) -> u32 {
        core::u32::MAX
    }

    #[inline(always)]
    fn extension_enabled(_ext: Extension) -> bool {
        false
    }
}

/// Result from ecall.
#[derive(Default)]
pub struct EcallOutput {
    pub new_pc: Option<u32>,
}

/// Exception.
///
/// Passed to `Host.raise_exception` when the interpreter caught an exception. However,
/// these exceptions *can* raise a hardware exception that is not handled by the interpreter,
/// and therefore need to be caught by host-side logic:
///
/// - Invalid opcode.
/// - Invalid memory reference.
/// - Division error.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Exception {
    InvalidOpcode,
    InvalidMemoryReference,
    InstructionAddressMisaligned,
    AtomicAddressMisaligned,
    InvalidLrscSequence,
    Ebreak,
    Wfi,
}

/// Exception type.
///
/// Not all

/// A token that represents safe access to the lower 32-bit address space.
pub struct LowerAddressSpaceToken(PhantomData<()>);

/// Program counter value aligned to 4 bytes.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct AlignedPC {
    _unsafe_inner: u32,
}

impl AlignedPC {
    #[inline(always)]
    fn new(inner: u32) -> Option<AlignedPC> {
        if inner & 0b11 == 0 {
            Some(AlignedPC {
                _unsafe_inner: inner,
            })
        } else {
            None
        }
    }

    #[inline(always)]
    fn next(self) -> AlignedPC {
        AlignedPC {
            _unsafe_inner: self._unsafe_inner + 4,
        }
    }
}

impl From<AlignedPC> for u32 {
    #[inline(always)]
    fn from(me: AlignedPC) -> u32 {
        // Hint to the compiler that the inner value is always a multiple of 4.
        if me._unsafe_inner & 0b11 != 0 {
            unsafe {
                unreachable_unchecked();
            }
        }
        me._unsafe_inner
    }
}

type OpcodeHandler<H> = fn(&mut Machine<H>, pc: AlignedPC, inst: u32);

struct OpcodeLut<H: Host> {
    opcode_funct3: [OpcodeHandler<H>; OPCODE_LUT_SIZE],
}

impl LowerAddressSpaceToken {
    /// Creates a `LowerAddressSpaceToken`.
    ///
    /// This function is unsafe because the caller must guarantee that:
    ///
    /// - The lower 32-bit address space is not used.
    /// - Exception handlers are properly set up to catch access faults from the lower 32-bit address space.
    pub unsafe fn new() -> LowerAddressSpaceToken {
        LowerAddressSpaceToken(PhantomData)
    }
}

impl<H: Host> GlobalContext<H> {
    pub const fn new() -> GlobalContext<H> {
        GlobalContext { lut: Once::new() }
    }
}

impl<H: Host> OpcodeLut<H> {
    /// Builds an opcode lookup table.
    fn new() -> OpcodeLut<H> {
        let mut opcode_funct3: [Option<OpcodeHandler<H>>; OPCODE_LUT_SIZE] =
            [None; OPCODE_LUT_SIZE];

        for i in 0..=0b111 {
            opcode_funct3[(0b0110111 << 3) + i] = Some(Machine::i_lui);
            opcode_funct3[(0b0010111 << 3) + i] = Some(Machine::i_auipc);
            opcode_funct3[(0b1101111 << 3) + i] = Some(Machine::i_jal);
        }

        opcode_funct3[0b1100111_000] = Some(Machine::i_jalr);
        opcode_funct3[0b1100011_000] = Some(Machine::i_beq);
        opcode_funct3[0b1100011_001] = Some(Machine::i_bne);
        opcode_funct3[0b1100011_100] = Some(Machine::i_blt);
        opcode_funct3[0b1100011_101] = Some(Machine::i_bge);
        opcode_funct3[0b1100011_110] = Some(Machine::i_bltu);
        opcode_funct3[0b1100011_111] = Some(Machine::i_bgeu);
        opcode_funct3[0b0000011_000] = Some(Machine::i_lb);
        opcode_funct3[0b0000011_001] = Some(Machine::i_lh);
        opcode_funct3[0b0000011_010] = Some(Machine::i_lw);
        opcode_funct3[0b0000011_100] = Some(Machine::i_lbu);
        opcode_funct3[0b0000011_101] = Some(Machine::i_lhu);
        opcode_funct3[0b0100011_000] = Some(Machine::i_sb);
        opcode_funct3[0b0100011_001] = Some(Machine::i_sh);
        opcode_funct3[0b0100011_010] = Some(Machine::i_sw);
        opcode_funct3[0b0010011_000] = Some(Machine::i_addi);
        opcode_funct3[0b0010011_010] = Some(Machine::i_slti);
        opcode_funct3[0b0010011_011] = Some(Machine::i_sltiu);
        opcode_funct3[0b0010011_100] = Some(Machine::i_xori);
        opcode_funct3[0b0010011_110] = Some(Machine::i_ori);
        opcode_funct3[0b0010011_111] = Some(Machine::i_andi);
        opcode_funct3[0b0010011_001] = Some(Machine::i_slli);
        opcode_funct3[0b0010011_101] = Some(Machine::i_srli_srai);
        opcode_funct3[0b0110011_000] = Some(Machine::i_add_sub_mul);
        opcode_funct3[0b0110011_001] = Some(Machine::i_sll_mulh);
        opcode_funct3[0b0110011_010] = Some(Machine::i_slt_mulhsu);
        opcode_funct3[0b0110011_011] = Some(Machine::i_sltu_mulhu);
        opcode_funct3[0b0110011_100] = Some(Machine::i_xor_div);
        opcode_funct3[0b0110011_101] = Some(Machine::i_srl_sra_divu);
        opcode_funct3[0b0110011_110] = Some(Machine::i_or_rem);
        opcode_funct3[0b0110011_111] = Some(Machine::i_and_remu);
        opcode_funct3[0b0001111_000] = Some(Machine::i_fence);
        opcode_funct3[0b1110011_000] = Some(Machine::i_ecallbreak);

        #[cfg(feature = "ext-a")]
        {
            if H::extension_enabled(Extension::A) {
                if crate::exec_arch::atomic_is_supported() {
                    opcode_funct3[0b0101111_010] = Some(Machine::i_amo_32);
                }
            }
        }

        // Replace nulls with -1.
        //
        // This is required because address 0 is controlled by user code and is unsafe to dereference
        // as a "null pointer".
        let opcode_funct3 = unsafe {
            use core::mem::transmute;
            let mut as_u64 = transmute::<_, [u64; OPCODE_LUT_SIZE]>(opcode_funct3);
            for f in as_u64.iter_mut() {
                if *f == 0 {
                    *f = core::u64::MAX;
                }
            }
            transmute::<_, [OpcodeHandler<H>; OPCODE_LUT_SIZE]>(as_u64)
        };

        OpcodeLut { opcode_funct3 }
    }
}

impl<H: Host> Machine<H> {
    /// Creates a clean `Machine` in its initial state.
    pub fn new(host: H) -> Machine<H> {
        let ctx = H::global_context();
        Machine {
            gregs: [0u32; 32],
            #[cfg(feature = "ext-a")]
            lr_sc: false,
            #[cfg(feature = "ext-a")]
            lr_sc_failed: false,
            lut: ctx.lut.call_once(OpcodeLut::new),
            host,
        }
    }

    /// Starts execution from `pc`.
    #[inline(never)]
    pub fn run(&mut self, pc: u32, _: &LowerAddressSpaceToken) {
        self.do_dispatch(pc)
    }

    /// Dispatches execution flow by instruction at `pc`.
    #[inline(always)]
    fn do_dispatch<T: Into<u32> + Copy>(&mut self, pc: T) {
        // Check PC alignment. Optimized out for non-branch opcodes.
        let pc = if let Some(x) = AlignedPC::new(pc.into()) {
            x
        } else {
            H::raise_exception(self, pc.into(), Exception::InstructionAddressMisaligned);
        };

        H::cycle_will_run(self, u32::from(pc));

        let inst = self.mem_load_32(pc, pc.into());
        let opcode = (inst & 0b1111111) as usize;
        let funct3 = ((inst >> 12) & 0b111) as usize;

        // `x0` should always be zero.
        self.gregs[0] = 0;
        self.lut.opcode_funct3[(opcode << 3) | funct3](self, pc, inst)
    }

    fn i_lui(&mut self, _pc: AlignedPC, _inst: u32) {
        self.gregs[inst_rd(_inst)] = inst_u_imm(_inst);
        self.do_dispatch(_pc.next())
    }

    fn i_auipc(&mut self, _pc: AlignedPC, _inst: u32) {
        self.gregs[inst_rd(_inst)] = inst_u_imm(_inst).wrapping_add(_pc.into());
        self.do_dispatch(_pc.next())
    }

    fn i_jal(&mut self, _pc: AlignedPC, _inst: u32) {
        let offset = sext21b(inst_j_imm(_inst));
        self.gregs[inst_rd(_inst)] = u32::from(_pc) + 4;
        self.do_dispatch(offset + u32::from(_pc))
    }

    fn i_jalr(&mut self, _pc: AlignedPC, _inst: u32) {
        let target = (self.gregs[inst_rs1(_inst)] + sext12b(inst_i_imm(_inst))) & !0b1u32;
        self.gregs[inst_rd(_inst)] = u32::from(_pc) + 4;
        self.do_dispatch(target)
    }
    fn i_beq(&mut self, _pc: AlignedPC, _inst: u32) {
        if self.gregs[inst_rs1(_inst)] == self.gregs[inst_rs2(_inst)] {
            let offset = sext13b(inst_b_imm(_inst));
            self.do_dispatch(offset + u32::from(_pc))
        } else {
            self.do_dispatch(_pc.next())
        }
    }
    fn i_bne(&mut self, _pc: AlignedPC, _inst: u32) {
        if self.gregs[inst_rs1(_inst)] != self.gregs[inst_rs2(_inst)] {
            let offset = sext13b(inst_b_imm(_inst));
            self.do_dispatch(offset + u32::from(_pc))
        } else {
            self.do_dispatch(_pc.next())
        }
    }
    fn i_blt(&mut self, _pc: AlignedPC, _inst: u32) {
        if (self.gregs[inst_rs1(_inst)] as i32) < (self.gregs[inst_rs2(_inst)] as i32) {
            let offset = sext13b(inst_b_imm(_inst));
            self.do_dispatch(offset + u32::from(_pc))
        } else {
            self.do_dispatch(_pc.next())
        }
    }
    fn i_bge(&mut self, _pc: AlignedPC, _inst: u32) {
        if (self.gregs[inst_rs1(_inst)] as i32) >= (self.gregs[inst_rs2(_inst)] as i32) {
            let offset = sext13b(inst_b_imm(_inst));
            self.do_dispatch(offset + u32::from(_pc))
        } else {
            self.do_dispatch(_pc.next())
        }
    }
    fn i_bltu(&mut self, _pc: AlignedPC, _inst: u32) {
        if self.gregs[inst_rs1(_inst)] < self.gregs[inst_rs2(_inst)] {
            let offset = sext13b(inst_b_imm(_inst));
            self.do_dispatch(offset + u32::from(_pc))
        } else {
            self.do_dispatch(_pc.next())
        }
    }
    fn i_bgeu(&mut self, _pc: AlignedPC, _inst: u32) {
        if self.gregs[inst_rs1(_inst)] >= self.gregs[inst_rs2(_inst)] {
            let offset = sext13b(inst_b_imm(_inst));
            self.do_dispatch(offset + u32::from(_pc))
        } else {
            self.do_dispatch(_pc.next())
        }
    }
    fn i_lb(&mut self, _pc: AlignedPC, _inst: u32) {
        self.deny_lr_sc(_pc);
        let offset = sext12b(inst_i_imm(_inst));
        let base = self.gregs[inst_rs1(_inst)];
        self.gregs[inst_rd(_inst)] = self.mem_load_8(_pc, base + offset) as i8 as i32 as u32;
        self.do_dispatch(_pc.next())
    }
    fn i_lh(&mut self, _pc: AlignedPC, _inst: u32) {
        self.deny_lr_sc(_pc);
        let offset = sext12b(inst_i_imm(_inst));
        let base = self.gregs[inst_rs1(_inst)];
        self.gregs[inst_rd(_inst)] = self.mem_load_16(_pc, base + offset) as i16 as i32 as u32;
        self.do_dispatch(_pc.next())
    }
    fn i_lw(&mut self, _pc: AlignedPC, _inst: u32) {
        self.deny_lr_sc(_pc);
        let offset = sext12b(inst_i_imm(_inst));
        let base = self.gregs[inst_rs1(_inst)];
        self.gregs[inst_rd(_inst)] = self.mem_load_32(_pc, base + offset);
        self.do_dispatch(_pc.next())
    }
    fn i_lbu(&mut self, _pc: AlignedPC, _inst: u32) {
        self.deny_lr_sc(_pc);
        let offset = sext12b(inst_i_imm(_inst));
        let base = self.gregs[inst_rs1(_inst)];
        self.gregs[inst_rd(_inst)] = self.mem_load_8(_pc, base + offset) as u32;
        self.do_dispatch(_pc.next())
    }
    fn i_lhu(&mut self, _pc: AlignedPC, _inst: u32) {
        self.deny_lr_sc(_pc);
        let offset = sext12b(inst_i_imm(_inst));
        let base = self.gregs[inst_rs1(_inst)];
        self.gregs[inst_rd(_inst)] = self.mem_load_16(_pc, base + offset) as u32;
        self.do_dispatch(_pc.next())
    }
    fn i_sb(&mut self, _pc: AlignedPC, _inst: u32) {
        self.deny_lr_sc(_pc);
        let offset = sext12b(inst_s_imm(_inst));
        let base = self.gregs[inst_rs1(_inst)];
        self.mem_store_8(_pc, base + offset, self.gregs[inst_rs2(_inst)] as u8);
        self.do_dispatch(_pc.next())
    }
    fn i_sh(&mut self, _pc: AlignedPC, _inst: u32) {
        self.deny_lr_sc(_pc);
        let offset = sext12b(inst_s_imm(_inst));
        let base = self.gregs[inst_rs1(_inst)];
        self.mem_store_16(_pc, base + offset, self.gregs[inst_rs2(_inst)] as u16);
        self.do_dispatch(_pc.next())
    }
    fn i_sw(&mut self, _pc: AlignedPC, _inst: u32) {
        self.deny_lr_sc(_pc);
        let offset = sext12b(inst_s_imm(_inst));
        let base = self.gregs[inst_rs1(_inst)];
        self.mem_store_32(_pc, base + offset, self.gregs[inst_rs2(_inst)]);
        self.do_dispatch(_pc.next())
    }
    fn i_addi(&mut self, _pc: AlignedPC, _inst: u32) {
        self.gregs[inst_rd(_inst)] =
            self.gregs[inst_rs1(_inst)].wrapping_add(sext12b(inst_i_imm(_inst)));
        self.do_dispatch(_pc.next())
    }
    fn i_slti(&mut self, _pc: AlignedPC, _inst: u32) {
        if (self.gregs[inst_rs1(_inst)] as i32) < (sext12b(inst_i_imm(_inst)) as i32) {
            self.gregs[inst_rd(_inst)] = 1;
        } else {
            self.gregs[inst_rd(_inst)] = 0;
        }
        self.do_dispatch(_pc.next())
    }
    fn i_sltiu(&mut self, _pc: AlignedPC, _inst: u32) {
        if self.gregs[inst_rs1(_inst)] < sext12b(inst_i_imm(_inst)) {
            self.gregs[inst_rd(_inst)] = 1;
        } else {
            self.gregs[inst_rd(_inst)] = 0;
        }
        self.do_dispatch(_pc.next())
    }
    fn i_xori(&mut self, _pc: AlignedPC, _inst: u32) {
        self.gregs[inst_rd(_inst)] = self.gregs[inst_rs1(_inst)] ^ sext12b(inst_i_imm(_inst));
        self.do_dispatch(_pc.next())
    }
    fn i_ori(&mut self, _pc: AlignedPC, _inst: u32) {
        self.gregs[inst_rd(_inst)] = self.gregs[inst_rs1(_inst)] | sext12b(inst_i_imm(_inst));
        self.do_dispatch(_pc.next())
    }
    fn i_andi(&mut self, _pc: AlignedPC, _inst: u32) {
        self.gregs[inst_rd(_inst)] = self.gregs[inst_rs1(_inst)] & sext12b(inst_i_imm(_inst));
        self.do_dispatch(_pc.next())
    }
    fn i_slli(&mut self, _pc: AlignedPC, _inst: u32) {
        self.gregs[inst_rd(_inst)] =
            self.gregs[inst_rs1(_inst)].wrapping_shl(inst_i_imm(_inst) & 0b11111);
        self.do_dispatch(_pc.next())
    }
    fn i_srli_srai(&mut self, _pc: AlignedPC, _inst: u32) {
        let imm = inst_i_imm(_inst);
        if imm & (1 << 10) == 0 {
            // srli
            self.gregs[inst_rd(_inst)] = self.gregs[inst_rs1(_inst)].wrapping_shr(imm & 0b11111);
        } else {
            // srai
            self.gregs[inst_rd(_inst)] =
                ((self.gregs[inst_rs1(_inst)] as i32).wrapping_shr(imm & 0b11111)) as u32;
        }
        self.do_dispatch(_pc.next())
    }
    fn i_add_sub_mul(&mut self, _pc: AlignedPC, _inst: u32) {
        let funct7 = inst_r_funct7(_inst);
        let left = self.gregs[inst_rs1(_inst)];
        let right = self.gregs[inst_rs2(_inst)];
        if likely(funct7 == 0) {
            // add
            self.gregs[inst_rd(_inst)] = left.wrapping_add(right);
        } else if likely(funct7 == 0b0100000) {
            // sub
            self.gregs[inst_rd(_inst)] = left.wrapping_sub(right);
        } else if likely(funct7 == 1) {
            // mul
            return self.i_mul(_pc, _inst);
        } else {
            H::raise_exception(self, _pc.into(), Exception::InvalidOpcode);
        }
        self.do_dispatch(_pc.next())
    }
    fn i_sll_mulh(&mut self, _pc: AlignedPC, _inst: u32) {
        let funct7 = inst_r_funct7(_inst);

        if unlikely(funct7 != 0) {
            if likely(funct7 == 1) {
                return self.i_mulh(_pc, _inst);
            }
            H::raise_exception(self, _pc.into(), Exception::InvalidOpcode);
        }

        self.gregs[inst_rd(_inst)] =
            self.gregs[inst_rs1(_inst)].wrapping_shl(self.gregs[inst_rs2(_inst)] & 0b11111);
        self.do_dispatch(_pc.next())
    }
    fn i_slt_mulhsu(&mut self, _pc: AlignedPC, _inst: u32) {
        let funct7 = inst_r_funct7(_inst);

        if unlikely(funct7 != 0) {
            if likely(funct7 == 1) {
                return self.i_mulhsu(_pc, _inst);
            }
            H::raise_exception(self, _pc.into(), Exception::InvalidOpcode);
        }

        if (self.gregs[inst_rs1(_inst)] as i32) < (self.gregs[inst_rs2(_inst)] as i32) {
            self.gregs[inst_rd(_inst)] = 1;
        } else {
            self.gregs[inst_rd(_inst)] = 0;
        }
        self.do_dispatch(_pc.next())
    }
    fn i_sltu_mulhu(&mut self, _pc: AlignedPC, _inst: u32) {
        let funct7 = inst_r_funct7(_inst);

        if unlikely(funct7 != 0) {
            if likely(funct7 == 1) {
                return self.i_mulhu(_pc, _inst);
            }
            H::raise_exception(self, _pc.into(), Exception::InvalidOpcode);
        }

        if self.gregs[inst_rs1(_inst)] < self.gregs[inst_rs2(_inst)] {
            self.gregs[inst_rd(_inst)] = 1;
        } else {
            self.gregs[inst_rd(_inst)] = 0;
        }
        self.do_dispatch(_pc.next())
    }
    fn i_xor_div(&mut self, _pc: AlignedPC, _inst: u32) {
        let funct7 = inst_r_funct7(_inst);

        if unlikely(funct7 != 0) {
            if likely(funct7 == 1) {
                return self.i_div(_pc, _inst);
            }
            H::raise_exception(self, _pc.into(), Exception::InvalidOpcode);
        }

        self.gregs[inst_rd(_inst)] = self.gregs[inst_rs1(_inst)] ^ self.gregs[inst_rs2(_inst)];
        self.do_dispatch(_pc.next())
    }
    fn i_srl_sra_divu(&mut self, _pc: AlignedPC, _inst: u32) {
        let funct7 = inst_r_funct7(_inst);
        let left = self.gregs[inst_rs1(_inst)];
        let right = self.gregs[inst_rs2(_inst)] & 0b11111;
        if likely(funct7 == 0) {
            // srl
            self.gregs[inst_rd(_inst)] = left.wrapping_shr(right);
        } else if likely(funct7 == 0b0100000) {
            // sra
            self.gregs[inst_rd(_inst)] = ((left as i32).wrapping_shr(right)) as u32;
        } else if likely(funct7 == 1) {
            // divu
            return self.i_divu(_pc, _inst);
        } else {
            H::raise_exception(self, _pc.into(), Exception::InvalidOpcode);
        }
        self.do_dispatch(_pc.next())
    }
    fn i_or_rem(&mut self, _pc: AlignedPC, _inst: u32) {
        let funct7 = inst_r_funct7(_inst);
        if unlikely(funct7 != 0) {
            if likely(funct7 == 1) {
                return self.i_rem(_pc, _inst);
            }
            H::raise_exception(self, _pc.into(), Exception::InvalidOpcode);
        }

        self.gregs[inst_rd(_inst)] = self.gregs[inst_rs1(_inst)] | self.gregs[inst_rs2(_inst)];
        self.do_dispatch(_pc.next())
    }
    fn i_and_remu(&mut self, _pc: AlignedPC, _inst: u32) {
        let funct7 = inst_r_funct7(_inst);
        if unlikely(funct7 != 0) {
            if likely(funct7 == 1) {
                return self.i_remu(_pc, _inst);
            }
            H::raise_exception(self, _pc.into(), Exception::InvalidOpcode);
        }

        self.gregs[inst_rd(_inst)] = self.gregs[inst_rs1(_inst)] & self.gregs[inst_rs2(_inst)];
        self.do_dispatch(_pc.next())
    }
    fn i_fence(&mut self, _pc: AlignedPC, _inst: u32) {
        core::sync::atomic::fence(Ordering::SeqCst);
        self.do_dispatch(_pc.next())
    }
    fn i_ecallbreak(&mut self, _pc: AlignedPC, _inst: u32) {
        self.deny_lr_sc(_pc);
        let func = inst_i_imm(_inst);
        if likely(func == 0) {
            let output = H::ecall(self, u32::from(_pc.next()));
            if let Some(new_pc) = output.new_pc {
                self.do_dispatch(new_pc)
            } else {
                self.do_dispatch(_pc.next())
            }
        } else if func == 1 {
            H::raise_exception(self, _pc.into(), Exception::Ebreak);
        } else if func == 0b0001_0000_0101 {
            H::raise_exception(self, _pc.into(), Exception::Wfi);
        } else {
            H::raise_exception(self, _pc.into(), Exception::InvalidOpcode);
        }
    }

    // M extension
    fn i_mul(&mut self, _pc: AlignedPC, _inst: u32) {
        self.gregs[inst_rd(_inst)] = self.gregs[inst_rs1(_inst)] * self.gregs[inst_rs2(_inst)];
        self.do_dispatch(_pc.next())
    }
    fn i_mulh(&mut self, _pc: AlignedPC, _inst: u32) {
        self.gregs[inst_rd(_inst)] = ((((self.gregs[inst_rs1(_inst)] as i32 as i64)
            * (self.gregs[inst_rs2(_inst)] as i32 as i64))
            as u64)
            >> 32) as u32;
        self.do_dispatch(_pc.next())
    }
    fn i_mulhsu(&mut self, _pc: AlignedPC, _inst: u32) {
        self.gregs[inst_rd(_inst)] = ((((self.gregs[inst_rs1(_inst)] as i32 as i64)
            * (self.gregs[inst_rs2(_inst)] as u32 as i64))
            as u64)
            >> 32) as u32;
        self.do_dispatch(_pc.next())
    }
    fn i_mulhu(&mut self, _pc: AlignedPC, _inst: u32) {
        self.gregs[inst_rd(_inst)] = ((((self.gregs[inst_rs1(_inst)] as u32 as i64)
            * (self.gregs[inst_rs2(_inst)] as u32 as i64))
            as u64)
            >> 32) as u32;
        self.do_dispatch(_pc.next())
    }
    fn i_div(&mut self, _pc: AlignedPC, _inst: u32) {
        self.gregs[inst_rd(_inst)] = unchecked_div_i32(
            self.gregs[inst_rs1(_inst)] as i32,
            self.gregs[inst_rs2(_inst)] as i32,
        ) as u32;
        self.do_dispatch(_pc.next())
    }
    fn i_divu(&mut self, _pc: AlignedPC, _inst: u32) {
        self.gregs[inst_rd(_inst)] =
            unchecked_div_u32(self.gregs[inst_rs1(_inst)], self.gregs[inst_rs2(_inst)]);
        self.do_dispatch(_pc.next())
    }
    fn i_rem(&mut self, _pc: AlignedPC, _inst: u32) {
        self.gregs[inst_rd(_inst)] = unchecked_rem_i32(
            self.gregs[inst_rs1(_inst)] as i32,
            self.gregs[inst_rs2(_inst)] as i32,
        ) as u32;
        self.do_dispatch(_pc.next())
    }
    fn i_remu(&mut self, _pc: AlignedPC, _inst: u32) {
        self.gregs[inst_rd(_inst)] =
            unchecked_rem_u32(self.gregs[inst_rs1(_inst)], self.gregs[inst_rs2(_inst)]);
        self.do_dispatch(_pc.next())
    }

    #[cfg(feature = "ext-a")]
    fn i_amo_32(&mut self, _pc: AlignedPC, _inst: u32) {
        use core::sync::atomic::{AtomicI32, AtomicU32};

        let funct7 = inst_r_funct7(_inst);
        let func = funct7 >> 2;

        // All orderings are supported by all possible Rust-native atomic operations used here.
        let ordering = match funct7 & 0b11 {
            0b00 => Ordering::Relaxed,
            0b01 => Ordering::Release,
            0b10 => Ordering::Acquire,
            0b11 => Ordering::SeqCst,
            _ => unreachable!(),
        };

        let memref = self.gregs[inst_rs1(_inst)];
        if unlikely(memref & 0b11 != 0) {
            H::raise_exception(self, _pc.into(), Exception::AtomicAddressMisaligned);
        }
        let memref =
            unsafe { &*(self.translate_ptr(_pc, memref) as *mut AtomicU32) };
        let rs2 = self.gregs[inst_rs2(_inst)];
        let rd = &mut self.gregs[inst_rd(_inst)];
        match func {
            0b00010 => {
                // LR.W
                self.deny_lr_sc(_pc);
                let rd = &mut self.gregs[inst_rd(_inst)];

                let (value, ok) = crate::exec_arch::atomic_lr(memref);

                self.lr_sc = true;
                self.lr_sc_failed = !ok;
                *rd = value;
            }
            0b00011 => {
                // SC.W
                if self.lr_sc && !self.lr_sc_failed {
                    crate::exec_arch::atomic_sc(memref, rs2);
                    *rd = 0;
                } else {
                    *rd = 1;
                }
                self.lr_sc = false;
            }
            0b00001 => {
                // AMOSWAP.W
                *rd = memref.swap(rs2, ordering);
            }
            0b00000 => {
                // AMOADD.W
                *rd = memref.fetch_add(rs2, ordering);
            }
            0b00100 => {
                // AMOXOR.W
                *rd = memref.fetch_xor(rs2, ordering);
            }
            0b01100 => {
                // AMOAND.W
                *rd = memref.fetch_and(rs2, ordering);
            }
            0b01000 => {
                // AMOOR.w
                *rd = memref.fetch_or(rs2, ordering);
            }
            0b10000 => {
                // AMOMIN.W
                *rd = unsafe { core::mem::transmute::<&AtomicU32, &AtomicI32>(memref) }
                    .fetch_min(rs2 as i32, ordering) as u32;
            }
            0b10100 => {
                // AMOMAX.W
                *rd = unsafe { core::mem::transmute::<&AtomicU32, &AtomicI32>(memref) }
                    .fetch_max(rs2 as i32, ordering) as u32;
            }
            0b11000 => {
                // AMOMINU.W
                *rd = memref.fetch_min(rs2, ordering);
            }
            0b11100 => {
                // AMOMAXU.W
                *rd = memref.fetch_max(rs2, ordering);
            }
            _ => H::raise_exception(self, _pc.into(), Exception::InvalidOpcode),
        }

        self.do_dispatch(_pc.next())
    }

    #[inline(always)]
    #[cfg(feature = "ext-a")]
    fn deny_lr_sc(&mut self, _pc: AlignedPC) {
        if unlikely(self.lr_sc) {
            H::raise_exception(self, _pc.into(), Exception::InvalidLrscSequence);
        }
    }

    #[inline(always)]
    #[cfg(not(feature = "ext-a"))]
    fn deny_lr_sc(&mut self, _pc: AlignedPC) {}

    #[inline(always)]
    unsafe fn translate_ptr<T>(&mut self, _pc: AlignedPC, ptr: u32) -> *mut T {
        if unlikely(ptr > H::mem_address_limit(self)) {
            H::raise_exception(self, _pc.into(), Exception::InvalidMemoryReference);
        }
        (H::mem_address_offset(self) + (ptr as usize)) as *mut T
    }

    #[inline(always)]
    fn mem_load_32(&mut self, _pc: AlignedPC, ptr: u32) -> u32 {
        unsafe { core::ptr::read_volatile(self.translate_ptr::<u32>(_pc, ptr)) }
    }

    #[inline(always)]
    fn mem_store_32(&mut self, _pc: AlignedPC, ptr: u32, val: u32) {
        unsafe { core::ptr::write_volatile(self.translate_ptr::<u32>(_pc, ptr), val) }
    }

    #[inline(always)]
    fn mem_load_16(&mut self, _pc: AlignedPC, ptr: u32) -> u16 {
        unsafe { core::ptr::read_volatile(self.translate_ptr::<u16>(_pc, ptr)) }
    }

    #[inline(always)]
    fn mem_store_16(&mut self, _pc: AlignedPC, ptr: u32, val: u16) {
        unsafe { core::ptr::write_volatile(self.translate_ptr::<u16>(_pc, ptr), val) }
    }

    #[inline(always)]
    fn mem_load_8(&mut self, _pc: AlignedPC, ptr: u32) -> u8 {
        unsafe { core::ptr::read_volatile(self.translate_ptr::<u8>(_pc, ptr)) }
    }

    #[inline(always)]
    fn mem_store_8(&mut self, _pc: AlignedPC, ptr: u32, val: u8) {
        unsafe { core::ptr::write_volatile(self.translate_ptr::<u8>(_pc, ptr), val) }
    }
}

#[inline(always)]
fn inst_rd(inst: u32) -> usize {
    ((inst >> 7) & 0b11111) as usize
}

#[inline(always)]
fn inst_rs1(inst: u32) -> usize {
    ((inst >> 15) & 0b11111) as usize
}

#[inline(always)]
fn inst_rs2(inst: u32) -> usize {
    ((inst >> 20) & 0b11111) as usize
}

#[inline(always)]
fn inst_u_imm(inst: u32) -> u32 {
    (inst >> 12) << 12
}

#[inline(always)]
fn inst_s_imm(inst: u32) -> u32 {
    ((inst >> 7) & 0b11111) | (((inst >> 25) & 0b1111111) << 5)
}

#[inline(always)]
fn inst_i_imm(inst: u32) -> u32 {
    inst >> 20
}

#[inline(always)]
fn inst_b_imm(inst: u32) -> u32 {
    (((inst >> 8) & 0b1111) << 1)
        | (((inst >> 7) & 0b1) << 11)
        | (((inst >> 25) & 0b111111) << 5)
        | (((inst >> 31) & 0b1) << 12)
}

#[inline(always)]
fn inst_j_imm(inst: u32) -> u32 {
    (((inst >> 12) & 0b11111111) << 12)
        | (((inst >> 20) & 0b1) << 11)
        | (((inst >> 21) & 0b1111111111) << 1)
        | (((inst >> 31) & 0b1) << 20)
}

#[inline(always)]
fn inst_r_funct7(inst: u32) -> u8 {
    (inst >> 25) as u8
}

#[inline(always)]
fn sext12b(src: u32) -> u32 {
    if src & (1 << 11) != 0 {
        src | !0b1111_1111_1111u32
    } else {
        src
    }
}

#[inline(always)]
fn sext13b(src: u32) -> u32 {
    if src & (1 << 12) != 0 {
        src | !0b1_1111_1111_1111u32
    } else {
        src
    }
}

#[inline(always)]
fn sext21b(src: u32) -> u32 {
    if src & (1 << 20) != 0 {
        src | !0b111_111_111_111_111_111_111u32
    } else {
        src
    }
}

#[inline(always)]
fn unchecked_div_u32(a: u32, b: u32) -> u32 {
    unsafe { core::intrinsics::unchecked_div(a, b) }
}

#[inline(always)]
fn unchecked_div_i32(a: i32, b: i32) -> i32 {
    unsafe { core::intrinsics::unchecked_div(a, b) }
}

#[inline(always)]
fn unchecked_rem_u32(a: u32, b: u32) -> u32 {
    unsafe { core::intrinsics::unchecked_rem(a, b) }
}

#[inline(always)]
fn unchecked_rem_i32(a: i32, b: i32) -> i32 {
    unsafe { core::intrinsics::unchecked_rem(a, b) }
}
