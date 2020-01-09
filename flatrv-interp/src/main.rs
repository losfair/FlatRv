extern crate flatrv;

use flatrv::exec::{Machine, Host, Exception, GlobalContext, LowerAddressSpaceToken, EcallOutput};
use flatrv::elf;
use nix::sys::mman::{mmap, mprotect, ProtFlags, MapFlags};
use std::{fs::File, env, io::Read};

struct TestHost {

}

impl Host for TestHost {
    #[inline(never)]
    fn raise_exception(m: &mut Machine<Self>, pc: u32, exc: Exception) -> ! {
        panic!("Exception: pc = {:08x}, exc = {:?}", pc, exc);
    }
    #[inline(never)]
    fn ecall(m: &mut Machine<Self>, pc: u32) -> EcallOutput {
        let gregs: Vec<String> = m.gregs.iter().enumerate().map(|(i, x)| format!("\t{}: {:08x}", i, x)).collect();
        println!("ecall @ {:08x}: \n{}", pc, gregs.join("\n"));
        EcallOutput::default()
    }
    fn global_context() -> &'static GlobalContext<Self> {
        static GC: GlobalContext<TestHost> = GlobalContext::new();
        &GC
    }
}

struct OsMemoryManager {

}

impl elf::MemoryManager for OsMemoryManager {
    unsafe fn mmap(&mut self, start: usize, len: usize, prot: elf::SegmentProtection) -> bool {
        let mut prot_flags = ProtFlags::PROT_READ;
        if prot.contains(elf::SegmentProtection::W) {
            prot_flags |= ProtFlags::PROT_WRITE;
        }
        if prot.contains(elf::SegmentProtection::X) {
            prot_flags |= ProtFlags::PROT_EXEC;
        }
        match mmap(
            start as *mut _,
            len as _,
            prot_flags,
            MapFlags::MAP_PRIVATE | MapFlags::MAP_ANON | MapFlags::MAP_FIXED,
            -1,
            0
        ) {
            Ok(_) => true,
            Err(_) => false
        }
    }
    unsafe fn mprotect(&mut self, start: usize, len: usize, prot: elf::SegmentProtection) -> bool {
        let mut prot_flags = ProtFlags::PROT_READ;
        if prot.contains(elf::SegmentProtection::W) {
            prot_flags |= ProtFlags::PROT_WRITE;
        }
        if prot.contains(elf::SegmentProtection::X) {
            prot_flags |= ProtFlags::PROT_EXEC;
        }
        match mprotect(
            start as *mut _,
            len as _,
            prot_flags
        ) {
            Ok(_) => true,
            Err(_) => false
        }
    }
}

fn main() {
    let image_path = env::args().nth(1).expect("Image path required");
    let start_address = {
        let mut f = File::open(&image_path).expect("Cannot open image");
        let mut data: Vec<u8> = Vec::new();
        f.read_to_end(&mut data).expect("Cannot read image");
        unsafe {
            elf::load(&data, 0..0x100000, &mut OsMemoryManager {}).expect("Cannot load ELF").entry_address
        }
    };

    let mut machine = Machine::new(TestHost {});
    let token = unsafe {
        LowerAddressSpaceToken::new()
    };
    machine.run(start_address, &token);
    println!("{:?}", machine.gregs);
}
