extern crate flatrv;

use flatrv::exec::{Machine, Host, Exception, GlobalContext, LowerAddressSpaceToken, EcallOutput};

struct TestHost {

}

impl Host for TestHost {
    #[inline(never)]
    fn raise_exception(m: &mut Machine<Self>, pc: u32, exc: Exception) -> ! {
        panic!("Exception: pc = {:08x}, exc = {:?}", pc, exc);
    }
    #[inline(never)]
    fn ecall(m: &mut Machine<Self>, pc: u32) -> EcallOutput {
        println!("ecall @ {:08x}", pc);
        EcallOutput::default()
    }
    fn global_context() -> &'static GlobalContext<Self> {
        static GC: GlobalContext<TestHost> = GlobalContext::new();
        &GC
    }
}

fn main() {
    let mut machine = Machine::new(TestHost {});
    let token = unsafe {
        LowerAddressSpaceToken::new()
    };
    machine.run(0x10000, &token);
    println!("{}", machine.gregs[0]);
}
