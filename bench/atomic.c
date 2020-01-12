#include <stdatomic.h>

unsigned char __stack[65536];
unsigned char *__stack_end = __stack + sizeof(__stack);

_Atomic unsigned int value = 0;

void __attribute__((noinline)) ecall_print(int _x) {
    register int x asm("a0");
    x = _x;
    __asm__ volatile("ecall" :: "r"(x));
}

int bench_fetch_add(int n) {
    value = 0;
    for(int i = 0; i < n; i++) {
        if(atomic_fetch_add(&value, 1) != i) {
            __asm__ volatile("ebreak\n");
        }
    }
    return 0;
}

int bench_cmpxchg(int n) {
    value = 0;
    for(int i = 0; i < n; i++) {
        if(!atomic_compare_exchange_strong(&value, &i, i + 1)) {
            __asm__ volatile("ebreak\n");
        }
    }
    return 0;
}

void main() {
    //int x = bench_fetch_add(100000000);
    //ecall_print(x);
    int x = bench_cmpxchg(100000000);
    ecall_print(x);
    __asm__ volatile("ebreak\n");
    while(1);
}

void __attribute__((naked)) _start() {
    __asm__ volatile(
        "lui a0, %hi(__stack_end)\n"
        "lw sp, %lo(__stack_end)(a0)\n"
        "jal ra, main\n"
    );
}

