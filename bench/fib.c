int __attribute__((noinline)) add(int a, int b) {
    return a + b;
}

long long add_ll(long long a, long long b) {
    return a + b;
}

int __attribute__((noinline)) mul(int a, int b) { return a * b; }
long long mul_ll(long long a, long long b) { return a * b; }

unsigned char __stack[65536];
unsigned char *__stack_end = __stack + sizeof(__stack);

void __attribute__((noinline)) ecall_print(int _x) {
    register int x asm("a0");
    x = _x;
    __asm__ volatile("ecall" :: "r"(x));
}

int fib(int n) {
    if(n == 1 || n == 2) return 1;
    else return fib(n - 1) + fib(n - 2);
}

void main() {
    int x = fib(40);
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

