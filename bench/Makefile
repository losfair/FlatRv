CC := riscv64-linux-gnu-gcc-9
CFLAGS := -O2 -static -nostartfiles -nostdlib -march=rv32ima -mabi=ilp32

all: fib atomic

fib: fib.o
	$(CC) $(CFLAGS) -o fib.elf fib.o

atomic: atomic.o
	$(CC) $(CFLAGS) -o atomic.elf atomic.o

.PHONY: clean
clean:
	rm -f *.o *.elf
