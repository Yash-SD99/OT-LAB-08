#!/usr/bin/env bash
# ==============================================================================
# Setup Script for xv6-riscv Development on GitHub Codespaces / Ubuntu Linux
# Track G: xv6 Operating System Scheduler Optimization
# ==============================================================================
set -e

echo "=== [Step 1/5] Updating package repositories ==="
sudo apt-get update -y

echo "=== [Step 2/5] Installing RISC-V GNU Toolchain and QEMU Emulator ==="
sudo apt-get install -y \
    build-essential \
    gcc-riscv64-unknown-elf \
    binutils-riscv64-unknown-elf \
    qemu-system-misc \
    gdb-multiarch \
    git

echo "=== [Step 3/5] Verifying toolchain installation ==="
riscv64-unknown-elf-gcc --version | head -n 1
qemu-system-riscv64 --version | head -n 1

echo "=== [Step 4/5] Cloning pristine MIT xv6-riscv repository ==="
XV6_DIR="$HOME/xv6-riscv"
if [ ! -d "$XV6_DIR" ]; then
    git clone https://github.com/mit-pdos/xv6-riscv.git "$XV6_DIR"
    echo "Cloned xv6-riscv to $XV6_DIR"
else
    echo "xv6-riscv already exists at $XV6_DIR"
fi

cd "$XV6_DIR"
git checkout riscv 2>/dev/null || true
git reset --hard HEAD
git clean -fd

echo "=== [Step 5/5] Applying Track G Lottery Scheduler & sys_schedstat Patch ==="
PATCH_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patches/0001_lottery_scheduler_and_schedstat.patch"

if [ -f "$PATCH_PATH" ]; then
    echo "Checking patch: $PATCH_PATH"
    git apply --check "$PATCH_PATH"
    git apply "$PATCH_PATH"
    echo "Patch applied successfully!"
else
    echo "ERROR: Patch file not found at $PATCH_PATH"
    exit 1
fi

echo "=== Copying user-level workload benchmarks into xv6 user/ ==="
USER_SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/user"
cp "$USER_SRC_DIR"/workload_cpu.c user/
cp "$USER_SRC_DIR"/workload_io.c user/
cp "$USER_SRC_DIR"/workload_mixed.c user/
cp "$USER_SRC_DIR"/sched_bench.c user/

echo "=== Building xv6 kernel and user space utilities ==="
make clean
make -j$(nproc)

echo ""
echo "=================================================================="
echo " xv6 Build Complete! Ready for Testing in GitHub Codespaces      "
echo "=================================================================="
echo "To run xv6 in QEMU, execute:"
echo "    cd $XV6_DIR"
echo "    make qemu"
echo ""
echo "Inside xv6 shell, run:"
echo "    $ usertests               # Verify kernel correctness and zero regressions"
echo "    $ sched_bench             # Benchmark proportional share lottery scheduler"
echo "    $ workload_cpu            # CPU-bound workload test"
echo "    $ workload_io             # I/O-bound workload test"
echo "    $ workload_mixed          # Mixed CPU & I/O workload test"
echo ""
echo "To exit QEMU: Press 'Ctrl-a' then 'x'."
