// ==============================================================================
// workload_mixed.c: Mixed CPU & I/O Benchmark for xv6
// Track G: xv6 Operating System Optimization
// ==============================================================================
#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"
#include "kernel/schedstat.h"

#define PHASES 30
#define CPU_BURST_OPS 2000000

int main(int argc, char *argv[]) {
    int tickets = 20;
    if (argc > 1) {
        tickets = atoi(argv[1]);
    }

    settickets(tickets);
    printf("[Mixed Workload] Running %d alternating CPU-burst and I/O phases with %d tickets...\n",
           PHASES, tickets);

    uint t_start = uptime();
    volatile unsigned long acc = 1;

    for (int p = 0; p < PHASES; p++) {
        // Phase 1: CPU-intensive burst (70% time)
        for (unsigned long i = 0; i < CPU_BURST_OPS; i++) {
            acc = (acc * 33 + i) ^ (acc >> 1);
        }

        // Phase 2: I/O yield (30% time)
        sleep(1);
    }

    uint elapsed = uptime() - t_start;
    struct schedstat st;
    if (schedstat(&st) == 0) {
        printf("[Mixed Workload Complete]\n");
        printf("  PID:            %d\n", st.pid);
        printf("  Tickets:        %d\n", st.tickets);
        printf("  Total Elapsed:  %d ticks\n", elapsed);
        printf("  Ticks Running:  %d ticks\n", st.ticks_run);
        printf("  Ticks Waiting:  %d ticks\n", st.ticks_wait);
        printf("  Sched Contexts: %d\n", st.sched_count);
        printf("  Accumulator:    %d\n", (int)acc);
    }

    exit(0);
}
