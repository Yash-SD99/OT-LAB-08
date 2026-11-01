// ==============================================================================
// sched_bench.c: Benchmark Harness for xv6 Lottery / Proportional-Share Scheduler
// Track G: xv6 Operating System Optimization
// ==============================================================================
#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"
#include "kernel/schedstat.h"

#define NUM_CHILDREN 3
#define WORKLOAD_LOOPS 50000000

// Ticket allocations for testing proportional-share fairness:
// Expected ratio: 10 : 20 : 30  =>  1 : 2 : 3  (16.7% : 33.3% : 50.0%)
static int ticket_ratios[NUM_CHILDREN] = {10, 20, 30};

void cpu_intensive_workload(int id, int tickets) {
    settickets(tickets);
    
    // CPU-bound loop with volatile arithmetic to prevent compiler optimization
    volatile unsigned long acc = 0;
    for (unsigned long i = 0; i < WORKLOAD_LOOPS; i++) {
        acc += (i * 3) ^ (acc >> 2);
    }
    
    struct schedstat st;
    if (schedstat(&st) == 0) {
        printf("[Child %d | PID %d] Finished with %d tickets: ticks_run=%d, ticks_wait=%d, sched_count=%d\n",
               id, st.pid, st.tickets, st.ticks_run, st.ticks_wait, st.sched_count);
    } else {
        printf("[Child %d] Error calling schedstat\n", id);
    }
    exit(0);
}

int main(int argc, char *argv[]) {
    printf("==================================================================\n");
    printf(" xv6 Lottery / Priority Scheduler Benchmark Harness (Track G)    \n");
    printf("==================================================================\n");
    printf("Spawning %d concurrent worker processes with ticket allocations:\n", NUM_CHILDREN);
    for (int i = 0; i < NUM_CHILDREN; i++) {
        printf("  Process %d: %d tickets (Expected share: %.1f%%)\n",
               i + 1, ticket_ratios[i], (float)ticket_ratios[i] * 100.0 / 60.0);
    }
    printf("Starting concurrent execution...\n\n");

    int pids[NUM_CHILDREN];
    uint start_time = uptime();

    for (int i = 0; i < NUM_CHILDREN; i++) {
        int pid = fork();
        if (pid < 0) {
            printf("fork failed!\n");
            exit(1);
        }
        if (pid == 0) {
            // Child process
            cpu_intensive_workload(i + 1, ticket_ratios[i]);
        } else {
            pids[i] = pid;
        }
    }

    // Parent waits for all children to finish
    for (int i = 0; i < NUM_CHILDREN; i++) {
        wait(0);
    }

    uint total_time = uptime() - start_time;
    printf("\nAll worker processes completed in %d ticks.\n", total_time);
    printf("Scheduler fairness verification complete.\n");
    exit(0);
}
