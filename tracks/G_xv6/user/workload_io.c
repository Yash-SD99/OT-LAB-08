// ==============================================================================
// workload_io.c: I/O-Bound Interactive Benchmark for xv6
// Track G: xv6 Operating System Optimization
// ==============================================================================
#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"
#include "kernel/schedstat.h"

#define NUM_ROUNDS 50
#define BUFFER_SIZE 64

int main(int argc, char *argv[]) {
    int tickets = 30; // High priority for interactive responsiveness
    if (argc > 1) {
        tickets = atoi(argv[1]);
    }

    settickets(tickets);
    printf("[I/O Workload] Running %d interactive sleep/pipe I/O rounds with %d tickets...\n",
           NUM_ROUNDS, tickets);

    int pipe_fd[2];
    if (pipe(pipe_fd) < 0) {
        printf("Failed to create pipe!\n");
        exit(1);
    }

    char send_buf[BUFFER_SIZE] = "xv6_sched_io_ping";
    char recv_buf[BUFFER_SIZE];

    uint t_start = uptime();
    uint total_sleep_ticks = 0;

    for (int r = 0; r < NUM_ROUNDS; r++) {
        // Write to pipe
        write(pipe_fd[1], send_buf, sizeof(send_buf));
        
        // Brief voluntary yield / sleep simulating interactive user wait
        sleep(1);
        total_sleep_ticks += 1;

        // Read from pipe
        read(pipe_fd[0], recv_buf, sizeof(recv_buf));
    }

    close(pipe_fd[0]);
    close(pipe_fd[1]);

    uint elapsed = uptime() - t_start;
    struct schedstat st;
    if (schedstat(&st) == 0) {
        printf("[I/O Workload Complete]\n");
        printf("  PID:            %d\n", st.pid);
        printf("  Tickets:        %d\n", st.tickets);
        printf("  Total Elapsed:  %d ticks\n", elapsed);
        printf("  Sleep Ticks:    %d ticks\n", total_sleep_ticks);
        printf("  Ticks Running:  %d ticks\n", st.ticks_run);
        printf("  Ticks Waiting:  %d ticks\n", st.ticks_wait);
        printf("  Sched Contexts: %d\n", st.sched_count);
    }

    exit(0);
}
