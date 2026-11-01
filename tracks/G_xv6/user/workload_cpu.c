// ==============================================================================
// workload_cpu.c: Pure CPU-Bound Benchmark for xv6
// Track G: xv6 Operating System Optimization
// ==============================================================================
#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"
#include "kernel/schedstat.h"

#define MATRIX_DIM 32
#define ITERATIONS 500

static int A[MATRIX_DIM][MATRIX_DIM];
static int B[MATRIX_DIM][MATRIX_DIM];
static int C[MATRIX_DIM][MATRIX_DIM];

void init_matrices() {
    for (int i = 0; i < MATRIX_DIM; i++) {
        for (int j = 0; j < MATRIX_DIM; j++) {
            A[i][j] = (i + j) % 17;
            B[i][j] = (i * j + 3) % 19;
            C[i][j] = 0;
        }
    }
}

void multiply_matrices() {
    for (int i = 0; i < MATRIX_DIM; i++) {
        for (int j = 0; j < MATRIX_DIM; j++) {
            int sum = 0;
            for (int k = 0; k < MATRIX_DIM; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] += sum % 1000;
        }
    }
}

int main(int argc, char *argv[]) {
    int tickets = 20;
    if (argc > 1) {
        tickets = atoi(argv[1]);
    }

    settickets(tickets);
    printf("[CPU Workload] Running with %d tickets, %d iterations of %dx%d matrix multiply...\n",
           tickets, ITERATIONS, MATRIX_DIM, MATRIX_DIM);

    init_matrices();
    uint t_start = uptime();

    for (int it = 0; it < ITERATIONS; it++) {
        multiply_matrices();
    }

    uint elapsed_ticks = uptime() - t_start;
    struct schedstat st;
    if (schedstat(&st) == 0) {
        printf("[CPU Workload Complete]\n");
        printf("  PID:            %d\n", st.pid);
        printf("  Tickets:        %d\n", st.tickets);
        printf("  Elapsed Ticks:  %d\n", elapsed_ticks);
        printf("  Ticks Running:  %d\n", st.ticks_run);
        printf("  Ticks Waiting:  %d\n", st.ticks_wait);
        printf("  Sched Contexts: %d\n", st.sched_count);
        printf("  Checksum:       %d\n", C[0][0]);
    }

    exit(0);
}
