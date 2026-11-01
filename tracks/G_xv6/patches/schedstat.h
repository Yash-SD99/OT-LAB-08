#ifndef _SCHEDSTAT_H_
#define _SCHEDSTAT_H_

// ==============================================================================
// schedstat.h: Shared user-kernel structure for scheduling statistics
// Track G: xv6 Operating System Optimization
// ==============================================================================

struct schedstat {
  int pid;         // Process ID
  int tickets;     // Number of lottery tickets held
  int ticks_run;   // Cumulative clock ticks spent executing on CPU
  int ticks_wait;  // Cumulative clock ticks spent waiting in RUNNABLE state
  int sched_count; // Number of times context-switched onto CPU
};

#endif // _SCHEDSTAT_H_
