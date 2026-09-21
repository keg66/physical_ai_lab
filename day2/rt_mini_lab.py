"""A minimal real-time loop with a 10 ms period."""

import time


def process(iteration: int) -> None:
    """Run the work for one iteration."""
    # print(f"iteration: {iteration}")
    pass


def main() -> None:
    """Run the process and wait loop."""
    period = 0.01
    start = time.perf_counter()
    previous_start = None
    measured_periods = []
    execution_times = []
    jitters = []
    deadline_misses = 0

    for iteration in range(1000):
        scheduled_start = start + iteration * period
        current_start = time.perf_counter()
        if previous_start is not None:
            measured_periods.append(current_start - previous_start)
        previous_start = current_start
        jitters.append(current_start - scheduled_start)

        process(iteration)
        process_end = time.perf_counter()
        execution_times.append(process_end - current_start)

        deadline = scheduled_start + period
        if process_end > deadline:
            deadline_misses += 1

        # Wait for an absolute deadline based on the start time,
        # rather than using a relative sleep.
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

    periods_ms = [measured_period * 1000 for measured_period in measured_periods]
    execution_times_ms = [execution_time * 1000 for execution_time in execution_times]
    jitters_ms = [jitter * 1000 for jitter in jitters]
    print(f"target period : {period * 1000:.3f} ms")
    print(f"mean period   : {sum(periods_ms) / len(periods_ms):.3f} ms")
    print(f"min period    : {min(periods_ms):.3f} ms")
    print(f"max period    : {max(periods_ms):.3f} ms")
    print()
    print(f"mean execution: {sum(execution_times_ms) / len(execution_times_ms):.3f} ms")
    print(f"min execution : {min(execution_times_ms):.3f} ms")
    print(f"max execution : {max(execution_times_ms):.3f} ms")
    print()
    print(f"mean jitter   : {sum(jitters_ms) / len(jitters_ms):.3f} ms")
    print(f"min jitter    : {min(jitters_ms):.3f} ms")
    print(f"max jitter    : {max(jitters_ms):.3f} ms")
    print()
    print(f"deadline misses: {deadline_misses} / 1000")


if __name__ == "__main__":
    main()
