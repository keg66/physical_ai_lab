"""A minimal real-time loop with a 10 ms period."""

import time


def process(iteration: int) -> None:
    """Run the work for one iteration."""
    pass


def main() -> None:
    """Run the process and wait loop."""
    period = 0.01
    start = time.perf_counter()
    previous_start = None
    measured_periods = []

    for iteration in range(1000):
        current_start = time.perf_counter()
        if previous_start is not None:
            measured_periods.append(current_start - previous_start)
        previous_start = current_start

        process(iteration)

        # Wait for an absolute deadline based on the start time,
        # rather than using a relative sleep.
        deadline = start + (iteration + 1) * period
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

    periods_ms = [measured_period * 1000 for measured_period in measured_periods]
    print(f"target period : {period * 1000:.3f} ms")
    print(f"mean period   : {sum(periods_ms) / len(periods_ms):.3f} ms")
    print(f"min period    : {min(periods_ms):.3f} ms")
    print(f"max period    : {max(periods_ms):.3f} ms")


if __name__ == "__main__":
    main()
