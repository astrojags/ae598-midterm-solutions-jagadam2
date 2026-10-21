"""HCW closed-form relative motion for Question 36."""

import csv

import numpy as np

from q36_setup import (
    mean_motion, period, time_grid, checkpoint_times,
    results_dir, get_initial_state,
)


def propagate(initial_state, time_grid):
    """Return HCW Hill-frame position and velocity at the given times."""
    x0, y0, z0, x_dot0, y_dot0, z_dot0 = initial_state
    n = mean_motion
    t = np.asarray(time_grid)
    sine = np.sin(n * t)
    cosine = np.cos(n * t)

    # HCW closed-form position solution
    x = ((4 - 3 * cosine) * x0
         + (sine / n) * x_dot0
         + (2 * (1 - cosine) / n) * y_dot0)

    y = (y0 + 6 * (sine - n * t) * x0
         - (2 * (1 - cosine) / n) * x_dot0
         + (4 * sine - 3 * n * t) / n * y_dot0)

    z = cosine * z0 + (sine / n) * z_dot0

    # Time derivatives of the closed-form solution
    x_dot = 3 * n * sine * x0 + cosine * x_dot0 + 2 * sine * y_dot0
    y_dot = (6 * n * (cosine - 1) * x0
             - 2 * sine * x_dot0 + (4 * cosine - 3) * y_dot0)
    z_dot = -n * sine * z0 + cosine * z_dot0

    return np.column_stack((x, y, z, x_dot, y_dot, z_dot))


def save_checkpoints(filename, checkpoints):
    """Save the 1-, 2-, and 5-orbit states."""
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "orbit", "time_s", "x_m", "y_m", "z_m",
            "x_dot_mps", "y_dot_mps", "z_dot_mps",
        ])
        for orbit, time, state in zip((1, 2, 5), checkpoint_times, checkpoints):
            writer.writerow([orbit, time, *state])


def main():
    results_dir.mkdir(exist_ok=True)

    for case in ("case_i", "case_ii"):
        initial_state = get_initial_state(case)
        states = propagate(initial_state, time_grid)
        checkpoints = propagate(initial_state, checkpoint_times)

        # Basic checks on the analytical solution
        if not np.allclose(states[0], initial_state, rtol=0, atol=1e-10):
            raise ValueError(f"HCW initial state does not match for {case}.")

        if case == "case_i":
            if not np.allclose(checkpoints[:, :3], initial_state[:3],
                               rtol=0, atol=1e-8):
                raise ValueError("Case (i) should be periodic in HCW.")
        else:
            drift_per_orbit = -6 * np.pi * (
                2 * initial_state[0] + initial_state[4] / mean_motion
            )
            expected_y = np.array([1, 2, 5]) * drift_per_orbit
            if not np.allclose(checkpoints[:, 1], expected_y,
                               rtol=0, atol=1e-8):
                raise ValueError("Case (ii) HCW drift check failed.")

        np.savez_compressed(
            results_dir / f"hcw_{case}.npz",
            time=time_grid,
            state=states,
            checkpoint_times=checkpoint_times,
            checkpoints=checkpoints,
        )
        save_checkpoints(
            results_dir / f"hcw_{case}_checkpoints.csv", checkpoints
        )

        print(f"\nHCW model — {case}")
        print(f"Initial-state error: {np.max(np.abs(states[0] - initial_state)):.2e}")
        for orbit, state in zip((1, 2, 5), checkpoints):
            print(f"After {orbit} orbit(s): x = {state[0]:.3f} m, "
                  f"y = {state[1]:.3f} m")

    print("\nHCW results saved in the results folder.")


if __name__ == "__main__":
    main()
