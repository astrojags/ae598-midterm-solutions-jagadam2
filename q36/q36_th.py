"""Tschauner-Hempel (TH) relative motion for Question 36."""

import csv

import numpy as np
from scipy.integrate import solve_ivp

from q36_setup import (
    e, mean_motion, p, h, r_perigee, thetadot_perigee,
    time_grid, checkpoint_times, results_dir, get_initial_state,
)


def time_to_anomaly(time_grid):
    """Convert time since perigee to continuous true anomaly (radians)."""
    mean_anomaly = mean_motion * np.asarray(time_grid)
    eccentric_anomaly = mean_anomaly.copy()

    # Solve Kepler's equation: M = E_anomaly - e*sin(E_anomaly).
    for _ in range(10):
        correction = (eccentric_anomaly - e * np.sin(eccentric_anomaly)
                      - mean_anomaly) / (1 - e * np.cos(eccentric_anomaly))
        eccentric_anomaly -= correction
        if np.max(np.abs(correction)) < 1e-13:
            break

    # This form keeps true anomaly continuous across multiple orbits.
    theta = eccentric_anomaly + 2 * np.arctan2(
        e * np.sin(eccentric_anomaly),
        1 + np.sqrt(1 - e**2) - e * np.cos(eccentric_anomaly),
    )
    return theta


def hill_to_th(initial_state):
    """Convert the initial Hill state at perigee to TH coordinates."""
    position = initial_state[:3] / r_perigee
    # At perigee the target's radial velocity is zero.
    anomaly_derivative = initial_state[3:] / (r_perigee * thetadot_perigee)
    return np.concatenate((position, anomaly_derivative))


def th_equations(theta, state):
    """TH equations with true anomaly as the independent variable."""
    u, v, w, u_prime, v_prime, w_prime = state
    k = 1 + e * np.cos(theta)

    return [
        u_prime,
        v_prime,
        w_prime,
        2 * v_prime + 3 * u / k,
        -2 * u_prime,
        -w,
    ]


def th_to_hill(theta, states):
    """Convert TH states back to dimensional Hill positions and velocities."""
    theta = np.asarray(theta)
    u, v, w, u_prime, v_prime, w_prime = states.T

    k = 1 + e * np.cos(theta)
    radius = p / k
    theta_dot = h / radius**2
    radius_prime = p * e * np.sin(theta) / k**2
    radius_dot = radius_prime * theta_dot

    x = radius * u
    y = radius * v
    z = radius * w

    x_dot = radius_dot * u + radius * theta_dot * u_prime
    y_dot = radius_dot * v + radius * theta_dot * v_prime
    z_dot = radius_dot * w + radius * theta_dot * w_prime

    return np.column_stack((x, y, z, x_dot, y_dot, z_dot))


def propagate(initial_state, time_grid):
    """Integrate TH and return Hill states at the requested physical times."""
    theta = time_to_anomaly(time_grid)
    initial_th = hill_to_th(initial_state)

    solution = solve_ivp(
        th_equations, (0.0, theta[-1]), initial_th,
        t_eval=theta, method="DOP853", rtol=1e-13, atol=1e-15,
    )
    if not solution.success:
        raise RuntimeError("TH integration failed: " + solution.message)

    return th_to_hill(theta, solution.y.T)


def save_checkpoints(filename, checkpoints):
    """Save the 1-, 2-, and 5-orbit states to a CSV file."""
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
        initial_th = hill_to_th(initial_state)
        states = propagate(initial_state, time_grid)
        checkpoints = propagate(initial_state, np.r_[0.0, checkpoint_times])[1:]

        if not np.allclose(states[0], initial_state, rtol=0, atol=1e-8):
            raise ValueError(f"TH initial state does not match for {case}.")
        if case == "case_ii" and not np.allclose(
            checkpoints[:, :3], initial_state[:3], rtol=0, atol=1e-6
        ):
            raise ValueError("Case (ii) should be periodic in TH.")

        np.savez_compressed(
            results_dir / f"th_{case}.npz",
            time=time_grid,
            state=states,
            checkpoint_times=checkpoint_times,
            checkpoints=checkpoints,
            initial_th=initial_th,
        )
        save_checkpoints(
            results_dir / f"th_{case}_checkpoints.csv", checkpoints
        )

        print(f"\nTH model — {case}")
        print("Initial TH state [u, v, w, u', v', w']:")
        print(np.array2string(initial_th, precision=10))
        print(f"Initial-state error: {np.max(np.abs(states[0] - initial_state)):.2e}")
        for orbit, state in zip((1, 2, 5), checkpoints):
            print(f"After {orbit} orbit(s): x = {state[0]:.3f} m, "
                  f"y = {state[1]:.3f} m")

    print("\nTH results saved in the results folder.")


if __name__ == "__main__":
    main()
