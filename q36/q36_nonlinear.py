"""Nonlinear two-body propagation for Question 36."""

import csv

import numpy as np
from scipy.integrate import solve_ivp

from q36_setup import (
    mu, r_perigee, h, thetadot_perigee,
    time_grid, checkpoint_times, results_dir, get_initial_state,
)


def two_body(t, state):
    """Inertial two-body equations for one spacecraft."""
    position = state[:3]
    velocity = state[3:]
    radius = np.linalg.norm(position)
    acceleration = -mu * position / radius**3
    return np.concatenate((velocity, acceleration))


def initial_inertial_states(hill_state):
    """Set the target at perigee and convert the deputy's Hill state."""
    target_position = np.array([r_perigee, 0.0, 0.0])
    target_velocity = np.array([0.0, h / r_perigee, 0.0])

    # The inertial axes and Hill axes coincide at t = 0.
    relative_position = hill_state[:3]
    relative_velocity = hill_state[3:]
    omega = np.array([0.0, 0.0, thetadot_perigee])

    deputy_position = target_position + relative_position
    deputy_velocity = (target_velocity + relative_velocity
                       + np.cross(omega, relative_position))

    target_state = np.concatenate((target_position, target_velocity))
    deputy_state = np.concatenate((deputy_position, deputy_velocity))
    return target_state, deputy_state


def convert_to_hill(target_states, deputy_states):
    """Convert inertial positions and velocities to the target's Hill frame."""
    target_position = target_states[:, :3]
    target_velocity = target_states[:, 3:]
    deputy_position = deputy_states[:, :3]
    deputy_velocity = deputy_states[:, 3:]

    relative_position = deputy_position - target_position
    relative_velocity = deputy_velocity - target_velocity

    radius = np.linalg.norm(target_position, axis=1)
    radial = target_position / radius[:, None]

    angular_momentum = np.cross(target_position, target_velocity)
    normal = angular_momentum / np.linalg.norm(
        angular_momentum, axis=1
    )[:, None]
    along_track = np.cross(normal, radial)

    # Remove the velocity caused by rotation of the Hill axes.
    omega = angular_momentum / radius[:, None]**2
    hill_velocity = relative_velocity - np.cross(omega, relative_position)

    x = np.sum(relative_position * radial, axis=1)
    y = np.sum(relative_position * along_track, axis=1)
    z = np.sum(relative_position * normal, axis=1)
    x_dot = np.sum(hill_velocity * radial, axis=1)
    y_dot = np.sum(hill_velocity * along_track, axis=1)
    z_dot = np.sum(hill_velocity * normal, axis=1)

    return np.column_stack((x, y, z, x_dot, y_dot, z_dot))


def propagate(initial_state, time_grid):
    """Propagate both spacecraft and return the deputy's Hill-frame state."""
    target_start, deputy_start = initial_inertial_states(initial_state)
    duration = (time_grid[0], time_grid[-1])

    target = solve_ivp(
        two_body, duration, target_start, t_eval=time_grid,
        method="DOP853", rtol=1e-12, atol=1e-10,
    )
    deputy = solve_ivp(
        two_body, duration, deputy_start, t_eval=time_grid,
        method="DOP853", rtol=1e-12, atol=1e-10,
    )

    if not target.success or not deputy.success:
        raise RuntimeError("Nonlinear two-body integration failed.")

    return convert_to_hill(target.y.T, deputy.y.T)


def save_checkpoints(filename, checkpoints):
    """Write the 1-, 2-, and 5-orbit states to a CSV file."""
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
    checkpoint_indices = np.searchsorted(time_grid, checkpoint_times)
    if not np.allclose(time_grid[checkpoint_indices], checkpoint_times,
                          rtol=0, atol=1e-8):
        raise ValueError("The time grid must contain the checkpoint times.")

    for case in ("case_i", "case_ii"):
        initial_state = get_initial_state(case)
        states = propagate(initial_state, time_grid)
        checkpoints = states[checkpoint_indices]

        np.savez_compressed(
            results_dir / f"nonlinear_{case}.npz",
            time=time_grid,
            state=states,
            checkpoint_times=checkpoint_times,
            checkpoints=checkpoints,
        )
        save_checkpoints(
            results_dir / f"nonlinear_{case}_checkpoints.csv", checkpoints
        )

        initial_error = np.max(np.abs(states[0] - initial_state))
        print(f"\nNonlinear model — {case}")
        print(f"Initial-state error: {initial_error:.2e}")
        for orbit, state in zip((1, 2, 5), checkpoints):
            print(f"After {orbit} orbit(s): x = {state[0]:.3f} m, "
                  f"y = {state[1]:.3f} m")

    print("\nNonlinear results saved in the results folder.")


if __name__ == "__main__":
    main()
