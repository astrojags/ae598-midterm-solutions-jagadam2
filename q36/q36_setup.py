"""Shared settings and initial conditions for Question 36."""

from pathlib import Path
import numpy as np


# Earth and target orbit (metres, seconds)
mu = 3.986004418e14        # Earth's gravitational parameter (m^3/s^2)
a = 7_500_000.0            # Semi-major axis (m)
e = 0.15                   # Eccentricity

mean_motion = np.sqrt(mu / a**3)
period = 2 * np.pi / mean_motion
p = a * (1 - e**2)         # Semi-latus rectum (m)
r_perigee = a * (1 - e)
h = np.sqrt(mu * p)        # Specific angular momentum (m^2/s)
thetadot_perigee = h / r_perigee**2


# Starting Hill-frame position and velocity (m, m/s)
x0 = 500.0
y0 = 0.0
z0 = 200.0
xdot0 = 0.0
zdot0 = 0.0

# Two choices for initial along-track velocity
ydot_case1 = -2 * mean_motion * x0
ydot_case2 = -(2 + e) / (1 + e) * thetadot_perigee * x0

initial_states = {
    "case_i": np.array([x0, y0, z0, xdot0, ydot_case1, zdot0]),
    "case_ii": np.array([x0, y0, z0, xdot0, ydot_case2, zdot0]),
}


# Simulation times
num_orbits = 5
num_points = 2001
end_time = num_orbits * period
time_grid = np.linspace(0, end_time, num_points)
checkpoint_times = np.array([1, 2, 5]) * period

results_dir = Path(__file__).resolve().parent / "results"


def get_initial_state(case):
    """Return a separate copy of the chosen Hill-frame initial state."""
    return initial_states[case].copy()


if __name__ == "__main__":
    print(f"Mean motion: {mean_motion:.9f} rad/s")
    print(f"Orbital period: {period:.3f} s")
    print(f"Perigee radius: {r_perigee / 1000:.3f} km")
    print(f"Perigee angular rate: {thetadot_perigee:.9f} rad/s")
    print(f"Propagation time: {end_time:.3f} s")
    print(f"Case (i)  y_dot0: {ydot_case1:.6f} m/s")
    print(f"Case (ii) y_dot0: {ydot_case2:.6f} m/s")
