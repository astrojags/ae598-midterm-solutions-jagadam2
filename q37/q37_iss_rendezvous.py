"""Question 37: ISS rendezvous using the HCW equations."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from scipy.integrate import solve_ivp


# Units: meters, seconds, m/s. LVLH: x outward, y along-track.
mu = 3.986004418e14
r_earth = 6378137.0
r_iss = r_earth + 420000.0
n = np.sqrt(mu / r_iss**3)
T = 2 * np.pi / n


def propagate(state, duration, acceleration=(0.0, 0.0), samples=200):
    """Integrate the planar HCW equations over one coast or thrust segment."""
    ax, ay = acceleration

    def hcw(t, s):
        x, y, vx, vy = s
        return [vx, vy, 3 * n**2 * x + 2 * n * vy + ax, -2 * n * vx + ay]

    times = np.linspace(0, duration, samples)
    solution = solve_ivp(
        hcw, (0, duration), state, t_eval=times,
        rtol=1e-11, atol=1e-10
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y.T


def impulse(state, dvx, dvy, time, label, burns):
    new_state = state.copy()
    new_state[2:] += [dvx, dvy]
    burns.append((time, dvx, dvy, label))
    return new_state


def check_state(label, actual, expected, tolerance=1e-4):
    error = np.max(np.abs(actual - expected))
    print(f"{label:24s} {np.array2string(actual, precision=7)}")
    assert np.allclose(actual, expected, atol=tolerance, rtol=0), label


# Phase 1: free drift and relative Hohmann transfer to S1 = (0, -3000 m).
state = np.array([-5000.0, -60000.0, 0.0, 1.5 * n * 5000.0])
burns = []
clock = 0.0

height_change = 5000.0
hohmann_dv = n * height_change / 4
hohmann_advance = 3 * np.pi * height_change / 4
burn_y = -3000.0 - hohmann_advance
drift_time = (burn_y - state[1]) / state[3]

free_drift = propagate(state, drift_time)
state = free_drift[-1].copy()
clock += drift_time
check_state("End of free drift", state, [-5000, burn_y, 0, 1.5 * n * 5000])

state = impulse(state, 0, hohmann_dv, clock, "Hohmann burn 1", burns)
hohmann = propagate(state, T / 2)
state = hohmann[-1].copy()
clock += T / 2
state = impulse(state, 0, hohmann_dv, clock, "Hohmann burn 2 / S1", burns)
check_state("S1 arrival", state, [0, -3000, 0, 0])
far_time = clock

# Phase 2: seven radial V-bar hops from -3000 m to -300 m.
n_hops = 7
hop_distance = 2700.0 / n_hops
hop_dv = -n * hop_distance / 4
hops = []
hold_points = [state[:2].copy()]

for k in range(n_hops):
    state = impulse(state, hop_dv, 0, clock, f"Hop {k + 1} start", burns)
    path = propagate(state, T / 2, samples=160)
    hops.append(path)
    state = path[-1].copy()
    clock += T / 2
    state = impulse(state, hop_dv, 0, clock, f"Hop {k + 1} stop", burns)
    hold_points.append(state[:2].copy())
    check_state(f"Hop {k + 1} arrival", state,
                [0, -3000 + (k + 1) * hop_distance, 0, 0])

max_excursion = max(np.max(np.abs(path[:, 0])) for path in hops)
assert max_excursion <= 100.0 + 1e-7
closing_time = clock - far_time

# Phase 3: forced straight V-bar approach from -300 m to -10 m.
closing_speed = 0.10
final_time = 290.0 / closing_speed
radial_acceleration = -2 * n * closing_speed

state = impulse(state, 0, closing_speed, clock, "Start final approach", burns)
final_approach = propagate(state, final_time,
                           acceleration=(radial_acceleration, 0.0))
state = final_approach[-1].copy()
clock += final_time
state = impulse(state, 0, -closing_speed, clock, "Final braking", burns)
check_state("Capture at 10 m", state, [0, -10, 0, 0])

impulsive_dv = sum(np.hypot(dvx, dvy) for _, dvx, dvy, _ in burns)
continuous_dv = abs(radial_acceleration) * final_time
total_dv = impulsive_dv + continuous_dv

print(f"\nISS mean motion       : {n:.9f} rad/s")
print(f"ISS period            : {T / 60:.3f} min")
print(f"First Hohmann burn at : y = {burn_y / 1000:.6f} km")
print(f"Drift time            : {drift_time / 60:.3f} min")
print(f"Hohmann burn          : +{hohmann_dv:.6f} m/s each")
print(f"Hop burn              : {hop_dv:.6f} m/s each (14 burns)")
print(f"Maximum hop excursion : {max_excursion:.3f} m")
print(f"Final acceleration    : {radial_acceleration:.9f} m/s^2")
print(f"Far-range duration    : {far_time / 60:.3f} min")
print(f"Closing duration      : {closing_time / 60:.3f} min")
print(f"Final duration        : {final_time / 60:.3f} min")
print(f"Total duration        : {clock / 3600:.6f} h")
print(f"Total delta-v         : {total_dv:.6f} m/s")

print("\nManeuver timeline (min, dvx, dvy, purpose):")
for time, dvx, dvy, label in burns:
    print(f"{time / 60:8.3f}  {dvx:+9.6f}  {dvy:+9.6f}  {label}")
print(f"{(far_time + closing_time) / 60:8.3f}-"
      f"{clock / 60:.3f}  {radial_acceleration * final_time:+9.6f}  "
      f"{0.0:+9.6f}  Continuous thrust (integrated)")

# Figure: overview, radial hops, and final approach (y horizontal, x vertical).
fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
ax1, ax2, ax3 = axes

for ax in axes:
    ax.axhline(0, color="0.65", linewidth=0.8)
    ax.axvline(0, color="0.65", linewidth=0.8)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Along-track y (km)")
    ax.set_ylabel("Radial x (km)")

ax1.plot(free_drift[:, 1] / 1000, free_drift[:, 0] / 1000,
         label="Free drift", linewidth=2)
ax1.plot(hohmann[:, 1] / 1000, hohmann[:, 0] / 1000,
         label="Relative Hohmann", linewidth=2)
for k, path in enumerate(hops):
    ax1.plot(path[:, 1] / 1000, path[:, 0] / 1000,
             color="tab:green", linewidth=1.6,
             label="Radial hops" if k == 0 else None)
ax1.plot(final_approach[:, 1] / 1000, final_approach[:, 0] / 1000,
         color="tab:red", linewidth=2.5, label="Forced V-bar")
ax1.add_patch(Ellipse((0, 0), 4, 2, fill=False,
                      linestyle="--", edgecolor="0.45"))
ax1.plot([-60, burn_y / 1000, -3, -0.3, -0.01],
         [-5, -5, 0, 0, 0], "ko", markersize=3)
ax1.annotate("Initial", (-60, -5), xytext=(-54, -4.6), fontsize=9)
ax1.annotate("Burn 1", (burn_y / 1000, -5),
             xytext=(-26, -4.2), arrowprops={"arrowstyle": "->"}, fontsize=9)
ax1.annotate("Burn 2 / S1", (-3, 0), xytext=(-14, 0.8),
             arrowprops={"arrowstyle": "->"}, fontsize=9)
ax1.set(xlim=(-62, 3), ylim=(-5.7, 1.5), title="(a) Complete rendezvous")
ax1.legend(loc="upper left", fontsize=8)

for path in hops:
    ax2.plot(path[:, 1] / 1000, path[:, 0] / 1000,
             color="tab:green", linewidth=2)
holds = np.array(hold_points)
ax2.scatter(holds[:, 1] / 1000, holds[:, 0] / 1000,
            color="black", s=17, zorder=4)
ax2.axvline(-2, color="0.5", linestyle="--", linewidth=1.3)
ax2.text(-1.97, 0.035, "AE begins on V-bar", fontsize=8)
ax2.annotate("S1", (-3, 0), xytext=(-3.05, 0.04), fontsize=9)
ax2.annotate("S2", (-0.3, 0), xytext=(-0.63, 0.04), fontsize=9)
ax2.text(-3.06, -0.125, "Radial start/stop burns at black dots", fontsize=8)
ax2.set(xlim=(-3.15, 0.05), ylim=(-0.135, 0.095),
        title="(b) Seven radial hops")

ax3.plot(final_approach[:, 1] / 1000, final_approach[:, 0] / 1000,
         color="tab:red", linewidth=3)
ax3.add_patch(Circle((0, 0), radius=0.2, fill=False,
                     linestyle="--", edgecolor="0.4", label="200 m KOZ (assumed)"))
ax3.plot([-0.3, -0.01, 0], [0, 0, 0], "ko", markersize=4)
ax3.annotate("S2 (-300 m)", (-0.3, 0), xytext=(-0.39, 0.065),
             arrowprops={"arrowstyle": "->"}, fontsize=9)
ax3.annotate("Capture (-10 m)", (-0.01, 0), xytext=(-0.33, -0.10),
             arrowprops={"arrowstyle": "->"}, fontsize=9)
ax3.annotate("ISS", (0, 0), xytext=(0.02, 0.035), fontsize=9)
ax3.set(xlim=(-0.42, 0.24), ylim=(-0.23, 0.23),
        title="(c) Final controlled approach")
ax3.set_aspect("equal", adjustable="box")
ax3.legend(loc="upper right", fontsize=8)

fig.suptitle("Question 37 — HCW ISS rendezvous", fontsize=14)
fig.savefig("q37_iss_trajectory.png", dpi=240)
fig.savefig("q37_iss_trajectory.pdf")
plt.close(fig)
