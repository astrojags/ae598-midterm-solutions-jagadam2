"""Compare the nonlinear, HCW, and TH results for Question 36."""

import csv

import matplotlib.pyplot as plt
import numpy as np

from q36_setup import time_grid, checkpoint_times, results_dir


models = ("nonlinear", "hcw", "th")
labels = {"nonlinear": "Nonlinear truth", "hcw": "HCW", "th": "TH"}
colors = {"nonlinear": "#222222", "hcw": "#d97727", "th": "#2179ad"}
styles = {"nonlinear": "-", "hcw": "--", "th": "-."}


def load_results(case):
    """Load saved results from all three model scripts."""
    results = {}

    for model in models:
        filename = results_dir / f"{model}_{case}.npz"
        if not filename.exists():
            raise FileNotFoundError(
                f"Missing {filename.name}. Run q36_{model}.py first."
            )

        with np.load(filename) as data:
            time = data["time"]
            states = data["state"]
            checkpoints = data["checkpoints"]
            checkpoint_times = data["checkpoint_times"]

        if time.shape != time_grid.shape or not np.allclose(
            time, time_grid, rtol=0, atol=1e-8
        ):
            raise ValueError(f"Time grid does not match for {model}, {case}.")
        if states.shape != (len(time_grid), 6):
            raise ValueError(f"State array has the wrong shape: {model}, {case}.")
        if checkpoints.shape != (3, 6) or not np.allclose(
            checkpoint_times, checkpoint_times, rtol=0, atol=1e-8
        ):
            raise ValueError(f"Checkpoint data does not match: {model}, {case}.")
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(checkpoints)):
            raise ValueError(f"Non-finite results found: {model}, {case}.")

        results[model] = {"state": states, "checkpoints": checkpoints}

    return results


def plot_trajectories(case, results):
    """Save full and near-target trajectory plots separately."""
    case_name = "i" if case == "case_i" else "ii"

    for view in ("full", "zoom"):
        fig, ax = plt.subplots(figsize=(9, 5.7))

        for model in models:
            state = results[model]["state"]
            ax.plot(state[:, 1], state[:, 0], label=labels[model],
                    color=colors[model], linestyle=styles[model], linewidth=1.8)

        ax.plot(0, 0, "x", color="gray", markersize=7, label="Target")
        ax.plot(0, 500, "o", markerfacecolor="white", markeredgecolor="black",
                markersize=6, label="Initial deputy")

        if view == "zoom":
            ax.set_xlim(-1200, 1200)
            ax.set_ylim(-650, 650)
            ax.set_title(f"Case ({case_name}): near-target trajectories")
        else:
            ax.set_title(f"Case ({case_name}): full trajectories over 5 target orbits")

        ax.set_xlabel("Along-track position, y (m)")
        ax.set_ylabel("Radial position, x (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)

        filename = results_dir / f"{case}_trajectory_{view}"
        fig.savefig(filename.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(filename.with_suffix(".png"), dpi=240, bbox_inches="tight")
        plt.close(fig)


def calculate_errors(results):
    """Return 3D position errors relative to the nonlinear truth."""
    truth = results["nonlinear"]["state"][:, :3]
    errors = {}
    for model in ("hcw", "th"):
        position = results[model]["state"][:, :3]
        errors[model] = np.linalg.norm(position - truth, axis=1)
    return errors


def plot_errors(case, errors):
    """Make a logarithmic plot of the position errors."""
    fig, ax = plt.subplots(figsize=(9, 5.3))
    time_in_orbits = time_grid / (checkpoint_times[0])

    for model in ("hcw", "th"):
        # The exact initial error is zero, so it cannot appear on a log axis.
        plot_values = errors[model].copy()
        plot_values[0] = np.nan
        ax.semilogy(time_in_orbits, plot_values, color=colors[model],
                    linestyle=styles[model], linewidth=1.9,
                    label=f"{labels[model]} vs nonlinear truth")

    ax.set_xlabel("Elapsed time (target orbits)")
    ax.set_ylabel("3D position error magnitude (m) — log scale")
    ax.set_title(f"Case ({'i' if case == 'case_i' else 'ii'}): position error over 5 target orbits")
    ax.set_xlim(0, 5)
    ax.set_xticks(np.arange(0, 6))
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(loc="best", fontsize=9)

    fig.savefig(results_dir / f"{case}_errors.pdf", bbox_inches="tight")
    fig.savefig(results_dir / f"{case}_errors.png", dpi=240,
                bbox_inches="tight")
    plt.close(fig)


def save_error_table(case, errors):
    """Save time and position errors to a CSV file."""
    filename = results_dir / f"{case}_errors.csv"
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", "target_orbits", "hcw_error_m", "th_error_m"])
        for time, hcw_error, th_error in zip(time_grid, errors["hcw"], errors["th"]):
            writer.writerow([time, time / checkpoint_times[0], hcw_error, th_error])


def save_checkpoint_table(case, results):
    """Save one table containing all models at 1, 2, and 5 orbits."""
    filename = results_dir / f"{case}_comparison_checkpoints.csv"
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["orbit", "model", "x_m", "y_m"])
        for index, orbit in enumerate((1, 2, 5)):
            for model in models:
                x, y = results[model]["checkpoints"][index, :2]
                writer.writerow([orbit, labels[model], x, y])


def main():
    results_dir.mkdir(exist_ok=True)

    for case in ("case_i", "case_ii"):
        results = load_results(case)
        errors = calculate_errors(results)

        plot_trajectories(case, results)
        plot_errors(case, errors)
        save_error_table(case, errors)
        save_checkpoint_table(case, results)

        print(f"\nCase ({'i' if case == 'case_i' else 'ii'})")
        print(f"{'Orbit':<7}{'Model':<19}{'x (m)':>14}{'y (m)':>16}")
        for index, orbit in enumerate((1, 2, 5)):
            for model in models:
                x, y = results[model]["checkpoints"][index, :2]
                print(f"{orbit:<7}{labels[model]:<19}{x:>14.3f}{y:>16.3f}")

        for model in ("hcw", "th"):
            print(f"{model.upper()} final position error: "
                  f"{errors[model][-1]:.3f} m")

    print("\nComparison plots and tables saved in the results folder.")


if __name__ == "__main__":
    main()
