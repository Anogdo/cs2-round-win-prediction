"""Render training and development curves for both nested draws."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

STYLES = {16: ("#A94D25", "--"), 24: ("#427455", ":"), 32: ("#145F80", "-")}


def render(run: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    seed = plan["primary_training_seed"]
    draws = plan["subset_seeds"]
    with (run / "curves.csv").open(encoding="utf-8", newline="") as file:
        curves = [row for row in csv.DictReader(file) if int(row["seed"]) == seed]

    ink, muted = "#142632", "#42606D"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#CFDADF",
            "axes.labelcolor": ink,
            "xtick.color": muted,
            "ytick.color": muted,
            "svg.fonttype": "path",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 7.65), sharex=True, sharey="row")
    fig.subplots_adjust(left=0.10, right=0.96, top=0.72, bottom=0.15, hspace=0.20, wspace=0.21)
    fig.text(0.10, 0.93, "Mirage  |  Training set size", color=ink, fontsize=28, weight="bold")
    fig.text(
        0.10,
        0.869,
        f"Training seed {seed} · 10 common development matches",
        color=muted,
        fontsize=16,
    )
    for index, draw in enumerate(draws):
        dev_ax, train_ax = axes[:, index]
        dev_ax.set_title(f"Subset draw {draw}", loc="left", fontsize=17, weight="bold", pad=13)
        for size, (color, linestyle) in STYLES.items():
            name = "full_n32" if size == 32 else f"draw_{draw}_n{size}"
            points = [row for row in curves if row["trial"] == name]
            epochs = [int(row["epoch"]) for row in points]
            if epochs != list(range(1, plan["learning_curve_epochs"] + 1)):
                raise ValueError(f"Incomplete plot data for {name}.")
            dev = [float(row["development_log_loss"]) for row in points]
            fit = [float(row["train_log_loss"]) for row in points]
            best = dev.index(min(dev))
            dev_ax.plot(epochs, dev, label=f"{size} matches", color=color, ls=linestyle, lw=2.6)
            dev_ax.scatter(epochs[best], dev[best], color=color, s=50, edgecolor="white", zorder=4)
            train_ax.plot(epochs, fit, color=color, ls=linestyle, lw=2.6)
        for ax in (dev_ax, train_ax):
            ax.set_xlim(0.65, 20.35)
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E3EAED", linewidth=0.8)
            ax.tick_params(axis="both", length=0, pad=5)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        train_ax.set_xlabel("Training epoch")
    axes[0, 0].set_ylabel("Development Log Loss", labelpad=11)
    axes[1, 0].set_ylabel("Training Log Loss", labelpad=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncols=3,
        loc="upper right",
        bbox_to_anchor=(0.97, 0.86),
        frameon=False,
        fontsize=13,
        labelcolor=ink,
        handlelength=2.2,
    )
    fig.text(
        0.10,
        0.058,
        "Each round has equal weight. Dots mark the first development minimum. "
        "Both draws share the full 32-match fit.",
        color=muted,
        fontsize=12,
    )
    for extension in ("png", "svg"):
        with (run / f"learning_curves.{extension}").open("xb") as handle:
            fig.savefig(handle, format=extension, dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    render(parser.parse_args().run)
