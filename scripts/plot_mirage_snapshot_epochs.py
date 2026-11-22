"""Render the primary seed's development curves for a 16:9 presentation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

COLORS = {"fit": "#0D6684", "dev": "#B64A26", "ink": "#142632", "muted": "#4A616E"}


def render(run: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, MultipleLocator

    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    seed = plan["primary_seed"]
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    best = selection["seeds"][str(seed)]["selected_epoch"]
    with (run / "development/curves.csv").open(encoding="utf-8", newline="") as file:
        curve = [row for row in csv.DictReader(file) if int(row["seed"]) == seed]
    epochs = [int(row["epoch"]) for row in curve]
    fit = [float(row["train_log_loss"]) for row in curve]
    development = [float(row["development_log_loss"]) for row in curve]
    if (
        epochs != list(range(1, plan["epochs"] + 1))
        or epochs[development.index(min(development))] != best
    ):
        raise ValueError("The presentation curves disagree with the locked experiment.")
    detail_limit = min(len(epochs), max(15, best + 3))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#CED9DE",
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "svg.fonttype": "path",
        }
    )
    fig, (overview, detail) = plt.subplots(
        1, 2, figsize=(13.6, 7.65), gridspec_kw={"width_ratios": [1.7, 1]}
    )
    fig.subplots_adjust(left=0.09, right=0.962, top=0.727, bottom=0.164, wspace=0.26)
    fig.text(
        0.09,
        0.929,
        "Mirage  |  Single-snapshot Transformer",
        color=COLORS["ink"],
        fontsize=27,
        weight="bold",
    )
    fig.text(
        0.09,
        0.862,
        f"Seed {seed} · 32 training matches · 10 development matches",
        color=COLORS["muted"],
        fontsize=16,
    )
    for ax, title in ((overview, "Full training run"), (detail, "Development detail")):
        ax.set_title(title, loc="left", fontsize=17, weight="bold", color=COLORS["ink"], pad=13)
        ax.set_xlabel("Training epoch", labelpad=11)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E2E9EC", linewidth=0.8)
        ax.tick_params(axis="both", length=0, pad=6)

    (fit_line,) = overview.plot(epochs, fit, color=COLORS["fit"], linewidth=2.5, label="Training")
    (dev_line,) = overview.plot(
        epochs, development, color=COLORS["dev"], linewidth=2.5, label="Development"
    )
    overview.set(xlim=(0, epochs[-1] + 2), ylim=(0, max(development + fit) * 1.09))
    overview.set_ylabel("Round-weighted Log Loss", labelpad=12)
    overview.set_xticks([1, 50, 100, 150, 200])
    overview.yaxis.set_major_locator(MaxNLocator(nbins=5))
    overview.axvspan(1, detail_limit, color="#F0F4F6", zorder=0)
    overview.legend(
        handles=[fit_line, dev_line],
        loc="upper left",
        frameon=False,
        fontsize=13,
        handlelength=1.6,
        labelcolor=COLORS["ink"],
    )
    overview.scatter(epochs[-1], development[-1], color=COLORS["dev"], s=44, zorder=5)

    early = development[:detail_limit]
    bottom, top = min(early), max(early)
    span = max(top - bottom, 0.005)
    detail.set(xlim=(0.55, detail_limit + 0.45), ylim=(bottom - 0.19 * span, top + 0.08 * span))
    detail.yaxis.set_major_locator(MaxNLocator(nbins=5))
    if detail_limit <= 18:
        detail.xaxis.set_major_locator(MultipleLocator(3))
        detail.set_xticks([1, *range(3, detail_limit + 1, 3)])
    else:
        detail.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    detail.plot(epochs[:detail_limit], early, color=COLORS["dev"], linewidth=2.5, marker="o", ms=4)
    detail.axvline(best, color=COLORS["muted"], linewidth=1, linestyle=(0, (3, 3)), alpha=0.65)
    detail.scatter(
        best,
        development[best - 1],
        color=COLORS["dev"],
        s=88,
        zorder=5,
        edgecolor="white",
        linewidth=1.6,
    )
    detail.annotate(
        f"Epoch {best} · {development[best - 1]:.3f}",
        xy=(best, development[best - 1]),
        xytext=(10 if best < detail_limit / 2 else -10, -24),
        textcoords="offset points",
        ha="left" if best < detail_limit / 2 else "right",
        color=COLORS["ink"],
        fontsize=13,
        weight="bold",
    )
    fig.text(
        0.09,
        0.063,
        f"Epoch 200 development loss  {development[-1]:.3f}     ·     "
        "Development events  BLAST Bounty and Esports World Cup",
        color=COLORS["muted"],
        fontsize=13,
    )
    for extension in ("png", "svg"):
        target = run / f"learning_curves.{extension}"
        with target.open("xb") as handle:
            fig.savefig(handle, format=extension, dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    render(parser.parse_args().run)
