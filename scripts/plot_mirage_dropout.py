"""Render the primary seed and three-seed mean dropout curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

STYLES = {"d010": ("#145F80", "-"), "d020": ("#AA4B26", "--")}


def render(run: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    with (run / "curves.csv").open(encoding="utf-8", newline="") as file:
        curve = list(csv.DictReader(file))
    with (run / "development_scores.csv").open(encoding="utf-8", newline="") as file:
        logistic = float(next(csv.DictReader(file))["logistic_log_loss"])

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
    fig.subplots_adjust(left=0.105, right=0.96, top=0.72, bottom=0.15, hspace=0.22, wspace=0.18)
    fig.text(0.105, 0.93, "Mirage  |  Transformer dropout", color=ink, fontsize=28, weight="bold")
    fig.text(
        0.105,
        0.869,
        "47 fit matches · 10 previously observed development matches",
        color=muted,
        fontsize=15,
    )
    primary = plan["primary_seed"]
    epochs = list(range(1, plan["epochs"] + 1))
    for column, heading in enumerate(("Mean of three seeds", f"Primary seed {primary}")):
        dev, fit = axes[:, column]
        dev.set_title(heading, loc="left", color=ink, fontsize=17, weight="bold", pad=12)
        dev.axhline(logistic, color="#687B84", lw=1.8, ls=":", label="Logistic regression")
        for arm, (color, line) in STYLES.items():
            rows = [
                row
                for row in curve
                if row["arm"] == arm and (column == 0 or int(row["seed"]) == primary)
            ]
            values = {
                metric: [
                    fmean(float(row[metric]) for row in rows if int(row["epoch"]) == epoch)
                    for epoch in epochs
                ]
                for metric in ("development_log_loss", "train_log_loss")
            }
            dev.plot(
                epochs,
                values["development_log_loss"],
                color=color,
                ls=line,
                lw=2.6,
                label=f"Dropout {plan['dropout'][arm]:.2f}",
            )
            fit.plot(epochs, values["train_log_loss"], color=color, ls=line, lw=2.6)
            chosen = selection["shared_epochs"][arm]
            dev.scatter(
                chosen,
                values["development_log_loss"][chosen - 1],
                s=58,
                color=color,
                edgecolor="white",
                zorder=5,
            )
        for ax in (dev, fit):
            ax.set(xlim=(0.55, epochs[-1] + 0.45), xticks=[1, 5, 10, 15, 20])
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E3EAED", linewidth=0.8)
            ax.tick_params(axis="both", length=0, pad=5)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        fit.set_xlabel("Training epoch")
    axes[0, 0].set_ylabel("Development Log Loss", labelpad=10)
    axes[1, 0].set_ylabel("Training Log Loss", labelpad=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [1, 2, 0]
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        ncols=3,
        loc="upper right",
        bbox_to_anchor=(0.97, 0.858),
        frameon=False,
        fontsize=13,
        labelcolor=ink,
    )
    fig.text(
        0.105,
        0.056,
        "Dots mark the shared epoch chosen by mean development loss. Each round has equal weight.",
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
