#!/usr/bin/env python3
"""Plot H4 as paired NLL effects with 95% confidence intervals.

Draws 24 comparisons. Use --output to choose where to save the figure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

import plot_results_revision as revision
from paper_plot_style import COLORS, paper_style

ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    ("Giving", "Social allocation:\ngiving variant"),
    ("Taking", "Social allocation:\ntaking variant"),
    ("Category choice", "Categorization"),
    ("Lottery tradeoffs", "Risky choice:\ntradeoff pairs"),
)
CONDITIONS = ("us", "pooled", "china")
COUNTRIES = ("United States", "China")


def load_comparisons() -> pd.DataFrame:
    rows = revision.revised_allocation_data()[1]
    for row in revision.binary_control_rows("finetuned_vs_semantic"):
        if row["task"] == "Categorization":
            row["task"] = "Category choice"
        rows.append(row)
    frame = pd.DataFrame(rows)
    keys = ["task", "country", "condition"]
    expected = {(task, country, condition) for task, _ in TASKS
                for country in COUNTRIES for condition in CONDITIONS}
    assert len(frame) == 24 and not frame.duplicated(keys).any()
    assert set(frame[keys].itertuples(index=False, name=None)) == expected
    assert frame[["value", "ci_low", "ci_high"]].notna().all().all()
    assert (frame.ci_low <= frame.ci_high).all()
    assert frame.ci_low.min() >= -.10 and frame.ci_high.max() <= .25
    return frame


def draw(frame: pd.DataFrame, output: Path) -> None:
    style = paper_style(2.75, nrows=1, ncols=2)
    # savefig restores the prior layout engine after each export. With a manual
    # layout, restoring None can re-enable the shared style's constrained layout
    # for the next format. Disable both automatic engines in the context itself.
    style.update({"figure.constrained_layout.use": False, "figure.autolayout": False})
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, sharey=True, layout="none")
        fig.subplots_adjust(left=.18, right=.985, bottom=.16, top=.92,
                            wspace=.12)
        centers = (8.8, 6.4, 4.0, 1.6)
        for ax, country, heading in zip(axes, COUNTRIES, ("US responses", "Chinese responses")):
            panel = frame[frame.country.eq(country)]
            for center, (task, _) in zip(centers, TASKS):
                for offset, condition in zip((.6, 0, -.6), CONDITIONS):
                    row = panel[panel.task.eq(task) & panel.condition.eq(condition)].iloc[0]
                    y = center + offset
                    color = COLORS[condition]
                    # Draw endpoints directly: percentile CIs need not contain
                    # the observed estimate, so asymmetric errorbar sizes can fail.
                    ax.plot([row.ci_low, row.ci_high], [y, y], color=color,
                            lw=1.15, solid_capstyle="round", zorder=3)
                    ax.plot(row.value, y, "o", color=color, ms=4.0,
                            markeredgecolor="white", markeredgewidth=.35, zorder=4)
            ax.axvline(0, color="black", lw=.65, zorder=1)
            ax.set_xlim(-.10, .25)
            ax.set_xticks([-.1, 0, .1, .2], ["-0.1", "0", "0.1", "0.2"])
            ax.set_ylim(.7, 9.7)
            ax.set_yticks(centers, [title for _, title in TASKS])
            ax.tick_params(axis="y", length=0, pad=8, labelsize=7.5)
            ax.tick_params(axis="x", labelsize=7)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_linewidth(.5)
            ax.set_title(heading, fontsize=8.5, pad=6)

        fig.text(.5825, .04, "NLL gain over adjustment (nats/response)",
                 ha="center", va="center", fontsize=8)
        handles = [Line2D([], [], color=COLORS[c], marker="o", markersize=4.0,
                          lw=1.1, label=label)
                   for c, label in zip(CONDITIONS, ("US", "Pooled", "China"))]
        legend = axes[1].legend(handles=handles, title="Fine-tuning population", ncol=1,
                       loc="lower right", bbox_to_anchor=(.985, .025),
                       title_fontsize=6, fontsize=7, handlelength=1.3,
                       handletextpad=.55, columnspacing=1.2, labelspacing=.4,
                       borderaxespad=0, frameon=True, fancybox=False,
                       facecolor="white", edgecolor="black", framealpha=1)
        legend.get_frame().set_linewidth(.5)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(output.with_suffix(".png"), dpi=300)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "paper/figures/response_adjustment_comparison")
    args = parser.parse_args()
    frame = load_comparisons()
    draw(frame, args.output)
    frame.to_csv(args.output.with_suffix(".data.csv"), index=False)
    sources = [
        "results/giving_taking/response_adjustment/prior_paired_intervals.csv",
        "results/giving_taking/response_adjustment/prior_metrics_by_seed.csv",
        "results/giving_taking/balanced_taking_20261029/response_adjustment_gains.csv",
        "results/cao_anllo/followup_20261029/contrasts.csv",
    ]
    metadata = {
        "estimand": "NLL of matched response adjustment minus NLL of fine-tuned model",
        "control": "original response adjustment; binary finetuned_vs_semantic",
        "comparisons": len(frame),
        "allocation_intervals": "95% paired participant-and-observed-seed bootstrap",
        "binary_intervals": "95% paired participant bootstrap conditional on fitted models and items",
        "sources": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
    }
    args.output.with_suffix(".provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved figure with {len(frame)} comparisons: {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
