#!/usr/bin/env python3
"""Draw Figure 2 from the saved response probabilities and NLL contrasts."""
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from paper_plot_style import paper_style

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("us", "china", "centaur", "llama")
COLORS = {"us": "#4878A8", "china": "#CF866F", "centaur": "#242424", "llama": "#939AA3"}
LABELS = {"us": "US responses", "china": "Chinese responses", "centaur": "Frozen Centaur", "llama": "Base Llama"}
SOURCE_MARKERS = {"us": "o", "china": "o", "centaur": "s", "llama": "D"}
TASK_MARKERS = {"Giving": "o", "Taking": "s", "Categorization": "^", "Risky choice": "D"}
TASK_COLORS = {"Giving": "#8064A2", "Taking": "#4B908B", "Categorization": "#C28E3E", "Risky choice": "#4878A8"}
TASK_LABELS = {"Giving": "Social allocation: giving variant", "Taking": "Social allocation: taking variant",
               "Categorization": "Categorization", "Risky choice": "Risky choice"}
STEM = "figure2_distributions_nll"


def clean_axis(axis):
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#939393")
        axis.spines[side].set_linewidth(.55)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(width=.5, length=2.5, pad=3)


def draw_allocations(axis, data, task):
    for population in ("llama", "centaur", "china", "us"):
        part = data.loc[data.task.eq(task) & data.source.eq(population)].copy()
        part["response"] = part.option.astype(int)
        part = part.sort_values("response")
        axis.plot(part.response, 100*part.probability, color=COLORS[population],
                  lw=.85, ls=(0, (3, 2)) if population == "llama" else "-",
                  marker=SOURCE_MARKERS[population], ms=1.8,
                  markeredgewidth=0, zorder=3 if population in ("us", "china") else 2)
    axis.set(xlim=(-.5, 20.5), ylim=(0, 31),
             xlabel="Tokens given" if task == "Giving" else "Tokens taken")
    axis.set_xticks([0, 5, 10, 15, 20])
    axis.set_yticks([0, 10, 20, 30])
    axis.set_title("Social allocation: " + task.lower() + " variant", pad=8)
    axis.xaxis.labelpad = 3
    clean_axis(axis)


def draw_binary(axis, data, task, items):
    item_ids = [str(i["item_id"]) for i in items]
    positions = np.arange(1, len(items)+1) if task == "Categorization" else np.array([1, 2, 3, 4, 6, 7, 8, 9])
    for population, offset in zip(SOURCES, [-.24, -.08, .08, .24]):
        part = data.loc[data.task.eq(task) & data.source.eq(population)].set_index("option").loc[item_ids]
        axis.plot(positions+offset, 100*part.probability, ls="none",
                  marker=SOURCE_MARKERS[population], color=COLORS[population],
                  ms=2.8, markeredgewidth=0)
    axis.set_ylim(0, 102)
    axis.set_yticks([0, 50, 100])
    axis.set_title(task, pad=4)
    if task == "Categorization":
        axis.set(xlim=(.5, 15.5), ylabel="Taxonomic choice (%)", xlabel="Item")
        axis.set_xticks([1, 5, 10, 15])
    else:
        axis.set(xlim=(.5, 9.5), ylabel="10-point choice (%)",
                 xlabel="Win probabilities (%)")
        labels = [f"{100*i['p_10']:g}/{100*i['p_1']:g}" for i in items]
        axis.set_xticks(positions, labels)
        plt.setp(axis.get_xticklabels(), rotation=55, ha="right", rotation_mode="anchor", fontsize=7)
        axis.xaxis.labelpad = 4
        axis.axvline(5, color="#D9D9D9", lw=.55, zorder=0)
        for x, label in [(2.5, "Tradeoff"), (7.5, "Dominance")]:
            axis.text(x, 13, label, ha="center", va="center", fontsize=7.5, color="#666666")
    clean_axis(axis)


def draw_nll(axis, gap):
    offsets = {"Giving": 0., "Taking": .025, "Categorization": -.025, "Risky choice": 0.}
    axis.axhline(0, color="#999999", lw=.65, ls=(0, (3, 3)), zorder=0)
    for task, marker in TASK_MARKERS.items():
        rows = gap.loc[gap.task.eq(task)].set_index("source").loc[["llama", "centaur"]]
        x = np.array([0., 1.]) + offsets[task]
        color = TASK_COLORS[task]
        axis.plot(x, rows.estimate, color=color, lw=1.0, zorder=1)
        axis.vlines(x, rows.ci_low, rows.ci_high, color=color, lw=.9, zorder=2)
        axis.plot(x, rows.estimate, color=color, marker=marker, ls="none",
                  ms=4.2 if marker == "D" else 5.0, markeredgecolor="white",
                  markeredgewidth=.4, zorder=3)
    axis.set(xlim=(-.19, 1.16), ylim=(-.065, .43), ylabel="China − US NLL (nats)")
    axis.set_xticks([0, 1], ["Base Llama", "Frozen Centaur"])
    axis.set_yticks([0, .1, .2, .3, .4], ["0", "0.1", "0.2", "0.3", "0.4"])
    clean_axis(axis)
    axis.tick_params(axis="x", length=0, pad=7, labelsize=9)


def load_inputs(input_dir: Path):
    """Read the two aggregate tables and check task and response coverage."""
    data_path = input_dir / "response_probabilities.csv"
    gap_path = input_dir / "population_nll_gaps.csv"
    metadata_path = input_dir / "metadata.json"
    data = pd.read_csv(data_path, dtype={"option": str})
    gap = pd.read_csv(gap_path)
    metadata = json.loads(metadata_path.read_text())
    labels = {
        "Categorization": [{"item_id": item} for item in metadata["categorization_item_order"]],
        "Risky choice": metadata["risky_choice_order"],
    }
    expected_options = {
        "Giving": {str(i) for i in range(21)},
        "Taking": {str(i) for i in range(21)},
        **{task: {str(item["item_id"]) for item in items} for task, items in labels.items()},
    }
    expected_rows = {(task, source, option) for task, options in expected_options.items()
                     for source in SOURCES for option in options}
    actual_rows = set(data[["task", "source", "option"]].itertuples(index=False, name=None))
    if len(data) != 260 or actual_rows != expected_rows:
        raise ValueError("Response data must cover every task, source, and response once.")
    if not data.probability.between(0, 1).all():
        raise ValueError("Response probabilities must be finite and between zero and one.")
    allocation = data.loc[data.task.isin(("Giving", "Taking"))]
    if not np.allclose(allocation.groupby(["task", "source"]).probability.sum(), 1):
        raise ValueError("Allocation probabilities must sum to one for each task and source.")
    expected_gaps = {(task, source) for task in TASK_MARKERS for source in ("llama", "centaur")}
    actual_gaps = set(gap[["task", "source"]].itertuples(index=False, name=None))
    if len(gap) != 8 or actual_gaps != expected_gaps:
        raise ValueError("NLL data must contain each task and model once.")
    if not np.isfinite(gap[["estimate", "ci_low", "ci_high"]]).all().all():
        raise ValueError("NLL estimates and intervals must be finite.")
    if not (gap.ci_low <= gap.ci_high).all():
        raise ValueError("An NLL interval has reversed endpoints.")
    paths = (data_path, gap_path, metadata_path)
    return data, gap, labels, paths


def draw(input_dir: Path, output_dir: Path):
    """Write the figure and record the hashes of its aggregate inputs."""
    data, gap, labels, paths = load_inputs(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    style = paper_style(3.95)
    style.update({"figure.constrained_layout.use": False, "font.size": 8.5,
                  "axes.labelsize": 8.5, "axes.titlesize": 9.5,
                  "xtick.labelsize": 8, "ytick.labelsize": 8,
                  "axes.labelpad": 5, "svg.fonttype": "none"})
    with plt.rc_context(style):
        fig = plt.figure(figsize=(8.4, 3.95))
        axes = {
            "Giving": fig.add_axes([.065, .550, .245, .250]),
            "Taking": fig.add_axes([.375, .550, .245, .250]),
            "Categorization": fig.add_axes([.065, .160, .245, .255]),
            "Risky choice": fig.add_axes([.375, .160, .245, .255]),
            "NLL": fig.add_axes([.730, .160, .245, .640]),
        }
        center_a, center_b = .3425, .8525
        title_y, legend_y = .974, .900
        fig.text(.012, title_y, "a", ha="left", va="center", fontsize=11, fontweight="bold")
        fig.text(center_a, title_y, "Observed and predicted responses", ha="center", va="center", fontsize=11)
        fig.text(.665, title_y, "b", ha="left", va="center", fontsize=11, fontweight="bold")
        fig.text(center_b, title_y, "Population NLL gap", ha="center", va="center", fontsize=11)

        human_order = ["us", "china", "centaur", "llama"]
        source_handles = [Line2D([], [], color=COLORS[s], marker=SOURCE_MARKERS[s],
                                 lw=.85, ls=(0, (3, 2)) if s == "llama" else "-", ms=3.5,
                                 label=LABELS[s]) for s in human_order]
        task_handles = [Line2D([], [], color=TASK_COLORS[t], marker=TASK_MARKERS[t],
                              ls="none", ms=4.1, label=TASK_LABELS[t]) for t in TASK_MARKERS]
        common_legend = dict(loc="center", fontsize=7.6, frameon=False,
                             handletextpad=.5, columnspacing=1.1, labelspacing=.65, borderaxespad=0)
        legend_a = fig.legend(handles=source_handles, bbox_to_anchor=(center_a, legend_y), ncol=4, handlelength=1.7, **common_legend)
        legend_b = fig.legend(handles=task_handles, bbox_to_anchor=(center_b, legend_y), ncol=2, handlelength=.8, **common_legend)

        for task in ("Giving", "Taking"):
            draw_allocations(axes[task], data, task)
        axes["Giving"].set_ylabel("Response probability (%)")
        for task in ("Categorization", "Risky choice"):
            draw_binary(axes[task], data, task, labels[task])
        draw_nll(axes["NLL"], gap)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_boxes = [l.get_window_extent(renderer).transformed(fig.transFigure.inverted()) for l in (legend_a, legend_b)]
        assert abs((legend_boxes[0].y0+legend_boxes[0].y1)/2 - (legend_boxes[1].y0+legend_boxes[1].y1)/2) < .001
        assert all(box.x0 >= 0 and box.x1 <= 1 for box in legend_boxes), [b.bounds for b in legend_boxes]
        for extension in ("png", "pdf", "svg"):
            metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else ({"Date": None} if extension == "svg" else {})
            fig.savefig(output_dir / f"{STEM}.{extension}", dpi=350, facecolor="white", metadata=metadata)
        plt.close(fig)

    (output_dir / f"{STEM}.provenance.json").write_text(json.dumps({
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "panels": {"a": "Observed and predicted response distributions", "b": "Population NLL gap"},
        "layout_inches": [8.4, 3.95],
        "risky_choice_items": labels["Risky choice"],
    }, indent=2) + "\n")
    print(output_dir / f"{STEM}.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/figure2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/figures")
    args = parser.parse_args()
    draw(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
