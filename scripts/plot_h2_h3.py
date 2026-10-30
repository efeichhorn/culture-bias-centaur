#!/usr/bin/env python3
"""Draw H2/H3 item predictions, averaged over the three fitted seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import figure_helpers as original
from paper_plot_style import COLORS, paper_style

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"
EXT = ROOT / "results/cao_anllo/followup_20261029"
CONDITIONS = ("us", "pooled", "china")
LABELS = ("US", "Pooled", "China")
TASKS = ("Categorization", "Lottery tradeoffs", "Giving", "Taking")
SOURCES: set[Path] = set()


def read(path):
    SOURCES.add(path)
    return pd.read_csv(path)


def one(frame, **where):
    for key, value in where.items():
        frame = frame.loc[frame[key].eq(value)]
    if len(frame) != 1:
        raise ValueError(f"Expected one aggregate for {where}, found {len(frame)}")
    return frame.iloc[0]


def prediction_data():
    binary = read(ROOT / "results/cao_anllo/item_predictions.csv")
    allocation = read(ROOT / "results/giving_taking/followup_20261029/h2_h3_item_predictions.csv")
    reference = read(EXT / "prediction_summary.csv")
    allocation_reference = read(ROOT / "results/giving_taking/figure_inputs/prediction_shifts_by_seed.csv")
    rows = []
    for task, dataset, stratum, count in (("Categorization", "cao", "critical", 15),
                                          ("Lottery tradeoffs", "anllo", "tradeoff", 4)):
        selected = binary.loc[(binary.dataset == dataset) & (binary.stratum == stratum)
                              & (binary.source == "finetuned_centaur")
                              & (binary.evaluation_item_scope == "unseen_items")]
        if not selected.is_heldout_item.all() or len(selected) != count*9:
            raise ValueError(f"Incomplete held-out predictions for {task}")
        for row in selected.itertuples(index=False):
            rows.append(dict(task=task, item=str(row.item_id), seed=int(row.seed),
                             condition=row.condition, prediction=100*row.p_positive, unit="percent"))
        for condition in CONDITIONS:
            old = one(reference, dataset=dataset, stratum="all" if dataset == "cao" else stratum,
                      condition=condition, source="finetuned_centaur", metric="predicted_p_positive")
            observed = selected.loc[selected.condition == condition, "p_positive"].mean()
            if not np.isclose(observed, old.estimate, rtol=0, atol=1e-12):
                raise ValueError(f"Binary prediction mean changed for {task}/{condition}")
    for row in allocation.itertuples(index=False):
        rows.append(dict(task=row.task, item=str(row.item), seed=int(row.seed),
                         condition=row.condition, prediction=row.prediction, unit="tokens"))
    frame = pd.DataFrame(rows)
    if frame.duplicated(["task", "item", "seed", "condition"]).any():
        raise ValueError("Duplicate item predictions")
    for task, count in zip(TASKS, (15, 4, 25, 25)):
        part = frame[frame.task == task]
        if len(part) != count*9 or part.item.nunique() != count:
            raise ValueError(f"Wrong item coverage for {task}")
        for _, group in part.groupby("item"):
            if set(zip(group.seed, group.condition)) != {(s, c) for s in (0, 1, 2) for c in CONDITIONS}:
                raise ValueError("Unpaired conditions or seeds")
        if task in ("Giving", "Taking"):
            for (seed, condition), group in part.groupby(["seed", "condition"]):
                old = one(allocation_reference, task=task, seed=seed, condition=condition)
                if not np.isclose(group.prediction.mean(), old.expected_tokens, rtol=0, atol=1e-10):
                    raise ValueError("Allocation prediction mean changed")
    if not np.isfinite(frame.prediction).all():
        raise ValueError("Nonfinite item prediction")
    return frame


def plot_predictions(frame):
    style = paper_style(2.35, nrows=1, ncols=4)
    style.update({"figure.constrained_layout.use": False, "font.size": 8,
                  "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 7.5})
    summaries = []
    with plt.rc_context(style):
        fig = plt.figure()
        grid = fig.add_gridspec(1, 4, left=.075, right=.99, bottom=.185, top=.789, wspace=.42)
        axes = [fig.add_subplot(grid[0, c]) for c in range(4)]
        labels = ("Taxonomic choice (%)", "10-point choice (%)", "Mean tokens given", "Mean tokens taken")
        for col, (ax, task, ylabel) in enumerate(zip(axes, TASKS, labels)):
            part = frame[frame.task == task]
            items = part.groupby(["item", "condition"]).prediction.mean().unstack("condition")[list(CONDITIONS)]
            for _, row in items.iterrows():
                ax.plot(range(3), row.to_numpy(), color="#B9BEC4", alpha=.75, lw=.65, zorder=1)
                ax.scatter(range(3), row.to_numpy(), s=4.5, color="#AEB5BC", alpha=.8, linewidths=0, zorder=2)
            means = items.mean()
            ax.plot(range(3), means, color="#333333", lw=1.6, zorder=4)
            for x, condition in enumerate(CONDITIONS):
                ax.plot(x, means[condition], "o", color=COLORS[condition], ms=5.7,
                        mec="white", mew=.7, zorder=5)
            delta = means["us"]-means["china"]
            title = f"Social allocation:\n{task.lower()} variant" if task in ("Giving", "Taking") else ("Risky choice:\ntradeoff pairs" if task == "Lottery tradeoffs" else task)
            ax.set_title(title, loc="left", fontsize=8.3, pad=5)
            ax.set_xlim(-.22, 2.22)
            ax.set_xticks(range(3), LABELS)
            for tick, condition in zip(ax.get_xticklabels(), CONDITIONS):
                tick.set_color(COLORS[condition])
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 100 if col < 2 else 12)
            ax.set_yticks([0, 25, 50, 75, 100] if col < 2 else [0, 4, 8, 12])
            ax.yaxis.grid(True, color="#E8EAEC", linewidth=.5, zorder=0)
            ax.tick_params(axis="x", length=0, pad=4)
            seed_means = part.groupby(["seed", "condition"]).prediction.mean().unstack("condition")
            for condition in CONDITIONS:
                summaries.append(dict(task=task, condition=condition, mean=means[condition],
                                      seed_min=seed_means[condition].min(), seed_max=seed_means[condition].max(),
                                      n_items=len(items), us_minus_china=delta))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax, letter, label in ((axes[0], "a", "Within-task effects"),
                                   (axes[3], "b", "Transfer")):
            panel_left = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).x0
            fig.text(panel_left, .945, letter, ha="left", fontsize=9, fontweight="bold")
            fig.text(ax.get_position().x0, .945, label, ha="left", fontsize=9)
        handles = [Line2D([], [], color="#B9BEC4", lw=.8, label="Individual item"),
                   Line2D([], [], color="#333333", lw=1.6, marker="o", ms=4, label="Mean across items")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5,.015), ncol=2,
                   frameon=False, fontsize=7.5, handlelength=1.7, columnspacing=1.6,
                   borderaxespad=0)
        original.save(fig, OUT, "population_effects_h2_h3")
    return pd.DataFrame(summaries)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    OUT = parser.parse_args().output_dir
    frame = prediction_data()
    summary = plot_predictions(frame)
    frame.to_csv(OUT / "population_effects_h2_h3.data.csv", index=False)
    summary.to_csv(OUT / "population_effects_h2_h3.summary.csv", index=False)
    SOURCES.add(Path(__file__).resolve())
    SOURCES.add(ROOT / "scripts/paper_plot_style.py")
    provenance = {
        "figure": "population_effects_h2_h3",
        "task_order": list(TASKS),
        "aggregation": "Each item averages three seeds; task means give each item equal weight.",
        "spread": "Light lines show fixed held-out items, not uncertainty intervals.",
        "response_definitions": {"Categorization": "taxonomic choice", "Lottery tradeoffs": "10-point choice",
                                 "Giving": "expected tokens given", "Taking": "expected tokens taken"},
        "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)},
    }
    (OUT / "population_effects_h2_h3.provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(OUT / "population_effects_h2_h3.pdf")


if __name__ == "__main__":
    main()
