#!/usr/bin/env python3
"""Draw the appendix frozen-model scores and answer-position control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

import figure_helpers as original
from paper_plot_style import COLORS, paper_style

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "results/cao_anllo/followup_20261029"
TAKING = ROOT / "results/giving_taking/balanced_taking_20261029"
OUT = ROOT / "paper/figures"
CONDITIONS = ("us", "pooled", "china")
COUNTRIES = ("us", "china")
COUNTRY_LABELS = ("US responses", "Chinese responses")
TASKS = (("cao", "all", "Categorization"), ("anllo", "tradeoff", "Lottery tradeoffs"))
INPUTS: set[Path] = set()


def read(path: Path) -> pd.DataFrame:
    INPUTS.add(path)
    return pd.read_csv(path)


def select(frame: pd.DataFrame, **where) -> pd.Series:
    chosen = frame
    for key, value in where.items():
        chosen = chosen[chosen[key].eq(value)]
    if len(chosen) != 1:
        raise ValueError(f"Expected exactly one row: {where}; found {len(chosen)}")
    return chosen.iloc[0]


def interval(ax, x, row, color, marker="o", horizontal=False, label=None):
    value = float(row["estimate"])
    low, high = float(row["ci_low"]), float(row["ci_high"])
    # A percentile interval need not contain its observed point estimate.
    if horizontal:
        ax.plot([low, high], [x, x], color=color, lw=.85, zorder=3)
        ax.plot(value, x, marker=marker, ms=3.3, color=color, label=label, zorder=4)
    else:
        ax.plot([x, x], [low, high], color=color, lw=.85, zorder=3)
        ax.plot(x, value, marker=marker, ms=3.3, color=color, label=label, zorder=4)


def save(fig, name):
    original.save(fig, OUT, name)


def binary_control_rows(contrast):
    contrasts = read(EXT / "contrasts.csv")
    rows = []
    for dataset, stratum, task in TASKS:
        for country, full in zip(COUNTRIES, original.COUNTRIES):
            for condition in CONDITIONS:
                row = select(contrasts, dataset=dataset, stratum=stratum, country=country,
                             condition=condition, contrast=contrast)
                rows.append(dict(task=task, country=full, condition=condition,
                                 value=row.estimate, ci_low=row.ci_low, ci_high=row.ci_high))
    return rows


def plot_position_control():
    contrasts = read(EXT / "contrasts.csv")
    style = paper_style(2.8)
    style["figure.constrained_layout.use"] = False
    with plt.rc_context(style):
        fig = plt.figure()
        axes = [fig.add_axes([x / 453.6, 15.319375 / 201.6,
                             182.645375 / 453.6, 150.150625 / 201.6])
                for x in (37.350625, 266.994625)]
        for panel, (ax, (dataset, stratum, title)) in enumerate(zip(axes, TASKS)):
            for x, country in enumerate(COUNTRIES):
                for offset, condition in zip((-.24, 0, .24), CONDITIONS):
                    for dx, name, marker in ((-.035, "finetuned_vs_semantic", "o"),
                                            (.035, "finetuned_vs_semantic_position", "s")):
                        row = select(contrasts, dataset=dataset, stratum=stratum, country=country,
                                     condition=condition, contrast=name)
                        interval(ax, x+offset+dx, row, COLORS[condition], marker=marker)
            ax.axhline(0, color="black", lw=.5)
            ax.set_xticks([0, 1], COUNTRY_LABELS)
            ax.set_xlim(-.55, 1.55)
            title = "Categorization" if dataset == "cao" else "Risky choice: tradeoff pairs"
            original.decorate(ax, f"{'ab'[panel]}  {title}", "NLL gain over adjustment (nats/response)")
        handles = [Patch(color=COLORS[c], label=l) for c, l in zip(CONDITIONS, ("US", "Pooled", "China"))]
        handles += [Line2D([], [], color="black", marker=m, lw=0, label=l)
                    for m, l in (("o", "Response"), ("s", "Response + position"))]
        fig.legend(handles=handles, ncol=5, loc="outside upper center", frameon=False,
                   handlelength=1, columnspacing=1)
        save(fig, "answer_position_control")


def revised_allocation_data():
    frozen, controls = original.allocation_data()
    # The balanced reanalysis exports plotting inputs with the original schema.
    for name, rows in (("frozen_sample_error.csv", frozen), ("response_adjustment_gains.csv", controls)):
        replacement = read(TAKING / name).to_dict("records")
        rows[:] = [r for r in rows if r["task"] != "Taking"] + replacement
    return frozen, controls


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    OUT = parser.parse_args().output_dir
    frozen, _ = revised_allocation_data()
    extension_frozen = original.binary_frozen_data()
    intervals = read(EXT / "frozen_intervals.csv")
    for row in extension_frozen:
        dataset = "cao" if row["task"] == "Category choice" else "anllo"
        ci = select(intervals, dataset=dataset, stratum="all" if dataset == "cao" else "tradeoff",
                    country="us" if row["Country"] == "United States" else "china",
                    source="frozen_centaur" if row["model"] == "centaur" else "base_llama")
        row.update(error_ci_low=ci.ci_low, error_ci_high=ci.ci_high)
    frozen.extend(extension_frozen)
    original.draw_frozen(frozen, OUT)
    plot_position_control()
    INPUTS.update(ROOT / p for p in (
        "results/giving_taking/figure_inputs/frozen_sample_error.csv",
        "results/giving_taking/frozen/metrics_summary.csv",
        "results/giving_taking/response_adjustment/prior_paired_intervals.csv",
        "results/giving_taking/response_adjustment/prior_metrics_by_seed.csv",
        "results/cao_anllo/frozen_baseline_country_metrics.csv"))
    provenance = {"style": "tueplots via paper_plot_style.paper_style", "figures": ["frozen_sample_fit", "answer_position_control"],
                  "taking_sample": "1329 per country, wave matched; see balanced Taking metadata",
                  "binary_intervals": "shared participant bootstrap, fixed fitted seeds and items",
                  "allocation_intervals": "shared participant and observed-seed bootstrap for adapter contrasts",
                  "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(INPUTS)}}
    (OUT / "results_revision.provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(f"Saved frozen_sample_fit and answer_position_control in {OUT}")


if __name__ == "__main__":
    main()
