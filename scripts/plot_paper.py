#!/usr/bin/env python3
"""Rebuild the starter paper figure from the checked-in aggregate results.

Run from any working directory: python /path/to/repo/scripts/plot_paper.py
Only aggregate predictions are read. No raw participant data or GPU is needed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tueplots import bundles


ROOT = Path(__file__).resolve().parents[1]
PANELS = (
    ("cao", "all", 15, "a  Category choice", "Category choice (%)"),
    ("anllo", "tradeoff", 4, "b  Lottery tradeoffs", "10-point lottery choice (%)"),
)
CONDITIONS = ("frozen", "us", "pooled", "china")
COLORS = ("#B9BEC4", "#4878A8", "#8E82A8", "#CF866F")
LABELS = ("Frozen", "US", "Pooled", "China")


def figure_data(source: Path) -> list[dict]:
    """Select unseen-item summaries without counting duplicate country rows."""
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        source_name = str(source.resolve().relative_to(ROOT))
    except ValueError:
        source_name = source.name
    data = []
    for dataset, stratum, n_items, _, _ in PANELS:
        selected = [
            row for row in rows
            if row["dataset"] == dataset
            and row["stratum"] == stratum
            and row["evaluation_item_scope"] == "unseen_items"
            and row["country"] == "balanced_us_china"
            and row["source"] in ("frozen_centaur", "finetuned_centaur")
        ]
        if len(selected) != 18:
            raise ValueError(f"Expected 18 source rows for {dataset}; found {len(selected)}")
        if any(row["coverage"] != "complete" or int(row["n_items"]) != n_items
               for row in selected):
            raise ValueError(f"Incomplete item coverage for {dataset}")
        frozen = [float(row["predicted_p_positive"]) for row in selected
                  if row["source"] == "frozen_centaur"]
        if len(frozen) != 9 or max(frozen) - min(frozen) > 1e-12:
            raise ValueError(f"Frozen predictions are inconsistent across runs for {dataset}")
        entries = [("frozen", "", frozen[0])]
        for condition in CONDITIONS[1:]:
            fitted = sorted(
                (row for row in selected
                 if row["source"] == "finetuned_centaur"
                 and row["condition"] == condition),
                key=lambda row: int(row["seed"]),
            )
            if [row["seed"] for row in fitted] != ["0", "1", "2"]:
                raise ValueError(f"Expected exactly seeds 0, 1, 2 for {dataset}/{condition}")
            entries.extend((condition, row["seed"], float(row["predicted_p_positive"]))
                           for row in fitted)
        for condition, seed, probability in entries:
            if not 0 <= probability <= 1:
                raise ValueError(f"Invalid probability for {dataset}/{condition}/{seed}")
            data.append({
                "dataset": dataset,
                "stratum": stratum,
                "n_items": n_items,
                "model": condition,
                "seed": seed,
                "predicted_probability": probability,
                "evaluation_scope": "unseen_items",
                "source_file": source_name,
                "source_sha256": digest,
            })
    return data


def draw(data: list[dict], output_dir: Path) -> None:
    style = bundles.neurips2024(usetex=False, nrows=1, ncols=2, family="sans-serif")
    style.update({
        "font.sans-serif": ["DejaVu Sans"],
        "figure.figsize": (6.3, 2.5),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": "#B8BFC5",
        "axes.linewidth": 0.6,
        "axes.labelcolor": "#26333D",
        "text.color": "#26333D",
        "xtick.color": "#26333D",
        "ytick.color": "#52606A",
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2)
        for ax, (dataset, _, _, title, ylabel) in zip(axes, PANELS):
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_xlim(-0.55, 3.55)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E7EBEE", linewidth=0.55)
            for x, (condition, color) in enumerate(zip(CONDITIONS, COLORS)):
                values = [100 * row["predicted_probability"] for row in data
                          if row["dataset"] == dataset and row["model"] == condition]
                mean = sum(values) / len(values)
                ax.bar(x, mean, width=0.64, color=color, edgecolor="none", zorder=2)
                if condition != "frozen":
                    ax.scatter([x - 0.12, x, x + 0.12], values, s=13,
                               color="#26333D", edgecolor="white", linewidth=0.45,
                               zorder=3)
            ax.set_xticks(range(4), LABELS)
            ax.set_xlabel("Centaur model", labelpad=7)
            ax.set_ylabel(ylabel, labelpad=5)
            ax.set_title(title, loc="left", fontweight="bold", pad=10)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_dir / "population_predictions"
        fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(stem.with_suffix(".png"), dpi=300)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/cao_anllo/seed_metrics.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/figures")
    args = parser.parse_args()
    data = figure_data(args.input)
    data_path = ROOT / "results/paper_figure_data.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    draw(data, args.output_dir)
    print(f"Wrote {args.output_dir / 'population_predictions.pdf'}")
    print(f"Wrote {args.output_dir / 'population_predictions.png'}")
    print(f"Wrote {data_path}")


if __name__ == "__main__":
    main()
