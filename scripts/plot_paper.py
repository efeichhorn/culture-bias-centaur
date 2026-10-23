#!/usr/bin/env python3
"""Rebuild the paper figures from the checked-in aggregate results.

Run from any working directory: python /path/to/repo/scripts/plot_paper.py
Only aggregate predictions are read. No raw participant data or GPU is needed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tueplots import bundles
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
PANELS = (
    ("cao", "all", 15, "a  Category choice", "Category choice (%)"),
    ("anllo", "tradeoff", 4, "b  Lottery tradeoffs", "10-point lottery choice (%)"),
)
CONDITIONS = ("frozen", "us", "pooled", "china")
COLORS = ("#B9BEC4", "#4878A8", "#8E82A8", "#CF866F")
LABELS = ("Frozen", "US", "Pooled", "China")
COUNTRIES = ("United States", "China")
COUNTRY_LABELS = ("US responses", "China responses")
GIVING_TAKING = ROOT / "results/giving_taking"


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


def paper_style(height: float = 2.5) -> dict:
    """Shared tueplots typography, with an ACL full-width figure canvas."""
    style = bundles.neurips2024(usetex=False, nrows=1, ncols=2, family="sans-serif")
    style.update({
        "font.sans-serif": ["DejaVu Sans"],
        "figure.figsize": (6.3, height),
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
    return style


def decorate(ax, title: str, ylabel: str) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E7EBEE", linewidth=0.55)
    ax.set_ylabel(ylabel, labelpad=5)
    ax.set_title(title, loc="left", fontweight="bold", pad=10)


def save(fig, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / name
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def draw(data: list[dict], output_dir: Path) -> None:
    with plt.rc_context(paper_style()):
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
        save(fig, output_dir, "population_predictions")


def read_csv(relative: str) -> list[dict]:
    with (GIVING_TAKING / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def one(rows: list[dict], **where) -> dict:
    found = [row for row in rows if all(row[key] == value for key, value in where.items())]
    if len(found) != 1:
        raise ValueError(f"Expected one matching aggregate, found {len(found)}: {where}")
    return found[0]


def require_equal(actual: float, expected: float, name: str) -> None:
    if abs(actual - expected) > 1e-10:
        raise ValueError(f"Inconsistent aggregate {name}: {actual} vs {expected}")


def allocation_data() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    frozen = read_csv("figure_inputs/frozen_sample_error.csv")
    frozen_reference = read_csv("frozen/metrics_summary.csv")
    for row in frozen:
        ref = one(frozen_reference, task=row["task"], Country=row["Country"],
                  condition=row["model"] + "_original")
        require_equal(float(row["prediction_error"]), -float(ref["conditional_ll"]),
                      "frozen prediction error")

    shifts = read_csv("figure_inputs/prediction_shifts_by_seed.csv")
    shifts_reference = read_csv("response_adjustment/human_model_means_by_seed.csv")
    if len(shifts) != 18:
        raise ValueError("Expected 2 tasks x 3 fine-tuning populations x 3 seeds")
    for row in shifts:
        for country in COUNTRIES:
            ref = one(shifts_reference, seed=row["seed"], task=row["task"].lower(),
                      sample="Combined", country=country, condition=row["condition"])
            require_equal(float(row["expected_tokens"]), float(ref["model_expected_tokens"]),
                          "expected tokens")
        require_equal(float(row["shift_tokens"]),
                      float(row["expected_tokens"]) - float(row["frozen_expected_tokens"]),
                      "shift from frozen")

    intervals = read_csv("bootstrap_intervals.csv")
    scores = read_csv("seed_metrics.csv")
    adjustment_intervals = read_csv("response_adjustment/prior_paired_intervals.csv")
    adjustment_scores = read_csv("response_adjustment/prior_metrics_by_seed.csv")
    gains, controls = [], []
    for task in ("Giving", "Taking"):
        for country in COUNTRIES:
            for condition in CONDITIONS[1:]:
                row = one(intervals, task=task, Country=country, metric="conditional_ll",
                          contrast=condition + "_gain_vs_base",
                          method="shared_participant_and_seed")
                values = []
                for seed in ("0", "1", "2"):
                    fitted = one(scores, seed=seed, task=task, Country=country, condition=condition)
                    base = one(scores, seed=seed, task=task, Country=country, condition="base")
                    values.append(float(fitted["conditional_ll"]) - float(base["conditional_ll"]))
                require_equal(float(row["mean"]), mean(values), "gain vs frozen")
                gains.append(dict(task=task, country=country, condition=condition,
                                  value=float(row["mean"]), ci_low=float(row["ci_low"]),
                                  ci_high=float(row["ci_high"])))

                row = one(adjustment_intervals, task=task, Country=country,
                          metric="conditional_ll", contrast=condition + "_prior_minus_lora",
                          method="shared_participant_and_seed")
                values = [float(one(adjustment_scores, seed=seed, task=task, country=country,
                                    condition=condition, metric="conditional_ll")["prior_minus_lora"])
                          for seed in ("0", "1", "2")]
                require_equal(float(row["mean"]), mean(values), "adjustment comparison")
                # Source is adjustment minus fine-tuning; reverse both endpoints.
                controls.append(dict(task=task, country=country, condition=condition,
                                     value=-float(row["mean"]), ci_low=-float(row["ci_high"]),
                                     ci_high=-float(row["ci_low"])))
    return frozen, shifts, gains, controls


def extension_summaries() -> tuple[list[dict], list[dict]]:
    with (ROOT / "results/cao_anllo/frozen_baseline_country_metrics.csv").open(newline="") as handle:
        baseline = list(csv.DictReader(handle))
    with (ROOT / "results/cao_anllo/seed_metrics.csv").open(newline="") as handle:
        scores = list(csv.DictReader(handle))
    frozen, controls = [], []
    for dataset, stratum, task in (("cao", "all", "Category choice"),
                                   ("anllo", "tradeoff", "Lottery tradeoffs")):
        for country, raw_country in zip(COUNTRIES, ("us", "china")):
            for model, source in (("centaur", "frozen_centaur"), ("llama", "base_llama")):
                row = one(baseline, dataset=dataset, stratum=stratum, source=source,
                          country=raw_country, sample_scope="all_retained_participants_frozen_audit")
                frozen.append(dict(task=task, Country=country, model=model,
                                   metric="conditional_ll", prediction_error=float(row["conditional_nll"])))
            for condition in CONDITIONS[1:]:
                values = []
                for seed in ("0", "1", "2"):
                    where = dict(dataset=dataset, stratum=stratum, condition=condition,
                                 country=raw_country, seed=seed, coverage="complete",
                                 evaluation_item_scope="unseen_items")
                    fitted = one(scores, source="finetuned_centaur", **where)
                    adjusted = one(scores, source="response_adjustment", **where)
                    values.append(float(adjusted["conditional_nll"]) - float(fitted["conditional_nll"]))
                controls.append(dict(task=task, country=country, condition=condition,
                                     value=mean(values), seed_values=values))
    return frozen, controls


def draw_frozen(rows: list[dict], output_dir: Path) -> None:
    with plt.rc_context(paper_style(4.35)):
        fig, axes = plt.subplots(2, 2)
        tasks = ("Giving", "Taking", "Category choice", "Lottery tradeoffs")
        for panel, (ax, task) in enumerate(zip(axes.flat, tasks)):
            for x, model in enumerate(("centaur", "llama")):
                for offset, country, color in zip((-.18, .18), COUNTRIES, (COLORS[1], COLORS[3])):
                    row = one(rows, task=task, model=model, Country=country, metric="conditional_ll")
                    value = float(row["prediction_error"])
                    ax.bar(x + offset, value, width=.32, color=color, zorder=2)
                    if "error_ci_low" in row:
                        ax.errorbar(x + offset, value,
                                    yerr=[[value - float(row["error_ci_low"])],
                                          [float(row["error_ci_high"]) - value]],
                                    fmt="none", color="#26333D", linewidth=.7, capsize=2, zorder=3)
            decorate(ax, f"{'abcd'[panel]}  {task}", "Error (nats; lower is better)")
            ax.set_ylim(0, 3.2 if panel < 2 else .8)
            ax.set_yticks([0, 1, 2, 3] if panel < 2 else [0, .2, .4, .6, .8])
            ax.set_xticks([0, 1], ["Centaur", "Llama"])
            ax.set_xlim(-.65, 1.65)
        fig.legend(handles=[Patch(color=COLORS[1], label="US responses"),
                            Patch(color=COLORS[3], label="China responses")],
                   frameon=False, ncols=2, loc="outside upper center",
                   handlelength=1, columnspacing=1.5)
        save(fig, output_dir, "frozen_sample_fit")


def grouped_gain(ax, rows: list[dict], task: str) -> None:
    for x, country in enumerate(COUNTRIES):
        for offset, condition, color in zip((-.24, 0, .24), CONDITIONS[1:], COLORS[1:]):
            row = one(rows, task=task, country=country, condition=condition)
            value = row["value"]
            ax.bar(x + offset, value, width=.21, color=color, zorder=2)
            if "ci_low" in row:
                ax.errorbar(x + offset, value,
                            yerr=[[value - row["ci_low"]], [row["ci_high"] - value]],
                            fmt="none", color="#26333D", linewidth=.7, capsize=2, zorder=3)
            else:
                ax.scatter([x+offset-.05, x+offset, x+offset+.05], row["seed_values"],
                           s=11, color="#26333D", edgecolor="white", linewidth=.4, zorder=3)
    ax.set_xticks([0, 1], COUNTRY_LABELS)
    ax.set_xlim(-.60, 1.60)
    ax.axhline(0, color="#A8B0B8", linewidth=.75, zorder=1)


def population_legend(fig) -> None:
    fig.legend(handles=[Patch(color=color, label=label) for color, label in zip(COLORS[1:], LABELS[1:])],
               title="Fine-tuning sample", frameon=False, ncols=3,
               loc="outside upper center", handlelength=1,
               columnspacing=1.5, title_fontsize=8)


def draw_allocation_effects(shifts: list[dict], gains: list[dict], output_dir: Path) -> None:
    with plt.rc_context(paper_style(4.65)):
        fig, axes = plt.subplots(2, 2)
        for panel, task in enumerate(("Giving", "Taking")):
            ax = axes[0, panel]
            for x, condition, color in zip(range(3), CONDITIONS[1:], COLORS[1:]):
                values = [float(one(shifts, task=task, condition=condition, seed=seed)["shift_tokens"])
                          for seed in ("0", "1", "2")]
                ax.bar(x, mean(values), width=.62, color=color, zorder=2)
                ax.scatter([x-.11, x, x+.11], values, s=13, color="#26333D",
                           edgecolor="white", linewidth=.45, zorder=3)
            decorate(ax, f"{'ab'[panel]}  {task}: prediction shift", "Shift from frozen (tokens)")
            ax.set_ylim(-4.5, .15)
            ax.set_yticks([-4, -3, -2, -1, 0])
            ax.set_xticks(range(3), LABELS[1:])
            ax.set_xlabel("Giving fine-tuning sample", labelpad=7)
            ax.set_xlim(-.6, 2.6)
            ax.axhline(0, color="#A8B0B8", linewidth=.75, zorder=1)

            ax = axes[1, panel]
            grouped_gain(ax, gains, task)
            decorate(ax, f"{'cd'[panel]}  {task}: prediction gain", "Gain over frozen\n(nats; higher is better)")
            ax.set_ylim(0, .56)
            ax.set_yticks([0, .2, .4])
        save(fig, output_dir, "giving_taking_population_effects")


def draw_controls(rows: list[dict], output_dir: Path) -> None:
    with plt.rc_context(paper_style(4.65)):
        fig, axes = plt.subplots(2, 2)
        tasks = ("Giving", "Taking", "Category choice", "Lottery tradeoffs")
        for panel, (ax, task) in enumerate(zip(axes.flat, tasks)):
            grouped_gain(ax, rows, task)
            decorate(ax, f"{'abcd'[panel]}  {task}", "Gain over adjustment\n(nats; higher is better)")
            ax.set_ylim(-.10, .25)
            ax.set_yticks([-.1, 0, .1, .2])
        population_legend(fig)
        save(fig, output_dir, "response_adjustment_comparison")


CAPTIONS = {
    "frozen_sample_fit": {
        "hypothesis": "H1",
        "caption": "Frozen-model prediction error by evaluated population; lower is better. Probabilities are normalized over valid responses (21 allocations or two choices). Giving uses held-out participants, Taking a separate sample, and category/lottery panels all retained participants. Allocation error bars are existing 95% participant-bootstrap intervals; the other panels are descriptive without intervals. Category choice covers 15 Cao items and lotteries four Anllo tradeoff pairs. English prompts omit country labels. Sample fit alone does not isolate cultural bias; absolute scores are not directly comparable across tasks.",
        "sources": ["results/giving_taking/figure_inputs/frozen_sample_error.csv", "results/giving_taking/frozen/metrics_summary.csv", "results/cao_anllo/frozen_baseline_country_metrics.csv"],
        "interpretation": "Centaur has lower error on US responses in Giving, Taking and category choice, and on Chinese responses in lottery tradeoffs. Pooled is inapplicable because these are frozen models and the colors identify evaluated human samples."
    },
    "giving_taking_population_effects": {
        "hypothesis": "H2-H3",
        "caption": "Effects of Giving-only fine-tuning on Giving and Taking. Top: expected allocation shifts from frozen Centaur across 25 held-out targets; dots show three seeds. Bottom: conditional log-score gains on held-out Giving participants or the separate Taking sample; higher is better. Colors identify fine-tuning samples, with 910 participants per fit (pooled: 455 per country). Error bars are existing 95% paired participant-and-seed bootstrap intervals (4,000 draws; fixed targets/folds). Negative shifts mean less given but less taken, with opposite implications for generosity.",
        "sources": ["results/giving_taking/figure_inputs/prediction_shifts_by_seed.csv", "results/giving_taking/response_adjustment/human_model_means_by_seed.csv", "results/giving_taking/bootstrap_intervals.csv", "results/giving_taking/seed_metrics.csv"],
        "interpretation": "The Chinese adapter shifts mean allocations down most in both tasks; pooled is intermediate. Population matching gives the highest Giving score, but the US adapter has the highest Taking score in both evaluated populations. Shifts and predictive benefit are separate quantities."
    },
    "response_adjustment_comparison": {
        "hypothesis": "H4",
        "caption": "Fine-tuning gain over a matched response adjustment; positive values favor fine-tuning. Both methods use identical observations: Giving only for panels a-b, category choices for c, and lottery choices for d. The adjustment fits target-independent response-logit offsets (20 for allocations, one for binary choices). All fine-tuning populations are shown, including 50:50 pooled. Allocation error bars are existing 95% paired participant-and-seed bootstrap intervals; category/lottery dots show three seeds without intervals. Panels c-d use held-out participants and items; d covers four tradeoff pairs. This control does not separately correct answer-position preferences.",
        "sources": ["results/giving_taking/response_adjustment/prior_paired_intervals.csv", "results/giving_taking/response_adjustment/prior_metrics_by_seed.csv", "results/cao_anllo/seed_metrics.csv"],
        "interpretation": "The benefit beyond response adjustment varies by task and population. The binary control does not separately adjust response position; fine-tuning's residual benefit is not yet isolated from answer-order debiasing."
    },
    "population_predictions": {
        "hypothesis": "H2, additional tasks",
        "caption": "Prediction shifts on unseen items: category choice across 15 Cao word triads and 10-point lottery choice across four Anllo tradeoff pairs. Bars show frozen Centaur or equal-budget US, pooled and China fine-tuning; dots show three seeds. Probabilities are normalized over the two choices and averaged over both answer orders and held-out items. Anllo fine-tuning uses six of eight pairs per fold, including dominance pairs; this panel evaluates only tradeoffs. These are prediction shifts, not accuracy improvements.",
        "sources": ["results/cao_anllo/seed_metrics.csv"],
        "interpretation": "US fine-tuning yields more category choices than China fine-tuning; China fine-tuning yields more 10-point tradeoff choices. Pooled is intermediate in both tasks. These are held-out-item tests, not transfer to a different task."
    },
}


def write_metadata(output_dir: Path) -> None:
    metadata = {"style": "tueplots.neurips2024, 6.3-inch ACL full-width canvas; vector PDF and 300-dpi PNG",
                "figures": {}, "data_export_sha256": {
                    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (ROOT / "results/paper_figure_data.csv", ROOT / "results/allocation_figure_data.csv")
                }}
    for name, description in CAPTIONS.items():
        metadata["figures"][name] = dict(description)
        metadata["figures"][name]["source_sha256"] = {
            source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
            for source in description["sources"]
        }
    (output_dir / "figure_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def write_allocation_data(frozen, shifts, gains, controls) -> None:
    records = []
    for row in frozen:
        records.append(dict(figure="frozen_sample_fit", panel=row["task"],
                            condition=row["model"], country=row["Country"],
                            mean=row["prediction_error"], ci_low=row.get("error_ci_low", ""),
                            ci_high=row.get("error_ci_high", "")))
    for task in ("Giving", "Taking"):
        for condition in CONDITIONS[1:]:
            values = [float(one(shifts, task=task, condition=condition, seed=seed)["shift_tokens"])
                      for seed in ("0", "1", "2")]
            records.append(dict(figure="giving_taking_population_effects", panel=task + " shift",
                                condition=condition, mean=mean(values),
                                seed_0=values[0], seed_1=values[1], seed_2=values[2]))
    for figure, rows in (("giving_taking_population_effects", gains),
                         ("response_adjustment_comparison", controls)):
        for row in rows:
            record = dict(figure=figure, panel=row["task"], condition=row["condition"],
                          country=row["country"], mean=row["value"],
                          ci_low=row.get("ci_low", ""), ci_high=row.get("ci_high", ""))
            if "seed_values" in row:
                record.update({f"seed_{seed}": value for seed, value in enumerate(row["seed_values"])})
            records.append(record)
    fields = ["figure", "panel", "condition", "country", "mean", "ci_low", "ci_high", "seed_0", "seed_1", "seed_2"]
    path = ROOT / "results/allocation_figure_data.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/cao_anllo/seed_metrics.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/figures")
    args = parser.parse_args()
    data = figure_data(args.input)
    data_path = ROOT / "results/paper_figure_data.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)
    draw(data, args.output_dir)
    frozen, shifts, gains, controls = allocation_data()
    extension_frozen, extension_controls = extension_summaries()
    frozen.extend(extension_frozen)
    controls.extend(extension_controls)
    write_allocation_data(frozen, shifts, gains, controls)
    draw_frozen(frozen, args.output_dir)
    draw_allocation_effects(shifts, gains, args.output_dir)
    draw_controls(controls, args.output_dir)
    write_metadata(args.output_dir)
    for name in CAPTIONS:
        print(f"Wrote {args.output_dir / (name + '.pdf')}")
        print(f"Wrote {args.output_dir / (name + '.png')}")
    print(f"Wrote {data_path}")


if __name__ == "__main__":
    main()
