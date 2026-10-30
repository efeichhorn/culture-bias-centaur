"""Shared data checks and drawing helpers for the paper figures."""
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation

from paper_plot_style import COLORS as PALETTE, paper_style

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("frozen", "us", "pooled", "china")
COLORS = tuple(PALETTE[condition] for condition in CONDITIONS)
COUNTRIES = ("United States", "China")
GIVING_TAKING = ROOT / "results/giving_taking"



def decorate(ax, title: str, ylabel: str) -> None:
    letter, title_text = title.split("  ", 1)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_title(title_text, loc="left", x=.065, fontweight="normal", pad=6)
    ax.text(0, 1, letter, transform=ax.transAxes + ScaledTranslation(0, 6 / 72, ax.figure.dpi_scale_trans),
            fontweight="bold", fontsize=9, ha="left", va="baseline")


def save(fig, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / name
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


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


def allocation_data() -> tuple[list[dict], list[dict]]:
    """Load frozen scores and response-adjustment comparisons for allocation."""
    frozen = read_csv("figure_inputs/frozen_sample_error.csv")
    frozen_reference = read_csv("frozen/metrics_summary.csv")
    for row in frozen:
        ref = one(frozen_reference, task=row["task"], Country=row["Country"],
                  condition=row["model"] + "_original")
        require_equal(float(row["prediction_error"]), -float(ref["conditional_ll"]),
                      "frozen prediction error")

    intervals = read_csv("response_adjustment/prior_paired_intervals.csv")
    scores = read_csv("response_adjustment/prior_metrics_by_seed.csv")
    controls = []
    for task in ("Giving", "Taking"):
        for country in COUNTRIES:
            for condition in CONDITIONS[1:]:
                row = one(intervals, task=task, Country=country, metric="conditional_ll",
                          contrast=condition + "_prior_minus_lora", method="shared_participant_and_seed")
                values = [float(one(scores, seed=seed, task=task, country=country,
                                    condition=condition, metric="conditional_ll")["prior_minus_lora"])
                          for seed in ("0", "1", "2")]
                require_equal(float(row["mean"]), mean(values), "adjustment comparison")
                # The source stores log-score differences; reverse the sign for NLL.
                controls.append(dict(task=task, country=country, condition=condition,
                                     value=-float(row["mean"]), ci_low=-float(row["ci_high"]),
                                     ci_high=-float(row["ci_low"])))
    return frozen, controls


def binary_frozen_data() -> list[dict]:
    """Load all-participant frozen scores for categorization and lottery tradeoffs."""
    with (ROOT / "results/cao_anllo/frozen_baseline_country_metrics.csv").open(newline="") as handle:
        baseline = list(csv.DictReader(handle))
    frozen = []
    for dataset, stratum, task in (("cao", "all", "Category choice"),
                                   ("anllo", "tradeoff", "Lottery tradeoffs")):
        for country, raw_country in zip(COUNTRIES, ("us", "china")):
            for model, source in (("centaur", "frozen_centaur"), ("llama", "base_llama")):
                row = one(baseline, dataset=dataset, stratum=stratum, source=source,
                          country=raw_country, sample_scope="all_retained_participants_frozen_audit")
                frozen.append(dict(task=task, Country=country, model=model,
                                   metric="conditional_ll", prediction_error=float(row["conditional_nll"])))
    return frozen


def draw_frozen(rows: list[dict], output_dir: Path) -> None:
    style = paper_style(3.7, nrows=2)
    style["figure.constrained_layout.use"] = False
    with plt.rc_context(style):
        fig = plt.figure()
        # Fixed panel bounds keep the long task labels clear of adjacent panels.
        bounds = [(29.881875, 138.595375), (259.525875, 138.595375),
                  (29.881875, 15.319375), (259.525875, 15.319375)]
        axes = [fig.add_axes([x / 453.6, y / 266.4, 190.114125 / 453.6, 91.674625 / 266.4])
                for x, y in bounds]
        tasks = ("Giving", "Taking", "Category choice", "Lottery tradeoffs")
        titles = ("Social allocation: giving variant", "Social allocation: taking variant",
                  "Categorization", "Risky choice: tradeoff pairs")
        for panel, (ax, task, title) in enumerate(zip(axes, tasks, titles)):
            for x, model in enumerate(("centaur", "llama")):
                for offset, country, color in zip((-.16, .16), COUNTRIES, (COLORS[1], COLORS[3])):
                    row = one(rows, task=task, model=model, Country=country, metric="conditional_ll")
                    value = float(row["prediction_error"])
                    ax.bar(x + offset, value, width=.27, color=color, zorder=2)
                    if "error_ci_low" in row:
                        ax.errorbar(x + offset, value,
                                    yerr=[[value - float(row["error_ci_low"])],
                                          [float(row["error_ci_high"]) - value]],
                                    fmt="none", color="black", linewidth=.55, capsize=1.5, capthick=.55, zorder=3)
            decorate(ax, f"{'abcd'[panel]}  {title}", "Conditional NLL (nats/response)")
            ax.set_ylim(0, 3.2 if panel < 2 else .8)
            ax.set_yticks([0, 1, 2, 3] if panel < 2 else [0, .2, .4, .6, .8])
            ax.set_xticks([0, 1], ["Frozen Centaur", "Base Llama"])
            ax.set_xlim(-.65, 1.65)
        fig.legend(handles=[Patch(color=COLORS[1], label="US responses"),
                            Patch(color=COLORS[3], label="Chinese responses")],
                   frameon=False, ncols=2, loc="outside upper center",
                   handlelength=1, columnspacing=1.5)
        save(fig, output_dir, "frozen_sample_fit")
