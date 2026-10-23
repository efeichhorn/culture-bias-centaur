#!/usr/bin/env python3
"""Plot illustrative social targets from corrected Giving/Taking results.

Default: draw the checked-in aggregate CSV. To rebuild it from private inputs,
pass --prepare-from-workspace /path/to/the/original/data/workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from tueplots import bundles

ROOT = Path(__file__).resolve().parents[1]
GROUPS = [23, 2, 13, 8, 11]
COLORS = {"us": "#4878A8", "china": "#CF866F", "pooled": "#8E82A8", "frozen": "#8B9299"}
BEHAVIOR_SHA = "65f8913600fddeae3ca99196c2b0e2db76f2bf2021e1b32f4419f954645a5ecd"


def prepare(workspace: Path, output: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.audit_dictator_a1 import shared_bootstrap

    inputs = {}
    def read(path):
        inputs[str(path.relative_to(workspace))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text()) if path.suffix == ".json" else pd.read_csv(path)

    behavior_path = workspace / "data_cache/dictator_Behavior_corrected_20260930.csv"
    if hashlib.sha256(behavior_path.read_bytes()).hexdigest() != BEHAVIOR_SHA:
        raise ValueError("Expected the verified corrected allocation data")
    behavior = read(behavior_path)
    behavior["value"] = behavior.Token / 5
    if not behavior.value.between(0, 20).all() or not (behavior.Token % 5 == 0).all():
        raise ValueError("Invalid allocation units")
    folds = read(workspace / "data_cache/dictator_centaur70b_item_holdout_corrected_20260930/item_fold_manifest.csv")
    fold_map = folds.set_index("GroupNo").fold.to_dict()
    names = folds.set_index("GroupNo").item.to_dict()
    run_root = workspace / "model_results/dictator_centaur70b_item_holdout_corrected_20260930"
    keys = ["Dataset", "Nation", "SubNo"]
    rows, sample_counts = [], []
    for task in ["Giving", "Taking"]:
        people = []
        for seed in range(3):
            if task == "Giving":
                split = read(workspace / f"data_cache/dictator_centaur70b_corrected_20260930/seed{seed}/participant_split_manifest.csv")
                ids = split.loc[split.split == "test", keys]
                part = behavior.merge(ids, on=keys, validate="many_to_one")
                if len(part) != len(ids) * 25 or not part.Dataset.isin([1, 2]).all():
                    raise ValueError("Giving test-participant join failed")
            else:
                part = behavior[behavior.Dataset.isin([3, 4])]
            people.append(part[part.GroupNo.isin(GROUPS)].assign(seed=seed))
        people = pd.concat(people, ignore_index=True)
        for nation, country in [(0, "us"), (1, "china")]:
            part = people[people.Nation == nation].copy()
            part["outcome"] = part.GroupNo.astype(str)
            draws, _, outcomes = shared_bootstrap(part, 4000, np.random.default_rng(2026102300 + nation))
            if not np.isfinite(draws).all():
                raise ValueError("Nonfinite participant bootstrap values")
            points = part.groupby(["seed", "GroupNo"]).value.mean().unstack()
            low, high = np.quantile(draws.mean(axis=1), [.025, .975], axis=0)
            n_unique = len(part[keys].drop_duplicates())
            sample_counts.append({"task": task, "country": country, "unique_participants": n_unique,
                                  "participants_per_seed": part[keys + ["seed"]].drop_duplicates().groupby("seed").size().tolist()})
            for group in GROUPS:
                index = outcomes.index(str(group))
                rows.append(dict(task=task, group=group, item=names[group], source="human", population=country,
                                 mean=points[group].mean(), lower=low[index], upper=high[index],
                                 interval="95% participant bootstrap CI", n_unique_participants=n_unique,
                                 **{f"seed{s}": points.loc[s, group] for s in range(3)}))
        for condition in ["us", "pooled", "china", "frozen"]:
            for group in GROUPS:
                estimates = []
                for seed in range(3):
                    directory = run_root / f"seed{seed}" / ("base" if condition == "frozen" else f"fold{fold_map[group]}/{condition}")
                    marker = read(directory / "verified_complete.json")
                    if marker["status"] != "verified_complete" or marker["source_behavior_sha256"] != BEHAVIOR_SHA:
                        raise ValueError("Unverified corrected evaluation")
                    if condition != "frozen" and (marker["optimizer_steps"] != 25 or group not in marker["groups"]):
                        raise ValueError("Item was not withheld from a completed fit")
                    table = read(directory / f"evaluations/{task.lower()}/item_predictions.csv")
                    selected = table[table.GroupNo == group]
                    if len(selected) != 1:
                        raise ValueError("Expected one held-out model prediction")
                    estimates.append(float(selected.iloc[0].expected_tokens))
                    check = read(directory / f"evaluations/{task.lower()}/human_item_comparison.csv")
                    check = check[(check.GroupNo == group) & (check["sample"] == "Combined")]
                    for country, nation in [("United States", 0), ("China", 1)]:
                        actual = people[(people.seed == seed) & (people.Nation == nation) & (people.GroupNo == group)].value.mean()
                        saved = float(check.loc[check.country == country, "human_expected_tokens"].iloc[0])
                        if not np.isclose(actual, saved, atol=1e-10, rtol=0):
                            raise ValueError("Human means differ from verified evaluation samples")
                spread = float(np.std(estimates, ddof=1))
                if condition == "frozen" and spread > 1e-10:
                    raise ValueError("Frozen predictions vary across evaluation splits")
                mean = float(np.mean(estimates))
                rows.append(dict(task=task, group=group, item=names[group], source="model", population=condition,
                                 mean=mean, lower=mean-spread, upper=mean+spread,
                                 interval="none" if condition == "frozen" else "one SD across three fitted seeds",
                                 n_unique_participants="", **{f"seed{s}": estimates[s] for s in range(3)}))
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    provenance = {
        "selection": "Five illustrative targets copied from CHM slide 19: Homeless, Girl, Civil servant, Pakistani, Greek. They are not a representative sample or a test of general transfer.",
        "training": "All adapters were fine-tuned on Giving only. Every displayed target was withheld from its adapter's Giving fine-tuning; Taking responses were never used for fitting.",
        "model_mean": "Expected tokens conditional on the 21 valid responses, averaged across three fixed fitted seeds. Error bars show one seed SD, not confidence intervals.",
        "human_mean": "Giving: equally average the three manifest-held-out test-sample means. Taking: all separate Taking samples, counted once.",
        "human_intervals": "4000 percentile participant-bootstrap resamples; fixed within-seed wave weights, shared participant multiplicities across overlapping Giving splits; no item or fitted-model resampling.",
        "sample_counts": sample_counts, "source_behavior_sha256": BEHAVIOR_SHA, "input_sha256": inputs,
        "aggregate_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def draw(data: pd.DataFrame, output_dir: Path) -> None:
    if len(data) != 60 or data.duplicated(["task", "group", "source", "population"]).any():
        raise ValueError("Expected six series for five targets in each task")
    if not np.isfinite(data[["mean", "lower", "upper"]]).all().all():
        raise ValueError("Nonfinite figure values")
    if not ((data.lower <= data["mean"]) & (data["mean"] <= data.upper)).all():
        raise ValueError("Interval does not contain its displayed mean")
    plt.rcParams.update(bundles.neurips2024(usetex=False, family="sans-serif"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 8,
                        "axes.spines.top": False, "axes.spines.right": False,
                        "axes.spines.left": False, "axes.edgecolor": "#BBC1C6", "axes.linewidth": .6,
                        "text.color": "#26333D", "axes.labelcolor": "#26333D", "pdf.fonttype": 42,
                        "figure.constrained_layout.use": False, "figure.autolayout": False})
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.6), sharex=True, sharey=True)
    positions = {group: 4-i for i, group in enumerate(GROUPS)}
    for ax, task, title, label in zip(axes, ["Giving", "Taking"],
                                    ["a  Held-out Giving", "b  Taking transfer"],
                                    ["Mean tokens given", "Mean tokens taken"]):
        for row in data[data.task == task].itertuples():
            human = row.source == "human"
            offset = (.235 if row.population == "us" else .165) if human else {"us": -.015, "pooled": -.065, "china": -.115, "frozen": -.32}[row.population]
            marker = "o" if human else "x" if row.population == "frozen" else "D" if row.population == "pooled" else "s"
            color = COLORS[row.population]
            y = positions[row.group] + offset
            if row.interval != "none":
                ax.errorbar(row.mean, y, xerr=[[row.mean-row.lower], [row.upper-row.mean]],
                            fmt="none", ecolor=color, lw=.85, capsize=1.8, capthick=.7, zorder=2)
            marker_style = {"linewidths": .9} if marker == "x" else {"linewidths": .45, "edgecolors": "white"}
            ax.scatter(row.mean, y, s=23 if marker == "D" else 21, marker=marker, color=color,
                       **marker_style, zorder=3)
        ax.set_title(title, loc="left", fontweight="bold", pad=9)
        ax.set_xlabel(label)
        ax.set_yticks([positions[g] for g in GROUPS], [str(data[data.group == g].item.iloc[0]).capitalize() for g in GROUPS])
        ax.tick_params(axis="y", length=0, pad=6)
        ax.tick_params(axis="x", width=.6, length=3)
        ax.grid(axis="x", color="#E5E8EB", linewidth=.5, zorder=0)
        ax.set_ylim(-.6, 4.65)
        ax.xaxis.set_major_locator(MultipleLocator(2))
    low = np.floor(data.lower.min()/2)*2
    high = np.ceil(data.upper.max()/2)*2
    axes[0].set_xlim(low-.35, high+.35)
    legend = []
    for population, marker, label in [("us", "o", "Human US"), ("us", "s", "US fine-tuning"),
                                       ("china", "o", "Human China"), ("china", "s", "China fine-tuning"),
                                       ("frozen", "x", "Frozen Centaur"), ("pooled", "D", "Pooled fine-tuning")]:
        legend.append(Line2D([], [], marker=marker, linestyle="none", color=COLORS[population],
                             markeredgecolor=COLORS[population] if marker == "x" else "white",
                             markeredgewidth=.6, markersize=4.5, label=label))
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=7.5,
               handletextpad=.35, columnspacing=1.6, bbox_to_anchor=(.54, .005))
    fig.subplots_adjust(left=.145, right=.995, top=.92, bottom=.24, wspace=.14)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ["pdf", "png"]:
        fig.savefig(output_dir / f"social_group_predictions.{extension}", dpi=300, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/giving_taking/figure_inputs/social_group_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/figures")
    parser.add_argument("--prepare-from-workspace", type=Path)
    args = parser.parse_args()
    if args.prepare_from_workspace:
        prepare(args.prepare_from_workspace.resolve(), args.input)
    draw(pd.read_csv(args.input), args.output_dir)
    print(args.output_dir / "social_group_predictions.pdf")


if __name__ == "__main__":
    main()
