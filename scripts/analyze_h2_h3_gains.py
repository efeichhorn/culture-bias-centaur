#!/usr/bin/env python3
"""Export paired binary-task gains for the combined H2/H3 figure.

Reconstructs participant scores from verified saved predictions and the original
held-out splits. Reuses the existing whole-participant bootstrap without model
training or new inference. Only aggregate rows and provenance are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_culture_extensions as original
from analyze_culture_followup import SharedBootstrap, difference

ROOT = Path(__file__).resolve().parents[1]
TASKS = (("cao", "all"), ("anllo", "tradeoff"))
CONDITIONS = ("us", "pooled", "china")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, default=ROOT.parent / "publication-bias-transfer-demo")
    parser.add_argument("--output", type=Path, default=ROOT / "results/cao_anllo/followup_20261029")
    parser.add_argument("--resamples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()

    repo = args.source_repo.resolve()
    run_index = repo / "data_cache/culture_extensions_20261021/runs/run_index.json"
    outputs = repo / "outputs/culture_extensions_20261021"
    statuses, predictions, _, _, datasets, splits, expected = original.collect(
        run_index, repo, outputs, outputs / "baseline_precision_aligned")
    original.require(len(statuses) == 81 and all(row["status"] == "verified_production" for row in statuses),
                     "The combined figure requires all 81 verified original runs")
    people, seeds = original.participant_and_seed_metrics(predictions, datasets, splits, expected)
    frame = pd.DataFrame(people)
    frame = frame[frame.country.isin(["us", "china"])]

    # Match inferential_tables' held-out group order and seed exactly. Its later
    # all-retained bootstraps do not affect any of these paired held-out draws.
    rng = np.random.default_rng(args.seed)
    bootstraps = {key: SharedBootstrap(part, args.resamples, rng)
                  for key, part in frame.groupby(["dataset", "country"], sort=True)}
    rows = []
    for dataset, stratum in TASKS:
        for country in ("us", "china"):
            boot = bootstraps[(dataset, country)]
            subset = frame[(frame.dataset == dataset) & (frame.stratum == stratum)
                           & (frame.country == country)]
            for condition in CONDITIONS:
                paired = subset[subset.condition == condition]
                frozen = boot.summarize(paired[paired.source == "frozen_centaur"], "conditional_nll")
                fitted = boot.summarize(paired[paired.source == "finetuned_centaur"], "conditional_nll")
                summary, _, _ = difference(frozen, fitted)
                rows.append(dict(dataset=dataset, stratum=stratum, country=country,
                                 condition=condition, metric="log_score_gain", **summary))
    gains = pd.DataFrame(rows)
    original.require(len(gains) == 12, "Expected two tasks by two populations by three adapters")

    # Retain the follow-up's exact observed-seed estimand and verify it against
    # the prior exports before adding any new aggregate output.
    reference = ROOT / "results/cao_anllo/followup_20261029/contrasts.csv"
    previous = pd.read_csv(reference)
    previous = previous[previous.contrast == "own_vs_frozen"]
    keys = ["dataset", "stratum", "country", "condition"]
    checked = gains.merge(previous, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
    original.require(len(checked) == 4, "Expected four existing matched-population contrasts")
    max_difference = 0.0
    for column in ("estimate", "ci_low", "ci_high", "seed_sd", "seed_min", "seed_max"):
        delta = float((checked[column + "_new"] - checked[column + "_old"]).abs().max())
        max_difference = max(max_difference, delta)
        original.require(delta < 1e-12, f"Existing own-vs-frozen {column} failed reproduction: {delta}")

    old_seeds_path = ROOT / "results/cao_anllo/followup_20261029/seed_metrics.csv"
    old_seeds = pd.read_csv(old_seeds_path)
    old_seeds = old_seeds[old_seeds.source.isin(["frozen_centaur", "finetuned_centaur"])]
    new_seeds = pd.DataFrame(seeds)
    new_seeds = new_seeds[new_seeds.source.isin(["frozen_centaur", "finetuned_centaur"])]
    seed_keys = ["dataset", "seed", "condition", "source", "stratum", "country"]
    checked_seeds = new_seeds.merge(old_seeds, on=seed_keys, suffixes=("_new", "_old"),
                                    validate="one_to_one")
    original.require(len(checked_seeds) == len(old_seeds) == len(new_seeds), "Saved score coverage changed")
    score_difference = float((checked_seeds.conditional_nll_new - checked_seeds.conditional_nll_old).abs().max())
    original.require(score_difference < 1e-12, "Existing frozen/fitted scores failed reproduction")

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "h2_h3_gains.csv"
    gains.to_csv(csv_path, index=False)
    provenance = dict(
        status="complete_validated_paired_binary_gains", source_repo="${WORKSPACE}/publication-bias-transfer-demo",
        resamples=args.resamples, bootstrap_seed=args.seed, verified_runs=len(statuses),
        metric="Frozen NLL minus fine-tuned NLL in nats; positive favors fine-tuning",
        evaluation="Original held-out participants and stitched held-out items; categorization all 15 items, lotteries four tradeoff pairs",
        uncertainty="Exploratory pointwise 95% whole-participant bootstrap, stratified by country. A person's multiplicity is shared across overlapping seed test sets; normalize within each original seed test set and average observed seeds equally. Fixed fitted models, seeds, items, folds, and splits; no refitting or seed resampling. Intervals exclude training-sample uncertainty and generalization to new items.",
        privacy="Participant scores remain in memory; exports contain only aggregates and input hashes",
        matched_contrasts_checked=len(checked), matched_contrast_max_absolute_difference=max_difference,
        saved_score_rows_checked=len(checked_seeds), saved_score_max_absolute_difference=score_difference,
        bootstrap_diagnostics=[dict(dataset=key[0], country=key[1], n_unique_participants=len(boot.ids),
                                    rejected_empty_subset_draws=boot.rejected)
                               for key, boot in bootstraps.items()],
        sources={
            "run_index": sha256(run_index),
            "scripts/analyze_h2_h3_gains.py": sha256(Path(__file__)),
            "scripts/analyze_culture_extensions.py": sha256(Path(original.__file__)),
            "scripts/analyze_culture_followup.py": sha256(ROOT / "scripts/analyze_culture_followup.py"),
            str(reference.relative_to(ROOT)): sha256(reference),
            str(old_seeds_path.relative_to(ROOT)): sha256(old_seeds_path),
        },
        verified_inputs=[{key: value for key, value in row.items()
                          if key in ("dataset", "seed", "fold", "condition") or key.endswith("sha256")}
                         for row in statuses],
        outputs={csv_path.name: sha256(csv_path)},
    )
    (args.output / "h2_h3_gains.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({key: provenance[key] for key in ("status", "verified_runs", "matched_contrasts_checked",
                                                     "matched_contrast_max_absolute_difference",
                                                     "saved_score_max_absolute_difference")}, indent=2))
    print(gains[["dataset", "stratum", "country", "condition", "estimate", "ci_low", "ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
