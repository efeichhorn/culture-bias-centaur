#!/usr/bin/env python3
"""Export held-out allocation predictions for the paired H2/H3 figure.

Reads existing verified Giving-trained fits only; no training or inference.
The portable CSV contains model predictions, without participant observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR_SHA = "65f8913600fddeae3ca99196c2b0e2db76f2bf2021e1b32f4419f954645a5ecd"
CONDITIONS = ["us", "pooled", "china"]
TASKS = ["Giving", "Taking"]
GROUPS = list(range(1, 26))


def export(workspace: Path, output: Path) -> None:
    inputs: dict[str, str] = {}

    def read(path: Path):
        try:
            name = str(path.relative_to(workspace))
        except ValueError:
            name = str(path)
        inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text()) if path.suffix == ".json" else pd.read_csv(path)

    folds = read(workspace / "data_cache/dictator_centaur70b_item_holdout_corrected_20260930/item_fold_manifest.csv")
    if sorted(folds.GroupNo.tolist()) != GROUPS or sorted(folds.fold.unique()) != list(range(5)):
        raise ValueError("Expected 25 distinct targets over five held-out folds")
    names = folds.set_index("GroupNo").item.to_dict()
    runs = workspace / "model_results/dictator_centaur70b_item_holdout_corrected_20260930"
    records, frozen_records = [], []

    def verified(directory: Path, seed: int, condition: str, fold: int | None = None,
                 groups: list[int] | None = None):
        marker = read(directory / "verified_complete.json")
        if marker["status"] != "verified_complete" or marker["source_behavior_sha256"] != BEHAVIOR_SHA:
            raise ValueError(f"Expected a verified corrected evaluation: {directory}")
        if marker["seed"] != seed or marker["condition"] != condition:
            raise ValueError(f"Evaluation identity mismatch: {directory}")
        if fold is not None and (marker["fold"] != fold or marker["optimizer_steps"] != 25
                                 or sorted(marker["groups"]) != groups):
            raise ValueError(f"Target fold was not withheld from a completed fit: {directory}")

    for seed in range(3):
        for condition in CONDITIONS:
            for fold, fold_table in folds.groupby("fold"):
                groups = sorted(fold_table.GroupNo.tolist())
                directory = runs / f"seed{seed}/fold{fold}/{condition}"
                verified(directory, seed, condition, int(fold), groups)
                for task in TASKS:
                    prediction = read(directory / f"evaluations/{task.lower()}/item_predictions.csv")
                    if sorted(prediction.GroupNo.tolist()) != groups:
                        raise ValueError(f"Prediction targets differ from the held-out fold: {directory}/{task}")
                    if not (prediction.training_seed.eq(seed).all() and prediction.item_fold.eq(fold).all()
                            and prediction.condition.eq(condition).all() and prediction.evaluation_task.eq(task).all()):
                        raise ValueError(f"Prediction metadata differs from its verified fit: {directory}/{task}")
                    for row in prediction.itertuples():
                        if row.item != names[row.GroupNo]:
                            raise ValueError("Target name differs from the fold manifest")
                        records.append(dict(task=task, item_id=row.GroupNo, item=row.item, seed=seed,
                                            condition=condition, prediction=float(row.expected_tokens)))
        base = runs / f"seed{seed}/base"
        verified(base, seed, "base")
        for task in TASKS:
            prediction = read(base / f"evaluations/{task.lower()}/item_predictions.csv")
            if sorted(prediction.GroupNo.tolist()) != GROUPS:
                raise ValueError("Frozen evaluation must include all 25 targets")
            frozen_records.append(dict(task=task, seed=seed, frozen_expected_tokens=float(prediction.expected_tokens.mean())))

    data = pd.DataFrame(records).sort_values(["task", "item_id", "seed", "condition"]).reset_index(drop=True)
    if len(data) != 450 or data.duplicated(["task", "item_id", "seed", "condition"]).any():
        raise ValueError("Expected exactly 450 distinct target/seed/condition predictions")
    if not data.prediction.between(0, 20).all() or not np.isfinite(data.prediction).all():
        raise ValueError("Expected finite conditional expected tokens on the 0--20 scale")
    if not data.groupby(["task", "seed", "condition"]).size().eq(25).all():
        raise ValueError("Incomplete target coverage")
    if not data.groupby(["task", "item_id", "condition"]).seed.nunique().eq(3).all():
        raise ValueError("Every item and condition must include all three fitted seeds")

    established = read(ROOT / "results/giving_taking/figure_inputs/prediction_shifts_by_seed.csv")
    means = data.groupby(["task", "seed", "condition"]).prediction.mean().rename("expected_tokens").reset_index()
    means = means.merge(pd.DataFrame(frozen_records), on=["task", "seed"], validate="many_to_one")
    means["shift_tokens"] = means.expected_tokens - means.frozen_expected_tokens
    comparison = means.merge(established, on=["task", "seed", "condition"], validate="one_to_one", suffixes=("_recomputed", "_saved"))
    if len(comparison) != 18 or len(established) != 18:
        raise ValueError("Expected all 18 established task/seed/condition means")
    validation = {}
    for field in ["expected_tokens", "frozen_expected_tokens", "shift_tokens"]:
        discrepancy = np.abs(comparison[f"{field}_recomputed"] - comparison[f"{field}_saved"])
        maximum = float(discrepancy.max())
        if not np.isfinite(maximum) or maximum > 1e-10:
            raise ValueError(f"Recomputed {field} differ from saved figure results: {maximum}")
        validation[field] = dict(values=18, maximum_absolute_error=maximum, tolerance=1e-10)

    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    provenance = {
        "description": "All 25 held-out social-target model predictions for Giving and Taking, exported from existing verified corrected evaluations.",
        "training": "All adapters were trained on Giving only. Each target is read only from its held-out Giving fold; Taking responses were never used for fitting.",
        "measurement": "Expected tokens conditional on the 21 valid responses (0--20). Each row retains one fitted seed; paired-figure item lines average three seeds and overall means weight all 25 items equally.",
        "coverage": {"tasks": TASKS, "conditions": CONDITIONS, "seeds": [0, 1, 2], "targets_per_task": 25, "rows": len(data)},
        "source_behavior_sha256_verified_in_completion_markers": BEHAVIOR_SHA,
        "validation_against_established_seed_means": validation,
        "input_sha256": inputs,
        "export_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "aggregate_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(output)
    print(means.groupby(["task", "condition"])["expected_tokens"].mean().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT.parent)
    parser.add_argument("--output", type=Path, default=ROOT / "results/giving_taking/followup_20261029/h2_h3_item_predictions.csv")
    args = parser.parse_args()
    export(args.workspace.resolve(), args.output.resolve())
