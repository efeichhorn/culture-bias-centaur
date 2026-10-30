#!/usr/bin/env python3
"""Fixed social-target folds and allocation training helpers."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dictator_culture_finetune import (
    train_distributional_lora_adapter,
    training_config,
)
from scripts.dictator_culture_mixed_split import (
    behavior_source_provenance,
    heldout_rows,
    require_corpus_behavior_source,
)
from scripts.evaluate_dictator_giving_base import (
    build_prompt,
    evaluate_against_humans,
    load_giving_data,
    score_model,
)
from scripts.evaluate_dictator_taking_transfer import (
    TAKING_DATASETS,
    build_taking_prompt,
    load_taking_data,
)
from src.llm_model import device_for_eval, load_base_model, load_tokenizer
from src.utils import ensure_dir


CONDITIONS = ("us", "china", "pooled")
ITEM_FOLDS = {
    0: (1, 5, 12, 16, 23),
    1: (2, 6, 13, 17, 24),
    2: (3, 7, 14, 18, 25),
    3: (4, 8, 10, 15, 19),
    4: (9, 11, 20, 21, 22),
}
FOLD_RATIONALE = {
    0: "boy; German; farmer; property developer; homeless",
    1: "girl; Japanese; civil servant; pharmaceutical executive; disabled",
    2: "male; North Korean; janitor; investment banker; no information",
    3: "female; Pakistani; Jewish; nanny; lawyer",
    4: "Australian; Greek; teacher; athlete; scientist",
}


def validate_item_folds() -> None:
    keys = sorted(ITEM_FOLDS)
    if keys != list(range(5)):
        raise ValueError("Item folds must be numbered 0 through 4.")
    flattened = [group_no for fold in keys for group_no in ITEM_FOLDS[fold]]
    if sorted(flattened) != list(range(1, 26)):
        raise ValueError("Every target group must appear in exactly one item fold.")
    if any(len(ITEM_FOLDS[fold]) != 5 for fold in keys):
        raise ValueError("Every item fold must contain exactly five target groups.")


def item_fold_manifest(items: pd.DataFrame) -> pd.DataFrame:
    validate_item_folds()
    mapping = {
        group_no: fold
        for fold, group_nos in ITEM_FOLDS.items()
        for group_no in group_nos
    }
    manifest = items[["GroupNo", "item", "item_chinese"]].copy()
    manifest["fold"] = manifest["GroupNo"].map(mapping)
    manifest["fold_rationale"] = manifest["fold"].map(FOLD_RATIONALE)
    if manifest["fold"].isna().any() or len(manifest) != 25:
        raise ValueError("Item metadata do not match the fixed 25-item fold manifest.")
    return manifest.sort_values(["fold", "GroupNo"]).reset_index(drop=True)


def prepare_manifest(args: argparse.Namespace) -> None:
    _, items = load_giving_data(args.behavior_path, args.group_path)
    output_dir = ensure_dir(args.output_dir)
    manifest = item_fold_manifest(items)
    manifest.to_csv(output_dir / "item_fold_manifest.csv", index=False)
    protocol = {
        **behavior_source_provenance(args.behavior_path),
        "fold_definition": "fixed before inspecting Taking outcomes; balanced by item semantics",
        "folds": {str(key): list(value) for key, value in ITEM_FOLDS.items()},
        "conditions": list(CONDITIONS),
        "training_task": "Giving",
        "training_items_per_fold": 20,
        "heldout_items_per_fold": 5,
        "evaluation_tasks": ["Giving", "Taking"],
        "taking_data_used_for_training_or_selection": False,
        "participant_manifests": "reuse the mixed-wave participant splits within each seed",
        "optimization_exposure": "20 steps x 5 items/step = 100 item exposures = 5 epochs",
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8",
    )


def train_fold_condition(args: argparse.Namespace) -> None:
    if args.condition not in CONDITIONS:
        raise ValueError("--condition is required for training.")
    require_corpus_behavior_source(
        args.corpus_dir, args.behavior_path,
        prepare_script="scripts/dictator_culture_mixed_split.py",
        group_path=args.group_path, seed=args.seed,
    )
    heldout_group_nos = set(ITEM_FOLDS[args.fold])
    counts_path = Path(args.corpus_dir) / f"counts_{args.condition}.csv"
    rows = pd.read_csv(counts_path)
    train_rows = rows.loc[~rows["GroupNo"].isin(heldout_group_nos)].copy()
    if train_rows["GroupNo"].nunique() != 20:
        raise ValueError("Every fold must train on exactly 20 target groups.")
    if set(train_rows["GroupNo"]).intersection(heldout_group_nos):
        raise ValueError("Held-out target groups leaked into the training rows.")

    output_dir = ensure_dir(args.output_dir)
    adapter_dir = ensure_dir(output_dir / "adapters" / args.condition)
    train_rows.to_csv(output_dir / f"training_counts_{args.condition}.csv", index=False)
    cfg = training_config(args)
    metrics_path = output_dir / f"train_metrics_{args.condition}.json"
    train_distributional_lora_adapter(
        train_rows,
        cfg=cfg,
        output_dir=adapter_dir,
        metrics_path=metrics_path,
        seed=args.seed,
    )
    (output_dir / f"train_config_{args.condition}.json").write_text(
        json.dumps(asdict(cfg), indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        **behavior_source_provenance(args.behavior_path),
        "seed": args.seed,
        "fold": args.fold,
        "condition": args.condition,
        "training_task": "Giving",
        "training_group_nos": sorted(train_rows["GroupNo"].unique().astype(int).tolist()),
        "heldout_group_nos": sorted(heldout_group_nos),
        "heldout_items_used_for_training": False,
    }
    (output_dir / f"train_metadata_{args.condition}.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_evaluation_tables(
    evaluation_dir: Path,
    tables: dict[str, pd.DataFrame],
    condition: str,
    seed: int,
    fold: int,
    task: str,
) -> None:
    evaluation_dir = ensure_dir(evaluation_dir)
    for filename, frame in tables.items():
        frame.insert(1, "condition", condition)
        frame.insert(2, "training_seed", seed)
        frame.insert(3, "item_fold", fold)
        frame.insert(4, "evaluation_task", task)
        frame.to_csv(evaluation_dir / filename, index=False)


def evaluate_fold_condition(args: argparse.Namespace) -> None:
    if args.condition not in CONDITIONS:
        raise ValueError("--condition is required for evaluation.")
    output_dir = Path(args.output_dir)
    adapter_path = output_dir / "adapters" / args.condition
    if not (adapter_path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Missing item-holdout adapter: {adapter_path}")

    from peft import PeftModel

    giving, items = load_giving_data(args.behavior_path, args.group_path)
    taking, _ = load_taking_data(args.behavior_path, args.group_path)
    split_manifest = pd.read_csv(Path(args.corpus_dir) / "participant_split_manifest.csv")
    heldout_group_nos = set(ITEM_FOLDS[args.fold])
    heldout_items = items.loc[items["GroupNo"].isin(heldout_group_nos)].copy()
    if heldout_items["GroupNo"].nunique() != 5:
        raise ValueError("Evaluation requires exactly five held-out target groups.")
    giving_test = heldout_rows(giving, split_manifest)
    giving_test = giving_test.loc[giving_test["GroupNo"].isin(heldout_group_nos)]
    taking_test = taking.loc[taking["GroupNo"].isin(heldout_group_nos)]

    tokenizer = load_tokenizer(args.model_name)
    base_model = load_base_model(args.model_name, args.bf16, args.attn_implementation)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    device = device_for_eval()
    model.to(device)
    model.eval()

    task_specs = [
        ("Giving", giving_test, build_prompt, (1, 2)),
        ("Taking", taking_test, build_taking_prompt, TAKING_DATASETS),
    ]
    for task, human_rows, prompt_builder, wave_datasets in task_specs:
        (
            candidates,
            predictions,
            invalid_tokens,
            aggregate_invalid_tokens,
        ) = score_model(
            model,
            tokenizer,
            device,
            heldout_items,
            args.model_name,
            prompt_builder=prompt_builder,
        )
        metrics, human_items = evaluate_against_humans(
            human_rows,
            candidates,
            wave_datasets=wave_datasets,
        )
        evaluation_dir = output_dir / "evaluations" / task.lower() / args.condition
        _write_evaluation_tables(
            evaluation_dir,
            {
                "candidate_probabilities.csv": candidates,
                "item_predictions.csv": predictions,
                "invalid_token_probabilities.csv": invalid_tokens,
                "aggregate_invalid_tokens.csv": aggregate_invalid_tokens,
                "model_metrics.csv": metrics,
                "human_item_comparison.csv": human_items,
            },
            condition=args.condition,
            seed=args.seed,
            fold=args.fold,
            task=task,
        )
        metadata = {
            "model_name": args.model_name,
            "adapter_path": str(adapter_path),
            "condition": args.condition,
            "training_seed": args.seed,
            "item_fold": args.fold,
            "evaluation_task": task,
            "heldout_group_nos": sorted(heldout_group_nos),
            "evaluation_items_seen_during_adapter_training": False,
            "taking_responses_used_for_training_or_selection": False,
            "giving_evaluation_uses_heldout_participants": True,
        }
        (evaluation_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
