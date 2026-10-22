#!/usr/bin/env python3
"""Held-out-item and Giving-to-Taking transfer experiment for Centaur-70B.

The fixed item folds are identical to the Minitaur experiment.  For each fold,
the culture adapter is trained on 20 Giving target groups and evaluated on the
five target groups that were excluded from adapter training.  The same adapter
is also evaluated zero-shot on the corresponding Taking items.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dictator_centaur70b_qlora import (
    COUNTRIES,
    EVALUATION_CONDITIONS,
    TRAIN_CONDITIONS,
    load_evaluation_model,
    load_tokenizer,
    resolve_saved_adapter_dir,
    runtime_versions,
    train as train_centaur70b,
)
from scripts.dictator_culture_mixed_split import (
    PARTICIPANT_KEYS,
    behavior_source_provenance,
    heldout_rows,
    require_corpus_behavior_source,
)
from scripts.dictator_item_holdout_transfer import (
    FOLD_RATIONALE,
    ITEM_FOLDS,
    item_fold_manifest,
)
from scripts.evaluate_dictator_giving_base import (
    RESPONSE_VALUES,
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
from src.utils import ensure_dir


GIVING_DATASETS = (1, 2)


def heldout_group_nos(fold: int) -> tuple[int, ...]:
    if fold not in ITEM_FOLDS:
        raise ValueError(f"Fold must be one of {sorted(ITEM_FOLDS)}; got {fold}.")
    return ITEM_FOLDS[fold]


def filter_training_rows(rows: pd.DataFrame, fold: int) -> pd.DataFrame:
    """Remove the five fixed fold items and validate the resulting corpus."""
    heldout = set(heldout_group_nos(fold))
    filtered = rows.loc[~rows["GroupNo"].isin(heldout)].copy()
    if filtered["GroupNo"].nunique() != 20:
        raise ValueError("A held-out fold must leave exactly 20 Giving items.")
    if set(filtered["GroupNo"]).intersection(heldout):
        raise ValueError("Held-out items leaked into the Centaur training corpus.")
    if "participant_key" in filtered:
        counts = filtered.groupby("participant_key").size()
        if not counts.eq(20).all():
            raise ValueError("Every selected participant must retain exactly 20 items.")
    return filtered.reset_index(drop=True)


def prepare_manifest(args: argparse.Namespace) -> None:
    _, items = load_giving_data(args.behavior_path, args.group_path)
    output_dir = ensure_dir(args.output_dir)
    item_fold_manifest(items).to_csv(
        output_dir / "item_fold_manifest.csv", index=False
    )
    protocol = {
        **behavior_source_provenance(args.behavior_path),
        "model": "Centaur-70B",
        "base_model": args.base_model,
        "centaur_adapter": args.centaur_adapter,
        "fold_definition": (
            "The same fixed, semantically balanced five folds used for Minitaur; "
            "defined before inspecting Taking outcomes."
        ),
        "folds": {str(key): list(value) for key, value in ITEM_FOLDS.items()},
        "fold_rationale": FOLD_RATIONALE,
        "conditions": list(TRAIN_CONDITIONS),
        "training_task": "Giving",
        "training_items_per_fold": 20,
        "heldout_items_per_fold": 5,
        "evaluation_tasks": ["Giving", "Taking"],
        "participant_split": "70/15/15, stratified by wave x culture",
        "giving_test_unit": "complete held-out participant",
        "taking_data_used_for_training_or_selection": False,
        "country_label_in_prompt": False,
        "training_objective": args.training_objective,
        "distribution_repeats": args.distribution_repeats,
        "optimizer_steps_per_adapter": args.distribution_repeats,
        "item_exposures_per_adapter": 20 * args.distribution_repeats,
        "quantization": "4-bit NF4 with double quantization; BF16 compute",
        "trainable_component": "new culture LoRA stacked on frozen Centaur adapter",
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )


def train_fold_condition(args: argparse.Namespace) -> None:
    if args.condition not in TRAIN_CONDITIONS:
        raise ValueError("Training requires --condition us, china, or pooled.")
    if args.fold not in ITEM_FOLDS:
        raise ValueError("Training requires a held-out fold from 0 through 4.")

    source_corpus = Path(args.corpus_dir)
    parent_protocol = require_corpus_behavior_source(
        source_corpus, args.behavior_path,
        prepare_script="scripts/dictator_centaur70b_qlora.py",
        group_path=args.group_path, seed=args.seed,
    )
    rows = pd.read_json(
        source_corpus / f"rows_{args.condition}.jsonl", lines=True
    )
    filtered = filter_training_rows(rows, args.fold)
    output_dir = ensure_dir(args.output_dir)
    filtered_corpus = ensure_dir(output_dir / "training_corpus")
    filtered.to_json(
        filtered_corpus / f"rows_{args.condition}.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    filtered_protocol = {
        **parent_protocol,
        "parent_corpus_dir": str(source_corpus.resolve()),
        "parent_protocol_sha256": hashlib.sha256((source_corpus / "protocol.json").read_bytes()).hexdigest(),
        "item_fold": args.fold,
        "equal_response_budget": int(len(filtered)),
        "training_items_per_fold": int(filtered["GroupNo"].nunique()),
        "training_group_nos": sorted(filtered["GroupNo"].unique().astype(int).tolist()),
        "heldout_group_nos": list(heldout_group_nos(args.fold)),
    }
    (filtered_corpus / "protocol.json").write_text(
        json.dumps(filtered_protocol, indent=2) + "\n", encoding="utf-8"
    )

    original_corpus_dir = args.corpus_dir
    args.corpus_dir = str(filtered_corpus)
    try:
        train_centaur70b(args)
    finally:
        args.corpus_dir = original_corpus_dir

    metadata = {
        **behavior_source_provenance(args.behavior_path),
        "seed": args.seed,
        "fold": args.fold,
        "condition": args.condition,
        "training_task": "Giving",
        "training_group_nos": sorted(
            filtered["GroupNo"].unique().astype(int).tolist()
        ),
        "heldout_group_nos": list(heldout_group_nos(args.fold)),
        "training_participants": int(filtered["participant_key"].nunique()),
        "training_responses": int(len(filtered)),
        "heldout_items_used_for_training": False,
        "taking_responses_used_for_training_or_selection": False,
    }
    (output_dir / "train_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def _annotate_and_write(
    evaluation_dir: Path,
    tables: dict[str, pd.DataFrame],
    args: argparse.Namespace,
    task: str,
) -> None:
    evaluation_dir = ensure_dir(evaluation_dir)
    for filename, frame in tables.items():
        frame.insert(1, "condition", args.condition)
        frame.insert(2, "training_seed", args.seed)
        frame.insert(3, "item_fold", args.fold)
        frame.insert(4, "evaluation_task", task)
        frame.to_csv(evaluation_dir / filename, index=False)


def evaluation_source_provenance(args: argparse.Namespace) -> dict[str, str]:
    """Require the evaluation population and trained adapter to share a source."""
    source = behavior_source_provenance(args.behavior_path)
    corpus = Path(args.corpus_dir)
    protocol = require_corpus_behavior_source(
        corpus, args.behavior_path,
        prepare_script="scripts/dictator_centaur70b_qlora.py",
        group_path=args.group_path, seed=args.seed,
    )
    if protocol.get("seed") != args.seed:
        raise ValueError("Evaluation corpus seed does not match --seed.")
    if args.condition != "base":
        metadata_path = Path(args.output_dir) / "train_metadata.json"
        if not metadata_path.is_file():
            raise ValueError("Adapter lacks train_metadata.json with verified behavior provenance.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "source_behavior_sha256": source["source_behavior_sha256"],
            "seed": args.seed,
            "fold": args.fold,
            "condition": args.condition,
            "heldout_group_nos": list(heldout_group_nos(args.fold)),
        }
        for field, value in expected.items():
            if metadata.get(field) != value:
                raise ValueError(f"Adapter training provenance mismatch for {field}.")
    return {
        **source,
        "corpus_protocol_sha256": hashlib.sha256((corpus / "protocol.json").read_bytes()).hexdigest(),
    }


def evaluate_condition(args: argparse.Namespace) -> None:
    if args.condition not in EVALUATION_CONDITIONS:
        raise ValueError("Evaluation requires base, us, china, or pooled.")
    provenance = evaluation_source_provenance(args)
    if args.condition == "base":
        if args.fold != -1:
            raise ValueError("Frozen-base evaluation uses --fold -1 (all 25 items).")
        evaluation_group_nos = tuple(range(1, 26))
    else:
        if args.fold not in ITEM_FOLDS:
            raise ValueError("Adapter evaluation requires a fold from 0 through 4.")
        evaluation_group_nos = heldout_group_nos(args.fold)
        resolve_saved_adapter_dir(Path(args.output_dir) / "adapter")

    giving, giving_items = load_giving_data(args.behavior_path, args.group_path)
    taking, _ = load_taking_data(args.behavior_path, args.group_path)
    items = giving_items.loc[
        giving_items["GroupNo"].isin(evaluation_group_nos)
    ].copy()
    if items["GroupNo"].nunique() != len(evaluation_group_nos):
        raise ValueError("Evaluation item metadata do not match the requested fold.")

    split_manifest = pd.read_csv(
        Path(args.corpus_dir) / "participant_split_manifest.csv"
    )
    giving_test = heldout_rows(giving, split_manifest)
    giving_test = giving_test.loc[
        giving_test["GroupNo"].isin(evaluation_group_nos)
    ].copy()
    taking_test = taking.loc[
        taking["GroupNo"].isin(evaluation_group_nos)
    ].copy()

    tokenizer = load_tokenizer(args.base_model)
    model = load_evaluation_model(args)
    import torch

    device = torch.device("cuda:0")
    specs = [
        ("Giving", giving_test, build_prompt, GIVING_DATASETS),
        ("Taking", taking_test, build_taking_prompt, TAKING_DATASETS),
    ]
    for task, human_rows, prompt_builder, wave_datasets in specs:
        candidates, predictions, invalid, aggregate_invalid = score_model(
            model,
            tokenizer,
            device,
            items,
            f"centaur70b-{args.condition}-item-holdout",
            prompt_builder=prompt_builder,
        )
        metrics, human_items = evaluate_against_humans(
            human_rows,
            candidates,
            wave_datasets=wave_datasets,
        )
        evaluation_dir = (
            Path(args.output_dir) / "evaluations" / task.lower()
        )
        _annotate_and_write(
            evaluation_dir,
            {
                "candidate_probabilities.csv": candidates,
                "item_predictions.csv": predictions,
                "invalid_token_probabilities.csv": invalid,
                "aggregate_invalid_tokens.csv": aggregate_invalid,
                "model_metrics.csv": metrics,
                "human_item_comparison.csv": human_items,
            },
            args,
            task,
        )
        metadata = {
            **provenance,
            "condition": args.condition,
            "training_seed": args.seed,
            "item_fold": args.fold,
            "evaluation_task": task,
            "evaluation_group_nos": [int(value) for value in evaluation_group_nos],
            "evaluation_items_seen_during_culture_adapter_training": (
                None if args.condition == "base" else False
            ),
            "giving_evaluation_uses_heldout_participants": True,
            "taking_responses_used_for_training_or_selection": False,
            "base_model": args.base_model,
            "centaur_adapter": args.centaur_adapter,
            "culture_adapter": (
                None
                if args.condition == "base"
                else str(
                    resolve_saved_adapter_dir(Path(args.output_dir) / "adapter")
                )
            ),
            "inference_passes_per_item": 1,
            "response_values": RESPONSE_VALUES,
            "evaluation_countries": list(COUNTRIES),
            "runtime_versions": runtime_versions(),
        }
        (evaluation_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Centaur-70B five-fold item holdout and Taking transfer."
    )
    parser.add_argument(
        "--stage", required=True, choices=["manifest", "train", "evaluate"]
    )
    parser.add_argument("--condition", choices=EVALUATION_CONDITIONS)
    parser.add_argument("--fold", type=int, choices=[-1, *ITEM_FOLDS], default=0)
    parser.add_argument("--behavior-path", required=True)
    parser.add_argument("--group-path", required=True)
    parser.add_argument("--corpus-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--base-model", default="unsloth/Meta-Llama-3.1-70B-bnb-4bit"
    )
    parser.add_argument(
        "--centaur-adapter",
        default="marcelbinz/Llama-3.1-Centaur-70B-adapter",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--micro-batch-size", type=int, default=20)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=768)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--use-rslora", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--optimizer", choices=["adamw_8bit", "adamw"], default="adamw")
    parser.add_argument(
        "--training-objective",
        choices=["item_distribution", "participant_rows"],
        default="item_distribution",
    )
    parser.add_argument("--distribution-repeats", type=int, default=25)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--step0-tolerance", type=float, default=1e-6)
    parser.add_argument("--step0-items", type=int, default=5)
    parser.add_argument("--smoke-rows", type=int, default=0)
    parser.add_argument("--max-optimizer-steps", type=int, default=-1)
    parser.add_argument("--bootstrap-resamples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "manifest":
        prepare_manifest(args)
        return
    if not args.corpus_dir:
        raise ValueError("--corpus-dir is required for training and evaluation.")
    if args.stage == "train":
        train_fold_condition(args)
    else:
        evaluate_condition(args)


if __name__ == "__main__":
    main()
