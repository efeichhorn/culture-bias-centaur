#!/usr/bin/env python3
"""Taking prompts, response loading, and evaluation helpers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_dictator_giving_base import (
    RESPONSE_VALUES,
    evaluate_against_humans,
    percent_to_token_choice,
    score_model,
)
from src.llm_model import device_for_eval, load_base_model, load_tokenizer
from src.utils import ensure_dir
from scripts.repair_wang_behavior_columns import ensure_not_mislabeled_source


CONDITIONS = ("base", "us", "china", "pooled", "pooled_full")
TAKING_DATASETS = (3, 4)
PARTICIPANT_KEYS = ["Dataset", "Nation", "SubNo"]


def build_taking_prompt(item: str) -> str:
    """Transcribe the released US Taking questionnaire for likelihood scoring.

    The source survey used a 0--20 dropdown. The final two lines expose that
    complete response set and add the same ``<<`` encoding used by Centaur-style
    scoring in the Giving analysis.
    """
    dropdown_choices = ", ".join(str(value) for value in RESPONSE_VALUES)
    encoded_choices = ", ".join(f"<<{value}>>" for value in RESPONSE_VALUES)
    return (
        "Please read the instructions carefully.\n\n"
        "Research on first impressions has shown that people are very good at making "
        "judgments about others based on very little information.\n\n"
        "In this study, you will learn just a little bit of information about a variety "
        "of different people, and you'll be asked to make 25 total decisions about how "
        "to divide resources with them. Then, you'll be asked some questions about your "
        "impressions of those people.\n\n"
        "For each decision, you'll learn one piece of information about a new person. "
        "For example, you might learn what the person's occupation or nationality is. "
        "When you see that information, try to imagine someone who fits that description. "
        "Sometimes you may see \"no information\", which means you don't know anything "
        "about the person this time.\n\n"
        "Then, you'll consider how you would make a decision if the given piece of "
        "information were all you knew about the person.\n\n"
        "Specifically, in the scenario in this study:\n\n"
        "- One of you is randomly assigned to start with 20 tokens on each decision, and "
        "the other person starts with 0 tokens. Each token is worth $1.5.\n"
        "- You have been selected to start with 0 tokens. The other person starts with "
        "20 tokens.\n"
        "- You decide how many of the other person's 20 tokens to take. If you decide to "
        "take x tokens, it means you receive x tokens and leave the other person 20-x "
        "tokens.\n"
        "- The other person cannot affect the outcome.\n\n"
        "You can take any number of tokens from 0 to 20.\n\n"
        "Although these decisions are hypothetical, please answer as honestly as you can, "
        "and please consider each decision individually. Because this is about first "
        "impressions, however, feel free to answer fairly quickly.\n\n"
        "Reminder: You decide how many tokens you want to take from the other person. "
        "The other person knows the original allocation of tokens, but that person cannot "
        "affect the outcome of your decision. Indicate how many you would take from the "
        "other person.\n\n"
        "For this decision, you are paired with a random person.\n"
        "Here is one piece of information about them:\n\n"
        f"{item}\n\n"
        "How many of the 20 tokens would you take from this person?\n\n"
        f"Available dropdown responses: {dropdown_choices}.\n"
        "For this natural-language transcription, responses are enclosed in double angle "
        f"brackets. The valid encoded responses are {encoded_choices}.\n"
        "You enter <<"
    )


def load_taking_data(
    behavior_path: str | Path,
    group_path: str | Path,
    *,
    allow_legacy_mislabeled: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Taking data; legacy opt-in is only for reproducing invalid archives."""
    behavior_path = Path(behavior_path)
    group_path = Path(group_path)
    if not allow_legacy_mislabeled:
        ensure_not_mislabeled_source(behavior_path)
    behavior = (
        pd.read_csv(behavior_path)
        if behavior_path.suffix.lower() == ".csv"
        else pd.read_excel(behavior_path)
    )
    item_source = (
        pd.read_csv(group_path)
        if group_path.suffix.lower() == ".csv"
        else pd.read_excel(group_path, sheet_name="Behavior studies")
    )
    items = item_source.rename(
        columns={"No.": "GroupNo", "English": "item", "Chinese": "item_chinese"}
    )
    taking = behavior.loc[behavior["Dataset"].isin(TAKING_DATASETS)].copy()
    taking["Country"] = taking["Nation"].map({0: "United States", 1: "China"})
    taking["TokenChoice"] = percent_to_token_choice(taking["Token"])
    taking["TokenChoicePercent"] = taking["TokenChoice"] * 5
    taking = taking.merge(
        items[["GroupNo", "item", "item_chinese"]],
        on="GroupNo",
        validate="many_to_one",
    )
    if taking["Country"].isna().any():
        raise ValueError("Unexpected Nation code in Taking data.")
    response_counts = taking.groupby(PARTICIPANT_KEYS).size()
    if not response_counts.eq(25).all():
        raise ValueError("Every retained participant must have exactly 25 Taking decisions.")
    expected_items = set(range(1, 26))
    if set(taking["GroupNo"].unique()) != expected_items:
        raise ValueError("Taking data do not contain the expected 25 target groups.")
    return taking, items.sort_values("GroupNo").reset_index(drop=True)


def evaluate_condition(args: argparse.Namespace) -> None:
    if args.condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {args.condition}")
    adapter_path = None
    if args.condition != "base":
        adapter_path = (
            Path(args.adapter_root) / f"seed{args.seed}" / "adapters" / args.condition
        )
        if not (adapter_path / "adapter_model.safetensors").exists():
            raise FileNotFoundError(f"Missing Giving-trained adapter: {adapter_path}")

    taking, items = load_taking_data(args.behavior_path, args.group_path)
    tokenizer = load_tokenizer(args.model_name)
    base_model = load_base_model(
        args.model_name,
        args.bf16,
        args.attn_implementation,
    )
    if adapter_path is None:
        model = base_model
    else:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, str(adapter_path))
    device = device_for_eval()
    model.to(device)
    model.eval()

    (
        candidate_probabilities,
        item_predictions,
        invalid_token_probabilities,
        aggregate_invalid_tokens,
    ) = score_model(
        model,
        tokenizer,
        device,
        items,
        args.model_name,
        prompt_builder=build_taking_prompt,
    )
    metrics, human_item_comparison = evaluate_against_humans(
        taking,
        candidate_probabilities,
        wave_datasets=TAKING_DATASETS,
    )

    tables = {
        "candidate_probabilities.csv": candidate_probabilities,
        "item_predictions.csv": item_predictions,
        "invalid_token_probabilities.csv": invalid_token_probabilities,
        "aggregate_invalid_tokens.csv": aggregate_invalid_tokens,
        "model_metrics.csv": metrics,
        "human_item_comparison.csv": human_item_comparison,
    }
    for frame in tables.values():
        frame.insert(1, "condition", args.condition)
        frame.insert(2, "training_seed", args.seed)
        frame.insert(3, "evaluation_task", "Taking")

    evaluation_dir = ensure_dir(
        Path(args.output_dir) / f"seed{args.seed}" / "evaluations" / args.condition
    )
    for filename, frame in tables.items():
        frame.to_csv(evaluation_dir / filename, index=False)
    item_predictions[["GroupNo", "item", "item_chinese", "prompt"]].to_csv(
        evaluation_dir / "prompt_manifest.csv",
        index=False,
    )

    participants = (
        taking[PARTICIPANT_KEYS + ["Country"]]
        .drop_duplicates()
        .groupby(["Dataset", "Country"])
        .size()
        .rename("participants")
        .reset_index()
    )
    metadata = {
        "model_name": args.model_name,
        "adapter_path": None if adapter_path is None else str(adapter_path),
        "adapter_training_task": (
            None if adapter_path is None else "Giving (Datasets 1 and 2 only)"
        ),
        "adapter_training_condition": args.condition,
        "adapter_training_seed": args.seed,
        "evaluation_task": "Taking",
        "evaluation_datasets": list(TAKING_DATASETS),
        "evaluation_uses_taking_responses_for_training": False,
        "response_values": RESPONSE_VALUES,
        "response_unit": "tokens_taken_out_of_20",
        "response_format": "prompt_ends_with_<<; score_response_value_token_only",
        "stimulus_version": "source-faithful_US_Taking_questionnaire_with_Centaur_encoding",
        "country_label_in_prompt": False,
        "evaluation_participants": participants.to_dict(orient="records"),
        "mean_reconstruction_error_percentage_points": float(
            np.mean(np.abs(taking["Token"] - taking["TokenChoice"] * 5.0))
        ),
    }
    (evaluation_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(metrics.to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)
