#!/usr/bin/env python3
"""Balanced participant splits and source checks for allocation models."""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dictator_culture_finetune import (
    train_distributional_lora_adapter,
    training_config,
)
from scripts.evaluate_dictator_giving_base import (
    RESPONSE_VALUES,
    evaluate_against_humans,
    greedy_generations,
    load_giving_data,
    score_model,
)
from scripts.repair_wang_behavior_columns import ensure_not_mislabeled_source
from src.llm_model import device_for_eval, load_base_model, load_tokenizer
from src.utils import ensure_dir


CONDITIONS = ("us", "china", "pooled", "pooled_full")
COUNTRIES = ("United States", "China")
COUNTRY_TO_CONDITION = {"United States": "us", "China": "china"}
PARTICIPANT_KEYS = ["Dataset", "Nation", "SubNo"]


def behavior_source_provenance(behavior_path: str | Path) -> dict[str, str]:
    """Record the exact checked input used to construct cached training targets."""
    path = Path(behavior_path)
    ensure_not_mislabeled_source(path)
    return {
        "source_behavior_path": str(path.resolve()),
        "source_behavior_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def require_corpus_behavior_source(
    corpus_dir: str | Path,
    behavior_path: str | Path,
    *,
    prepare_script: str,
    group_path: str | Path,
    seed: int = 0,
) -> dict[str, object]:
    """Reject stale or untracked cached targets before any training/model loading."""
    provenance = behavior_source_provenance(behavior_path)
    corpus_dir = Path(corpus_dir)
    protocol_path = corpus_dir / "protocol.json"
    protocol = {}
    if protocol_path.is_file():
        try:
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    if not isinstance(protocol, dict) or protocol.get("source_behavior_sha256") != provenance["source_behavior_sha256"]:
        new_corpus = corpus_dir.with_name(corpus_dir.name + "_corrected")
        command = shlex.join([
            "python", prepare_script, "--stage", "prepare",
            "--behavior-path", str(behavior_path), "--group-path", str(group_path),
            "--output-dir", str(new_corpus), "--seed", str(seed),
        ])
        raise ValueError(
            f"Cached corpus {corpus_dir} has missing or stale source_behavior_sha256; "
            "its targets are not verified against the requested behavior file. "
            "Regenerate from corrected behavior before training, then use the new --corpus-dir:\n"
            + command
        )
    return protocol


def add_participant_key(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["participant_key"] = (
        output["Dataset"].astype(str)
        + ":"
        + output["Nation"].astype(str)
        + ":"
        + output["SubNo"].astype(str)
    )
    return output


def balanced_participant_split(
    giving: pd.DataFrame,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> pd.DataFrame:
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Training and validation fractions must be positive.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Training plus validation fraction must be below one.")
    participants = (
        giving[PARTICIPANT_KEYS + ["Country"]]
        .drop_duplicates()
        .sort_values(PARTICIPANT_KEYS)
        .reset_index(drop=True)
    )
    parts = []
    for wave in sorted(participants["Dataset"].unique()):
        wave_frame = participants.loc[participants["Dataset"].eq(wave)]
        limiting_count = int(
            wave_frame.groupby("Country").size().reindex(COUNTRIES).min()
        )
        train_count = int(np.floor(limiting_count * train_fraction))
        validation_count = int(np.floor(limiting_count * validation_fraction))
        test_count = limiting_count - train_count - validation_count
        split_labels = (
            ["train"] * train_count
            + ["validation"] * validation_count
            + ["test"] * test_count
        )
        for country_index, country in enumerate(COUNTRIES):
            country_frame = wave_frame.loc[wave_frame["Country"].eq(country)]
            selected = country_frame.sample(
                n=limiting_count,
                replace=False,
                random_state=seed + wave * 10_000 + country_index * 1_000,
            ).reset_index(drop=True)
            selected = selected.sample(
                frac=1.0,
                random_state=seed + wave * 20_000 + country_index * 2_000,
            ).reset_index(drop=True)
            selected["split"] = split_labels
            parts.append(selected)
    manifest = add_participant_key(pd.concat(parts, ignore_index=True))
    counts = manifest.groupby(["Dataset", "split", "Country"]).size().unstack()
    if not counts[COUNTRIES[0]].eq(counts[COUNTRIES[1]]).all():
        raise ValueError("Participant split is not balanced by culture within wave.")
    if manifest[PARTICIPANT_KEYS].duplicated().any():
        raise ValueError("A participant appears in more than one split.")
    return manifest.sort_values(["split", "Dataset", "Nation", "SubNo"]).reset_index(
        drop=True
    )


def select_training_conditions(split_manifest: pd.DataFrame, seed: int) -> pd.DataFrame:
    train = split_manifest.loc[split_manifest["split"].eq("train")].copy()
    country_counts = train.groupby("Country")["participant_key"].nunique()
    if len(set(country_counts.reindex(COUNTRIES))) != 1:
        raise ValueError("Single-culture training pools must have equal sizes.")
    single_budget = int(country_counts.iloc[0])
    if single_budget % 2:
        raise ValueError("The single-culture training budget must be even.")
    pooled_per_culture = single_budget // 2
    parts = []
    for country_index, country in enumerate(COUNTRIES):
        country_train = train.loc[train["Country"].eq(country)].copy()
        single = country_train.copy()
        single["condition"] = COUNTRY_TO_CONDITION[country]
        single["selection_role"] = "single_culture"
        parts.append(single)

        pooled = country_train.sample(
            n=pooled_per_culture,
            replace=False,
            random_state=seed + 50_000 + country_index * 5_000,
        ).copy()
        pooled["condition"] = "pooled"
        pooled["selection_role"] = "balanced_equal_total_budget"
        parts.append(pooled)

        pooled_full = country_train.copy()
        pooled_full["condition"] = "pooled_full"
        pooled_full["selection_role"] = "balanced_full_per_culture_budget"
        parts.append(pooled_full)
    manifest = pd.concat(parts, ignore_index=True)
    totals = manifest.groupby("condition")["participant_key"].nunique()
    expected = {
        "us": single_budget,
        "china": single_budget,
        "pooled": single_budget,
        "pooled_full": 2 * single_budget,
    }
    if totals.to_dict() != expected:
        raise ValueError(f"Unexpected training condition budgets: {totals.to_dict()}")
    return manifest.sort_values(
        ["condition", "Dataset", "Nation", "SubNo"]
    ).reset_index(drop=True)


def aggregate_training_counts(
    giving: pd.DataFrame,
    condition_manifest: pd.DataFrame,
    condition: str,
) -> pd.DataFrame:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    selected = condition_manifest.loc[
        condition_manifest["condition"].eq(condition), PARTICIPANT_KEYS
    ].drop_duplicates()
    rows = giving.merge(
        selected,
        on=PARTICIPANT_KEYS,
        how="inner",
        validate="many_to_one",
    )
    participant_trials = rows.groupby(PARTICIPANT_KEYS).size()
    if not participant_trials.eq(25).all():
        raise ValueError("Every training participant must contribute all 25 items.")
    participant_count = int(len(participant_trials))
    counts = (
        rows.groupby(["GroupNo", "item", "TokenChoice"])
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"TokenChoice": "response_value"})
    )
    counts["condition"] = condition
    counts["participant_count"] = participant_count
    counts["trial_count"] = participant_count * 25
    return counts[
        [
            "condition",
            "GroupNo",
            "item",
            "response_value",
            "count",
            "participant_count",
            "trial_count",
        ]
    ]


def prepare_corpora(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    giving, _ = load_giving_data(args.behavior_path, args.group_path)
    split_manifest = balanced_participant_split(
        giving,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    condition_manifest = select_training_conditions(split_manifest, seed=args.seed)
    split_manifest.to_csv(output_dir / "participant_split_manifest.csv", index=False)
    condition_manifest.to_csv(
        output_dir / "training_participant_manifest.csv", index=False
    )

    corpus_rows = []
    for condition in CONDITIONS:
        counts = aggregate_training_counts(giving, condition_manifest, condition)
        counts.to_csv(output_dir / f"counts_{condition}.csv", index=False)
        selected = condition_manifest.loc[
            condition_manifest["condition"].eq(condition)
        ]
        by_country = (
            selected.groupby(["condition", "Country"])
            .agg(participants=("participant_key", "nunique"))
            .reset_index()
        )
        by_country["trials"] = by_country["participants"] * 25
        corpus_rows.append(by_country)
    pd.concat(corpus_rows, ignore_index=True).to_csv(
        output_dir / "corpus_manifest.csv", index=False
    )
    split_counts = (
        split_manifest.groupby(["Dataset", "split", "Country"])
        .size()
        .rename("participants")
        .reset_index()
    )
    split_counts.to_csv(output_dir / "split_counts.csv", index=False)
    protocol = {
        **behavior_source_provenance(args.behavior_path),
        "seed": args.seed,
        "waves": [1, 2],
        "split_unit": "complete participant (Dataset, Nation, SubNo)",
        "stratification": "balanced within wave x culture",
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "test_fraction": 1.0 - args.train_fraction - args.validation_fraction,
        "conditions": list(CONDITIONS),
        "country_label_in_prompt": False,
        "pooled_equal_budget": "half US, half China; same total participants as a single-culture adapter",
        "pooled_full": "all US and China training participants; twice the total participant budget",
        "training_objective": "empirical 21-category soft-target cross-entropy",
        "primary_evaluation": "Combined-wave held-out participant conditional KL / paired log loss",
        "secondary_evaluation": "Wave-specific held-out metrics and ordinal Wasserstein distance",
    }
    (Path(output_dir) / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )


def train_condition(args: argparse.Namespace) -> None:
    if args.condition not in CONDITIONS:
        raise ValueError("--condition is required for training.")
    require_corpus_behavior_source(
        args.corpus_dir, args.behavior_path,
        prepare_script="scripts/dictator_culture_mixed_split.py",
        group_path=args.group_path, seed=args.seed,
    )
    counts_path = Path(args.corpus_dir) / f"counts_{args.condition}.csv"
    rows = pd.read_csv(counts_path)
    cfg = training_config(args)
    adapter_dir = ensure_dir(Path(args.output_dir) / "adapters" / args.condition)
    metrics_path = Path(args.output_dir) / f"train_metrics_{args.condition}.json"
    train_distributional_lora_adapter(
        rows,
        cfg=cfg,
        output_dir=adapter_dir,
        metrics_path=metrics_path,
        seed=args.seed,
    )
    (Path(args.output_dir) / f"train_config_{args.condition}.json").write_text(
        json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8"
    )


def heldout_rows(giving: pd.DataFrame, split_manifest: pd.DataFrame) -> pd.DataFrame:
    test_participants = split_manifest.loc[
        split_manifest["split"].eq("test"), PARTICIPANT_KEYS
    ]
    rows = giving.merge(
        test_participants,
        on=PARTICIPANT_KEYS,
        how="inner",
        validate="many_to_one",
    )
    counts = rows.groupby(PARTICIPANT_KEYS).size()
    if not counts.eq(25).all():
        raise ValueError("Every held-out participant must contribute 25 responses.")
    return rows


def evaluate_condition(args: argparse.Namespace) -> None:
    if args.condition not in CONDITIONS:
        raise ValueError("--condition is required for evaluation.")
    adapter_path = Path(args.output_dir) / "adapters" / args.condition
    if not (adapter_path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Missing adapter: {adapter_path}")

    from peft import PeftModel

    giving, items = load_giving_data(args.behavior_path, args.group_path)
    split_manifest = pd.read_csv(
        Path(args.corpus_dir) / "participant_split_manifest.csv"
    )
    test_rows = heldout_rows(giving, split_manifest)
    tokenizer = load_tokenizer(args.model_name)
    base_model = load_base_model(
        args.model_name, args.bf16, args.attn_implementation
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    device = device_for_eval()
    model.to(device)
    model.eval()
    (
        candidate_probabilities,
        item_predictions,
        invalid_token_probabilities,
        aggregate_invalid_tokens,
    ) = score_model(model, tokenizer, device, items, args.model_name)
    metrics, human_item_comparison = evaluate_against_humans(
        test_rows, candidate_probabilities
    )
    generations = greedy_generations(
        model,
        tokenizer,
        device,
        item_predictions,
        batch_size=args.generation_batch_size,
    )
    for frame in [
        candidate_probabilities,
        item_predictions,
        invalid_token_probabilities,
        aggregate_invalid_tokens,
        metrics,
        human_item_comparison,
        generations,
    ]:
        frame.insert(1, "condition", args.condition)
    evaluation_dir = ensure_dir(
        Path(args.output_dir) / "evaluations" / args.condition
    )
    for filename, frame in {
        "candidate_probabilities.csv": candidate_probabilities,
        "item_predictions.csv": item_predictions,
        "invalid_token_probabilities.csv": invalid_token_probabilities,
        "aggregate_invalid_tokens.csv": aggregate_invalid_tokens,
        "model_metrics.csv": metrics,
        "human_item_comparison.csv": human_item_comparison,
        "greedy_generations.csv": generations,
    }.items():
        frame.to_csv(evaluation_dir / filename, index=False)
    metadata = {
        "model_name": args.model_name,
        "adapter_path": str(adapter_path),
        "condition": args.condition,
        "evaluation_participants": int(
            test_rows[PARTICIPANT_KEYS].drop_duplicates().shape[0]
        ),
        "primary_sample": "Combined",
        "response_values": RESPONSE_VALUES,
        "country_label_in_prompt": False,
    }
    (evaluation_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def candidate_matrix(candidates: pd.DataFrame, column: str) -> np.ndarray:
    return (
        candidates.pivot(index="GroupNo", columns="response_value", values=column)
        .reindex(index=range(1, 26), columns=RESPONSE_VALUES)
        .to_numpy(dtype=float)
    )


def human_distribution(rows: pd.DataFrame) -> np.ndarray:
    counts = (
        rows.groupby(["GroupNo", "TokenChoice"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(1, 26), columns=RESPONSE_VALUES, fill_value=0)
    )
    return counts.div(counts.sum(axis=1), axis=0).to_numpy(dtype=float)


def conditional_kl(q: np.ndarray, p: np.ndarray) -> float:
    positive = q > 0
    log_q = np.zeros_like(q)
    log_q[positive] = np.log(q[positive])
    entropy = float(-(q * log_q).sum(axis=1).mean())
    nll = float(
        -(q * np.log(np.clip(p, np.finfo(float).tiny, 1.0))).sum(axis=1).mean()
    )
    return nll - entropy


def tie_aware_mode_metrics(
    rows: pd.DataFrame, candidates: pd.DataFrame
) -> tuple[float, float]:
    counts = (
        rows.groupby(["GroupNo", "TokenChoice"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(1, 26), columns=RESPONSE_VALUES, fill_value=0)
    )
    model_modes = (
        candidates.pivot(
            index="GroupNo",
            columns="response_value",
            values="conditional_probability",
        )
        .reindex(index=range(1, 26), columns=RESPONSE_VALUES)
        .idxmax(axis=1)
        .astype(int)
    )
    correct = []
    distances = []
    for group_no, model_mode in model_modes.items():
        item_counts = counts.loc[group_no]
        human_modes = item_counts.index[item_counts.eq(item_counts.max())].to_numpy(
            dtype=int
        )
        correct.append(int(model_mode in human_modes))
        distances.append(int(np.min(np.abs(human_modes - model_mode))))
    return float(np.mean(correct)), float(np.mean(distances))


def analyze(args: argparse.Namespace) -> None:
    run_dir = Path(args.output_dir)
    base_dir = Path(args.base_result_dir)
    candidate_frames = {
        "base": pd.read_csv(base_dir / "candidate_probabilities.csv")
    }
    for condition in CONDITIONS:
        candidate_frames[condition] = pd.read_csv(
            run_dir / "evaluations" / condition / "candidate_probabilities.csv"
        )
    probabilities = {
        condition: candidate_matrix(frame, "conditional_probability")
        for condition, frame in candidate_frames.items()
    }

    giving, _ = load_giving_data(args.behavior_path, args.group_path)
    split_manifest = pd.read_csv(
        Path(args.corpus_dir) / "participant_split_manifest.csv"
    )
    test_rows = heldout_rows(giving, split_manifest)

    metric_parts = []
    human_parts = []
    for condition, candidates in candidate_frames.items():
        metrics, human_items = evaluate_against_humans(test_rows, candidates)
        metrics.insert(1, "condition", condition)
        human_items.insert(1, "condition", condition)
        metric_parts.append(metrics)
        human_parts.append(human_items)
    all_metrics = pd.concat(metric_parts, ignore_index=True)
    all_human_items = pd.concat(human_parts, ignore_index=True)
    all_metrics.to_csv(run_dir / "all_model_metrics.csv", index=False)
    all_human_items.to_csv(run_dir / "all_human_item_comparison.csv", index=False)

    secondary_rows = []
    combined_metrics = all_metrics.loc[all_metrics["sample"].eq("Combined")]
    for condition, candidates in candidate_frames.items():
        for country in COUNTRIES:
            country_rows = test_rows.loc[test_rows["Country"].eq(country)]
            mode_accuracy, mode_distance = tie_aware_mode_metrics(
                country_rows, candidates
            )
            record = combined_metrics.loc[
                combined_metrics["condition"].eq(condition)
                & combined_metrics["country"].eq(country)
            ].iloc[0]
            secondary_rows.append(
                {
                    "condition": condition,
                    "country": country,
                    "participants": int(record["participants"]),
                    "tie_aware_item_mode_accuracy": mode_accuracy,
                    "item_mode_mae_tokens": mode_distance,
                    "item_expected_mae_tokens": float(record["item_mean_mae_tokens"]),
                    "ordinal_wasserstein_tokens": float(
                        record["ordinal_wasserstein_tokens"]
                    ),
                }
            )
    pd.DataFrame(secondary_rows).to_csv(
        run_dir / "heldout_secondary_metrics.csv", index=False
    )

    participant_loss_parts = []
    for condition, candidates in candidate_frames.items():
        lookup = candidates[
            [
                "GroupNo",
                "response_value",
                "conditional_probability",
                "full_vocab_log_probability",
            ]
        ].rename(columns={"response_value": "TokenChoice"})
        scored = test_rows.merge(
            lookup,
            on=["GroupNo", "TokenChoice"],
            how="left",
            validate="many_to_one",
        )
        scored["conditional_nll_nats"] = -np.log(
            scored["conditional_probability"].clip(lower=np.finfo(float).tiny)
        )
        scored["centaur_nll_nats"] = -scored["full_vocab_log_probability"]
        participant = (
            scored.groupby(PARTICIPANT_KEYS + ["Country"], as_index=False)[
                ["conditional_nll_nats", "centaur_nll_nats"]
            ]
            .mean()
        )
        participant.insert(0, "condition", condition)
        participant_loss_parts.append(participant)
    participant_losses = pd.concat(participant_loss_parts, ignore_index=True)
    participant_losses.to_csv(run_dir / "participant_test_losses.csv", index=False)

    combined = all_metrics.loc[all_metrics["sample"].eq("Combined")]
    loss_matrix = combined.pivot(
        index="condition", columns="country", values="conditional_kl_nats"
    ).rename(columns={"United States": "kl_us", "China": "kl_china"})
    loss_matrix["mean_kl"] = loss_matrix[["kl_us", "kl_china"]].mean(axis=1)
    loss_matrix["alignment_gap_china_minus_us"] = (
        loss_matrix["kl_china"] - loss_matrix["kl_us"]
    )
    loss_matrix.reset_index().to_csv(run_dir / "heldout_loss_matrix.csv", index=False)

    participant_ids = {
        country: test_rows.loc[test_rows["Country"].eq(country), PARTICIPANT_KEYS]
        .drop_duplicates()
        .reset_index(drop=True)
        for country in COUNTRIES
    }
    country_rows = {
        country: test_rows.loc[test_rows["Country"].eq(country)]
        for country in COUNTRIES
    }
    rng = np.random.default_rng(args.seed + 90_000)
    bootstrap_rows = []
    for bootstrap_index in range(args.bootstrap_resamples):
        q_by_country = {}
        kl = {}
        for country in COUNTRIES:
            ids = participant_ids[country]
            sampled = ids.iloc[
                rng.integers(0, len(ids), size=len(ids))
            ].copy()
            sampled["bootstrap_occurrence"] = np.arange(len(sampled))
            sampled_rows = sampled.merge(
                country_rows[country][PARTICIPANT_KEYS + ["GroupNo", "TokenChoice"]],
                on=PARTICIPANT_KEYS,
                how="left",
                validate="many_to_many",
            )
            q = human_distribution(sampled_rows)
            q_by_country[country] = q
            for condition, p in probabilities.items():
                kl[(condition, country)] = conditional_kl(q, p)
                bootstrap_rows.append(
                    {
                        "bootstrap": bootstrap_index,
                        "estimand": "conditional_kl",
                        "condition": condition,
                        "country": country,
                        "value": kl[(condition, country)],
                    }
                )
                if condition != "base":
                    bootstrap_rows.append(
                        {
                            "bootstrap": bootstrap_index,
                            "estimand": "conditional_kl_change_vs_base",
                            "condition": condition,
                            "country": country,
                            "value": kl[(condition, country)] - kl[("base", country)],
                        }
                    )
        gaps = {}
        for condition in probabilities:
            gaps[condition] = (
                kl[(condition, "China")]
                - kl[(condition, "United States")]
            )
            bootstrap_rows.append(
                {
                    "bootstrap": bootstrap_index,
                    "estimand": "alignment_gap_china_minus_us",
                    "condition": condition,
                    "country": "China - United States",
                    "value": gaps[condition],
                }
            )
        bootstrap_rows.append(
            {
                "bootstrap": bootstrap_index,
                "estimand": "culture_specialization_interaction",
                "condition": "us_minus_china_adapters",
                "country": "difference-in-differences",
                "value": gaps["us"] - gaps["china"],
            }
        )
        for pooled_condition in ["pooled", "pooled_full"]:
            for country in COUNTRIES:
                specialist = COUNTRY_TO_CONDITION[country]
                bootstrap_rows.append(
                    {
                        "bootstrap": bootstrap_index,
                        "estimand": "pooled_minus_matching_specialist_kl",
                        "condition": pooled_condition,
                        "country": country,
                        "value": kl[(pooled_condition, country)] - kl[(specialist, country)],
                    }
                )
        balanced_mixture = 0.5 * (
            q_by_country["United States"] + q_by_country["China"]
        )
        for condition, p in probabilities.items():
            bootstrap_rows.append(
                {
                    "bootstrap": bootstrap_index,
                    "estimand": "balanced_mixture_kl",
                    "condition": condition,
                    "country": "50% United States + 50% China",
                    "value": conditional_kl(balanced_mixture, p),
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(run_dir / "participant_bootstrap_samples.csv", index=False)
    summary = (
        bootstrap.groupby(["estimand", "condition", "country"])["value"]
        .agg(
            estimate="mean",
            ci_low=lambda values: values.quantile(0.025),
            ci_high=lambda values: values.quantile(0.975),
        )
        .reset_index()
    )
    summary.to_csv(run_dir / "participant_bootstrap_summary.csv", index=False)

    matrix = loss_matrix
    effects = {
        "seed": args.seed,
        "us_own_change_vs_base": float(matrix.loc["us", "kl_us"] - matrix.loc["base", "kl_us"]),
        "us_cross_change_vs_base": float(matrix.loc["us", "kl_china"] - matrix.loc["base", "kl_china"]),
        "china_own_change_vs_base": float(matrix.loc["china", "kl_china"] - matrix.loc["base", "kl_china"]),
        "china_cross_change_vs_base": float(matrix.loc["china", "kl_us"] - matrix.loc["base", "kl_us"]),
        "culture_specialization_interaction": float(
            matrix.loc["us", "alignment_gap_china_minus_us"]
            - matrix.loc["china", "alignment_gap_china_minus_us"]
        ),
        "pooled_us_minus_us_specialist": float(matrix.loc["pooled", "kl_us"] - matrix.loc["us", "kl_us"]),
        "pooled_china_minus_china_specialist": float(matrix.loc["pooled", "kl_china"] - matrix.loc["china", "kl_china"]),
        "pooled_full_us_minus_us_specialist": float(matrix.loc["pooled_full", "kl_us"] - matrix.loc["us", "kl_us"]),
        "pooled_full_china_minus_china_specialist": float(matrix.loc["pooled_full", "kl_china"] - matrix.loc["china", "kl_china"]),
    }
    (run_dir / "contrast_effects.json").write_text(
        json.dumps(effects, indent=2) + "\n", encoding="utf-8"
    )
