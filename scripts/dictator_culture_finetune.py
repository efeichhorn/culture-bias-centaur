#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_dictator_giving_base import (
    RESPONSE_VALUES,
    build_prompt,
    evaluate_against_humans,
    greedy_generations,
    load_giving_data,
    percent_to_token_choice,
    score_model,
)
from src.config import ExperimentConfig
from src.llm_model import (
    build_lora_model,
    device_for_eval,
    load_base_model,
    load_tokenizer,
)
from src.llm_train import _base_causal_lm, train_lora_adapter
from src.utils import ensure_dir, save_json, set_global_seed


CONDITIONS = ("us", "china", "pooled")
COUNTRIES = ("United States", "China")


def participant_table(giving: pd.DataFrame, wave: int = 1) -> pd.DataFrame:
    return (
        giving.loc[giving["Dataset"].eq(wave), ["Dataset", "Nation", "SubNo", "Country"]]
        .drop_duplicates()
        .sort_values(["Nation", "SubNo"])
        .reset_index(drop=True)
    )


def select_training_participants(
    giving: pd.DataFrame,
    seed: int,
    culture_participants: int = 500,
    pooled_per_culture: int = 250,
) -> pd.DataFrame:
    if pooled_per_culture * 2 != culture_participants:
        raise ValueError(
            "The pooled condition must contain the same total participants as each "
            "single-culture condition."
        )
    participants = participant_table(giving, wave=1)
    selected_parts = []
    for country_index, country in enumerate(COUNTRIES):
        available = participants.loc[participants["Country"].eq(country)].copy()
        if len(available) < culture_participants:
            raise ValueError(
                f"Need {culture_participants} Wave-1 participants for {country}, "
                f"found {len(available)}."
            )
        selected = available.sample(
            n=culture_participants,
            random_state=seed + 10_000 * (country_index + 1),
            replace=False,
        ).sort_values("SubNo")
        selected["condition"] = "us" if country == "United States" else "china"
        selected["selection_role"] = "single_culture"
        selected_parts.append(selected)

        pooled = selected.sample(
            n=pooled_per_culture,
            random_state=seed + 30_000 * (country_index + 1),
            replace=False,
        ).copy()
        pooled["condition"] = "pooled"
        pooled["selection_role"] = "balanced_pooled"
        selected_parts.append(pooled)
    manifest = pd.concat(selected_parts, ignore_index=True)
    manifest["participant_key"] = (
        manifest["Dataset"].astype(str)
        + ":"
        + manifest["Nation"].astype(str)
        + ":"
        + manifest["SubNo"].astype(str)
    )
    return manifest.sort_values(["condition", "Nation", "SubNo"]).reset_index(drop=True)


def training_prompt(item: str) -> str:
    prompt = build_prompt(item)
    if not prompt.endswith("<<"):
        raise ValueError("Expected the evaluation prompt to end with <<.")
    return prompt[:-2]


def build_training_corpus(
    giving: pd.DataFrame,
    manifest: pd.DataFrame,
    condition: str,
    seed: int,
) -> pd.DataFrame:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    chosen = manifest.loc[
        manifest["condition"].eq(condition), ["Dataset", "Nation", "SubNo"]
    ].drop_duplicates()
    rows = giving.merge(
        chosen,
        on=["Dataset", "Nation", "SubNo"],
        how="inner",
        validate="many_to_one",
    ).copy()
    if not rows["Dataset"].eq(1).all():
        raise ValueError("Training corpus contains a non-Wave-1 row.")
    participant_counts = rows.groupby(["Dataset", "Nation", "SubNo"]).size()
    if not participant_counts.eq(25).all():
        raise ValueError("Every selected training participant must contribute 25 trials.")
    rows["response_value"] = percent_to_token_choice(rows["Token"])
    rows["prompt"] = rows["item"].map(training_prompt)
    rows["response_text"] = rows["response_value"].map(lambda value: f"<<{value}>>.")
    rows["condition"] = condition
    rows["participant_key"] = (
        rows["Dataset"].astype(str)
        + ":"
        + rows["Nation"].astype(str)
        + ":"
        + rows["SubNo"].astype(str)
    )
    rows = rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return rows[
        [
            "condition",
            "Dataset",
            "Nation",
            "Country",
            "SubNo",
            "participant_key",
            "GroupNo",
            "item",
            "Token",
            "response_value",
            "prompt",
            "response_text",
        ]
    ]


def prepare_corpora(
    behavior_path: str | Path,
    group_path: str | Path,
    output_dir: str | Path,
    seed: int,
    culture_participants: int,
    pooled_per_culture: int,
) -> None:
    output_dir = ensure_dir(output_dir)
    giving, _ = load_giving_data(behavior_path, group_path)
    manifest = select_training_participants(
        giving,
        seed=seed,
        culture_participants=culture_participants,
        pooled_per_culture=pooled_per_culture,
    )
    manifest.to_csv(output_dir / "participant_manifest.csv", index=False)

    corpus_rows = []
    prepared_corpora = {}
    for condition in CONDITIONS:
        corpus = build_training_corpus(giving, manifest, condition, seed)
        prepared_corpora[condition] = corpus
        corpus.to_json(
            output_dir / f"rows_{condition}.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )
        counts = (
            corpus.groupby(["condition", "Country"])
            .agg(
                participants=("participant_key", "nunique"),
                trials=("response_value", "size"),
                mean_response_tokens=("response_value", "mean"),
            )
            .reset_index()
        )
        corpus_rows.append(counts)
    corpus_manifest = pd.concat(corpus_rows, ignore_index=True)
    corpus_manifest.to_csv(output_dir / "corpus_manifest.csv", index=False)

    smoke_candidates = prepared_corpora["us"].loc[
        prepared_corpora["us"]["response_value"].eq(10)
    ]
    if smoke_candidates.empty:
        raise ValueError("Could not construct a response-10 masking smoke corpus.")
    smoke = pd.concat([smoke_candidates.iloc[[0]]] * 32, ignore_index=True)
    smoke.to_json(
        output_dir / "rows_smoke_us.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )

    total_trials = corpus_manifest.groupby("condition")["trials"].sum()
    total_participants = corpus_manifest.groupby("condition")["participants"].sum()
    if not total_trials.eq(culture_participants * 25).all():
        raise ValueError("Training conditions do not have equal trial budgets.")
    if not total_participants.eq(culture_participants).all():
        raise ValueError("Training conditions do not have equal participant budgets.")
    protocol = {
        "seed": seed,
        "training_wave": 1,
        "evaluation_wave": 2,
        "culture_participants_per_condition": culture_participants,
        "pooled_participants_per_culture": pooled_per_culture,
        "trials_per_participant": 25,
        "trials_per_condition": int(culture_participants * 25),
        "released_to_source_mapping": "floor(Token/5 + 0.5), clipped to [0,20]",
        "training_prompt_ending": "You enter ",
        "training_completion": "<<value>>.",
        "loss_mask": "numeric response value only",
        "evaluation_prompt_ending": "You enter <<",
        "evaluation_response": "numeric response token",
        "country_label_in_prompt": False,
    }
    (Path(output_dir) / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )


def training_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        output_dir=str(args.output_dir),
        model_name=args.model_name,
        num_train_epochs=1.0,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        dynamic_padding=True,
        length_bucketed_batches=True,
        data_parallel=False,
        bf16=args.bf16,
        gradient_checkpointing=True,
        train_on_eos=True,
        train_answer_value_only=True,
        attn_implementation=args.attn_implementation,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
    )


def distributional_training_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if "count" in rows.columns:
        counts = (
            rows.groupby(["GroupNo", "item", "response_value"], as_index=False)[
                "count"
            ]
            .sum()
        )
    else:
        counts = (
            rows.groupby(["GroupNo", "item", "response_value"])
            .size()
            .rename("count")
            .reset_index()
        )
    item_names = rows[["GroupNo", "item"]].drop_duplicates()
    if item_names.empty or item_names["GroupNo"].duplicated().any():
        raise ValueError("Distributional training requires one name per retained item.")
    group_nos = sorted(item_names["GroupNo"].astype(int).unique())
    grid = pd.MultiIndex.from_product(
        [group_nos, RESPONSE_VALUES],
        names=["GroupNo", "response_value"],
    ).to_frame(index=False)
    grid = grid.merge(item_names, on="GroupNo", validate="many_to_one")
    grid = grid.merge(
        counts,
        on=["GroupNo", "item", "response_value"],
        how="left",
    ).fillna({"count": 0})
    probabilities = (
        grid.pivot(index="GroupNo", columns="response_value", values="count")
        .reindex(index=group_nos, columns=RESPONSE_VALUES, fill_value=0)
    )
    probabilities = probabilities.div(probabilities.sum(axis=1), axis=0)
    items = (
        item_names.sort_values("GroupNo")
        .assign(prompt=lambda frame: frame["item"].map(build_prompt))
        .reset_index(drop=True)
    )
    return items, probabilities.to_numpy(dtype=np.float32)


def train_distributional_lora_adapter(
    rows: pd.DataFrame,
    cfg: ExperimentConfig,
    output_dir: str | Path,
    metrics_path: str | Path,
    seed: int,
) -> dict[str, object]:
    import torch
    from transformers import get_linear_schedule_with_warmup

    def encode_prompts(prompts: list[str]) -> dict[str, torch.Tensor]:
        encoded_rows = []
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        for prompt in prompts:
            ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            if len(ids) > cfg.max_seq_length:
                has_bos = bool(ids) and ids[0] == bos_token_id
                if has_bos:
                    ids = [ids[0], *ids[-(cfg.max_seq_length - 1) :]]
                else:
                    ids = ids[-cfg.max_seq_length :]
            encoded_rows.append(ids)
        maximum = max(len(ids) for ids in encoded_rows)
        input_rows = []
        mask_rows = []
        for ids in encoded_rows:
            padding = maximum - len(ids)
            input_rows.append(ids + [tokenizer.pad_token_id] * padding)
            mask_rows.append([1] * len(ids) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(mask_rows, dtype=torch.long, device=device),
        }

    set_global_seed(seed)
    output_dir = ensure_dir(output_dir)
    items, target_probabilities = distributional_training_rows(rows)
    tokenizer = load_tokenizer(cfg.model_name)
    candidate_token_ids = []
    for value in RESPONSE_VALUES:
        ids = tokenizer(str(value), add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"Response {value} is not one token: {ids}")
        candidate_token_ids.append(int(ids[0]))
    if len(set(candidate_token_ids)) != len(candidate_token_ids):
        raise ValueError("Response values map to duplicate token IDs.")

    model = load_base_model(cfg.model_name, cfg.bf16, cfg.attn_implementation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(model, "config"):
        model.config.use_cache = False
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = build_lora_model(model, cfg)
    if cfg.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.to(device)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    planned_steps = cfg.max_steps if cfg.max_steps > 0 else 25
    warmup_steps = int(planned_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=planned_steps,
    )
    causal_lm = _base_causal_lm(model)
    backbone = causal_lm.model
    lm_head = causal_lm.lm_head
    candidate_ids_tensor = torch.tensor(
        candidate_token_ids, dtype=torch.long, device=device
    )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items)).tolist()
    cursor = 0
    losses = []
    diagnostic_encoded = encode_prompts([str(items.iloc[0]["prompt"])])
    diagnostic_target = torch.tensor(
        target_probabilities[[0]], dtype=torch.float32, device=device
    )
    model.eval()
    with torch.no_grad():
        ordinary_logits = model(**diagnostic_encoded).logits[0, -1].float()
        diagnostic_outputs = backbone(
            input_ids=diagnostic_encoded["input_ids"],
            attention_mask=diagnostic_encoded["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        selective_logits = lm_head(
            diagnostic_outputs.last_hidden_state[:, -1, :]
        )[0].float()
        maximum_logit_difference = float(
            (ordinary_logits - selective_logits).abs().max().detach().cpu()
        )
        diagnostic_selected_logprobs = torch.log_softmax(
            selective_logits, dim=-1
        ).index_select(dim=0, index=candidate_ids_tensor)
        diagnostic_nll = float(
            -(diagnostic_target[0] * diagnostic_selected_logprobs).sum().detach().cpu()
        )
    prompt_lengths = [
        len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        for prompt in items["prompt"]
    ]
    print(
        "[distributional_lora] preflight "
        f"padding_side={tokenizer.padding_side} "
        f"prompt_tokens={min(prompt_lengths)}-{max(prompt_lengths)} "
        f"max_seq_length={cfg.max_seq_length} "
        f"ordinary_selective_max_logit_diff={maximum_logit_difference:.8f} "
        f"first_item_nll={diagnostic_nll:.6f}",
        flush=True,
    )
    if maximum_logit_difference > 1e-4:
        raise ValueError(
            "Selective final-position logits do not match the ordinary model forward."
        )
    if diagnostic_nll > 6.0:
        raise ValueError(
            "Distributional preflight NLL is implausibly high; refusing to train."
        )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(planned_steps):
        if cursor + cfg.per_device_train_batch_size > len(order):
            order = rng.permutation(len(items)).tolist()
            cursor = 0
        batch_indices = order[cursor : cursor + cfg.per_device_train_batch_size]
        cursor += cfg.per_device_train_batch_size
        batch_prompts = items.iloc[batch_indices]["prompt"].tolist()
        encoded = encode_prompts(batch_prompts)
        targets = torch.tensor(
            target_probabilities[batch_indices],
            dtype=torch.float32,
            device=device,
        )
        outputs = backbone(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        last_indices = encoded["attention_mask"].sum(dim=1) - 1
        last_hidden = outputs.last_hidden_state[
            torch.arange(len(batch_indices), device=device), last_indices
        ]
        logits = lm_head(last_hidden).float()
        selected_logprobs = torch.log_softmax(logits, dim=-1).index_select(
            dim=1, index=candidate_ids_tensor
        )
        loss = -(targets * selected_logprobs).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        print(
            f"[distributional_lora] step={step + 1}/{planned_steps} "
            f"loss={loss_value:.6f} items={batch_indices}",
            flush=True,
        )

    effective_train_rows = (
        int(rows["count"].sum()) if "count" in rows.columns else int(len(rows))
    )
    if "participant_count" in rows.columns:
        participant_count = int(rows["participant_count"].max())
    elif "participant_key" in rows.columns:
        participant_count = int(rows["participant_key"].nunique())
    else:
        raise ValueError(
            "Distributional rows require participant_key or participant_count metadata."
        )
    metrics = {
        "objective": "empirical_21_category_soft_target_cross_entropy",
        "train_rows": effective_train_rows,
        "aggregated_training_rows": int(len(rows)),
        "train_items": int(len(items)),
        "participant_count": participant_count,
        "planned_steps": int(planned_steps),
        "train_steps": int(planned_steps),
        "item_batch_size": int(cfg.per_device_train_batch_size),
        "item_exposures": int(planned_steps * cfg.per_device_train_batch_size),
        "first_train_loss": float(losses[0]),
        "last_train_loss": float(losses[-1]),
        "mean_train_loss": float(np.mean(losses)),
        "mean_labeled_tokens_per_item": 1.0,
        "response_values": RESPONSE_VALUES,
        "adapter_path": str(output_dir),
    }
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    save_json(metrics_path, metrics)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def train_condition(args: argparse.Namespace) -> None:
    if args.condition not in CONDITIONS:
        raise ValueError("--condition is required for training.")
    corpus_path = (
        Path(args.corpus_path)
        if args.corpus_path
        else Path(args.corpus_dir) / f"rows_{args.condition}.jsonl"
    )
    rows = pd.read_json(corpus_path, lines=True)
    cfg = training_config(args)
    output_dir = ensure_dir(Path(args.output_dir) / "adapters" / args.condition)
    metrics_path = Path(args.output_dir) / f"train_metrics_{args.condition}.json"
    if args.training_objective == "distributional":
        train_distributional_lora_adapter(
            rows,
            cfg=cfg,
            output_dir=output_dir,
            metrics_path=metrics_path,
            seed=args.seed,
        )
    else:
        train_lora_adapter(
            rows,
            cfg=cfg,
            output_dir=output_dir,
            metrics_path=metrics_path,
            seed=args.seed,
        )
    (Path(args.output_dir) / f"train_config_{args.condition}.json").write_text(
        json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8"
    )


def evaluate_condition(args: argparse.Namespace) -> None:
    if args.condition not in CONDITIONS:
        raise ValueError("--condition is required for evaluation.")
    adapter_path = Path(args.output_dir) / "adapters" / args.condition
    if not (adapter_path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Missing trained adapter: {adapter_path}")

    from peft import PeftModel

    giving, items = load_giving_data(args.behavior_path, args.group_path)
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
        giving, candidate_probabilities
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
    candidate_probabilities.to_csv(
        evaluation_dir / "candidate_probabilities.csv", index=False
    )
    item_predictions.to_csv(evaluation_dir / "item_predictions.csv", index=False)
    invalid_token_probabilities.to_csv(
        evaluation_dir / "invalid_token_probabilities.csv", index=False
    )
    aggregate_invalid_tokens.to_csv(
        evaluation_dir / "aggregate_invalid_tokens.csv", index=False
    )
    metrics.to_csv(evaluation_dir / "model_metrics.csv", index=False)
    human_item_comparison.to_csv(
        evaluation_dir / "human_item_comparison.csv", index=False
    )
    generations.to_csv(evaluation_dir / "greedy_generations.csv", index=False)
    metadata = {
        "model_name": args.model_name,
        "adapter_path": str(adapter_path),
        "condition": args.condition,
        "evaluation_protocol": "source-faithful Giving prompt ending with <<; numeric response-token likelihood",
        "primary_sample": "Wave 2",
        "response_values": RESPONSE_VALUES,
    }
    (evaluation_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def _candidate_matrix(path: Path) -> np.ndarray:
    candidates = pd.read_csv(path)
    if "condition" in candidates:
        candidates = candidates.drop(columns="condition")
    return (
        candidates.pivot(
            index="GroupNo",
            columns="response_value",
            values="conditional_probability",
        )
        .reindex(index=range(1, 26), columns=RESPONSE_VALUES)
        .to_numpy(dtype=float)
    )


def summarize_and_bootstrap(args: argparse.Namespace) -> None:
    run_dir = Path(args.output_dir)
    base_result_dir = Path(args.base_result_dir)
    prediction_paths = {
        "base": base_result_dir / "candidate_probabilities.csv",
        **{
            condition: run_dir
            / "evaluations"
            / condition
            / "candidate_probabilities.csv"
            for condition in CONDITIONS
        },
    }
    missing = [str(path) for path in prediction_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing evaluation files: " + ", ".join(missing))
    probabilities = {
        condition: _candidate_matrix(path)
        for condition, path in prediction_paths.items()
    }

    metric_parts = []
    base_metrics = pd.read_csv(base_result_dir / "model_metrics.csv")
    base_metrics.insert(1, "condition", "base")
    metric_parts.append(base_metrics)
    for condition in CONDITIONS:
        metric_parts.append(
            pd.read_csv(
                run_dir / "evaluations" / condition / "model_metrics.csv"
            )
        )
    all_metrics = pd.concat(metric_parts, ignore_index=True)
    all_metrics.to_csv(run_dir / "all_model_metrics.csv", index=False)
    wave2 = all_metrics.loc[all_metrics["sample"].eq("Wave 2")].copy()
    wide = wave2.pivot(
        index="condition", columns="country", values="conditional_kl_nats"
    ).rename(
        columns={
            "United States": "conditional_kl_us",
            "China": "conditional_kl_china",
        }
    )
    wide["alignment_gap_china_minus_us"] = (
        wide["conditional_kl_china"] - wide["conditional_kl_us"]
    )
    wide = wide.reset_index()
    wide.to_csv(run_dir / "culture_alignment_summary.csv", index=False)

    giving, _ = load_giving_data(args.behavior_path, args.group_path)
    wave2_rows = giving.loc[giving["Dataset"].eq(2)].copy()
    rng = np.random.default_rng(args.seed + 90_000)
    bootstrap_rows = []
    for bootstrap_index in range(args.bootstrap_resamples):
        culture_kl: dict[tuple[str, str], float] = {}
        for country in COUNTRIES:
            country_rows = wave2_rows.loc[wave2_rows["Country"].eq(country)]
            participant_ids = (
                country_rows[["Nation", "SubNo"]].drop_duplicates().reset_index(drop=True)
            )
            sampled = participant_ids.iloc[
                rng.integers(0, len(participant_ids), size=len(participant_ids))
            ].copy()
            sampled["bootstrap_occurrence"] = np.arange(len(sampled))
            sampled_rows = sampled.merge(
                country_rows[["Nation", "SubNo", "GroupNo", "TokenChoice"]],
                on=["Nation", "SubNo"],
                how="left",
                validate="many_to_many",
            )
            counts = (
                sampled_rows.groupby(["GroupNo", "TokenChoice"])
                .size()
                .unstack(fill_value=0)
                .reindex(index=range(1, 26), columns=RESPONSE_VALUES, fill_value=0)
            )
            q = counts.div(counts.sum(axis=1), axis=0).to_numpy(dtype=float)
            log_q = np.zeros_like(q)
            positive = q > 0
            log_q[positive] = np.log(q[positive])
            entropy = float(-(q * log_q).sum(axis=1).mean())
            for condition, p in probabilities.items():
                conditional_nll = float(
                    -(q * np.log(np.clip(p, np.finfo(float).tiny, 1.0)))
                    .sum(axis=1)
                    .mean()
                )
                culture_kl[(condition, country)] = conditional_nll - entropy
        gaps = {}
        for condition in probabilities:
            for country in COUNTRIES:
                bootstrap_rows.append(
                    {
                        "bootstrap": bootstrap_index,
                        "estimand": "conditional_kl",
                        "condition": condition,
                        "country": country,
                        "value": culture_kl[(condition, country)],
                    }
                )
                if condition != "base":
                    bootstrap_rows.append(
                        {
                            "bootstrap": bootstrap_index,
                            "estimand": "conditional_kl_change_vs_base",
                            "condition": condition,
                            "country": country,
                            "value": (
                                culture_kl[(condition, country)]
                                - culture_kl[("base", country)]
                            ),
                        }
                    )
        for condition in probabilities:
            gap = (
                culture_kl[(condition, "China")]
                - culture_kl[(condition, "United States")]
            )
            gaps[condition] = gap
            bootstrap_rows.append(
                {
                    "bootstrap": bootstrap_index,
                    "estimand": "alignment_gap_china_minus_us",
                    "condition": condition,
                    "country": "China - United States",
                    "value": gap,
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
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(run_dir / "participant_bootstrap_samples.csv", index=False)
    bootstrap_summary = (
        bootstrap.groupby(["estimand", "condition", "country"])["value"]
        .agg(
            estimate="mean",
            ci_low=lambda values: values.quantile(0.025),
            ci_high=lambda values: values.quantile(0.975),
        )
        .reset_index()
    )
    bootstrap_summary.to_csv(
        run_dir / "participant_bootstrap_summary.csv", index=False
    )

    observed_gaps = wide.set_index("condition")[
        "alignment_gap_china_minus_us"
    ]
    observed = {
        "culture_specialization_interaction": float(
            observed_gaps.loc["us"] - observed_gaps.loc["china"]
        ),
        "us_adapter_alignment_gap": float(observed_gaps.loc["us"]),
        "china_adapter_alignment_gap": float(observed_gaps.loc["china"]),
        "pooled_adapter_alignment_gap": float(observed_gaps.loc["pooled"]),
        "base_alignment_gap": float(observed_gaps.loc["base"]),
    }
    (run_dir / "culture_effects.json").write_text(
        json.dumps(observed, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wave-1 to Wave-2 culture-specific Giving fine-tuning pilot."
    )
    parser.add_argument(
        "--stage", required=True, choices=["prepare", "train", "evaluate", "analyze"]
    )
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--behavior-path", required=True)
    parser.add_argument("--group-path", required=True)
    parser.add_argument("--corpus-dir")
    parser.add_argument("--corpus-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--base-result-dir",
        default="../model_results/dictator_giving_base/marcelbinz_Llama-3.1-Minitaur-8B",
    )
    parser.add_argument(
        "--model-name", default="marcelbinz/Llama-3.1-Minitaur-8B"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--culture-participants", type=int, default=500)
    parser.add_argument("--pooled-per-culture", type=int, default=250)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--training-objective",
        choices=["distributional", "participant_rows"],
        default="distributional",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=768)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        prepare_corpora(
            args.behavior_path,
            args.group_path,
            args.output_dir,
            args.seed,
            args.culture_participants,
            args.pooled_per_culture,
        )
    elif args.stage == "train":
        if not args.corpus_dir and not args.corpus_path:
            raise ValueError("--corpus-dir or --corpus-path is required for training.")
        train_condition(args)
    elif args.stage == "evaluate":
        evaluate_condition(args)
    else:
        summarize_and_bootstrap(args)


if __name__ == "__main__":
    main()
