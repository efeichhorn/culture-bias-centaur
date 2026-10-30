#!/usr/bin/env python3
"""Giving prompts, response loading, and candidate scoring helpers."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# Direct script execution places scripts/, rather than the repository root, on
# sys.path. Add the root explicitly so the shared src package is importable in
# both local and containerized Slurm runs.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm_generate import generate_completions
from scripts.repair_wang_behavior_columns import ensure_not_mislabeled_source


RESPONSE_VALUES = list(range(21))
RESPONSE_PREFIX = "<<"
TOP_INVALID_TOKENS_PER_ITEM = 50
TOP_AGGREGATE_INVALID_TOKENS = 200
VALID_GENERATED_TOKEN = re.compile(r"^\s*(?P<value>20|1[0-9]|[0-9])\s*$")


def build_prompt(item: str) -> str:
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
        "- One of you is randomly assigned to start with 20 tokens on each decision. "
        "Throughout this study, that person is YOU. Each token is worth $1.5.\n"
        "- You start with 20 tokens. The other person knows that you start with 20 tokens.\n"
        "- You decide how to split the 20 tokens with the other person. If you decide to "
        "share x tokens with him, it means you leave yourself 20-x tokens.\n"
        "- The other person cannot affect the outcome.\n\n"
        "You can split the initial 20 tokens however you wish.\n\n"
        "Although these decisions are hypothetical, please answer as honestly as you can, "
        "and please consider each decision individually. Because this is about first "
        "impressions, however, feel free to answer fairly quickly.\n\n"
        "Reminder: You decide how to split 20 tokens with the other person. The other "
        "person knows that you start with 20 tokens, but that person cannot affect the "
        "outcome of your decision. Indicate how many you would give to the other person.\n\n"
        "For this decision, you are paired with a random person.\n"
        "Here is one piece of information about them:\n\n"
        f"{item}\n\n"
        "How many of the 20 tokens would you give to this person?\n\n"
        f"Available dropdown responses: {dropdown_choices}.\n"
        "For this natural-language transcription, responses are enclosed in double angle "
        f"brackets. The valid encoded responses are {encoded_choices}.\n"
        "You enter <<"
    )


def percent_to_token_choice(values: pd.Series | np.ndarray) -> np.ndarray:
    """Map released 0--100 allocations to the source questionnaire's 0--20 scale."""
    raw = np.asarray(values, dtype=float)
    return np.clip(np.floor(raw / 5.0 + 0.5), 0, 20).astype(int)


def stable_softmax(values: list[float] | np.ndarray) -> np.ndarray:
    # Stabilization is in-place, so preserve the full-vocabulary log
    # probabilities retained by the caller for the Centaur-style NLL.
    scores = np.array(values, dtype=np.float64, copy=True)
    scores -= np.max(scores)
    probabilities = np.exp(scores)
    return probabilities / probabilities.sum()


def candidate_probabilities_from_logprobs(
    logprobs: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    logprobs_array = np.asarray(logprobs, dtype=np.float64)
    full_vocab_probabilities = np.exp(logprobs_array)
    valid_response_mass = float(full_vocab_probabilities.sum())
    conditional_probabilities = stable_softmax(logprobs_array)
    return full_vocab_probabilities, conditional_probabilities, valid_response_mass


def parse_generated_token(text: str) -> int | None:
    match = VALID_GENERATED_TOKEN.fullmatch(text)
    return int(match.group("value")) if match else None


def score_single_token_candidates(
    model,
    tokenizer,
    prompt: str,
    response_values: list[int],
    device,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Return full-vocabulary log probabilities at the response position.

    Psych-101 supplies the opening ``<<`` delimiter as context and applies loss only
    to the human response token(s). All source response values are single tokens for
    the evaluated Llama tokenizers, so one forward pass yields the exact Centaur-style
    likelihood for every possible response.
    """
    import torch

    candidate_token_ids: list[int] = []
    for value in response_values:
        token_ids = tokenizer(str(value), add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise ValueError(
                f"Response {value!r} must be one token for this evaluator; got {token_ids}."
            )
        candidate_token_ids.append(int(token_ids[0]))
    if len(set(candidate_token_ids)) != len(candidate_token_ids):
        raise ValueError("Distinct response values mapped to duplicate tokenizer tokens.")

    encoded = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        next_token_logits = model(**encoded).logits[0, -1].float()
        next_token_logprobs = torch.log_softmax(next_token_logits, dim=-1)
        candidate_logprobs = next_token_logprobs[candidate_token_ids]
    return (
        candidate_token_ids,
        candidate_logprobs.detach().cpu().numpy().astype(np.float64),
        next_token_logprobs.exp().detach().cpu().numpy().astype(np.float64),
    )


def token_diagnostic_fields(tokenizer, token_id: int) -> dict[str, object]:
    return {
        "token_id": int(token_id),
        "tokenizer_piece": str(tokenizer.convert_ids_to_tokens(int(token_id))),
        "decoded_text": tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    }


def load_giving_data(
    behavior_path: str | Path,
    group_path: str | Path,
    *,
    allow_legacy_mislabeled: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Giving data; legacy opt-in is only for reproducing invalid archives."""
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
    giving = behavior.loc[behavior["Dataset"].isin([1, 2])].copy()
    giving["Country"] = giving["Nation"].map({0: "United States", 1: "China"})
    giving["TokenChoice"] = percent_to_token_choice(giving["Token"])
    giving["TokenChoicePercent"] = giving["TokenChoice"] * 5
    giving = giving.merge(
        items[["GroupNo", "item", "item_chinese"]],
        on="GroupNo",
        validate="many_to_one",
    )
    if giving["Country"].isna().any():
        raise ValueError("Unexpected Nation code in behavior data.")
    response_counts = giving.groupby(["Dataset", "Nation", "SubNo"]).size()
    if not response_counts.eq(25).all():
        raise ValueError("Every retained participant must have exactly 25 Giving decisions.")
    return giving, items.sort_values("GroupNo").reset_index(drop=True)


def score_model(
    model,
    tokenizer,
    device,
    items: pd.DataFrame,
    model_name: str,
    prompt_builder: Callable[[str], str] = build_prompt,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_rows: list[dict[str, object]] = []
    item_rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []
    vocabulary_probability_sum: np.ndarray | None = None
    valid_token_ids: list[int] | None = None
    for index, row in items.iterrows():
        prompt = prompt_builder(str(row["item"]))
        token_ids, logprobs, vocabulary_probabilities = score_single_token_candidates(
            model, tokenizer, prompt, RESPONSE_VALUES, device
        )
        if valid_token_ids is None:
            valid_token_ids = token_ids
        elif token_ids != valid_token_ids:
            raise ValueError("Valid response token IDs changed across items.")
        if vocabulary_probability_sum is None:
            vocabulary_probability_sum = np.zeros_like(vocabulary_probabilities)
        vocabulary_probability_sum += vocabulary_probabilities

        invalid_probabilities = vocabulary_probabilities.copy()
        invalid_probabilities[token_ids] = 0.0
        invalid_mass = float(invalid_probabilities.sum())
        invalid_order = np.argsort(invalid_probabilities)[::-1][
            :TOP_INVALID_TOKENS_PER_ITEM
        ]
        for rank, token_id in enumerate(invalid_order, start=1):
            probability = float(invalid_probabilities[token_id])
            invalid_rows.append(
                {
                    "model_name": model_name,
                    "GroupNo": int(row["GroupNo"]),
                    "item": row["item"],
                    "rank_within_item": rank,
                    **token_diagnostic_fields(tokenizer, int(token_id)),
                    "full_vocab_probability": probability,
                    "fraction_of_invalid_mass": probability / invalid_mass,
                    "item_invalid_mass": invalid_mass,
                }
            )
        full_vocab_probabilities, conditional_probabilities, valid_response_mass = (
            candidate_probabilities_from_logprobs(logprobs)
        )
        expected_tokens = float(
            np.dot(conditional_probabilities, np.asarray(RESPONSE_VALUES, dtype=float))
        )
        mode_index = int(np.argmax(conditional_probabilities))
        conditional_entropy = float(
            -np.sum(conditional_probabilities * np.log(conditional_probabilities))
        )
        item_rows.append(
            {
                "model_name": model_name,
                "GroupNo": int(row["GroupNo"]),
                "item": row["item"],
                "item_chinese": row["item_chinese"],
                "expected_tokens": expected_tokens,
                "expected_percentage": expected_tokens * 5.0,
                "modal_tokens": RESPONSE_VALUES[mode_index],
                "modal_percentage": RESPONSE_VALUES[mode_index] * 5,
                "conditional_modal_probability": float(conditional_probabilities[mode_index]),
                "conditional_entropy_nats": conditional_entropy,
                "valid_response_mass": valid_response_mass,
                "valid_mass_penalty_nats": float(-np.log(valid_response_mass)),
                "prompt": prompt,
                "response_prefix_in_prompt": RESPONSE_PREFIX,
            }
        )
        for value, token_id, logprob, full_probability, conditional_probability in zip(
            RESPONSE_VALUES,
            token_ids,
            logprobs,
            full_vocab_probabilities,
            conditional_probabilities,
        ):
            candidate_rows.append(
                {
                    "model_name": model_name,
                    "GroupNo": int(row["GroupNo"]),
                    "item": row["item"],
                    "response_value": value,
                    "response_percentage": value * 5,
                    "response_text": f"<<{value}>>",
                    "scored_response_text": str(value),
                    "scored_token_ids": json.dumps([token_id]),
                    "scored_token_count": 1,
                    "full_vocab_log_probability": float(logprob),
                    "full_vocab_probability": float(full_probability),
                    "conditional_probability": float(conditional_probability),
                    "valid_response_mass": valid_response_mass,
                }
            )
        print(f"[dictator-base] item={index + 1}/{len(items)} {row['item']}", flush=True)
    if vocabulary_probability_sum is None or valid_token_ids is None:
        raise ValueError("No items were scored.")
    mean_vocabulary_probabilities = vocabulary_probability_sum / len(items)
    mean_vocabulary_probabilities[valid_token_ids] = 0.0
    mean_invalid_mass = float(mean_vocabulary_probabilities.sum())
    aggregate_order = np.argsort(mean_vocabulary_probabilities)[::-1][
        :TOP_AGGREGATE_INVALID_TOKENS
    ]
    aggregate_rows = []
    for rank, token_id in enumerate(aggregate_order, start=1):
        probability = float(mean_vocabulary_probabilities[token_id])
        aggregate_rows.append(
            {
                "model_name": model_name,
                "aggregate_rank": rank,
                **token_diagnostic_fields(tokenizer, int(token_id)),
                "mean_full_vocab_probability": probability,
                "fraction_of_mean_invalid_mass": probability / mean_invalid_mass,
                "mean_invalid_mass": mean_invalid_mass,
            }
        )
    return (
        pd.DataFrame(candidate_rows),
        pd.DataFrame(item_rows),
        pd.DataFrame(invalid_rows),
        pd.DataFrame(aggregate_rows),
    )


def greedy_generations(
    model,
    tokenizer,
    device,
    item_predictions: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    prompts = item_predictions["prompt"].tolist()
    texts: list[str] = []
    for start in range(0, len(prompts), batch_size):
        texts.extend(
            generate_completions(
                model,
                tokenizer,
                prompts[start : start + batch_size],
                device,
                max_new_tokens=1,
                temperature=0.0,
                top_p=1.0,
            )
    )
    output = item_predictions[["model_name", "GroupNo", "item"]].copy()
    output["generated_token"] = texts
    output["parsed_value"] = [parse_generated_token(text) for text in texts]
    output["valid_response_token"] = output["parsed_value"].notna()
    return output


def _human_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    counts = (
        frame.groupby(["GroupNo", "TokenChoice"])
        .size()
        .rename("count")
        .reset_index()
    )
    grid = pd.MultiIndex.from_product(
        [sorted(frame["GroupNo"].unique()), RESPONSE_VALUES],
        names=["GroupNo", "TokenChoice"],
    ).to_frame(index=False)
    counts = grid.merge(counts, on=["GroupNo", "TokenChoice"], how="left").fillna({"count": 0})
    counts["probability"] = counts["count"] / counts.groupby("GroupNo")["count"].transform("sum")
    return counts


def evaluate_against_humans(
    giving: pd.DataFrame,
    candidate_probabilities: pd.DataFrame,
    wave_datasets: tuple[int, int] = (1, 2),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    conditional_probabilities = candidate_probabilities.pivot(
        index="GroupNo", columns="response_value", values="conditional_probability"
    ).reindex(columns=RESPONSE_VALUES)
    full_vocab_log_probabilities = candidate_probabilities.pivot(
        index="GroupNo", columns="response_value", values="full_vocab_log_probability"
    ).reindex(columns=RESPONSE_VALUES)
    valid_response_mass = (
        candidate_probabilities.groupby("GroupNo")["valid_response_mass"].first()
        .reindex(conditional_probabilities.index)
    )
    model_expected = conditional_probabilities.mul(RESPONSE_VALUES, axis=1).sum(axis=1)
    model_mode = conditional_probabilities.idxmax(axis=1)

    wave_1_dataset, wave_2_dataset = wave_datasets
    samples = [
        ("Wave 1", giving["Dataset"].eq(wave_1_dataset)),
        ("Wave 2", giving["Dataset"].eq(wave_2_dataset)),
        ("Combined", giving["Dataset"].isin(wave_datasets)),
    ]
    metric_rows: list[dict[str, object]] = []
    human_item_rows: list[dict[str, object]] = []
    model_name = str(candidate_probabilities["model_name"].iloc[0])
    for sample_name, sample_mask in samples:
        sample = giving.loc[sample_mask]
        for country in ["United States", "China"]:
            frame = sample.loc[sample["Country"].eq(country)]
            distribution = _human_distribution(frame)
            q = distribution.pivot(
                index="GroupNo", columns="TokenChoice", values="probability"
            ).reindex(index=conditional_probabilities.index, columns=RESPONSE_VALUES, fill_value=0.0)
            q_values = q.to_numpy(dtype=float)
            p_values = conditional_probabilities.to_numpy(dtype=float)
            full_logp_values = full_vocab_log_probabilities.to_numpy(dtype=float)
            safe_p_values = np.clip(p_values, np.finfo(np.float64).tiny, 1.0)
            log_q_values = np.zeros_like(q_values)
            positive_q = q_values > 0
            log_q_values[positive_q] = np.log(q_values[positive_q])
            centaur_nll_by_item = -(q_values * full_logp_values).sum(axis=1)
            conditional_nll_by_item = -(q_values * np.log(safe_p_values)).sum(axis=1)
            entropy_by_item = -(q_values * log_q_values).sum(axis=1)
            conditional_kl_by_item = conditional_nll_by_item - entropy_by_item
            mass_penalty_by_item = -np.log(valid_response_mass.to_numpy(dtype=float))
            centaur_excess_by_item = centaur_nll_by_item - entropy_by_item
            if not np.allclose(
                centaur_excess_by_item,
                conditional_kl_by_item + mass_penalty_by_item,
                atol=1e-6,
            ):
                raise ValueError("Centaur NLL decomposition check failed.")
            human_expected = q.mul(RESPONSE_VALUES, axis=1).sum(axis=1)
            human_mode = q.idxmax(axis=1)
            error = model_expected - human_expected
            cdf_distance = (
                np.abs(np.cumsum(p_values, axis=1)[:, :-1] - np.cumsum(q_values, axis=1)[:, :-1])
                .sum(axis=1)
            )
            participant_count = frame[["Dataset", "Nation", "SubNo"]].drop_duplicates().shape[0]
            metric_rows.append(
                {
                    "model_name": model_name,
                    "sample": sample_name,
                    "country": country,
                    "participants": participant_count,
                    "items": len(q),
                    "centaur_nll_nats": float(centaur_nll_by_item.mean()),
                    "human_entropy_nats": float(entropy_by_item.mean()),
                    "centaur_excess_nll_nats": float(centaur_excess_by_item.mean()),
                    "conditional_nll_nats": float(conditional_nll_by_item.mean()),
                    "conditional_kl_nats": float(conditional_kl_by_item.mean()),
                    "valid_mass_penalty_nats": float(mass_penalty_by_item.mean()),
                    "mean_valid_response_mass": float(valid_response_mass.mean()),
                    "ordinal_wasserstein_tokens": float(cdf_distance.mean()),
                    "human_mean_tokens": float(human_expected.mean()),
                    "model_mean_tokens": float(model_expected.mean()),
                    "human_mean_percentage": float(human_expected.mean() * 5.0),
                    "model_mean_percentage": float(model_expected.mean() * 5.0),
                    "mean_error_tokens": float(error.mean()),
                    "item_mean_mae_tokens": float(error.abs().mean()),
                    "item_mean_rmse_tokens": float(np.sqrt(np.mean(np.square(error)))),
                    "item_mean_correlation": float(model_expected.corr(human_expected)),
                    "item_mode_accuracy": float((model_mode == human_mode).mean()),
                }
            )
            for group_no in q.index:
                human_item_rows.append(
                    {
                        "model_name": model_name,
                        "sample": sample_name,
                        "country": country,
                        "GroupNo": int(group_no),
                        "human_expected_tokens": float(human_expected.loc[group_no]),
                        "human_modal_tokens": int(human_mode.loc[group_no]),
                        "model_expected_tokens": float(model_expected.loc[group_no]),
                        "model_modal_tokens": int(model_mode.loc[group_no]),
                        "expected_error_tokens": float(error.loc[group_no]),
                        "centaur_nll_nats": float(centaur_nll_by_item[list(q.index).index(group_no)]),
                        "conditional_kl_nats": float(conditional_kl_by_item[list(q.index).index(group_no)]),
                        "valid_response_mass": float(valid_response_mass.loc[group_no]),
                        "ordinal_wasserstein_tokens": float(cdf_distance[list(q.index).index(group_no)]),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(human_item_rows)
