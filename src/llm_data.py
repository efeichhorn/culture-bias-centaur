from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd


ANSWER_VALUE_PATTERN = re.compile(r"^<<(?P<value>A|B|[0-9]+)>>\.$")


def _answer_value_labels(tokenizer, completion: str, completion_ids: list[int]) -> list[int]:
    match = ANSWER_VALUE_PATTERN.fullmatch(completion)
    if match is None:
        raise ValueError(
            "Answer-value-only training requires a strict completion such as "
            f"'<<A>>.' or '<<50>>.', got {completion!r}."
        )
    prefix = completion[: match.start("value")]
    through_value = completion[: match.end("value")]
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    through_value_ids = tokenizer(through_value, add_special_tokens=False)["input_ids"]
    if (
        completion_ids[: len(prefix_ids)] != prefix_ids
        or completion_ids[: len(through_value_ids)] != through_value_ids
        or len(through_value_ids) <= len(prefix_ids)
    ):
        raise ValueError(
            "Tokenizer does not preserve the answer-value boundary in the strict response wrapper."
        )
    labels = [-100] * len(completion_ids)
    labels[len(prefix_ids) : len(through_value_ids)] = completion_ids[
        len(prefix_ids) : len(through_value_ids)
    ]
    return labels


def prepare_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode_prompt_completion(
    tokenizer,
    prompt: str,
    completion: str,
    max_seq_length: int,
    train_on_eos: bool = True,
    label_token_ids: set[int] | None = None,
    answer_value_only: bool = False,
) -> dict[str, list[int]]:
    # Include the model's beginning-of-text token exactly once on every
    # prompt/completion training sequence.
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if train_on_eos and tokenizer.eos_token_id is not None:
        completion_ids = completion_ids + [tokenizer.eos_token_id]

    if len(completion_ids) >= max_seq_length:
        raise ValueError("Completion is longer than max_seq_length.")
    available_prompt = max_seq_length - len(completion_ids)
    if len(prompt_ids) > available_prompt:
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        has_bos = bool(prompt_ids) and prompt_ids[0] == bos_token_id
        if has_bos and available_prompt > 0:
            tail_count = available_prompt - 1
            prompt_tail = prompt_ids[-tail_count:] if tail_count else []
            prompt_ids = [prompt_ids[0], *prompt_tail]
        else:
            prompt_ids = prompt_ids[-available_prompt:]

    input_ids = prompt_ids + completion_ids
    if answer_value_only and label_token_ids is not None:
        raise ValueError("Choose answer_value_only or label_token_ids, not both.")
    if answer_value_only:
        completion_labels = _answer_value_labels(tokenizer, completion, completion_ids)
    elif label_token_ids is None:
        completion_labels = completion_ids
    else:
        completion_labels = [
            token_id if token_id in label_token_ids else -100
            for token_id in completion_ids
        ]
        if all(label == -100 for label in completion_labels):
            raise ValueError("No completion token matched label_token_ids.")
    labels = [-100] * len(prompt_ids) + completion_labels
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class PromptCompletionDataset:
    def __init__(
        self,
        rows: pd.DataFrame,
        tokenizer,
        max_seq_length: int,
        train_on_eos: bool = True,
        answer_value_only: bool = False,
    ):
        self.examples = []
        required = {"prompt", "response_text"}
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"Priming training rows are missing required columns: {sorted(missing)}")
        for row in rows.to_dict("records"):
            if pd.isna(row["response_text"]):
                raise ValueError("Priming training row has an empty response_text.")
            self.examples.append(
                encode_prompt_completion(
                    tokenizer,
                    str(row["prompt"]),
                    str(row["response_text"]),
                    max_seq_length,
                    train_on_eos=train_on_eos,
                    answer_value_only=answer_value_only,
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]


@dataclass
class CompletionOnlyCollator:
    tokenizer: object
    pad_to_length: int | None = None

    def __call__(self, features: list[dict[str, list[int]]]):
        import torch

        max_len = self.pad_to_length or max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        pad_id = self.tokenizer.pad_token_id
        for feature in features:
            if len(feature["input_ids"]) > max_len:
                raise ValueError("Feature is longer than collator pad length.")
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
