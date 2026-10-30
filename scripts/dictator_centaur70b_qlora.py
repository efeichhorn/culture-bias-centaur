#!/usr/bin/env python3
"""Centaur-70B culture QLoRA experiment for the US--China dictator task.

The frozen model is the released 4-bit Llama base plus the released Centaur
adapter. A second, zero-initialized LoRA adapter is the only trainable part.
By default, training uses each item's empirical response distribution and
computes loss at the numeric response position. Participant-row training
is also available.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dictator_culture_finetune import training_prompt
from scripts.dictator_culture_mixed_split import (
    COUNTRY_TO_CONDITION,
    PARTICIPANT_KEYS,
    balanced_participant_split,
    behavior_source_provenance,
    candidate_matrix,
    conditional_kl,
    heldout_rows,
    human_distribution,
    require_corpus_behavior_source,
    select_training_conditions,
)
from scripts.evaluate_dictator_giving_base import (
    RESPONSE_VALUES,
    candidate_probabilities_from_logprobs,
    evaluate_against_humans,
    load_giving_data,
    score_model,
    score_single_token_candidates,
)
from src.llm_data import CompletionOnlyCollator, PromptCompletionDataset, prepare_tokenizer
from src.llm_train import (
    LengthGroupedBatchSampler,
    _base_causal_lm,
    completion_only_causal_lm_loss,
)
from src.utils import ensure_dir, set_global_seed


TRAIN_CONDITIONS = ("us", "china", "pooled")
EVALUATION_CONDITIONS = ("base", *TRAIN_CONDITIONS)
COUNTRIES = ("United States", "China")
CENTAUR_ADAPTER_NAME = "centaur"
CULTURE_ADAPTER_NAME = "culture"


def aggregate_item_response_distributions(rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated prompts to empirical response probabilities.

    Because training loss is applied only to the numeric response token, this
    count-weighted categorical objective is exactly the participant-row NLL,
    without repeating an identical prefix hundreds of times.
    """
    required = {"GroupNo", "item", "prompt", "response_value"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Training rows are missing columns: {sorted(missing)}")
    counts = (
        rows.groupby(["GroupNo", "item", "prompt", "response_value"])
        .size()
        .unstack("response_value", fill_value=0)
        .reindex(columns=RESPONSE_VALUES, fill_value=0)
        .sort_index()
    )
    totals = counts.sum(axis=1)
    if bool(totals.le(0).any()):
        raise ValueError("Every aggregated item must have at least one response.")
    probabilities = counts.div(totals, axis=0)
    aggregated = probabilities.reset_index()[["GroupNo", "item", "prompt"]]
    aggregated["participant_count"] = totals.to_numpy(dtype=np.int64)
    aggregated["target_probabilities"] = probabilities.to_numpy().tolist()
    return aggregated


def response_token_ids(tokenizer) -> list[int]:
    token_ids = []
    for value in RESPONSE_VALUES:
        encoded = tokenizer(str(value), add_special_tokens=False)["input_ids"]
        if len(encoded) != 1:
            raise ValueError(f"Response {value} is not one tokenizer token: {encoded}")
        token_ids.append(int(encoded[0]))
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("Distinct responses map to duplicate tokenizer tokens.")
    return token_ids


class ItemDistributionDataset:
    def __init__(
        self,
        rows: pd.DataFrame,
        tokenizer,
        max_seq_length: int,
        repeats: int,
        seed: int,
    ):
        if repeats <= 0:
            raise ValueError("Distribution training repeats must be positive.")
        base_examples = []
        for row in rows.to_dict("records"):
            prefix = str(row["prompt"]) + "<<"
            input_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
            if len(input_ids) > max_seq_length:
                bos_token_id = getattr(tokenizer, "bos_token_id", None)
                has_bos = bool(input_ids) and input_ids[0] == bos_token_id
                if has_bos:
                    input_ids = [input_ids[0], *input_ids[-(max_seq_length - 1) :]]
                else:
                    input_ids = input_ids[-max_seq_length:]
            base_examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "target_probabilities": row["target_probabilities"],
                    "participant_count": float(row["participant_count"]),
                }
            )
        rng = np.random.default_rng(seed)
        self.items_per_repeat = len(base_examples)
        self.repeats = repeats
        self.examples = []
        for _ in range(repeats):
            order = rng.permutation(len(base_examples))
            self.examples.extend(base_examples[index] for index in order)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.examples[index]


class FullItemSetBatchSampler:
    """Yield one shuffled, complete set of item contexts per optimizer update."""

    def __init__(self, items_per_repeat: int, repeats: int, seed: int):
        self.items_per_repeat = items_per_repeat
        self.repeats = repeats
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        repeat_order = rng.permutation(self.repeats)
        for repeat_index in repeat_order:
            start = int(repeat_index) * self.items_per_repeat
            yield list(range(start, start + self.items_per_repeat))

    def __len__(self) -> int:
        return self.repeats


class ItemDistributionCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: list[dict[str, object]]):
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_length)
            attention_mask.append(feature["attention_mask"] + [0] * pad_length)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "target_probabilities": torch.tensor(
                [feature["target_probabilities"] for feature in features],
                dtype=torch.float32,
            ),
            "participant_count": torch.tensor(
                [feature["participant_count"] for feature in features],
                dtype=torch.float32,
            ),
        }


def item_distribution_causal_lm_loss(model, batch, candidate_token_ids):
    import torch
    import torch.nn.functional as F

    causal_lm = _base_causal_lm(model)
    backbone = getattr(causal_lm, "model", None)
    lm_head = getattr(causal_lm, "lm_head", None)
    if backbone is None or lm_head is None:
        raise RuntimeError("Model does not expose its causal backbone and LM head.")
    outputs = backbone(
        input_ids=batch["input_ids"],
        # Padding is strictly to the right and causal states at the selected
        # final real token cannot attend to it. Omitting the 2-D mask keeps the
        # fast fused causal SDPA path used by participant-row training.
        attention_mask=None,
        use_cache=False,
        return_dict=True,
    )
    last_indices = batch["attention_mask"].sum(dim=1) - 1
    row_indices = torch.arange(outputs.last_hidden_state.shape[0], device=last_indices.device)
    final_hidden = outputs.last_hidden_state[row_indices, last_indices]
    logits = lm_head(final_hidden).float()
    log_probabilities = F.log_softmax(logits, dim=-1)
    choice_log_probabilities = log_probabilities.index_select(
        dim=-1, index=candidate_token_ids
    )
    item_losses = -(
        batch["target_probabilities"] * choice_log_probabilities
    ).sum(dim=-1)
    weights = batch["participant_count"]
    return (item_losses * weights).sum() / weights.sum()


def package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "accelerate": package_version("accelerate"),
        "peft": package_version("peft"),
        "bitsandbytes": package_version("bitsandbytes"),
        "safetensors": package_version("safetensors"),
    }


def validate_qlora_runtime() -> None:
    """Fail early when the Viper image lacks the required modern PEFT/BNB APIs."""
    import inspect

    import bitsandbytes  # noqa: F401
    import torch
    from peft import LoraConfig, PeftModel
    from transformers import BitsAndBytesConfig  # noqa: F401

    if not torch.cuda.is_available():
        raise RuntimeError("No ROCm/CUDA device is visible to PyTorch.")
    if "use_rslora" not in inspect.signature(LoraConfig).parameters:
        raise RuntimeError("PEFT is too old: LoraConfig lacks use_rslora.")
    if "selected_adapters" not in inspect.signature(PeftModel.save_pretrained).parameters:
        raise RuntimeError(
            "PEFT is too old: save_pretrained lacks selected_adapters."
        )


def build_participant_rows(
    giving: pd.DataFrame,
    condition_manifest: pd.DataFrame,
    condition: str,
    seed: int,
) -> pd.DataFrame:
    if condition not in TRAIN_CONDITIONS:
        raise ValueError(f"Unknown training condition: {condition}")
    selected = condition_manifest.loc[
        condition_manifest["condition"].eq(condition), PARTICIPANT_KEYS
    ].drop_duplicates()
    rows = giving.merge(
        selected,
        on=PARTICIPANT_KEYS,
        how="inner",
        validate="many_to_one",
    ).copy()
    participant_trials = rows.groupby(PARTICIPANT_KEYS).size()
    if not participant_trials.eq(25).all():
        raise ValueError("Every selected participant must contribute all 25 Giving items.")
    rows["response_value"] = rows["TokenChoice"].astype(int)
    rows["prompt"] = rows["item"].map(training_prompt)
    rows["response_text"] = rows["response_value"].map(lambda value: f"<<{value}>>.")
    rows["participant_key"] = (
        rows["Dataset"].astype(str)
        + ":"
        + rows["Nation"].astype(str)
        + ":"
        + rows["SubNo"].astype(str)
    )
    rows["condition"] = condition
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


def prepare_corpora(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    giving, _ = load_giving_data(args.behavior_path, args.group_path)
    split_manifest = balanced_participant_split(
        giving,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    all_conditions = select_training_conditions(split_manifest, seed=args.seed)
    condition_manifest = all_conditions.loc[
        all_conditions["condition"].isin(TRAIN_CONDITIONS)
    ].copy()
    split_manifest.to_csv(output_dir / "participant_split_manifest.csv", index=False)
    condition_manifest.to_csv(output_dir / "training_participant_manifest.csv", index=False)

    corpus_summary = []
    for condition in TRAIN_CONDITIONS:
        rows = build_participant_rows(giving, condition_manifest, condition, args.seed)
        rows.to_json(
            output_dir / f"rows_{condition}.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )
        for country, country_rows in rows.groupby("Country"):
            corpus_summary.append(
                {
                    "condition": condition,
                    "country": country,
                    "participants": int(country_rows["participant_key"].nunique()),
                    "responses": int(len(country_rows)),
                    "mean_tokens": float(country_rows["response_value"].mean()),
                }
            )
    summary = pd.DataFrame(corpus_summary)
    summary.to_csv(output_dir / "corpus_manifest.csv", index=False)
    totals = summary.groupby("condition")["responses"].sum()
    if totals.nunique() != 1:
        raise ValueError(f"Primary conditions do not have equal response budgets: {totals}")

    split_counts = (
        split_manifest.groupby(["Dataset", "Country", "split"])
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
        "stratification": "wave x culture",
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "test_fraction": 1.0 - args.train_fraction - args.validation_fraction,
        "conditions": list(TRAIN_CONDITIONS),
        "equal_response_budget": int(totals.iloc[0]),
        "pooled_composition": "50% United States, 50% China",
        "training_rows": "one row per human Giving response",
        "training_prompt_ending": "You enter ",
        "training_completion": "<<value>>.",
        "loss_mask": "numeric response value only",
        "country_label_in_prompt": False,
        "taking_used_for_training_or_selection": False,
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )


def quantization_config():
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_tokenizer(base_model_name: str):
    from transformers import AutoTokenizer

    return prepare_tokenizer(
        AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    )


def load_4bit_base(args: argparse.Namespace, for_training: bool):
    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("Centaur-70B QLoRA requires a visible ROCm/CUDA GPU.")
    kwargs = {
        "quantization_config": quantization_config(),
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
        "dtype": torch.bfloat16,
    }
    if args.attn_implementation != "auto":
        kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **kwargs)
    if not bool(getattr(model, "is_loaded_in_4bit", False)):
        raise RuntimeError("The frozen Llama base was not loaded in 4-bit mode.")
    if hasattr(model, "config"):
        model.config.use_cache = False
    if for_training:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    return model


def activate_adapters(model, adapter_names: list[str]) -> None:
    """Activate multiple LoRA adapters with PEFT's tuner-level API."""
    if len(adapter_names) == 1:
        model.set_adapter(adapter_names[0])
    else:
        setter = getattr(getattr(model, "base_model", None), "set_adapter", None)
        if setter is None:
            raise RuntimeError(
                "This PEFT version cannot compose the frozen Centaur and culture adapters."
            )
        setter(adapter_names)
    raw_active = getattr(model, "active_adapters", [])
    active = [raw_active] if isinstance(raw_active, str) else list(raw_active)
    if set(active) != set(adapter_names):
        raise RuntimeError(
            f"Adapter activation failed: requested={adapter_names}, active={active}"
        )


def load_frozen_centaur(args: argparse.Namespace, for_training: bool):
    from peft import PeftModel

    base = load_4bit_base(args, for_training=for_training)
    model = PeftModel.from_pretrained(
        base,
        args.centaur_adapter,
        adapter_name=CENTAUR_ADAPTER_NAME,
        is_trainable=False,
    )
    activate_adapters(model, [CENTAUR_ADAPTER_NAME])
    return model


def culture_lora_config(args: argparse.Namespace):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_rslora=args.use_rslora,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights=True,
    )


def add_trainable_culture_adapter(model, args: argparse.Namespace) -> list[str]:
    model.add_adapter(CULTURE_ADAPTER_NAME, culture_lora_config(args))
    activate_adapters(model, [CENTAUR_ADAPTER_NAME, CULTURE_ADAPTER_NAME])
    trainable_names = []
    for name, parameter in model.named_parameters():
        trainable = f".{CULTURE_ADAPTER_NAME}." in name and "lora_" in name
        parameter.requires_grad = trainable
        if trainable:
            trainable_names.append(name)
    if not trainable_names:
        raise RuntimeError("No culture-adapter parameters were made trainable.")
    leaked = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and f".{CULTURE_ADAPTER_NAME}." not in name
    ]
    if leaked:
        raise RuntimeError(f"Non-culture parameters are trainable: {leaked[:10]}")
    return trainable_names


def resolve_saved_adapter_dir(path: str | Path) -> Path:
    path = Path(path)
    candidates = [path, path / CULTURE_ADAPTER_NAME]
    for candidate in candidates:
        if (candidate / "adapter_config.json").exists() and (
            (candidate / "adapter_model.safetensors").exists()
            or (candidate / "adapter_model.bin").exists()
        ):
            return candidate
    raise FileNotFoundError(f"Could not locate a saved culture adapter under {path}")


def save_culture_adapter(model, path: str | Path) -> Path:
    path = ensure_dir(path)
    model.save_pretrained(
        str(path),
        safe_serialization=True,
        selected_adapters=[CULTURE_ADAPTER_NAME],
    )
    return resolve_saved_adapter_dir(path)


def choice_probability_table(model, tokenizer, items: pd.DataFrame, source: str) -> pd.DataFrame:
    import torch

    device = torch.device("cuda:0")
    rows = []
    model.eval()
    for item in items.itertuples(index=False):
        token_ids, logprobs, _ = score_single_token_candidates(
            model,
            tokenizer,
            item.prompt,
            RESPONSE_VALUES,
            device,
        )
        _, conditional, valid_mass = candidate_probabilities_from_logprobs(logprobs)
        for value, token_id, probability in zip(
            RESPONSE_VALUES, token_ids, conditional
        ):
            rows.append(
                {
                    "source": source,
                    "GroupNo": int(item.GroupNo),
                    "item": item.item,
                    "response_value": value,
                    "token_id": token_id,
                    "conditional_probability": float(probability),
                    "valid_response_mass": valid_mass,
                }
            )
    return pd.DataFrame(rows)


def step0_equivalence_check(
    model,
    tokenizer,
    items: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, object]:
    if args.step0_items <= 0:
        raise ValueError("--step0-items must be positive.")
    items = items.sort_values("GroupNo").head(args.step0_items).copy()
    baseline_items = items.assign(prompt=items["item"].map(lambda item: training_prompt(item) + "<<"))
    activate_adapters(model, [CENTAUR_ADAPTER_NAME])
    centaur = choice_probability_table(model, tokenizer, baseline_items, "centaur")
    add_trainable_culture_adapter(model, args)
    culture = choice_probability_table(model, tokenizer, baseline_items, "zero_init_culture")
    merged = centaur.merge(
        culture,
        on=["GroupNo", "item", "response_value", "token_id"],
        suffixes=("_centaur", "_culture"),
        validate="one_to_one",
    )
    difference = (
        merged["conditional_probability_culture"]
        - merged["conditional_probability_centaur"]
    )
    merged["probability_difference"] = difference
    merged.to_csv(output_dir / "step0_choice_probabilities.csv", index=False)
    expected = merged.assign(
        centaur_weighted=merged["response_value"]
        * merged["conditional_probability_centaur"],
        culture_weighted=merged["response_value"]
        * merged["conditional_probability_culture"],
    ).groupby("GroupNo")[["centaur_weighted", "culture_weighted"]].sum()
    metrics = {
        "n_items_checked": int(items["GroupNo"].nunique()),
        "group_numbers_checked": [int(value) for value in items["GroupNo"]],
        "max_abs_probability_difference": float(difference.abs().max()),
        "mean_abs_probability_difference": float(difference.abs().mean()),
        "max_abs_expected_token_difference": float(
            (expected["culture_weighted"] - expected["centaur_weighted"]).abs().max()
        ),
    }
    if metrics["max_abs_probability_difference"] > args.step0_tolerance:
        raise RuntimeError(
            "Zero-initialized culture adapter changed Centaur predictions beyond "
            f"tolerance: {metrics}"
        )
    return metrics


def train(args: argparse.Namespace) -> None:
    if args.condition not in TRAIN_CONDITIONS:
        raise ValueError("Training requires --condition us, china, or pooled.")
    require_corpus_behavior_source(
        args.corpus_dir, args.behavior_path,
        prepare_script="scripts/dictator_centaur70b_qlora.py",
        group_path=args.group_path, seed=args.seed,
    )
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    validate_qlora_runtime()
    set_global_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    rows_path = Path(args.corpus_dir) / f"rows_{args.condition}.jsonl"
    rows = pd.read_json(rows_path, lines=True)
    if args.smoke_rows > 0:
        rows = rows.iloc[: args.smoke_rows].copy()
    participant_training_rows = len(rows)
    giving, items = load_giving_data(args.behavior_path, args.group_path)
    del giving

    tokenizer = load_tokenizer(args.base_model)
    model = load_frozen_centaur(args, for_training=True)
    step0_metrics = step0_equivalence_check(
        model, tokenizer, items, args, output_dir
    )
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if args.training_objective == "item_distribution":
        aggregated_rows = aggregate_item_response_distributions(rows)
        if args.micro_batch_size != len(aggregated_rows):
            raise ValueError(
                "Item-distribution training requires one complete item set per "
                f"batch: --micro-batch-size must be {len(aggregated_rows)}."
            )
        if args.gradient_accumulation_steps != 1:
            raise ValueError(
                "Item-distribution training uses one full-item batch per update; "
                "set --gradient-accumulation-steps 1."
            )
        dataset = ItemDistributionDataset(
            aggregated_rows,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            repeats=args.distribution_repeats,
            seed=args.seed,
        )
        collator = ItemDistributionCollator(tokenizer)
        candidate_token_tensor = torch.tensor(
            response_token_ids(tokenizer), dtype=torch.long, device="cuda:0"
        )
    else:
        aggregated_rows = None
        dataset = PromptCompletionDataset(
            rows,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            train_on_eos=True,
            answer_value_only=True,
        )
        collator = CompletionOnlyCollator(tokenizer)
        candidate_token_tensor = None
    if args.training_objective == "item_distribution":
        batch_sampler = FullItemSetBatchSampler(
            items_per_repeat=dataset.items_per_repeat,
            repeats=dataset.repeats,
            seed=args.seed,
        )
    else:
        lengths = [len(example["input_ids"]) for example in dataset.examples]
        batch_sampler = LengthGroupedBatchSampler(
            lengths,
            batch_size=args.micro_batch_size,
            seed=args.seed,
        )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        pin_memory=False,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if args.optimizer == "adamw_8bit":
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(
            trainable_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    planned_steps = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if args.max_optimizer_steps > 0:
        planned_steps = min(planned_steps, args.max_optimizer_steps)
    warmup_steps = min(args.warmup_steps, max(0, planned_steps - 1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=planned_steps,
    )
    checkpoint_steps = {
        max(1, math.ceil(planned_steps * fraction)): label
        for fraction, label in [(0.25, "quarter"), (0.50, "half"), (1.0, "one_epoch")]
    }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    optimizer_step = 0
    micro_step = 0
    pending = 0
    total_loss = 0.0
    first_loss = None
    last_loss = None
    stop = False
    for batch_index, batch in enumerate(dataloader):
        batch = {key: value.to("cuda:0") for key, value in batch.items()}
        if args.training_objective == "item_distribution":
            loss = item_distribution_causal_lm_loss(
                model, batch, candidate_token_tensor
            )
        else:
            loss = completion_only_causal_lm_loss(model, batch)
        raw_loss = float(loss.detach().cpu())
        if not math.isfinite(raw_loss):
            raise RuntimeError(f"Nonfinite training loss at batch {batch_index}: {raw_loss}")
        if first_loss is None:
            first_loss = raw_loss
        last_loss = raw_loss
        total_loss += raw_loss
        (loss / args.gradient_accumulation_steps).backward()
        micro_step += 1
        pending += 1
        is_last_batch = batch_index + 1 == len(dataloader)
        if pending == args.gradient_accumulation_steps or is_last_batch:
            if pending < args.gradient_accumulation_steps:
                scale = args.gradient_accumulation_steps / pending
                for parameter in trainable_parameters:
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            pending = 0
            elapsed = time.perf_counter() - started
            print(
                f"[centaur70b] condition={args.condition} "
                f"step={optimizer_step}/{planned_steps} loss={raw_loss:.6f} "
                f"elapsed_seconds={elapsed:.1f}",
                flush=True,
            )
            if optimizer_step in checkpoint_steps:
                save_culture_adapter(
                    model,
                    output_dir / "checkpoints" / checkpoint_steps[optimizer_step],
                )
            if optimizer_step >= planned_steps:
                stop = True
                break
        if stop:
            break

    final_adapter_dir = save_culture_adapter(model, output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    elapsed = time.perf_counter() - started
    trainable_count = int(sum(parameter.numel() for parameter in trainable_parameters))
    total_count = int(sum(parameter.numel() for parameter in model.parameters()))
    metrics = {
        "condition": args.condition,
        "seed": args.seed,
        "training_rows": int(participant_training_rows),
        "unique_items": int(rows["GroupNo"].nunique()),
        "training_objective": args.training_objective,
        "optimization_examples": int(len(dataset)),
        "distribution_repeats": (
            args.distribution_repeats
            if args.training_objective == "item_distribution"
            else None
        ),
        "aggregated_item_contexts": (
            int(len(aggregated_rows)) if aggregated_rows is not None else None
        ),
        "epochs": (
            args.distribution_repeats
            if args.training_objective == "item_distribution"
            else 1
        ),
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.micro_batch_size
        * args.gradient_accumulation_steps,
        "optimizer_steps": optimizer_step,
        "micro_steps": micro_step,
        "planned_optimizer_steps": planned_steps,
        "warmup_steps": warmup_steps,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "mean_microbatch_loss": total_loss / max(1, micro_step),
        "elapsed_seconds": elapsed,
        "trainable_parameters": trainable_count,
        "total_loaded_parameters": total_count,
        "trainable_fraction": trainable_count / max(1, total_count),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "saved_adapter_dir": str(final_adapter_dir),
        "step0_equivalence": step0_metrics,
        "runtime_versions": runtime_versions(),
        "base_model": args.base_model,
        "centaur_adapter": args.centaur_adapter,
        "quantization": "4-bit NF4 with double quantization; BF16 compute",
        "active_frozen_adapter": CENTAUR_ADAPTER_NAME,
        "trainable_adapter": CULTURE_ADAPTER_NAME,
    }
    (output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


def load_evaluation_model(args: argparse.Namespace):
    model = load_frozen_centaur(args, for_training=False)
    if args.condition != "base":
        from peft import PeftConfig

        adapter_dir = resolve_saved_adapter_dir(Path(args.output_dir) / "adapter")
        # Reading the config first gives a clearer failure than load_adapter when
        # an incomplete Slurm output directory is supplied.
        PeftConfig.from_pretrained(str(adapter_dir))
        model.load_adapter(
            str(adapter_dir),
            adapter_name=CULTURE_ADAPTER_NAME,
            is_trainable=False,
        )
        activate_adapters(model, [CENTAUR_ADAPTER_NAME, CULTURE_ADAPTER_NAME])
    model.eval()
    return model


def evaluate(args: argparse.Namespace) -> None:
    validate_qlora_runtime()
    if args.condition not in EVALUATION_CONDITIONS:
        raise ValueError("Evaluation requires --condition base, us, china, or pooled.")
    evaluation_dir = ensure_dir(Path(args.output_dir) / "evaluation")
    giving, items = load_giving_data(args.behavior_path, args.group_path)
    split_manifest = pd.read_csv(
        Path(args.corpus_dir) / "participant_split_manifest.csv"
    )
    test_rows = heldout_rows(giving, split_manifest)
    tokenizer = load_tokenizer(args.base_model)
    model = load_evaluation_model(args)
    import torch

    device = torch.device("cuda:0")
    candidates, predictions, invalid, aggregate_invalid = score_model(
        model,
        tokenizer,
        device,
        items,
        f"centaur70b-{args.condition}",
    )
    metrics, human_items = evaluate_against_humans(test_rows, candidates)
    candidates.to_csv(evaluation_dir / "candidate_probabilities.csv", index=False)
    predictions.to_csv(evaluation_dir / "item_predictions.csv", index=False)
    invalid.to_csv(evaluation_dir / "invalid_token_probabilities.csv", index=False)
    aggregate_invalid.to_csv(
        evaluation_dir / "aggregate_invalid_tokens.csv", index=False
    )
    metrics.to_csv(evaluation_dir / "model_metrics.csv", index=False)
    human_items.to_csv(evaluation_dir / "human_item_comparison.csv", index=False)
    metadata = {
        "condition": args.condition,
        "seed": args.seed,
        "base_model": args.base_model,
        "centaur_adapter": args.centaur_adapter,
        "culture_adapter": None
        if args.condition == "base"
        else str(resolve_saved_adapter_dir(Path(args.output_dir) / "adapter")),
        "evaluation_participants": int(
            test_rows[PARTICIPANT_KEYS].drop_duplicates().shape[0]
        ),
        "test_participants_used_for_training": False,
        "inference_passes_per_item": 1,
        "response_values": RESPONSE_VALUES,
        "runtime_versions": runtime_versions(),
    }
    (evaluation_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def analyze(args: argparse.Namespace) -> None:
    seed_dir = Path(args.output_dir)
    candidate_frames = {
        condition: pd.read_csv(
            seed_dir / condition / "evaluation" / "candidate_probabilities.csv"
        )
        for condition in EVALUATION_CONDITIONS
    }
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
    participant_parts = []
    for condition, candidates in candidate_frames.items():
        metrics, _ = evaluate_against_humans(test_rows, candidates)
        metrics.insert(0, "condition", condition)
        metric_parts.append(metrics)
        lookup = candidates[
            ["GroupNo", "response_value", "conditional_probability"]
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
        participant = (
            scored.groupby(PARTICIPANT_KEYS + ["Country"], as_index=False)[
                "conditional_nll_nats"
            ]
            .mean()
        )
        participant.insert(0, "condition", condition)
        participant_parts.append(participant)
    all_metrics = pd.concat(metric_parts, ignore_index=True)
    all_metrics.to_csv(seed_dir / "all_model_metrics.csv", index=False)
    pd.concat(participant_parts, ignore_index=True).to_csv(
        seed_dir / "participant_test_losses.csv", index=False
    )

    combined = all_metrics.loc[all_metrics["sample"].eq("Combined")]
    matrix = combined.pivot(
        index="condition", columns="country", values="conditional_kl_nats"
    ).rename(columns={"United States": "kl_us", "China": "kl_china"})
    matrix["mean_kl"] = matrix[["kl_us", "kl_china"]].mean(axis=1)
    matrix.reset_index().to_csv(seed_dir / "heldout_loss_matrix.csv", index=False)
    matching_advantage = 0.5 * (
        matrix.loc["us", "kl_china"]
        + matrix.loc["china", "kl_us"]
        - matrix.loc["us", "kl_us"]
        - matrix.loc["china", "kl_china"]
    )
    pooled_penalty = 0.5 * (
        matrix.loc["pooled", "kl_us"]
        - matrix.loc["us", "kl_us"]
        + matrix.loc["pooled", "kl_china"]
        - matrix.loc["china", "kl_china"]
    )
    effects = {
        "seed": args.seed,
        "matching_culture_advantage_kl_nats": float(matching_advantage),
        "pooled_penalty_vs_matching_specialists_kl_nats": float(pooled_penalty),
        "base_mean_kl_nats": float(matrix.loc["base", "mean_kl"]),
        "us_adapter_own_kl_nats": float(matrix.loc["us", "kl_us"]),
        "china_adapter_own_kl_nats": float(matrix.loc["china", "kl_china"]),
        "pooled_us_kl_nats": float(matrix.loc["pooled", "kl_us"]),
        "pooled_china_kl_nats": float(matrix.loc["pooled", "kl_china"]),
    }

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
    rng = np.random.default_rng(args.seed + 700_000)
    bootstrap_rows = []
    for bootstrap_index in range(args.bootstrap_resamples):
        kl = {}
        for country in COUNTRIES:
            ids = participant_ids[country]
            sampled = ids.iloc[rng.integers(0, len(ids), size=len(ids))].copy()
            sampled["bootstrap_occurrence"] = np.arange(len(sampled))
            sampled_rows = sampled.merge(
                country_rows[country][PARTICIPANT_KEYS + ["GroupNo", "TokenChoice"]],
                on=PARTICIPANT_KEYS,
                how="left",
                validate="many_to_many",
            )
            q = human_distribution(sampled_rows)
            for condition, p in probabilities.items():
                kl[(condition, country)] = conditional_kl(q, p)
        bootstrap_rows.extend(
            [
                {
                    "bootstrap": bootstrap_index,
                    "estimand": "matching_culture_advantage_kl_nats",
                    "value": 0.5
                    * (
                        kl[("us", "China")]
                        + kl[("china", "United States")]
                        - kl[("us", "United States")]
                        - kl[("china", "China")]
                    ),
                },
                {
                    "bootstrap": bootstrap_index,
                    "estimand": "pooled_penalty_vs_matching_specialists_kl_nats",
                    "value": 0.5
                    * (
                        kl[("pooled", "United States")]
                        - kl[("us", "United States")]
                        + kl[("pooled", "China")]
                        - kl[("china", "China")]
                    ),
                },
            ]
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(seed_dir / "participant_bootstrap_samples.csv", index=False)
    summary = (
        bootstrap.groupby("estimand")["value"]
        .agg(
            estimate="mean",
            ci_low=lambda values: values.quantile(0.025),
            ci_high=lambda values: values.quantile(0.975),
        )
        .reset_index()
    )
    summary.to_csv(seed_dir / "participant_bootstrap_summary.csv", index=False)
    effects["participant_bootstrap_resamples"] = args.bootstrap_resamples
    (seed_dir / "contrast_effects.json").write_text(
        json.dumps(effects, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Centaur-70B stacked culture-adapter QLoRA experiment."
    )
    parser.add_argument(
        "--stage", required=True, choices=["prepare", "train", "evaluate", "analyze"]
    )
    parser.add_argument("--condition", choices=EVALUATION_CONDITIONS)
    parser.add_argument("--behavior-path", required=True)
    parser.add_argument("--group-path", required=True)
    parser.add_argument("--corpus-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
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
    parser.add_argument("--micro-batch-size", type=int, default=25)
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
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        prepare_corpora(args)
    elif args.stage == "train":
        if not args.corpus_dir:
            raise ValueError("--corpus-dir is required for training.")
        train(args)
    elif args.stage == "evaluate":
        if not args.corpus_dir:
            raise ValueError("--corpus-dir is required for evaluation.")
        evaluate(args)
    else:
        if not args.corpus_dir:
            raise ValueError("--corpus-dir is required for analysis.")
        analyze(args)


if __name__ == "__main__":
    main()
