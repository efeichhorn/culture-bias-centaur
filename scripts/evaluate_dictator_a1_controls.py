#!/usr/bin/env python3
"""Inference-only Llama/Centaur controls with the original and country-label prompts.

Score the numeric token after the source prompt's prefilled ``<<``. Each item is
independent; country labels describe the participant, not their interaction partner.
No culture adapter is loaded and no parameters or checkpoints are updated.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dictator_centaur70b_qlora import (
    CENTAUR_ADAPTER_NAME,
    activate_adapters,
    load_4bit_base,
    load_tokenizer,
    runtime_versions,
)
from scripts.evaluate_dictator_giving_base import (
    RESPONSE_VALUES,
    build_prompt,
    candidate_probabilities_from_logprobs,
    score_single_token_candidates,
)
from scripts.evaluate_dictator_taking_transfer import build_taking_prompt


PROMPT_VARIANTS = {
    "original": ("unspecified", ""),
    "us": ("United States", "This participant lives in the United States."),
    "china": ("China", "This participant lives in China."),
}
TASK_BUILDERS = {"giving": build_prompt, "taking": build_taking_prompt}
MODEL_CONDITIONS = ("llama", "centaur")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def control_prompt(task: str, item: str, prompt_variant: str) -> str:
    original = TASK_BUILDERS[task](item)
    _, prefix = PROMPT_VARIANTS[prompt_variant]
    return f"{prefix}\n\n{original}" if prefix else original


def load_items(group_path: str | Path) -> pd.DataFrame:
    path = Path(group_path)
    items = (
        pd.read_csv(path)
        if path.suffix.lower() == ".csv"
        else pd.read_excel(path, sheet_name="Behavior studies")
    ).rename(columns={"No.": "GroupNo", "English": "item"})
    items = items[["GroupNo", "item"]].copy()
    if len(items) != 25 or set(items["GroupNo"]) != set(range(1, 26)):
        raise ValueError("Expected exactly one row for each target group 1..25.")
    if items["item"].isna().any() or not items["item"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise ValueError("Every target group must have nonempty English item text.")
    return items.sort_values("GroupNo").reset_index(drop=True)


def build_prompt_records(items: pd.DataFrame) -> list[dict[str, object]]:
    records = []
    for task in TASK_BUILDERS:
        for variant, (country, prefix) in PROMPT_VARIANTS.items():
            for item in items.itertuples(index=False):
                prompt = control_prompt(task, str(item.item), variant)
                records.append(
                    {
                        "task": task,
                        "prompt_variant": variant,
                        "country_label": country,
                        "country_sentence": prefix,
                        "GroupNo": int(item.GroupNo),
                        "item": str(item.item),
                        "prompt_sha256": sha256_text(prompt),
                        "prompt": prompt,
                    }
                )
    return records


def score_rows(
    record: dict[str, object],
    model_condition: str,
    model_name: str,
    token_ids: list[int],
    logprobs: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    logprobs = np.asarray(logprobs, dtype=np.float64)
    if len(token_ids) != len(RESPONSE_VALUES) or len(set(token_ids)) != len(token_ids):
        raise ValueError("Expected 21 distinct numeric response tokens.")
    if logprobs.shape != (len(RESPONSE_VALUES),) or not np.isfinite(logprobs).all():
        raise ValueError("Expected 21 finite candidate log probabilities.")
    full, conditional, mass = candidate_probabilities_from_logprobs(logprobs)
    if (logprobs > 1e-6).any() or not 0 < mass <= 1 + 1e-6:
        raise ValueError("Candidate probabilities must have positive mass no greater than 1.")
    common = {
        "model_condition": model_condition,
        "model_name": model_name,
        **{key: value for key, value in record.items() if key != "prompt"},
    }
    candidates = [
        {
            **common,
            "response_value": value,
            "response_text": str(value),
            "token_id": int(token_id),
            "full_vocab_log_probability": float(logprob),
            "full_vocab_probability": float(probability),
            "conditional_probability": float(conditional_probability),
            "valid_response_mass": mass,
        }
        for value, token_id, logprob, probability, conditional_probability in zip(
            RESPONSE_VALUES, token_ids, logprobs, full, conditional
        )
    ]
    prediction = {
        **common,
        "expected_tokens": float(np.dot(conditional, RESPONSE_VALUES)),
        "modal_tokens": int(RESPONSE_VALUES[int(np.argmax(conditional))]),
        "conditional_modal_probability": float(conditional.max()),
        "valid_response_mass": mass,
        "prompt": record["prompt"],
    }
    return candidates, prediction


def assert_adapter_state(model, *, disabled: bool) -> int:
    """Check PEFT's actual LoRA layers, including the absence of merged weights."""
    layers = [module for module in model.modules() if hasattr(module, "lora_A")]
    if not layers:
        raise RuntimeError("No LoRA layers found; cannot verify the requested model state.")
    for layer in layers:
        if getattr(layer, "merged_adapters", []):
            raise RuntimeError("Merged adapter weights would invalidate the plain Llama control.")
        if getattr(layer, "disable_adapters", None) is not disabled:
            raise RuntimeError(f"A LoRA layer does not have disable_adapters={disabled}.")
        if not disabled:
            active = getattr(layer, "active_adapters", [])
            if isinstance(active, str):
                active = [active]
            if list(active) != [CENTAUR_ADAPTER_NAME]:
                raise RuntimeError(f"Unexpected active adapters in a LoRA layer: {active}")
    return len(layers)


def verify_disabled_equivalence(
    model, tokenizer, records, reference_scores, device, tolerance: float
) -> dict[str, object]:
    differences = []
    with model.disable_adapter():
        layer_count = assert_adapter_state(model, disabled=True)
        for record, (reference_ids, reference_logprobs) in zip(records, reference_scores):
            token_ids, logprobs, _ = score_single_token_candidates(
                model, tokenizer, record["prompt"], RESPONSE_VALUES, device
            )
            if token_ids != reference_ids:
                raise RuntimeError("Numeric response tokens changed after attaching Centaur.")
            differences.extend(np.abs(logprobs - reference_logprobs).tolist())
    assert_adapter_state(model, disabled=False)
    max_difference = max(differences)
    if not np.isfinite(max_difference) or max_difference > tolerance:
        raise RuntimeError(
            "Disabling Centaur did not reproduce the pre-adapter base logits: "
            f"max absolute candidate log-probability difference={max_difference}, "
            f"tolerance={tolerance}."
        )
    return {
        "passed": True,
        "reference_prompt_sha256": [record["prompt_sha256"] for record in records],
        "n_reference_prompts": len(records),
        "n_verified_lora_layers": layer_count,
        "max_abs_candidate_logprob_difference": max_difference,
        "tolerance": tolerance,
        "adapter_state_restored": True,
    }


def cached_checkpoint_info(name: str, filename: str) -> dict[str, object]:
    """Read cached revision evidence only; never contact or download from the Hub."""
    from huggingface_hub import try_to_load_from_cache

    local_file = Path(name) / filename
    cached = str(local_file.resolve()) if local_file.is_file() else try_to_load_from_cache(name, filename)
    path = Path(cached).absolute() if isinstance(cached, str) else None
    revision = None
    if path is not None and "snapshots" in path.parts:
        revision = path.parts[path.parts.index("snapshots") + 1]
    return {
        "requested_name_or_path": name,
        "cached_config_path": str(path) if path is not None else None,
        "cached_snapshot_revision": revision,
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path else None,
    }


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "candidate_probabilities.csv").exists():
        raise FileExistsError(f"Refusing to overwrite existing scores in {output_dir}.")
    records = build_prompt_records(load_items(args.group_path))
    with (output_dir / "prompts.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata = {
        "status": "prepared" if args.prepare_only else "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "runtime_versions": runtime_versions(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "group_source_sha256": hashlib.sha256(Path(args.group_path).read_bytes()).hexdigest(),
        "models": list(MODEL_CONDITIONS),
        "tasks": list(TASK_BUILDERS),
        "prompt_variants": PROMPT_VARIANTS,
        "response_values": RESPONSE_VALUES,
        "n_prompts": len(records),
        "expected_candidate_rows": 2 * len(records) * len(RESPONSE_VALUES),
        "scoring": "full-vocabulary next-token log probability of numeric response after prefilled <<",
        "conditional_probability_definition": "numeric token probability divided by valid_response_mass over 0..20",
        "valid_response_mass_definition": "sum of full-vocabulary next-token probabilities for numeric tokens 0..20; excludes closing delimiter",
        "context": "Each item is scored independently without previous participant responses.",
        "quantization": {"load_in_4bit": True, "type": "nf4", "double_quant": True, "compute_dtype": "bfloat16"},
    }
    write_json(output_dir / "metadata.json", metadata)
    if args.prepare_only:
        print(f"[a1-controls] prepared {len(records)} prompts in {output_dir}", flush=True)
        return

    import torch
    from peft import PeftModel

    tokenizer = load_tokenizer(args.base_model)
    base = load_4bit_base(args, for_training=False)
    base.eval()
    device = torch.device("cuda:0")
    references = [record for record in records if record["prompt_variant"] == "original" and record["GroupNo"] == 1]
    reference_scores = []
    for record in references:
        token_ids, logprobs, _ = score_single_token_candidates(
            base, tokenizer, record["prompt"], RESPONSE_VALUES, device
        )
        reference_scores.append((token_ids, logprobs))

    # Match load_frozen_centaur's loader exactly, but retain pre-adapter evidence
    # from this same base allocation so disable_adapter is verified numerically.
    model = PeftModel.from_pretrained(
        base, args.centaur_adapter, adapter_name=CENTAUR_ADAPTER_NAME, is_trainable=False
    )
    activate_adapters(model, [CENTAUR_ADAPTER_NAME])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metadata["adapter_disable_equivalence"] = verify_disabled_equivalence(
        model, tokenizer, references, reference_scores, device, args.equivalence_tolerance
    )
    metadata["checkpoints"] = {
        "base": cached_checkpoint_info(args.base_model, "config.json"),
        "centaur_adapter": cached_checkpoint_info(args.centaur_adapter, "adapter_config.json"),
        "loaded_base_commit_hash": getattr(base.config, "_commit_hash", None),
        "tokenizer_commit_hash": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
    }
    metadata["device"] = torch.cuda.get_device_name(0)
    metadata["n_trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_json(output_dir / "metadata.json", metadata)

    candidate_rows, prediction_rows = [], []
    for condition in MODEL_CONDITIONS:
        context = model.disable_adapter() if condition == "llama" else nullcontext()
        with context:
            assert_adapter_state(model, disabled=condition == "llama")
            model_name = args.base_model if condition == "llama" else args.centaur_adapter
            for index, record in enumerate(records, start=1):
                token_ids, logprobs, _ = score_single_token_candidates(
                    model, tokenizer, record["prompt"], RESPONSE_VALUES, device
                )
                candidates, prediction = score_rows(record, condition, model_name, token_ids, logprobs)
                candidate_rows.extend(candidates)
                prediction_rows.append(prediction)
                if index % 25 == 0:
                    pd.DataFrame(candidate_rows).to_csv(output_dir / "candidate_probabilities.csv", index=False)
                    pd.DataFrame(prediction_rows).to_csv(output_dir / "item_predictions.csv", index=False)
                    print(f"[a1-controls] model={condition} prompts={index}/{len(records)} task={record['task']} variant={record['prompt_variant']}", flush=True)
        assert_adapter_state(model, disabled=False)

    if len(candidate_rows) != metadata["expected_candidate_rows"]:
        raise RuntimeError("Incomplete candidate output.")
    metadata.update({
        "status": "complete",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_candidate_rows": len(candidate_rows),
        "n_item_predictions": len(prediction_rows),
    })
    write_json(output_dir / "metadata.json", metadata)
    print(f"[a1-controls] complete: {len(candidate_rows)} candidate rows in {output_dir}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="unsloth/Meta-Llama-3.1-70B-bnb-4bit")
    parser.add_argument("--centaur-adapter", default="marcelbinz/Llama-3.1-Centaur-70B-adapter")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--equivalence-tolerance", type=float, default=1e-6)
    parser.add_argument("--prepare-only", action="store_true", help="Write and validate the prompt manifest without loading any model.")
    args = parser.parse_args(argv)
    if not np.isfinite(args.equivalence_tolerance) or args.equivalence_tolerance < 0:
        parser.error("--equivalence-tolerance must be finite and nonnegative")
    return args


if __name__ == "__main__":
    run(parse_args())
