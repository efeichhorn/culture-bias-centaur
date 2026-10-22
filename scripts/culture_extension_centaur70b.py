#!/usr/bin/env python3
"""Fixed-protocol Centaur adaptation for Cao triads and Anllo lotteries.

Only training distributions enter optimization. Evaluation consumes item prompts,
never held-out human responses. Each update averages all training items and both
answer orders, with full-vocabulary (not conditional A/B) cross-entropy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

UPDATES = 25
MAX_SEQUENCE_LENGTH = 768
STEP_ZERO_TOLERANCE = 1e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def manifest_path(manifest_file: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (manifest_file.parent / path).resolve()


def canonical_country(value: str) -> str:
    aliases = {"us": "us", "usa": "us", "united states": "us", "cn": "china", "china": "china"}
    try:
        return aliases[value.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unexpected participant country: {value!r}") from exc


def validate_manifest(path: Path) -> tuple[dict, list[dict], list[dict]]:
    """Verify immutable inputs, whole-item separation and training membership."""
    path = path.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "dataset", "seed", "fold", "condition", "items_path", "items_sha256",
        "training_targets_path", "training_targets_sha256", "output_dir",
        "heldout_item_ids", "train_item_ids", "source_hashes", "participant_split_path",
        "participant_split_sha256", "training_participants_path", "training_participants_sha256",
    }
    if required - manifest.keys():
        raise ValueError(f"Missing manifest fields: {sorted(required - manifest.keys())}")
    dataset = manifest["dataset"]
    if dataset not in {"cao", "anllo"} or manifest["seed"] not in {0, 1, 2}:
        raise ValueError("Dataset/seed is outside the approved protocol")
    if manifest["condition"] not in {"us", "china", "pooled_equal_budget", "pooled"}:
        raise ValueError("Unknown training condition")
    n_items, n_train, n_folds, n_people, repeats = (15, 12, 5, 124, 1) if dataset == "cao" else (8, 6, 4, 34, 4)
    if not isinstance(manifest["fold"], int) or not 0 <= manifest["fold"] < n_folds:
        raise ValueError("Fold is outside the approved protocol")
    hashes = manifest["source_hashes"]
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Nonempty source_hashes are required")
    for label, digest in hashes.items():
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid source SHA256 for {label}")
    for stem in ["items", "training_targets", "participant_split", "training_participants"]:
        file_path = manifest_path(path, manifest[f"{stem}_path"])
        if sha256_file(file_path) != manifest[f"{stem}_sha256"]:
            raise ValueError(f"SHA256 mismatch for {stem}")
        manifest[f"{stem}_path"] = str(file_path)
    if "responses_path" in manifest:
        response_path = manifest_path(path, manifest["responses_path"])
        if sha256_file(response_path) != manifest.get("responses_sha256"):
            raise ValueError("SHA256 mismatch for responses")
        # Hash bytes only. Held-out human responses are never parsed by this runner.
        manifest["responses_path"] = str(response_path)
    manifest["output_dir"] = str(manifest_path(path, manifest["output_dir"]))
    items = json.loads(Path(manifest["items_path"]).read_text(encoding="utf-8"))
    if not isinstance(items, list) or len(items) != n_items:
        raise ValueError(f"Expected {n_items} items")
    ids = [str(item["item_id"]) for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate item IDs")
    for item in items:
        item["item_id"] = str(item["item_id"])
        for field in ("stratum", "positive_response", "prompt_ab", "prompt_ba"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"Missing item field {field}")
        if item["prompt_ab"] == item["prompt_ba"]:
            raise ValueError("Mirrored prompts must differ")
        if not all(item[field].endswith("You choose <<") for field in ("prompt_ab", "prompt_ba")):
            raise ValueError("Every prompt must end with the exact answer prefix 'You choose <<'")
    train = [str(value) for value in manifest["train_item_ids"]]
    heldout = [str(value) for value in manifest["heldout_item_ids"]]
    if len(train) != len(set(train)) or len(heldout) != len(set(heldout)):
        raise ValueError("Duplicate train/heldout item IDs")
    if set(train) & set(heldout) or set(train) | set(heldout) != set(ids) or len(train) != n_train:
        raise ValueError("Training and held-out items must form the approved disjoint partition")
    manifest["train_item_ids"], manifest["heldout_item_ids"] = train, heldout
    split_rows = read_csv(Path(manifest["participant_split_path"]))
    split = {}
    for row in split_rows:
        key = (canonical_country(row["country"]), row["participant_id"])
        if key in split:
            raise ValueError("Participant appears more than once in split manifest")
        if row["split"] not in {"train", "validation", "test", "unused"}:
            raise ValueError("Unknown participant partition")
        split[key] = row["split"]
    training_rows = read_csv(Path(manifest["training_participants_path"]))
    training_people = [(canonical_country(row["country"]), row["participant_id"]) for row in training_rows]
    if len(training_people) != n_people or len(set(training_people)) != n_people:
        raise ValueError("Wrong or duplicate training participant budget")
    if any(split.get(key) != "train" for key in training_people):
        raise ValueError("Training includes a validation/test/unknown participant")
    country_counts = {country: sum(key[0] == country for key in training_people) for country in ("us", "china")}
    expected = ({"us": n_people, "china": 0} if manifest["condition"] == "us" else
                {"us": 0, "china": n_people} if manifest["condition"] == "china" else
                {"us": n_people // 2, "china": n_people // 2})
    if country_counts != expected:
        raise ValueError("Training participant composition differs from the condition")
    targets = read_csv(Path(manifest["training_targets_path"]))
    if len(targets) != n_train or {str(row["item_id"]) for row in targets} != set(train):
        raise ValueError("Targets must contain each training item once and no held-out items")
    for row in targets:
        row["p_positive"] = float(row["p_positive"])
        row["n_responses"] = int(row["n_responses"])
        row["n_participants"] = int(row["n_participants"])
        if not math.isfinite(row["p_positive"]) or not 0 <= row["p_positive"] <= 1:
            raise ValueError("Target probability must be finite and in [0, 1]")
        if row["n_participants"] != n_people or row["n_responses"] != n_people * repeats:
            raise ValueError("Incomplete or unequal observed-response budget")
    manifest["run_manifest_path"] = str(path)
    manifest["run_manifest_sha256"] = sha256_file(path)
    if "protocol_path" in manifest:
        protocol_path = manifest_path(path, manifest["protocol_path"])
        if sha256_file(protocol_path) != manifest.get("protocol_sha256"):
            raise ValueError("SHA256 mismatch for protocol")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        spec = protocol["datasets"][dataset]
        folds = spec["heldout_item_folds"] if dataset == "cao" else spec["heldout_condition_folds"]
        if set(heldout) != {str(value) for value in folds[manifest["fold"]]}:
            raise ValueError("Item fold differs from the frozen protocol")
        manifest["protocol_path"] = str(protocol_path)
    return manifest, items, targets


def verified_candidate_ids(tokenizer, prompt: str) -> tuple[list[int], list[int]]:
    """Ensure actual prefix+answer tokenization is prefix IDs plus one A/B token."""
    prefix_ids = list(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    if not prefix_ids or len(prefix_ids) > MAX_SEQUENCE_LENGTH:
        raise ValueError(f"Prompt length {len(prefix_ids)} exceeds the limit; truncation is forbidden")
    candidates = []
    for letter in ("A", "B"):
        complete = list(tokenizer(prompt + letter, add_special_tokens=True)["input_ids"])
        if complete[:-1] != prefix_ids or len(complete) != len(prefix_ids) + 1:
            raise ValueError(f"Answer {letter} is not exactly one token after the actual prefix")
        candidates.append(int(complete[-1]))
    if candidates[0] == candidates[1]:
        raise ValueError("A and B map to the same token")
    return prefix_ids, candidates


def prepare_examples(items: list[dict], targets: list[dict], tokenizer) -> tuple[list[dict], list[dict], list[int]]:
    target_map = {str(row["item_id"]): float(row["p_positive"]) for row in targets}
    all_examples, training_examples = [], []
    common_ids = None
    for item in items:
        for order in ("ab", "ba"):
            input_ids, candidate_ids = verified_candidate_ids(tokenizer, item[f"prompt_{order}"])
            if common_ids is not None and candidate_ids != common_ids:
                raise ValueError("Candidate token IDs vary across prompts")
            common_ids = candidate_ids
            example = {"item_id": item["item_id"], "stratum": item["stratum"], "order": order,
                       "positive_response": item["positive_response"], "input_ids": input_ids}
            all_examples.append(example)
            if item["item_id"] in target_map:
                p = target_map[item["item_id"]]
                p_a = p if order == "ab" else 1 - p
                training_examples.append({**example, "target_probabilities": [p_a, 1 - p_a],
                                          "weight": 1 / (2 * len(targets))})
    return all_examples, training_examples, common_ids


def soft_target_loss_from_logprobs(logp_a: float, logp_b: float, p_positive: float, order: str) -> float:
    """Reference scalar loss used to verify mirrored-order weighting in tests."""
    p_a = p_positive if order == "ab" else 1 - p_positive
    return -(p_a * logp_a + (1 - p_a) * logp_b)


def probability_metrics(logp_a: float, logp_b: float, order: str) -> dict[str, float]:
    peak = max(logp_a, logp_b)
    log_mass = peak + math.log(math.exp(logp_a - peak) + math.exp(logp_b - peak))
    return {"logp_a": logp_a, "logp_b": logp_b,
            "p_positive": math.exp((logp_a if order == "ab" else logp_b) - log_mass),
            "valid_mass": math.exp(log_mass), "log_valid_mass": log_mass}


def collate(examples: list[dict], pad_token_id: int):
    import torch
    max_length = max(len(row["input_ids"]) for row in examples)
    return {"input_ids": torch.tensor([row["input_ids"] + [pad_token_id] * (max_length - len(row["input_ids"]))
                                       for row in examples], dtype=torch.long, device="cuda:0"),
            "lengths": torch.tensor([len(row["input_ids"]) for row in examples], dtype=torch.long, device="cuda:0")}


def candidate_logprobs(model, batch, candidate_ids):
    """LM head only on the answer-position states; gradients still span prefix."""
    import torch
    from src.llm_train import _base_causal_lm
    causal_lm = _base_causal_lm(model)
    outputs = causal_lm.model(input_ids=batch["input_ids"], attention_mask=None, use_cache=False, return_dict=True)
    # Right padding cannot affect the earlier real tokens under causal attention.
    row_ids = torch.arange(len(batch["lengths"]), device="cuda:0")
    hidden = outputs.last_hidden_state[row_ids, batch["lengths"] - 1]
    logits = causal_lm.lm_head(hidden).float()
    return torch.nn.functional.log_softmax(logits, dim=-1).index_select(-1, candidate_ids)


def score_examples(model, examples, candidate_ids, tokenizer, source: str, manifest: dict) -> list[dict]:
    import torch
    model.eval()
    rows = []
    with torch.inference_mode():
        for example in examples:
            values = candidate_logprobs(model, collate([example], tokenizer.pad_token_id), candidate_ids)[0].cpu().tolist()
            rows.append({"dataset": manifest["dataset"], "seed": manifest["seed"], "fold": manifest["fold"],
                         "condition": manifest["condition"], "source": source, "item_id": example["item_id"],
                         "stratum": example["stratum"], "order": example["order"],
                         "positive_response": example["positive_response"],
                         "heldout_item": example["item_id"] in manifest["heldout_item_ids"],
                         "token_id_a": int(candidate_ids[0]), "token_id_b": int(candidate_ids[1]),
                         **probability_metrics(*values, example["order"])})
    return rows


def verify_trainables(model) -> tuple[list, list[str]]:
    pairs = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not pairs or any(".culture." not in name or "lora_" not in name for name, _ in pairs):
        raise RuntimeError("Trainable parameters must consist exclusively of the culture LoRA adapter")
    if not any("lora_A.culture." in name for name, _ in pairs) or not any("lora_B.culture." in name for name, _ in pairs):
        raise RuntimeError("Both LoRA factors must be trainable")
    return [parameter for _, parameter in pairs], [name for name, _ in pairs]


def training_arguments(args):
    return argparse.Namespace(base_model=args.base_model, centaur_adapter=args.centaur_adapter,
                              attn_implementation=args.attn_implementation, gradient_checkpointing=True,
                              lora_r=8, lora_alpha=8, use_rslora=True, lora_dropout=0.0)


def run(args) -> None:
    # Validate provenance before loading any expensive model.
    manifest, items, targets = validate_manifest(Path(args.run_manifest))
    if args.micro_batch_size < 1:
        raise ValueError("--micro-batch-size must be positive")
    if args.max_optimizer_steps is not None and args.stage != "smoke":
        raise ValueError("Production training is fixed to 25 updates; step override is smoke-only")
    steps = args.max_optimizer_steps or (1 if args.stage == "smoke" else UPDATES)
    if not 1 <= steps <= UPDATES:
        raise ValueError("Smoke update count must be in [1, 25]")
    if args.output_dir:
        output = Path(args.output_dir).expanduser().resolve()
    else:
        output = Path(manifest["output_dir"])
        if args.stage in {"smoke", "baseline"}:
            output /= args.stage
    if args.stage in {"train", "smoke"} and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty training output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / f"{args.stage}_manifest.json", manifest)
    import torch
    from transformers import get_linear_schedule_with_warmup
    from scripts.dictator_centaur70b_qlora import (
        activate_adapters, add_trainable_culture_adapter, load_frozen_centaur,
        load_tokenizer, resolve_saved_adapter_dir, runtime_versions, save_culture_adapter,
        validate_qlora_runtime,
    )
    from src.utils import set_global_seed
    validate_qlora_runtime()
    set_global_seed(manifest["seed"])
    total_started = time.perf_counter()
    model_args = training_arguments(args)
    tokenizer = load_tokenizer(args.base_model)
    examples, training_examples, ids = prepare_examples(items, targets, tokenizer)
    write_json(output / "tokenization.json", {"candidate_token_ids": ids,
        "n_training_prompts": len(training_examples), "max_prompt_length": max(len(row["input_ids"]) for row in examples),
        "prompt_lengths": [{"item_id": row["item_id"], "order": row["order"], "tokens": len(row["input_ids"])} for row in examples]})
    candidate_ids = torch.tensor(ids, dtype=torch.long, device="cuda:0")
    # Use the same numerical preparation for frozen, trained and reloaded models.
    # k-bit preparation casts nonquantized weights to float32; skipping it only
    # for standalone evaluation changes predictions before any fine-tuning.
    model = load_frozen_centaur(model_args, for_training=True)
    model_load_seconds = time.perf_counter() - total_started
    print(f"[extension] loaded {manifest['dataset']} in {model_load_seconds:.1f}s", flush=True)
    if args.stage == "baseline":
        centaur = score_examples(model, examples, candidate_ids, tokenizer, "frozen_centaur", manifest)
        with model.disable_adapter():
            llama = score_examples(model, examples, candidate_ids, tokenizer, "base_llama", manifest)
        write_csv(output / "candidate_probabilities.csv", centaur + llama)
        write_json(output / "baseline_metrics.json", {"stage": args.stage, "dataset": manifest["dataset"],
                   "seed": manifest["seed"], "fold": manifest["fold"], "condition": manifest["condition"],
                   "run_manifest_sha256": manifest["run_manifest_sha256"],
                   "model_load_seconds": model_load_seconds,
                   "elapsed_seconds": time.perf_counter() - total_started, "runtime_versions": runtime_versions(),
                   "base_model": args.base_model, "centaur_adapter": args.centaur_adapter})
        return
    if args.stage == "evaluate":
        adapter = resolve_saved_adapter_dir(args.adapter_dir or (Path(manifest["output_dir"]) / "adapter"))
        model.load_adapter(str(adapter), adapter_name="culture", is_trainable=False)
        activate_adapters(model, ["centaur", "culture"])
        rows = score_examples(model, examples, candidate_ids, tokenizer, "finetuned_centaur", manifest)
        write_csv(output / "evaluation" / "candidate_probabilities.csv", rows)
        write_json(output / "evaluation" / "metadata.json", {"adapter_dir": str(adapter), "heldout_human_outcomes_loaded": False,
            "runtime_versions": runtime_versions(), "elapsed_seconds": time.perf_counter() - total_started})
        return
    check_started = time.perf_counter()
    frozen = score_examples(model, examples, candidate_ids, tokenizer, "frozen_centaur", manifest)
    add_trainable_culture_adapter(model, model_args)
    zero = score_examples(model, examples, candidate_ids, tokenizer, "zero_init_culture", manifest)
    differences = [abs(left["p_positive"] - right["p_positive"]) for left, right in zip(frozen, zero)]
    step0 = {"all_items": len(items), "both_orders": True, "max_abs_probability_difference": max(differences),
             "tolerance": STEP_ZERO_TOLERANCE, "elapsed_seconds": time.perf_counter() - check_started}
    write_csv(output / "step0_choice_probabilities.csv", frozen + zero)
    write_json(output / "step0_metrics.json", step0)
    if max(differences) > STEP_ZERO_TOLERANCE:
        raise RuntimeError(f"Zero-adapter equivalence failed: {step0}")
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    parameters, trainable_names = verify_trainables(model)
    write_json(output / "trainable_parameters.json", {"names": trainable_names, "numel": sum(p.numel() for p in parameters)})
    optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=0.01)
    # One smoke step must exercise an actual weight update; it is explicitly not
    # a production checkpoint. Production preserves the approved three-step warmup.
    warmup_steps = 0 if args.stage == "smoke" else 3
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=UPDATES)
    checkpoint_steps = {7: "quarter", 13: "half", 25: "final"}
    torch.cuda.reset_peak_memory_stats()
    train_started = time.perf_counter()
    rng = random.Random(manifest["seed"])
    step_logs = []
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        order = list(training_examples)
        rng.shuffle(order)
        step_started, full_loss = time.perf_counter(), 0.0
        used_lr = optimizer.param_groups[0]["lr"]
        for start in range(0, len(order), args.micro_batch_size):
            batch_rows = order[start : start + args.micro_batch_size]
            batch = collate(batch_rows, tokenizer.pad_token_id)
            logprobs = candidate_logprobs(model, batch, candidate_ids)
            target = torch.tensor([row["target_probabilities"] for row in batch_rows], dtype=torch.float32, device="cuda:0")
            weights = torch.tensor([row["weight"] for row in batch_rows], dtype=torch.float32, device="cuda:0")
            # Sum per-example contributions; each has fixed 1/(2*n_items) weight.
            # This remains exact when the final microbatch is smaller.
            loss = (-(target * logprobs).sum(dim=-1) * weights).sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at update {step}")
            full_loss += float(loss.detach().cpu())
            loss.backward()
        if not any(parameter.grad is not None for parameter in parameters):
            raise RuntimeError("No adapter gradients were produced")
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in parameters):
            raise RuntimeError(f"Nonfinite adapter gradients at update {step}")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step_log = {"step": step, "full_item_loss": full_loss, "learning_rate": used_lr,
                    "update_seconds": time.perf_counter() - step_started,
                    "elapsed_seconds": time.perf_counter() - train_started}
        step_logs.append(step_log)
        print("[extension] " + json.dumps(step_log), flush=True)
        with (output / "optimizer_steps.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(step_log) + "\n")
        if step in checkpoint_steps:
            save_culture_adapter(model, output / "checkpoints" / checkpoint_steps[step])
    train_seconds = time.perf_counter() - train_started
    adapter = save_culture_adapter(model, output / "adapter")
    tokenizer.save_pretrained(output / "tokenizer")
    trained = score_examples(model, examples, candidate_ids, tokenizer, "finetuned_centaur", manifest)
    write_csv(output / "evaluation" / "candidate_probabilities.csv", trained)
    write_csv(output / "evaluation" / "frozen_centaur_probabilities.csv", frozen)
    metrics = {"stage": args.stage, "complete_production_training": args.stage == "train" and steps == UPDATES,
        "dataset": manifest["dataset"], "condition": manifest["condition"], "seed": manifest["seed"], "fold": manifest["fold"],
        "optimizer_steps": steps, "planned_production_steps": UPDATES, "training_items": len(targets),
        "mirrored_prompts_per_update": len(training_examples), "micro_batch_size": args.micro_batch_size,
        "observed_training_responses": sum(row["n_responses"] for row in targets),
        "objective": "Equal-item, equal-order full-vocabulary soft-target cross-entropy", "warmup_steps": warmup_steps,
        "learning_rate": 5e-5, "weight_decay": 0.01, "optimizer": "AdamW", "schedule": "linear warmup then linear decay over 25 updates",
        "model_load_seconds": model_load_seconds, "training_seconds": train_seconds,
        "elapsed_seconds": time.perf_counter() - total_started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "first_loss": step_logs[0]["full_item_loss"], "last_loss": step_logs[-1]["full_item_loss"],
        "step0_equivalence": step0, "saved_adapter_dir": str(adapter), "runtime_versions": runtime_versions(),
        "base_model": args.base_model, "centaur_adapter": args.centaur_adapter,
        "test_human_outcomes_loaded": False, "run_manifest_sha256": manifest["run_manifest_sha256"],
        "trainer_sha256": sha256_file(Path(__file__))}
    write_json(output / "train_metrics.json", metrics)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--stage", choices=["smoke", "train", "evaluate", "baseline"], required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--adapter-dir")
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--base-model", default="unsloth/Meta-Llama-3.1-70B-bnb-4bit")
    parser.add_argument("--centaur-adapter", default="marcelbinz/Llama-3.1-Centaur-70B-adapter")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
