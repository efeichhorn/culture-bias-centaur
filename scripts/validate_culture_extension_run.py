#!/usr/bin/env python3
"""Verify complete GPU outputs without loading PyTorch or a model.

Only a successful validation writes verified_complete.json. A smoke result is
always marked non-production, even if its outputs otherwise look complete.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import culture_extension_centaur70b as trainer


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {label}") from exc
    require(math.isfinite(number), f"Nonfinite {label}")
    return number


def close(value, expected, label: str, *, abs_tol=1e-10) -> None:
    require(math.isclose(finite(value, label), finite(expected, label), rel_tol=1e-8, abs_tol=abs_tol),
            f"Inconsistent {label}: {value!r} != {expected!r}")


def bool_field(value, label: str) -> bool:
    require(value in (True, False, "True", "False", "true", "false", "0", "1"),
            f"Invalid boolean {label}")
    return value in (True, "True", "true", "1")


def validate_rows(path: Path, expected_sources: set[str], manifest: dict,
                  items: list[dict], token_ids: list[int]) -> list[dict]:
    rows = trainer.read_csv(path)
    item_map = {str(item["item_id"]): item for item in items}
    expected = {(source, item, order) for source in expected_sources for item in item_map for order in ("ab", "ba")}
    keys = [(row.get("source"), row.get("item_id"), row.get("order")) for row in rows]
    require(Counter(keys) == Counter(expected),
            f"{path.name}: each source/item/order must occur exactly once with no missing or extra rows")
    for row in rows:
        for field in ("dataset", "condition", "seed", "fold"):
            require(str(row.get(field)) == str(manifest[field]), f"{path.name}: incorrect {field}")
        item = item_map[row["item_id"]]
        for field in ("stratum", "positive_response"):
            require(row.get(field) == item[field], f"{path.name}: incorrect {field}")
        require(bool_field(row.get("heldout_item"), "heldout_item") == (row["item_id"] in manifest["heldout_item_ids"]),
                f"{path.name}: incorrect heldout flag")
        require([int(row["token_id_a"]), int(row["token_id_b"])] == token_ids,
                f"{path.name}: candidate token IDs differ from tokenization")
        values = {key: finite(row.get(key), key) for key in
                  ("logp_a", "logp_b", "p_positive", "valid_mass", "log_valid_mass")}
        require(values["logp_a"] <= 1e-7 and values["logp_b"] <= 1e-7,
                f"{path.name}: invalid full-vocabulary log probability")
        require(0 <= values["p_positive"] <= 1 and 0 <= values["valid_mass"] <= 1 + 1e-7,
                f"{path.name}: probability outside [0,1]")
        expected_values = trainer.probability_metrics(values["logp_a"], values["logp_b"], row["order"])
        for field in ("p_positive", "valid_mass", "log_valid_mass"):
            close(values[field], expected_values[field], f"{path.name} {field}")
    return rows


def validate_tokenization(path: Path, items: list[dict], n_training_items: int) -> list[int]:
    tokenization = json.loads(path.read_text())
    ids = tokenization.get("candidate_token_ids")
    require(isinstance(ids, list) and len(ids) == 2
            and all(type(value) is int and value >= 0 for value in ids) and ids[0] != ids[1],
            "Invalid distinct A/B candidate token IDs")
    lengths = tokenization.get("prompt_lengths", [])
    require(Counter((row["item_id"], row["order"]) for row in lengths)
            == Counter((str(item["item_id"]), order) for item in items for order in ("ab", "ba")),
            "Tokenization must cover every item in both orders exactly once")
    require(all(type(row["tokens"]) is int and 0 < row["tokens"] <= 768 for row in lengths),
            "Tokenization contains an invalid or truncated prompt length")
    require(tokenization.get("max_prompt_length") == max(row["tokens"] for row in lengths),
            "Incorrect maximum prompt length")
    require(tokenization.get("n_training_prompts") == 2 * n_training_items,
            "Incorrect mirrored training prompt count")
    return ids


def validate_adapter(output: Path, metrics: dict) -> list[Path]:
    candidates = [output / "adapter", output / "adapter" / "culture"]
    matches = [path for path in candidates if (path / "adapter_config.json").is_file()
               and any((path / filename).is_file() and (path / filename).stat().st_size > 0
                       for filename in ("adapter_model.safetensors", "adapter_model.bin"))]
    require(len(matches) == 1, "Expected exactly one saved adapter with config and nonempty weights")
    adapter = matches[0]
    require(Path(metrics.get("saved_adapter_dir", "")).resolve() == adapter.resolve(),
            "Saved adapter path differs from training metadata")
    config = json.loads((adapter / "adapter_config.json").read_text())
    for key, expected in {"peft_type": "LORA", "r": 8, "lora_alpha": 8, "lora_dropout": 0,
                          "use_rslora": True}.items():
        require(config.get(key) == expected, f"Adapter configuration differs from protocol: {key}")
    require(set(config.get("target_modules", [])) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
        "Adapter target modules differ from protocol")
    return [adapter / "adapter_config.json"] + [path for path in adapter.glob("adapter_model.*") if path.is_file()]


def validate_run(run_manifest: Path, output: Path, stage: str) -> dict:
    require(stage in {"smoke", "train", "baseline"}, "Unknown validation stage")
    output = output.resolve()
    marker = output / "verified_complete.json"
    # A previously successful check cannot remain visible after a failed recheck.
    marker.unlink(missing_ok=True)
    manifest, items, targets = trainer.validate_manifest(run_manifest)
    if "responses_path" in manifest:
        responses = trainer.manifest_path(run_manifest.resolve(), manifest["responses_path"])
        require(trainer.sha256_file(responses) == manifest["responses_sha256"], "SHA256 mismatch for responses")
    checked = [output / f"{stage}_manifest.json", output / "tokenization.json"]
    snapshot = json.loads(checked[0].read_text())
    require(snapshot == manifest, "Output manifest differs from the currently verified run manifest")
    token_ids = validate_tokenization(checked[1], items, len(targets))
    metrics_path = output / ("baseline_metrics.json" if stage == "baseline" else "train_metrics.json")
    checked.append(metrics_path)
    metrics = json.loads(metrics_path.read_text())
    require(metrics.get("run_manifest_sha256") == manifest["run_manifest_sha256"], "Metrics run-manifest hash mismatch")
    for field in ("dataset", "condition", "seed", "fold"):
        require(metrics.get(field) == manifest[field], f"Metrics {field} mismatch")
    require(metrics.get("stage") == stage, "Metrics stage mismatch; smoke cannot count as production")
    for field in ("base_model", "centaur_adapter"):
        require(isinstance(metrics.get(field), str) and bool(metrics[field].strip()), f"Missing model source {field}")
    for field in ("model_load_seconds", "elapsed_seconds"):
        require(finite(metrics.get(field), field) >= 0, f"Negative {field}")
    row_count = 0
    if stage == "baseline":
        path = output / "candidate_probabilities.csv"
        checked.append(path)
        rows = validate_rows(path, {"frozen_centaur", "base_llama"}, manifest, items, token_ids)
        row_count = len(rows)
        require(metrics.get("complete_production_training", False) is False,
                "Baseline cannot be marked as complete production training")
    else:
        steps = 25 if stage == "train" else 1
        require(metrics.get("complete_production_training") is (stage == "train"),
                "Production-completion flag does not match stage")
        require(metrics.get("optimizer_steps") == steps and metrics.get("planned_production_steps") == 25,
                "Wrong optimizer step count for stage")
        require(metrics.get("training_items") == len(targets)
                and metrics.get("mirrored_prompts_per_update") == 2 * len(targets), "Training item/prompt budget mismatch")
        require(metrics.get("observed_training_responses") == sum(row["n_responses"] for row in targets),
                "Observed training-response budget mismatch")
        require(metrics.get("test_human_outcomes_loaded") is False, "Held-out human outcomes were loaded during fitting")
        require(metrics.get("warmup_steps") == (3 if stage == "train" else 0), "Warmup differs from stage protocol")
        require(metrics.get("optimizer") == "AdamW", "Unexpected optimizer")
        close(metrics.get("learning_rate"), 5e-5, "learning rate")
        close(metrics.get("weight_decay"), .01, "weight decay")
        require(metrics.get("objective") == "Equal-item, equal-order full-vocabulary soft-target cross-entropy",
                "Unexpected training objective")
        step_path = output / "optimizer_steps.jsonl"
        checked.append(step_path)
        step_logs = [json.loads(line) for line in step_path.read_text().splitlines() if line.strip()]
        require([row.get("step") for row in step_logs] == list(range(1, steps + 1)),
                "Optimizer log must contain each expected step exactly once")
        previous_time = -1.
        for row in step_logs:
            require(finite(row.get("full_item_loss"), "full-item loss") >= 0, "Negative loss")
            require(finite(row.get("update_seconds"), "update seconds") >= 0, "Negative update time")
            elapsed = finite(row.get("elapsed_seconds"), "step elapsed seconds")
            require(elapsed >= previous_time, "Nonmonotonic step elapsed time")
            previous_time = elapsed
            step = row["step"]
            expected_lr = (5e-5 if stage == "smoke" else
                           5e-5 * (step - 1) / 3 if step <= 3 else 5e-5 * (26 - step) / 22)
            close(row.get("learning_rate"), expected_lr, "recorded update learning rate", abs_tol=1e-12)
        if stage == "smoke":
            require(finite(step_logs[0].get("learning_rate"), "smoke learning rate") > 0,
                    "Smoke must exercise a nonzero-learning-rate update")
        close(metrics.get("first_loss"), step_logs[0]["full_item_loss"], "first loss")
        close(metrics.get("last_loss"), step_logs[-1]["full_item_loss"], "last loss")
        for name in ("training_seconds", "peak_gpu_memory_bytes", "peak_gpu_memory_reserved_bytes"):
            require(finite(metrics.get(name), name) > 0, f"Missing or invalid {name}")
        zero_path, zero_metrics_path = output / "step0_choice_probabilities.csv", output / "step0_metrics.json"
        checked += [zero_path, zero_metrics_path]
        zero_rows = validate_rows(zero_path, {"frozen_centaur", "zero_init_culture"}, manifest, items, token_ids)
        by_key = {(row["source"], row["item_id"], row["order"]): row for row in zero_rows}
        max_difference = max(abs(float(by_key[("frozen_centaur", str(item["item_id"]), order)]["p_positive"])
                                 - float(by_key[("zero_init_culture", str(item["item_id"]), order)]["p_positive"]))
                             for item in items for order in ("ab", "ba"))
        zero_metrics = json.loads(zero_metrics_path.read_text())
        require(zero_metrics == metrics.get("step0_equivalence"), "Step-zero metadata disagree")
        require(zero_metrics.get("all_items") == len(items) and zero_metrics.get("both_orders") is True,
                "Step-zero check did not include all items in both orders")
        close(zero_metrics.get("tolerance"), 1e-6, "step-zero tolerance", abs_tol=1e-12)
        close(zero_metrics.get("max_abs_probability_difference"), max_difference, "step-zero maximum difference")
        require(max_difference <= 1e-6, "Step-zero equivalence exceeds 1e-6")
        for filename, source in (("candidate_probabilities.csv", "finetuned_centaur"),
                                 ("frozen_centaur_probabilities.csv", "frozen_centaur")):
            path = output / "evaluation" / filename
            checked.append(path)
            evaluated = validate_rows(path, {source}, manifest, items, token_ids)
            row_count += len(evaluated)
            if source == "frozen_centaur":
                for row in evaluated:
                    original = by_key[(source, row["item_id"], row["order"])]
                    for field in ("logp_a", "logp_b", "p_positive", "valid_mass", "log_valid_mass"):
                        close(row[field], original[field], f"frozen evaluation {field}")
        trainables_path = output / "trainable_parameters.json"
        checked.append(trainables_path)
        trainables = json.loads(trainables_path.read_text())
        names = trainables.get("names", [])
        require(names and all(".culture." in name and "lora_" in name for name in names)
                and any("lora_A.culture." in name for name in names)
                and any("lora_B.culture." in name for name in names)
                and type(trainables.get("numel")) is int and trainables["numel"] > 0,
                "Trainable parameter record is not exclusively the culture adapter")
        checked += validate_adapter(output, metrics)
    result = {
        "verified": True, "stage": stage, "complete_production_training": stage == "train",
        "complete_baseline_evaluation": stage == "baseline",
        **{key: manifest[key] for key in ("dataset", "seed", "fold", "condition", "run_manifest_sha256")},
        "optimizer_steps": 25 if stage == "train" else 1 if stage == "smoke" else 0,
        "items": len(items), "evaluation_rows": row_count,
        "verified_files_sha256": {str(path.relative_to(output)): trainer.sha256_file(path) for path in checked},
    }
    temporary = marker.with_suffix(".json.tmp")
    trainer.write_json(temporary, result)
    temporary.replace(marker)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=["smoke", "train", "baseline"])
    args = parser.parse_args()
    result = validate_run(args.run_manifest, args.output_dir, args.stage)
    print(json.dumps({key: result[key] for key in ("verified", "stage", "dataset", "seed", "fold", "condition",
                                                  "complete_production_training", "evaluation_rows")}))


if __name__ == "__main__":
    main()
