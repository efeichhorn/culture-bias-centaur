#!/usr/bin/env python3
"""Standard-library provenance and completion gates for corrected 70B jobs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

FOLDS = {0: [1, 5, 12, 16, 23], 1: [2, 6, 13, 17, 24],
         2: [3, 7, 14, 18, 25], 3: [4, 8, 10, 15, 19], 4: [9, 11, 20, 21, 22]}
CONDITIONS = ["us", "china", "pooled"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def task_identity(index):
    if not 0 <= index < 45:
        raise ValueError("Full array index must be 0..44.")
    return index // 15, (index % 15) // 3, CONDITIONS[index % 3]


def verify_fields(actual, expected, label):
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"{label}: {key}={actual.get(key)!r}, expected {value!r}")


def validate_corpus(behavior, corpus, seed):
    sha = digest(behavior)
    protocol = read_json(Path(corpus) / "protocol.json")
    verify_fields(protocol, {"source_behavior_sha256": sha, "seed": seed}, "corpus")
    for name in ["participant_split_manifest.csv", "rows_us.jsonl", "rows_china.jsonl", "rows_pooled.jsonl"]:
        if (Path(corpus) / name).stat().st_size == 0:
            raise ValueError(f"Empty corpus file: {name}")
    return sha


def validate_probabilities(path, groups):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(g, value) for g in groups for value in range(21)}
    observed = [(int(r["GroupNo"]), int(r["response_value"])) for r in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError(f"Missing/duplicate candidate responses: {path}")
    for group in groups:
        entries = [r for r in rows if int(r["GroupNo"]) == group]
        conditional = [float(r["conditional_probability"]) for r in entries]
        raw = [float(r["full_vocab_log_probability"]) for r in entries]
        masses = [float(r["valid_response_mass"]) for r in entries]
        if not all(math.isfinite(x) for x in conditional + raw + masses):
            raise ValueError(f"Nonfinite candidate probabilities: {path}")
        if not all(0 < x <= 1 for x in conditional) or not math.isclose(sum(conditional), 1, abs_tol=1e-7):
            raise ValueError(f"Invalid conditional probability normalization: {path}")
        mass = sum(math.exp(x) for x in raw)
        if not 0 < mass <= 1 + 1e-6 or not all(math.isclose(x, mass, abs_tol=1e-7) for x in masses):
            raise ValueError(f"Invalid valid-response mass: {path}")
        if not all(math.isclose(p, math.exp(ll) / mass, abs_tol=1e-7) for p, ll in zip(conditional, raw)):
            raise ValueError(f"Conditional/raw probability mismatch: {path}")


def validate_complete(behavior, corpus, output, seed, fold, condition, steps):
    output = Path(output)
    sha = validate_corpus(behavior, corpus, seed)
    groups = list(range(1, 26)) if condition == "base" else FOLDS[fold]
    if condition != "base":
        train = read_json(output / "train_metadata.json")
        verify_fields(train, {"source_behavior_sha256": sha, "seed": seed, "fold": fold,
                              "condition": condition, "heldout_group_nos": groups}, "training")
        metrics = read_json(output / "train_metrics.json")
        verify_fields(metrics, {"optimizer_steps": steps, "planned_optimizer_steps": steps,
                                "micro_batch_size": 20, "gradient_accumulation_steps": 1}, "training metrics")
        for field in ["first_loss", "last_loss", "mean_microbatch_loss"]:
            if not math.isfinite(float(metrics[field])):
                raise ValueError(f"Nonfinite training loss: {field}")
        adapters = [output / "adapter/culture/adapter_model.safetensors", output / "adapter/adapter_model.safetensors"]
        if not any(p.is_file() and p.stat().st_size > 0 and (p.parent / "adapter_config.json").is_file() for p in adapters):
            raise ValueError("Missing saved culture adapter weights/configuration.")
    for task in ["giving", "taking"]:
        directory = output / "evaluations" / task
        metadata = read_json(directory / "run_metadata.json")
        verify_fields(metadata, {"source_behavior_sha256": sha, "training_seed": seed,
                                 "item_fold": fold, "condition": condition,
                                 "evaluation_task": task.title(),
                                 "evaluation_group_nos": groups,
                                 "corpus_protocol_sha256": digest(Path(corpus) / "protocol.json")}, "evaluation")
        validate_probabilities(directory / "candidate_probabilities.csv", groups)
        with (directory / "model_metrics.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if {r["country"] for r in rows if r["sample"] == "Combined"} != {"United States", "China"}:
            raise ValueError(f"Incomplete country metrics: {directory}")
        for row in rows:
            for field in ["centaur_nll_nats", "conditional_nll_nats"]:
                if not math.isfinite(float(row[field])):
                    raise ValueError(f"Nonfinite evaluation loss: {directory}/{field}")
    return {"status": "verified_complete", "source_behavior_sha256": sha, "seed": seed,
            "fold": fold, "condition": condition, "optimizer_steps": steps,
            "evaluation_tasks": ["giving", "taking"], "groups": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior-path", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--condition", choices=["base", *CONDITIONS], default="base")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--write-completion", action="store_true")
    args = parser.parse_args()
    if args.output_dir:
        result = validate_complete(args.behavior_path, args.corpus_dir, args.output_dir,
                                   args.seed, args.fold, args.condition, args.steps)
        if args.write_completion:
            (Path(args.output_dir) / "verified_complete.json").write_text(json.dumps(result, indent=2) + "\n")
    else:
        result = {"source_behavior_sha256": validate_corpus(args.behavior_path, args.corpus_dir, args.seed)}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
