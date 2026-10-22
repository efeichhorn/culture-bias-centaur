import csv
import json
import math
from pathlib import Path

import pytest

from scripts.dictator_centaur70b_item_holdout import ITEM_FOLDS
from scripts.validate_dictator_centaur70b_corrected import (
    FOLDS, digest, task_identity, validate_complete,
)


def test_array_is_exactly_three_seeds_five_folds_three_conditions():
    assert {task_identity(i) for i in range(45)} == {
        (seed, fold, condition) for seed in range(3) for fold in range(5)
        for condition in ["us", "china", "pooled"]
    }
    assert FOLDS == {k: list(v) for k, v in ITEM_FOLDS.items()}
    with pytest.raises(ValueError):
        task_identity(45)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def complete_fixture(tmp_path, steps=25):
    behavior = tmp_path / "Behavior.csv"
    behavior.write_text("corrected fixture")
    corpus = tmp_path / "corpus"
    protocol = {"seed": 0, "source_behavior_sha256": digest(behavior)}
    write_json(corpus / "protocol.json", protocol)
    for name in ["participant_split_manifest.csv", "rows_us.jsonl", "rows_china.jsonl", "rows_pooled.jsonl"]:
        (corpus / name).write_text("fixture")
    output = tmp_path / "output"
    write_json(output / "train_metadata.json", {**protocol, "fold": 0, "condition": "us", "heldout_group_nos": FOLDS[0]})
    write_json(output / "train_metrics.json", {"optimizer_steps": steps, "planned_optimizer_steps": steps,
        "micro_batch_size": 20, "gradient_accumulation_steps": 1, "first_loss": 4., "last_loss": 2., "mean_microbatch_loss": 3.})
    write_json(output / "adapter/culture/adapter_config.json", {})
    (output / "adapter/culture/adapter_model.safetensors").write_bytes(b"fixture weights")
    for task in ["giving", "taking"]:
        directory = output / "evaluations" / task
        write_json(directory / "run_metadata.json", {"source_behavior_sha256": digest(behavior), "training_seed": 0,
            "item_fold": 0, "condition": "us", "evaluation_task": task.title(), "evaluation_group_nos": FOLDS[0],
            "corpus_protocol_sha256": digest(corpus / "protocol.json")})
        with (directory / "candidate_probabilities.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["GroupNo", "response_value", "conditional_probability", "full_vocab_log_probability", "valid_response_mass"])
            writer.writeheader()
            writer.writerows({"GroupNo": g, "response_value": value, "conditional_probability": 1/21,
                "full_vocab_log_probability": math.log(.9/21), "valid_response_mass": .9} for g in FOLDS[0] for value in range(21))
        with (directory / "model_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample", "country", "centaur_nll_nats", "conditional_nll_nats"])
            writer.writeheader()
            writer.writerows({"sample": "Combined", "country": country, "centaur_nll_nats": 3.1,
                "conditional_nll_nats": 3.0} for country in ["United States", "China"])
    return behavior, corpus, output


def test_complete_outputs_pass_but_smoke_cannot_masquerade_as_full(tmp_path):
    paths = complete_fixture(tmp_path, steps=1)
    assert validate_complete(*paths, 0, 0, "us", 1)["status"] == "verified_complete"
    with pytest.raises(ValueError, match="optimizer_steps"):
        validate_complete(*paths, 0, 0, "us", 25)


def test_completion_rejects_mixed_source_metadata(tmp_path):
    behavior, corpus, output = complete_fixture(tmp_path)
    behavior.write_text("different source")
    with pytest.raises(ValueError, match="source_behavior_sha256"):
        validate_complete(behavior, corpus, output, 0, 0, "us", 25)


def test_completion_rejects_nonfinite_or_missing_candidate_predictions(tmp_path):
    paths = complete_fixture(tmp_path)
    candidate = paths[2] / "evaluations/giving/candidate_probabilities.csv"
    original = candidate.read_text()
    candidate.write_text(original.replace(str(math.log(.9/21)), "nan", 1))
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_complete(*paths, 0, 0, "us", 25)
    candidate.write_text("\n".join(original.splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="Missing/duplicate"):
        validate_complete(*paths, 0, 0, "us", 25)
