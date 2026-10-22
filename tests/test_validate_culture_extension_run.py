"""Completion gates reject incomplete, mismatched, or numerically invalid runs."""
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import culture_extension_centaur70b as trainer
from scripts.validate_culture_extension_run import validate_run


def make_result(tmp_path: Path, stage: str):
    items = [{"item_id": str(i), "stratum": "tradeoff", "positive_response": "10_point",
              "prompt_ab": f"{i}: A ten B one. You choose <<",
              "prompt_ba": f"{i}: A one B ten. You choose <<"} for i in range(8)]
    trainer.write_json(tmp_path / "items.json", items)
    people = [{"participant_id": str(i), "country": "us"} for i in range(34)]
    trainer.write_csv(tmp_path / "people.csv", people)
    trainer.write_csv(tmp_path / "split.csv", [{**row, "split": "train"} for row in people])
    trainer.write_csv(tmp_path / "targets.csv", [{"item_id": str(i), "p_positive": .6,
                       "n_responses": 136, "n_participants": 34} for i in range(6)])
    manifest = {"dataset": "anllo", "seed": 0, "fold": 0, "condition": "us", "output_dir": "output",
                "train_item_ids": [str(i) for i in range(6)], "heldout_item_ids": ["6", "7"],
                "source_hashes": {"raw": "0" * 64}}
    for stem, filename in (("items", "items.json"), ("training_participants", "people.csv"),
                           ("participant_split", "split.csv"), ("training_targets", "targets.csv")):
        manifest[f"{stem}_path"] = filename
        manifest[f"{stem}_sha256"] = trainer.sha256_file(tmp_path / filename)
    manifest_path = tmp_path / "run_manifest.json"
    trainer.write_json(manifest_path, manifest)
    manifest, _, _ = trainer.validate_manifest(manifest_path)
    output = tmp_path / "output"
    output.mkdir()
    trainer.write_json(output / f"{stage}_manifest.json", manifest)
    trainer.write_json(output / "tokenization.json", {"candidate_token_ids": [32, 33],
        "n_training_prompts": 12, "max_prompt_length": 30,
        "prompt_lengths": [{"item_id": row["item_id"], "order": order, "tokens": 30}
                           for row in items for order in ("ab", "ba")]})

    def table(source):
        return [{**{key: manifest[key] for key in ("dataset", "seed", "fold", "condition")},
                 "source": source, "item_id": item["item_id"], "stratum": item["stratum"],
                 "order": order, "positive_response": item["positive_response"],
                 "heldout_item": item["item_id"] in manifest["heldout_item_ids"],
                 "token_id_a": 32, "token_id_b": 33,
                 **trainer.probability_metrics(math.log(.3), math.log(.2), order)}
                for item in items for order in ("ab", "ba")]

    metrics = {**{key: manifest[key] for key in ("dataset", "seed", "fold", "condition", "run_manifest_sha256")},
               "stage": stage, "base_model": "base-model", "centaur_adapter": "centaur",
               "model_load_seconds": 100, "elapsed_seconds": 200}
    if stage == "baseline":
        trainer.write_csv(output / "candidate_probabilities.csv", table("frozen_centaur") + table("base_llama"))
        trainer.write_json(output / "baseline_metrics.json", metrics)
        return manifest_path, output
    n_steps = 25 if stage == "train" else 1
    logs = [{"step": step, "full_item_loss": 1., "update_seconds": 2., "elapsed_seconds": step * 2.,
             "learning_rate": (5e-5 if stage == "smoke" else
                               5e-5 * (step - 1) / 3 if step <= 3 else 5e-5 * (26 - step) / 22)}
            for step in range(1, n_steps + 1)]
    (output / "optimizer_steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in logs))
    zero = {"all_items": 8, "both_orders": True, "max_abs_probability_difference": 0., "tolerance": 1e-6,
            "elapsed_seconds": 10.}
    trainer.write_json(output / "step0_metrics.json", zero)
    trainer.write_csv(output / "step0_choice_probabilities.csv", table("frozen_centaur") + table("zero_init_culture"))
    trainer.write_csv(output / "evaluation/candidate_probabilities.csv", table("finetuned_centaur"))
    trainer.write_csv(output / "evaluation/frozen_centaur_probabilities.csv", table("frozen_centaur"))
    trainer.write_json(output / "trainable_parameters.json", {
        "names": ["base.q_proj.lora_A.culture.weight", "base.q_proj.lora_B.culture.weight"], "numel": 100})
    adapter = output / "adapter/culture"
    trainer.write_json(adapter / "adapter_config.json", {
        "peft_type": "LORA", "r": 8, "lora_alpha": 8, "lora_dropout": 0, "use_rslora": True,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]})
    (adapter / "adapter_model.safetensors").write_bytes(b"nonempty fixture weights")
    metrics.update(complete_production_training=stage == "train", optimizer_steps=n_steps,
                   planned_production_steps=25, training_items=6, mirrored_prompts_per_update=12,
                   observed_training_responses=816, test_human_outcomes_loaded=False,
                   warmup_steps=3 if stage == "train" else 0, optimizer="AdamW", learning_rate=5e-5,
                   weight_decay=.01, objective="Equal-item, equal-order full-vocabulary soft-target cross-entropy",
                   first_loss=1., last_loss=1., training_seconds=50., peak_gpu_memory_bytes=100,
                   peak_gpu_memory_reserved_bytes=110, step0_equivalence=zero, saved_adapter_dir=str(adapter))
    trainer.write_json(output / "train_metrics.json", metrics)
    return manifest_path, output


@pytest.mark.parametrize("stage", ["smoke", "train", "baseline"])
def test_complete_results_get_stage_specific_verification(tmp_path, stage):
    manifest, output = make_result(tmp_path, stage)
    result = validate_run(manifest, output, stage)
    assert result["verified"] is True
    assert result["complete_production_training"] is (stage == "train")
    assert result["complete_baseline_evaluation"] is (stage == "baseline")
    assert result["evaluation_rows"] == 32
    assert json.loads((output / "verified_complete.json").read_text()) == result


def test_source_change_invalidates_previously_verified_marker(tmp_path):
    manifest, output = make_result(tmp_path, "train")
    validate_run(manifest, output, "train")
    with (tmp_path / "items.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_run(manifest, output, "train")
    assert not (output / "verified_complete.json").exists()


@pytest.mark.parametrize("field,bad_value,error", [
    ("p_positive", "nan", "Nonfinite"),
    ("valid_mass", "0.9", "Inconsistent"),
    ("heldout_item", "True", "heldout flag"),
    ("condition", "china", "incorrect condition"),
    ("token_id_a", "33", "candidate token IDs"),
])
def test_probability_records_are_numerically_and_semantically_checked(tmp_path, field, bad_value, error):
    manifest, output = make_result(tmp_path, "train")
    path = output / "evaluation/candidate_probabilities.csv"
    rows = trainer.read_csv(path)
    rows[0][field] = bad_value
    trainer.write_csv(path, rows)
    with pytest.raises(ValueError, match=error):
        validate_run(manifest, output, "train")
    assert not (output / "verified_complete.json").exists()


def test_both_baseline_models_must_cover_each_item_in_both_orders(tmp_path):
    manifest, output = make_result(tmp_path, "baseline")
    path = output / "candidate_probabilities.csv"
    rows = trainer.read_csv(path)
    trainer.write_csv(path, rows[:-1] + [rows[0]])
    with pytest.raises(ValueError, match="exactly once"):
        validate_run(manifest, output, "baseline")


def test_smoke_cannot_be_relabelled_production(tmp_path):
    manifest, output = make_result(tmp_path, "smoke")
    path = output / "train_metrics.json"
    metrics = json.loads(path.read_text())
    metrics["complete_production_training"] = True
    trainer.write_json(path, metrics)
    with pytest.raises(ValueError, match="Production-completion flag"):
        validate_run(manifest, output, "smoke")


def test_smoke_requires_actual_nonzero_learning_rate(tmp_path):
    manifest, output = make_result(tmp_path, "smoke")
    path = output / "optimizer_steps.jsonl"
    row = json.loads(path.read_text())
    row["learning_rate"] = 0
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="recorded update learning rate"):
        validate_run(manifest, output, "smoke")


def test_train_requires_all_twenty_five_logged_updates(tmp_path):
    manifest, output = make_result(tmp_path, "train")
    path = output / "optimizer_steps.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]))
    with pytest.raises(ValueError, match="each expected step"):
        validate_run(manifest, output, "train")


def test_step_zero_drift_is_recomputed_from_probabilities(tmp_path):
    manifest, output = make_result(tmp_path, "train")
    path = output / "step0_choice_probabilities.csv"
    rows = trainer.read_csv(path)
    for row in rows:
        if row["source"] == "zero_init_culture":
            row.update(trainer.probability_metrics(math.log(.4), math.log(.1), row["order"]))
    trainer.write_csv(path, rows)
    with pytest.raises(ValueError, match="step-zero maximum difference"):
        validate_run(manifest, output, "train")


def test_empty_saved_adapter_is_not_complete(tmp_path):
    manifest, output = make_result(tmp_path, "train")
    (output / "adapter/culture/adapter_model.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="nonempty weights"):
        validate_run(manifest, output, "train")


def test_validator_import_is_torch_free():
    completed = subprocess.run([sys.executable, "-c", "import sys; import scripts.validate_culture_extension_run; "
                                "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],
                               cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
