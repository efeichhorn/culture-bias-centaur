"""Protocol guards independent of PyTorch and GPU/model availability."""
import csv
import json
import math
from pathlib import Path

import pytest

from scripts.culture_extension_centaur70b import (
    prepare_examples,
    probability_metrics,
    sha256_file,
    soft_target_loss_from_logprobs,
    validate_manifest,
    verified_candidate_ids,
)


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": ([1] if add_special_tokens else []) + [ord(char) + 2 for char in text]}


def test_actual_prefix_and_continuation_tokenization_is_verified():
    prefix, ids = verified_candidate_ids(TinyTokenizer(), "Choose <<")
    assert len(prefix) == 10
    assert ids == [ord("A") + 2, ord("B") + 2]

    class MergingTokenizer(TinyTokenizer):
        def __call__(self, text, add_special_tokens=True):
            result = super().__call__(text, add_special_tokens)
            if text.endswith("<<A"):
                result["input_ids"] = result["input_ids"][:-2] + [900]
            return result

    with pytest.raises(ValueError, match="exactly one token"):
        verified_candidate_ids(MergingTokenizer(), "Choose <<")
    with pytest.raises(ValueError, match="truncation is forbidden"):
        verified_candidate_ids(TinyTokenizer(), "x" * 768)


def test_mirrored_orders_preserve_semantic_targets_and_total_item_weight():
    items = [{"item_id": str(i), "stratum": "tradeoff", "positive_response": "ten points",
              "prompt_ab": f"{i}: A ten B one. You choose <<",
              "prompt_ba": f"{i}: A one B ten. You choose <<"} for i in range(3)]
    targets = [{"item_id": "0", "p_positive": .8}, {"item_id": "2", "p_positive": .3}]
    all_examples, training, _ = prepare_examples(items, targets, TinyTokenizer())
    assert len(all_examples) == 6
    assert len(training) == 4
    assert sum(row["weight"] for row in training) == 1
    assert {row["item_id"] for row in training} == {"0", "2"}
    assert training[0]["target_probabilities"] == pytest.approx([.8, .2])
    assert training[1]["target_probabilities"] == pytest.approx([.2, .8])
    # Arbitrary microbatch boundaries preserve the full-item, equal-order loss.
    contributions = [row["weight"] * soft_target_loss_from_logprobs(-.7, -1.4,
                     .8 if row["item_id"] == "0" else .3, row["order"]) for row in training]
    assert sum(contributions[:3]) + sum(contributions[3:]) == pytest.approx(sum(contributions))


def test_loss_is_full_vocabulary_and_prediction_is_conditional():
    logp_a, logp_b = math.log(.2), math.log(.1)
    assert soft_target_loss_from_logprobs(logp_a, logp_b, .7, "ab") == pytest.approx(-.7 * logp_a - .3 * logp_b)
    assert soft_target_loss_from_logprobs(logp_a, logp_b, .7, "ba") == pytest.approx(-.3 * logp_a - .7 * logp_b)
    ab = probability_metrics(logp_a, logp_b, "ab")
    ba = probability_metrics(logp_a, logp_b, "ba")
    assert ab["valid_mass"] == pytest.approx(.3)
    assert ab["p_positive"] == pytest.approx(2 / 3)
    assert ba["p_positive"] == pytest.approx(1 / 3)
    # Tiny total mass is safe: conditional probability is computed in log space.
    assert probability_metrics(-1000, -1001, "ab")["p_positive"] == pytest.approx(1 / (1 + math.exp(-1)))


def dump_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def manifest_files(tmp_path):
    items = [{"item_id": str(i), "stratum": "critical", "positive_response": "category",
              "prompt_ab": f"{i}: A category B relation. You choose <<",
              "prompt_ba": f"{i}: A relation B category. You choose <<"} for i in range(15)]
    (tmp_path / "items.json").write_text(json.dumps(items))
    split = [{"participant_id": str(i), "country": "us", "split": "train" if i < 124 else "test"}
             for i in range(126)]
    split += [{"participant_id": "unused", "country": "us", "split": "unused"}]
    dump_csv(tmp_path / "split.csv", split)
    training = [{"participant_id": str(i), "country": "us"} for i in range(124)]
    dump_csv(tmp_path / "people.csv", training)
    targets = [{"item_id": str(i), "p_positive": .6, "n_responses": 124, "n_participants": 124} for i in range(12)]
    dump_csv(tmp_path / "targets.csv", targets)
    manifest = {"dataset": "cao", "seed": 0, "fold": 0, "condition": "us", "output_dir": "output",
                "train_item_ids": [str(i) for i in range(12)], "heldout_item_ids": ["12", "13", "14"],
                "source_hashes": {"raw": "0" * 64}}
    for stem, filename in (("items", "items.json"), ("participant_split", "split.csv"),
                           ("training_participants", "people.csv"), ("training_targets", "targets.csv")):
        manifest[f"{stem}_path"] = filename
        manifest[f"{stem}_sha256"] = sha256_file(tmp_path / filename)
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_manifest_paths_and_complete_training_partition(manifest_files):
    path, _ = manifest_files
    manifest, items, targets = validate_manifest(path)
    assert Path(manifest["items_path"]).is_absolute()
    assert len(items) == 15 and len(targets) == 12
    assert manifest["run_manifest_sha256"] == sha256_file(path)


def test_source_mutation_is_rejected(manifest_files):
    path, _ = manifest_files
    with (path.parent / "items.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_manifest(path)


def test_heldout_item_leakage_is_rejected(manifest_files):
    path, manifest = manifest_files
    manifest["heldout_item_ids"] = ["0", "13", "14"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="disjoint partition"):
        validate_manifest(path)


def test_test_participant_leakage_is_rejected_even_with_fresh_hash(manifest_files):
    path, manifest = manifest_files
    people_path = path.parent / "people.csv"
    people = [{"participant_id": str(i), "country": "us"} for i in range(1, 125)]
    dump_csv(people_path, people)
    manifest["training_participants_sha256"] = sha256_file(people_path)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="validation/test/unknown"):
        validate_manifest(path)


def test_observed_response_budget_cannot_be_upsampled(manifest_files):
    path, manifest = manifest_files
    targets_path = path.parent / "targets.csv"
    targets = [{"item_id": str(i), "p_positive": .6, "n_responses": 248, "n_participants": 124} for i in range(12)]
    dump_csv(targets_path, targets)
    manifest["training_targets_sha256"] = sha256_file(targets_path)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="observed-response budget"):
        validate_manifest(path)
