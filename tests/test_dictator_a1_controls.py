from contextlib import contextmanager
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_dictator_a1_controls as controls
from scripts.evaluate_dictator_giving_base import build_prompt
from scripts.evaluate_dictator_taking_transfer import build_taking_prompt


def test_original_prompts_are_identical_and_country_prefix_is_only_change():
    for task, builder in [("giving", build_prompt), ("taking", build_taking_prompt)]:
        original = builder("a farmer")
        assert controls.control_prompt(task, "a farmer", "original") == original
        for variant, sentence in [
            ("us", "This participant lives in the United States."),
            ("china", "This participant lives in China."),
        ]:
            assert controls.control_prompt(task, "a farmer", variant) == sentence + "\n\n" + original


def test_load_items_rejects_missing_or_duplicate_items(tmp_path):
    path = tmp_path / "groups.csv"
    pd.DataFrame({"No.": [1] * 25, "English": ["farmer"] * 25}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="exactly one row"):
        controls.load_items(path)
    pd.DataFrame({"No.": range(1, 26), "English": ["farmer"] * 24 + [""]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="nonempty"):
        controls.load_items(path)


def test_prepare_only_has_complete_manifest_and_does_not_load_models(tmp_path, monkeypatch):
    path = tmp_path / "groups.csv"
    pd.DataFrame({"No.": range(25, 0, -1), "English": [f"item {i}" for i in range(25, 0, -1)]}).to_csv(path, index=False)

    def unexpected_load(*args, **kwargs):
        raise AssertionError("Model loading attempted in prepare-only mode")

    monkeypatch.setattr(controls, "load_4bit_base", unexpected_load)
    monkeypatch.setattr(controls, "load_tokenizer", unexpected_load)
    output = tmp_path / "output"
    controls.run(controls.parse_args(["--group-path", str(path), "--output-dir", str(output), "--prepare-only"]))
    records = [json.loads(line) for line in (output / "prompts.jsonl").read_text().splitlines()]
    metadata = json.loads((output / "metadata.json").read_text())
    assert len(records) == 150
    assert len({(r["task"], r["prompt_variant"], r["GroupNo"]) for r in records}) == 150
    assert all(r["prompt_sha256"] == hashlib.sha256(r["prompt"].encode()).hexdigest() for r in records)
    assert records[0]["GroupNo"] == 1
    assert metadata["status"] == "prepared"
    assert metadata["expected_candidate_rows"] == 6300
    assert not (output / "candidate_probabilities.csv").exists()


def test_scores_preserve_raw_probability_mass_and_response_mapping():
    record = controls.build_prompt_records(pd.DataFrame({"GroupNo": [1], "item": ["farmer"]}))[0]
    full = np.arange(1, 22, dtype=float)
    full = full / full.sum() * 0.3
    logprobs = np.log(full)
    original = logprobs.copy()
    candidates, prediction = controls.score_rows(record, "llama", "base", list(range(100, 121)), logprobs)
    assert [row["response_value"] for row in candidates] == list(range(21))
    assert [row["token_id"] for row in candidates] == list(range(100, 121))
    assert np.allclose([row["full_vocab_log_probability"] for row in candidates], original)
    assert np.allclose([row["full_vocab_probability"] for row in candidates], full)
    assert np.allclose([row["conditional_probability"] for row in candidates], full / 0.3)
    assert np.isclose(prediction["valid_response_mass"], 0.3)
    assert prediction["expected_tokens"] == pytest.approx(np.dot(full / 0.3, np.arange(21)))
    assert prediction["modal_tokens"] == 20
    assert prediction["prompt"] == record["prompt"]
    assert all(row["prompt_sha256"] == record["prompt_sha256"] for row in candidates)
    assert np.array_equal(logprobs, original)


@pytest.mark.parametrize("logprobs", [np.zeros(21), np.full(21, np.nan), np.full(20, -5.)])
def test_invalid_scores_are_rejected(logprobs):
    with pytest.raises(ValueError):
        controls.score_rows({"prompt": "x"}, "llama", "base", list(range(21)), logprobs)


class FakePeftModel:
    def __init__(self):
        self.layer = SimpleNamespace(lora_A={}, disable_adapters=False, merged_adapters=[], active_adapters=[controls.CENTAUR_ADAPTER_NAME])

    def modules(self):
        return iter([self.layer])

    @contextmanager
    def disable_adapter(self):
        self.layer.disable_adapters = True
        try:
            yield
        finally:
            self.layer.disable_adapters = False


def test_disabling_adapter_is_numerically_checked_and_restored(monkeypatch):
    model = FakePeftModel()
    reference = np.linspace(-9., -2., 21)
    records = [{"prompt": "original giving", "prompt_sha256": "giving"}, {"prompt": "original taking", "prompt_sha256": "taking"}]
    references = [(list(range(21)), reference)] * 2

    def scorer(model, *_):
        assert model.layer.disable_adapters is True
        return list(range(21)), reference.copy(), None

    monkeypatch.setattr(controls, "score_single_token_candidates", scorer)
    result = controls.verify_disabled_equivalence(model, None, records, references, None, 1e-6)
    assert result["passed"]
    assert result["n_reference_prompts"] == 2
    assert result["max_abs_candidate_logprob_difference"] == 0
    assert not model.layer.disable_adapters

    monkeypatch.setattr(controls, "score_single_token_candidates", lambda *_: (list(range(21)), reference + 0.01, None))
    with pytest.raises(RuntimeError, match="did not reproduce"):
        controls.verify_disabled_equivalence(model, None, records, references, None, 1e-6)
    assert not model.layer.disable_adapters


def test_adapter_state_rejects_merged_or_active_layers_in_base_control():
    model = FakePeftModel()
    with pytest.raises(RuntimeError, match="disable_adapters=True"):
        controls.assert_adapter_state(model, disabled=True)
    model.layer.merged_adapters = [controls.CENTAUR_ADAPTER_NAME]
    with pytest.raises(RuntimeError, match="Merged adapter weights"):
        controls.assert_adapter_state(model, disabled=False)
