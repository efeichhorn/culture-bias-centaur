from __future__ import annotations

import numpy as np

from scripts.evaluate_dictator_giving_base import (
    RESPONSE_VALUES,
    build_prompt,
    candidate_probabilities_from_logprobs,
    parse_generated_token,
    percent_to_token_choice,
    stable_softmax,
)


def test_prompt_uses_closed_double_angle_response_set() -> None:
    prompt = build_prompt("farmer")
    assert "farmer" in prompt
    assert "20 tokens" in prompt
    assert "<<0>>" in prompt
    assert "<<20>>" in prompt
    assert "<<1>>, <<2>>" in prompt
    assert "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20" in prompt
    assert "..." not in prompt
    assert prompt.endswith("You enter <<")
    assert len(RESPONSE_VALUES) == 21


def test_percent_to_token_choice_rounds_half_up_and_preserves_boundaries() -> None:
    values = np.array([0, 2, 3, 5, 7, 8, 97, 98, 100])
    assert percent_to_token_choice(values).tolist() == [0, 0, 1, 1, 1, 2, 19, 20, 20]


def test_generated_token_parser_requires_one_valid_source_response() -> None:
    assert parse_generated_token("0") == 0
    assert parse_generated_token(" 20 ") == 20
    assert parse_generated_token("21") is None
    assert parse_generated_token("<<10>>") is None
    assert parse_generated_token("10.") is None


def test_stable_softmax_is_normalized() -> None:
    probabilities = stable_softmax([-1000.0, -1001.0, -1002.0])
    assert np.isclose(probabilities.sum(), 1.0)
    assert probabilities.argmax() == 0


def test_candidate_probabilities_keep_full_vocab_mass_separate() -> None:
    logprobs = np.log(np.array([0.1, 0.2, 0.05]))
    original = logprobs.copy()
    full, conditional, mass = candidate_probabilities_from_logprobs(logprobs)
    assert np.allclose(full, [0.1, 0.2, 0.05])
    assert np.isclose(mass, 0.35)
    assert np.allclose(conditional, np.array([0.1, 0.2, 0.05]) / 0.35)
    assert np.array_equal(logprobs, original)
