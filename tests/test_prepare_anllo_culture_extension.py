"""Response-side and prompt-fidelity guards for the lottery preparation."""
import json

import pytest

from scripts.prepare_anllo_culture_extension import decode_response, make_prompt, normalize_country


def raw_response(*, high_on_left, chose_left, condition=8, p_10=0.25, p_1=0.75):
    left = [p_10, 10] if high_on_left else [p_1, 1]
    right = [p_1, 1] if high_on_left else [p_10, 10]
    return {
        "Participant.Private.ID": "stable-person", "ParticipantID": "not-the-identifier",
        "Country": "USA", "WhichPhase": "3", "Condition": str(condition),
        "InvertedPosition": "0", "Magnitude1": str(left[1]), "Magnitude2": str(right[1]),
        "Proba1": str(left[0]), "Proba2": str(right[0]),
        "Option1": json.dumps(left), "Option2": json.dumps(right),
        "ChoseLorR": "-1" if chose_left else "1",
        "ChoseGoodorBad": str(int(high_on_left == chose_left)),
        "TrialPerCond": "2", "TrialNumber": "18",
    }


@pytest.mark.parametrize("high_on_left", [False, True])
@pytest.mark.parametrize("chose_left", [False, True])
def test_reconstructs_all_sides_without_using_unstable_person_id(high_on_left, chose_left):
    row = raw_response(high_on_left=high_on_left, chose_left=chose_left)
    decoded = decode_response(row, {"condition": 8, "p_10": 0.25, "p_1": 0.75})
    assert decoded == {
        "participant_id": "stable-person", "country": "us", "item_id": "8",
        "response_positive": int(high_on_left == chose_left), "repeat_id": 1,
        "positive_on_left": int(high_on_left), "trialnum": 18,
    }


@pytest.mark.parametrize("field,bad_value,error", [
    ("Option1", "[0.75,10]", "Serialized lottery"),
    ("Proba1", "0.5", "Serialized lottery"),
    ("ChoseLorR", "0", "Invalid side choice"),
    ("ChoseGoodorBad", "0", "disagrees with released"),
    ("InvertedPosition", "1", "re-audit side mapping"),
    ("TrialPerCond", "5", "Invalid within-pair repetition"),
])
def test_rejects_ambiguous_or_inconsistent_raw_rows(field, bad_value, error):
    row = raw_response(high_on_left=True, chose_left=True)
    row[field] = bad_value
    with pytest.raises(ValueError, match=error):
        decode_response(row, {"condition": 8, "p_10": 0.25, "p_1": 0.75})


def test_different_item_probability_cannot_silently_relabel_a_response():
    row = raw_response(high_on_left=True, chose_left=True)
    with pytest.raises(ValueError, match="disagrees with frozen protocol"):
        decode_response(row, {"condition": 8, "p_10": 0.75, "p_1": 0.75})


def test_mirrored_prompts_keep_outcomes_probabilities_and_semantic_mapping():
    item = {"condition": 9, "p_10": 0.25, "p_1": 1}
    ab, ba = make_prompt(item, True), make_prompt(item, False)
    high = "10 points with a 25% chance; 0 points with a 75% chance."
    low = "1 point with a 100% chance; 0 points with a 0% chance."
    assert f"A: {high}" in ab and f"B: {low}" in ab
    assert f"A: {low}" in ba and f"B: {high}" in ba
    assert ab.endswith("You choose <<") and ba.endswith("You choose <<")
    for prompt in (ab, ba):
        assert not any(word in prompt.lower() for word in ["china", "usa", "previous", "received", "feedback"])


def test_country_labels_are_shared_with_the_existing_us_china_analysis():
    assert normalize_country("USA") == "us"
    assert normalize_country("China") == "china"
    assert normalize_country("Morocco") == "morocco"
