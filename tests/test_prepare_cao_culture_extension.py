"""Scientific-label, stimulus mapping, and raw reconstruction checks for Cao."""

import json
from collections import Counter

import pytest

from scripts import prepare_cao_culture_extension as cao


def raw_page(response="hammer ", *, options="plank,hammer "):
    return {
        "subject": "retained_person", "culture": "US", "trial_index": "143",
        "prompt": "<b>saw</b>,<b>choose cat</b>",
        "options": options + ",dog,cat", "stimulus_type": "triads,catch",
        "question_order": "1,0",
        "response": json.dumps({"saw_triads": response, "choose cat_catch": "cat"}),
    }


def saw_mapping():
    return {("US", "saw"): {
        "item_id": "saw", "positive": "hammer", "negative": "plank",
        "source_positive_untrimmed": "hammer",
    }}


def test_whitespace_correction_preserves_semantic_choice_and_actual_display_order():
    responses, excluded = cao.parse_raw_page(raw_page(), saw_mapping())
    assert responses[0]["response_positive"] == 1
    assert responses[0]["source_untrimmed_key_match"] == 0
    assert responses[0]["source_positive_first"] == 0
    assert responses[0]["positive_on_left"] == ""  # original alternatives were vertical
    assert responses[0]["source_question_position"] == 1
    assert responses[0]["repeat_id"] == 0
    assert len(excluded) == 1 and excluded[0]["stimulus_type"] == "catch"


def test_response_not_displayed_and_wrong_alternative_pair_fail_closed():
    with pytest.raises(ValueError, match="not a displayed alternative"):
        cao.parse_raw_page(raw_page("screwdriver"), saw_mapping())
    with pytest.raises(ValueError, match="disagree with validated key"):
        cao.parse_raw_page(raw_page("hammer", options="screwdriver,hammer"), saw_mapping())


def test_retained_membership_does_not_depend_on_derived_labels_or_corrupt_item_text(tmp_path):
    path = tmp_path / "processed.csv"
    path.write_bytes(
        b"subject,culture,task_name,trial_info,resp\n"
        b"p1,US,TD,anything,FALSE\n"
        b"p1,US,TD,other,TRUE\n"
        b"p2,CN,TD,\xad\x00\xe1,broken\n"
        b"p3,CN,OTHER,irrelevant,TRUE\n"
    )
    assert cao.retained_participants(path) == {"p1": "us", "p2": "china"}


@pytest.fixture(scope="module")
def raw_dir():
    directory = cao.REPO_ROOT / "data_cache/culture_extensions_20261021/cao/raw"
    if not all((directory / name).exists() for name in cao.SOURCES):
        pytest.skip("Pinned source cache absent; run prepare_cao_culture_extension.py first")
    cao.ensure_sources(directory, offline=True)
    return directory


def test_all_bilingual_stimuli_match_the_key_and_counterbalanced_prompts(raw_dir):
    mapping = cao.load_item_mapping(raw_dir)
    assert len(mapping) == 30
    items = cao.make_items(mapping)
    assert {item["item_id"] for item in items} == set(cao.TRANSLATIONS)
    for item in items:
        source = item["source_details"]
        category, relation = source["english_category_option"], source["english_relation_option"]
        assert f"A: {category}\nB: {relation}" in item["prompt_ab"]
        assert f"A: {relation}\nB: {category}" in item["prompt_ba"]
        assert item["prompt_ab"].endswith("You choose <<")
        assert item["prompt_ba"].endswith("You choose <<")
        assert "most closely related" in item["prompt_ab"]
        assert "category" not in item["prompt_ab"]
        assert "China" not in item["prompt_ab"] and "United States" not in item["prompt_ab"]
        assert source["mandarin_cue"] == cao.TRANSLATIONS[item["item_id"]]


def test_complete_raw_reconstruction_reproduces_frozen_audit_and_excludes_fillers(raw_dir):
    participants = cao.retained_participants(raw_dir / "tidy_main.csv")
    mapping = cao.load_item_mapping(raw_dir)
    responses, excluded = cao.reconstruct(raw_dir, participants, mapping)
    protocol = json.loads((cao.REPO_ROOT / "designs/culture_extension_protocol_20261021.json").read_text())
    result = cao.audit(participants, responses, excluded, protocol)
    assert result["total_critical_responses"] == 7095
    assert result["category_choices"] == {"us": 3266, "china": 1240}
    assert result["participants"] == {"us": 295, "china": 178}
    assert result["whitespace_trimmed_responses"] == 304
    assert Counter(row["stimulus_type"] for row in excluded) == {"filler": 7095, "catch": 946}
    assert all(row["item_id"] in cao.TRANSLATIONS for row in responses)
    assert all(count == 15 for count in Counter(row["participant_id"] for row in responses).values())


def test_duplicate_or_missing_items_cannot_pass_audit(raw_dir):
    participants = cao.retained_participants(raw_dir / "tidy_main.csv")
    responses, excluded = cao.reconstruct(raw_dir, participants, cao.load_item_mapping(raw_dir))
    protocol = json.loads((cao.REPO_ROOT / "designs/culture_extension_protocol_20261021.json").read_text())
    # Replace an item with a duplicate while preserving all response totals.
    tampered = [dict(row) for row in responses]
    tampered[1]["item_id"] = tampered[0]["item_id"]
    with pytest.raises(ValueError, match="Repeated participant/item"):
        cao.audit(participants, tampered, excluded, protocol)
