"""Check semantic calibration, honest fold coverage, and paired aggregation."""

import json
import math

import pandas as pd
import pytest

from scripts import analyze_culture_extensions as analysis


def pair(item, p_ab, p_ba=None, source="frozen_centaur"):
    p_ba = p_ab if p_ba is None else p_ba
    rows = []
    for order, probability in (("ab", p_ab), ("ba", p_ba)):
        a = probability if order == "ab" else 1-probability
        rows.append({"source": source, "item_id": item, "stratum": "critical", "order": order,
                     "logp_a": math.log(a * 0.8), "logp_b": math.log((1-a) * 0.8),
                     "p_positive": probability, "valid_mass": .8, "log_valid_mass": math.log(.8)})
    return rows


def test_orders_are_mapped_to_semantics_before_averaging_and_offsetting():
    rows = pair("i", .8, .6)
    averaged = analysis.average_orders(rows)
    assert averaged["p_positive"] == pytest.approx(.7)
    assert averaged["order_difference_ab_minus_ba"] == pytest.approx(.2)
    shifted = analysis.average_orders(rows, math.log(2))
    assert shifted["p_positive"] == pytest.approx(((8/9) + .75)/2)
    assert shifted["original_valid_mass"] == pytest.approx(.8)
    with pytest.raises(ValueError, match="one ab and one ba"):
        analysis.average_orders([rows[0], rows[0]])


def test_scalar_prior_fits_semantic_preference_not_a_position_bias():
    rows = pair("x", .5) + pair("y", .5)
    targets = [{"item_id": item, "p_positive": .8, "n_responses": 100} for item in ("x", "y")]
    delta, diagnostics = analysis.fit_semantic_offset(targets, rows)
    assert 1.37 < delta < math.log(4)
    assert abs(diagnostics["gradient_at_solution"]) < 1e-10
    assert diagnostics["objective_at_solution"] < diagnostics["objective_at_zero"]
    assert analysis.average_orders(rows[:2], delta)["p_positive"] > .79
    # Mirroring which prompt is labeled ab/ba while swapping A/B probabilities
    # leaves semantic logits and the fitted scalar unchanged.
    mirrored = [{**row, "order": "ba" if row["order"] == "ab" else "ab",
                 "logp_a": row["logp_b"], "logp_b": row["logp_a"]} for row in rows]
    assert analysis.fit_semantic_offset(targets, mirrored)[0] == pytest.approx(delta)


def test_missing_orders_and_unequal_budgets_do_not_silently_enter_prior_fit():
    targets = [{"item_id": "x", "p_positive": .7, "n_responses": 100}]
    with pytest.raises(ValueError, match="one ab and one ba"):
        analysis.fit_semantic_offset(targets, pair("x", .5)[:1])
    targets.append({"item_id": "y", "p_positive": .7, "n_responses": 99})
    with pytest.raises(ValueError, match="Unequal training"):
        analysis.fit_semantic_offset(targets, pair("x", .5) + pair("y", .5))


def test_smoke_and_short_training_never_pass_production_gate(tmp_path):
    marker = {"verified": True, "stage": "smoke", "optimizer_steps": 1,
              "complete_production_training": False, "verified_files_sha256": {}}
    (tmp_path / "verified_complete.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="smoke is excluded"):
        analysis.verified_output(tmp_path, "train")
    marker.update(stage="train", complete_production_training=True)
    (tmp_path / "verified_complete.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="25-update"):
        analysis.verified_output(tmp_path, "train")


def test_modified_probability_file_is_rejected_even_with_a_completion_marker(tmp_path):
    path = tmp_path / "candidate_probabilities.csv"
    path.write_text("source,p_positive\nfrozen_centaur,0.5\n")
    marker = {"verified_files_sha256": {"candidate_probabilities.csv": analysis.sha256(path)}}
    path.write_text("source,p_positive\nfrozen_centaur,0.9\n")
    with pytest.raises(ValueError, match="changed verified probabilities"):
        analysis.verified_csv(tmp_path, "candidate_probabilities.csv", marker)


def prediction(condition, item, fold, p):
    return {"dataset": "cao", "seed": 0, "condition": condition, "source": "finetuned_centaur",
            "item_id": item, "stratum": "critical", "fold": fold,
            **analysis.average_orders(pair(item, p))}


def synthetic_inputs():
    responses = pd.DataFrame([
        {"participant_id": person, "country": country, "item_id": item, "response_positive": answer,
         "repeat_id": repeat}
        for person, country, answer in (("u1", "us", 1), ("u2", "us", 1), ("c1", "china", 0))
        for item in ("x", "y") for repeat in (0, 1)
    ])
    datasets = {"cao": ({"x": {}, "y": {}}, responses)}
    splits = {("cao", 0): [{"participant_id": person, "country": country, "split": "test"}
                           for person, country in (("u1", "us"), ("u2", "us"), ("c1", "china"))]}
    expected = {("cao", 0, condition): {0, 1} for condition in ("us", "china")}
    return datasets, splits, expected


def test_partial_fold_coverage_and_country_balanced_participant_scores():
    datasets, splits, expected = synthetic_inputs()
    predictions = [prediction("us", "x", 0, .8)]
    people, seeds = analysis.participant_and_seed_metrics(predictions, datasets, splits, expected)
    assert len(people) == 6  # all and critical, three complete participants
    assert all(row["coverage"] == "partial" and row["n_items"] == 1 for row in people)
    balanced = next(row for row in seeds if row["country"] == "balanced_us_china" and row["stratum"] == "all")
    assert balanced["human_p_positive"] == .5  # not the sample-size-weighted 2/3
    assert balanced["conditional_nll"] == pytest.approx(-.5*(math.log(.8)+math.log(.2)))
    assert analysis.coverage({0, 1}, {0, 1}) == "complete"
    assert analysis.coverage(set(), {0, 1}) == "missing"


def test_population_contrast_pairs_identical_folds_instead_of_unmatched_partial_means():
    datasets, splits, expected = synthetic_inputs()
    predictions = [prediction("us", "x", 0, .8), prediction("us", "y", 1, .9),
                   prediction("china", "x", 0, .6)]
    contrasts, people = analysis.paired_contrasts(predictions, datasets, splits, expected)
    row = next(row for row in contrasts if row["stratum"] == "all")
    assert row["prediction_shift_pp"] == pytest.approx(20)
    assert row["item_ids"] == "x" and row["paired_folds"] == "0"
    assert row["coverage"] == "partial"
    assert len(people) == 6


def test_duplicate_heldout_item_across_folds_is_rejected():
    datasets, splits, expected = synthetic_inputs()
    with pytest.raises(ValueError, match="multiple folds"):
        analysis.participant_and_seed_metrics(
            [prediction("us", "x", 0, .8), prediction("us", "x", 1, .9)], datasets, splits, expected)


def test_seen_items_remain_separate_per_fold_secondary_results():
    datasets, splits, expected = synthetic_inputs()
    predictions = [{**prediction("us", "x", fold, probability), "evaluation_item_scope": "seen_items"}
                   for fold, probability in ((0, .8), (1, .6))]
    people, summaries = analysis.seen_item_secondary_metrics(predictions, datasets, splits, expected)
    assert len(people) == 12  # two folds x two strata x three test participants
    assert {row["fold"] for row in summaries} == {0, 1}
    assert all(row["coverage"] == "per_fold_secondary" and row["evaluation_item_scope"] == "seen_items" for row in summaries)
    assert all(row["n_items"] == 1 for row in summaries)


def test_frozen_audit_covers_all_countries_without_any_production_training(tmp_path):
    output = tmp_path / "baseline/anllo"
    output.mkdir(parents=True)
    rows = pair("5", .7, source="frozen_centaur") + pair("5", .6, source="base_llama")
    for row in rows:
        row["stratum"] = "tradeoff"
    path = output / "candidate_probabilities.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    marker = {"verified": True, "stage": "baseline", "complete_baseline_evaluation": True,
              "dataset": "anllo", "run_manifest_sha256": "frozen_manifest",
              "verified_files_sha256": {"candidate_probabilities.csv": analysis.sha256(path)}}
    (output / "verified_complete.json").write_text(json.dumps(marker))
    responses = pd.DataFrame([
        {"participant_id": str(country), "country": f"country_{country}", "item_id": "5",
         "response_positive": country % 2, "repeat_id": repeat}
        for country in range(11) for repeat in range(4)
    ])
    datasets = {"anllo": ({"5": {"stratum": "tradeoff"}}, responses)}
    people, summaries, statuses = analysis.frozen_country_audit(
        datasets, tmp_path, {"runs": [{"dataset": "anllo", "manifest_sha256": "frozen_manifest"}]})
    assert statuses[0]["status"] == "verified_baseline"
    assert len(statuses[0]["countries"]) == 11
    assert len(people) == len(summaries) == 44  # 11 countries x 2 models x all/tradeoff
    assert all(row["sample_scope"] == "all_retained_participants_frozen_audit" for row in summaries)
    assert all(row["n_participants"] == 1 for row in summaries)


def test_frozen_reference_check_cannot_hide_opposing_order_drift():
    key = ("frozen_centaur", "i")
    ref = {key: pair("i", .5, .5)}
    drift = {key: pair("i", .52, .48)}
    assert analysis.average_orders(ref[key])["p_positive"] == analysis.average_orders(drift[key])["p_positive"]
    assert analysis.frozen_reference_difference(ref, drift, ["i"]) == pytest.approx(.02)
    assert analysis.frozen_reference_difference(ref, ref, ["i"]) == 0
    invalid_mass = {key: [{**row, "valid_mass": .7} for row in ref[key]]}
    assert analysis.frozen_reference_difference(ref, invalid_mass, ["i"]) == pytest.approx(.1)
