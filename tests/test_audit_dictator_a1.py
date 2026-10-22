from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.audit_dictator_a1 import Audit, check_candidates, histogram_candidates, score_participants, shared_bootstrap


def test_laplace_histogram_counts_and_unseen_responses() -> None:
    training = pd.DataFrame([{"GroupNo": group, "TokenChoice": response} for group in range(1, 26) for response in [0, 0, 20]])
    candidates = histogram_candidates(training)
    check_candidates(candidates)
    one = candidates.loc[candidates.GroupNo.eq(1)].set_index("response_value")
    assert one.loc[0, "conditional_probability"] == 3 / 24
    assert one.loc[20, "conditional_probability"] == 2 / 24
    assert one.loc[7, "conditional_probability"] == 1 / 24


def test_scoring_preserves_raw_conditional_distinction_and_units() -> None:
    candidates = histogram_candidates(pd.DataFrame([{"GroupNo": group, "TokenChoice": 0} for group in range(1, 26)]))
    candidates["full_vocab_log_probability"] += np.log(.8)
    people = pd.DataFrame([dict(Dataset=1, Nation=0, SubNo=1, Country="United States", GroupNo=group, TokenChoice=0) for group in range(1, 26)])
    scored = score_participants(people, candidates)
    assert np.isclose(scored.conditional_ll.iloc[0], np.log(2 / 22))
    assert np.isclose(scored.raw_response_ll.iloc[0], np.log(.8 * 2 / 22))


def test_reject_duplicate_fold_predictions() -> None:
    candidates = histogram_candidates(pd.DataFrame([{"GroupNo": group, "TokenChoice": 0} for group in range(1, 26)]))
    with pytest.raises(ValueError, match="Duplicate"):
        check_candidates(pd.concat([candidates, candidates.iloc[:1]]))


def test_shared_participants_get_identical_resamples_across_seeds() -> None:
    rows = [dict(seed=seed, Dataset=wave, Nation=0, SubNo=person, outcome="x", value=float(person + wave)) for seed in range(3) for wave in [1, 2] for person in range(12)]
    samples, seeds, outcomes = shared_bootstrap(pd.DataFrame(rows), 200, np.random.default_rng(11))
    assert seeds == [0, 1, 2]
    assert outcomes == ["x"]
    np.testing.assert_array_equal(samples[:,0], samples[:,1])
    np.testing.assert_array_equal(samples[:,1], samples[:,2])
    assert samples[:,0,0].std() > 0


def test_shared_bootstrap_keeps_paired_contrast_and_original_wave_weights() -> None:
    rows = [dict(seed=seed, Dataset=wave, Nation=0, SubNo=person, outcome=outcome, value=float(wave * 10 + offset)) for seed in range(2) for wave,n in [(1,20),(2,10)] for person in range(n) for outcome,offset in [("a",0),("b",3)]]
    samples, _, _ = shared_bootstrap(pd.DataFrame(rows), 100, np.random.default_rng(12))
    np.testing.assert_allclose(samples[:,:,0], 40 / 3)
    np.testing.assert_allclose(samples[:,:,1] - samples[:,:,0], 3)


def test_control_analysis_distinguishes_format_gain_from_conditional_gain(tmp_path) -> None:
    audit = Audit.__new__(Audit)
    audit.root = tmp_path
    audit.output = tmp_path / "audit"
    audit.output.mkdir()
    audit.inputs = set()
    audit.seed, audit.resamples = 17, 100
    humans = pd.DataFrame([dict(Dataset=1, Nation=nation, SubNo=1, Country=country, GroupNo=group, TokenChoice=0) for nation,country in [(0,"United States"),(1,"China")] for group in range(1,26)])
    audit.taking = humans.assign(Dataset=3)
    audit.test_rows = lambda _: humans
    parts = []
    for model, model_offset in [("llama",0),("centaur",.2)]:
        for task in ["giving", "taking"]:
            for variant, variant_offset in [("original",0),("us",.04),("china",.07)]:
                frame = pd.DataFrame([dict(GroupNo=group, response_value=response, conditional_probability=1/21, full_vocab_log_probability=np.log(.5/21)+model_offset+variant_offset, valid_response_mass=.5*np.exp(model_offset+variant_offset)) for group in range(1,26) for response in range(21)])
                parts.append(frame.assign(model_condition=model, task=task, prompt_variant=variant))
                if model == "centaur" and variant == "original":
                    previous = tmp_path / f"model_results/dictator_centaur70b_item_holdout/seed0/base/evaluations/{task}/candidate_probabilities.csv"
                    previous.parent.mkdir(parents=True)
                    frame.to_csv(previous,index=False)
    path = tmp_path / "model_results/dictator_a1_controls_20260930/candidate_probabilities.csv"
    path.parent.mkdir(parents=True)
    pd.concat(parts).to_csv(path,index=False)
    assert audit.control_analysis()
    summary = pd.read_csv(audit.output / "control_contrast_summary.csv")
    raw = summary.loc[summary.metric.eq("raw_response_ll")]
    base_gain = raw.loc[raw.contrast.eq("centaur_minus_llama_original"), "mean"]
    np.testing.assert_allclose(base_gain,.2)
    np.testing.assert_allclose(summary.loc[summary.metric.eq("conditional_ll"), "mean"],0,atol=1e-14)
    matched = raw.loc[raw.Country.eq("China") & raw.contrast.eq("centaur_matched_minus_opposite"), "mean"]
    np.testing.assert_allclose(matched,.03)
