from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.audit_dictator_a1 import check_candidates, histogram_candidates, score_participants, shared_bootstrap


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
