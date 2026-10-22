import numpy as np
import pandas as pd

from scripts.audit_dictator_response_prior import (
    apply_offsets, fit_response_offsets, log_softmax, objective_derivatives, training_counts,
)


def test_fit_recovers_known_shared_response_shift():
    base = np.array([[.2, .3, .5], [.5, .4, .1], [.1, .7, .2]])
    truth = np.array([-.4, .8, -.4])
    target = np.exp(log_softmax(np.log(base) + truth))
    estimated, diagnostics = fit_response_offsets(np.log(base), target * 100000, ridge=0)
    np.testing.assert_allclose(estimated, truth, atol=1e-7)
    assert diagnostics["converged"]


def test_objective_gradient_and_hessian_match_finite_differences():
    base = np.log(np.array([[.2, .3, .5], [.5, .4, .1]]))
    counts = np.array([[4, 2, 1], [1, 3, 9]])
    theta = np.array([.2, -.3])
    value, gradient, hessian = objective_derivatives(theta, base, counts, .001)
    epsilon = 1e-5
    for j in range(2):
        delta = np.eye(2)[j] * epsilon
        plus = objective_derivatives(theta + delta, base, counts, .001)
        minus = objective_derivatives(theta - delta, base, counts, .001)
        np.testing.assert_allclose(gradient[j], (plus[0] - minus[0]) / (2 * epsilon), atol=1e-9)
        np.testing.assert_allclose(hessian[:, j], (plus[1] - minus[1]) / (2 * epsilon), atol=1e-9)


def test_calibration_retains_mass_and_base_between_target_log_odds():
    rng = np.random.default_rng(42)
    base = np.exp(log_softmax(rng.normal(size=(2, 21))))
    rows = [dict(GroupNo=i + 1, response_value=v, conditional_probability=base[i, v],
                 valid_response_mass=[.8, .65][i]) for i in range(2) for v in range(21)]
    calibrated = apply_offsets(pd.DataFrame(rows), rng.normal(size=21))
    np.testing.assert_allclose(calibrated.groupby("GroupNo").full_vocab_probability.sum(), [.8, .65])
    new = calibrated.conditional_probability.to_numpy().reshape(2, 21)
    old_delta = np.log(base[0, 0] / base[0, 1]) - np.log(base[1, 0] / base[1, 1])
    new_delta = np.log(new[0, 0] / new[0, 1]) - np.log(new[1, 0] / new[1, 1])
    np.testing.assert_allclose(old_delta, new_delta, atol=1e-12)


def test_training_counts_exclude_heldout_people_targets_and_taking():
    behavior = pd.DataFrame([
        dict(Dataset=d, Nation=0, SubNo=s, GroupNo=g, response_value=(s + g + d) % 21)
        for d in [1, 3] for s in [1, 2] for g in [1, 2, 3]
    ])
    selected = pd.DataFrame([dict(Dataset=1, Nation=0, SubNo=1)])
    initial, people = training_counts(behavior, selected, [1, 2])
    altered = behavior.copy()
    excluded = (altered.Dataset != 1) | (altered.SubNo != 1) | (altered.GroupNo == 3)
    altered.loc[excluded, "response_value"] = 20
    after, _ = training_counts(altered, selected, [1, 2])
    np.testing.assert_array_equal(initial, after)
    assert people == 1 and initial.sum() == 2
