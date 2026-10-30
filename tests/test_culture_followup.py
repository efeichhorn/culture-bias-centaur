"""Meaningful pairing and control invariants for the October CPU follow-up."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_culture_followup import SharedBootstrap, adjusted_orders, difference, fit_position_control


def row(item, order, semantic_logit):
    sign = 1 if order == "ab" else -1
    return dict(item_id=item, order=order, logp_a=sign * semantic_logit / 2,
                logp_b=-sign * semantic_logit / 2, valid_mass=1., log_valid_mass=0.)


def test_position_control_corrects_order_bias_and_ignores_heldout_items():
    targets = [dict(item_id="train", p_positive=.7, n_responses=100)]
    frozen = [row("train", "ab", 3.), row("train", "ba", -3.)]
    semantic, position, diagnostics = fit_position_control(targets, frozen)
    prediction = adjusted_orders(frozen, semantic, position)
    assert abs(prediction["p_positive"] - .7) < .002
    assert abs(prediction["order_difference_ab_minus_ba"]) < .005
    assert position < -2.9 and .8 < semantic < .9
    assert diagnostics["objective_at_solution"] < diagnostics["objective_at_zero"]
    with_heldout = frozen + [row("heldout", "ab", 100.), row("heldout", "ba", 100.)]
    assert fit_position_control(targets, with_heldout)[:2] == (semantic, position)


def test_swapping_answer_positions_changes_only_position_offset():
    targets = [dict(item_id="x", p_positive=.7, n_responses=100)]
    a = [row("x", "ab", 2.), row("x", "ba", -1.)]
    b = [row("x", "ab", -1.), row("x", "ba", 2.)]
    s1, p1, _ = fit_position_control(targets, a)
    s2, p2, _ = fit_position_control(targets, b)
    assert abs(s1-s2) < 1e-12 and abs(p1+p2) < 1e-12
    assert abs(adjusted_orders(a, s1, p1)["p_positive"] - adjusted_orders(b, s2, p2)["p_positive"]) < 1e-12


def test_shared_participant_weights_pair_models_and_repeated_seeds():
    # Person b occurs in both splits and must get the very same multiplicity.
    frame = pd.DataFrame(dict(seed=[0, 0, 1, 1], participant_id=["a", "b", "b", "c"],
                              score=[1., 2., 2., 4.]))
    boot = SharedBootstrap(frame, 500, np.random.default_rng(7))
    original = boot.summarize(frame, "score")
    shifted = boot.summarize(frame.assign(score=frame.score + .3), "score")
    contrast = difference(shifted, original)
    assert np.allclose(contrast[1], .3)
    assert np.allclose(difference(original, original)[1], 0.)
    for j in range(500):
        w = {p: boot.weights[j, i] for p, i in boot.positions.items()}
        manual = .5 * ((w["a"]+2*w["b"])/(w["a"]+w["b"]) +
                       (2*w["b"]+4*w["c"])/(w["b"]+w["c"]))
        assert abs(original[1][j]-manual) < 1e-12


def test_duplicating_an_identical_seed_does_not_narrow_person_uncertainty():
    one = pd.DataFrame(dict(seed=[0, 0, 0], participant_id=["a", "b", "c"], score=[1., 2., 4.]))
    three = pd.concat([one.assign(seed=s) for s in range(3)])
    a = SharedBootstrap(one, 500, np.random.default_rng(8)).summarize(one, "score")
    b = SharedBootstrap(three, 500, np.random.default_rng(8)).summarize(three, "score")
    assert np.allclose(a[1], b[1])
