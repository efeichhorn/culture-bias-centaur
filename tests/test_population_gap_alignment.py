import numpy as np
import pytest

from scripts.analyze_population_gap_alignment import alignment_metrics, combine_waves


def test_combined_wave_weights_are_seed_specific_and_preserve_target_pairing():
    # Seeds have opposite wave weights; all targets share each participant draw.
    wave1 = np.array([[[1., 4.], [2., 5.]], [[3., 6.], [4., 7.]]])
    wave2 = wave1 + 10
    combined = combine_waves([wave1, wave2], np.array([[1, 3], [3, 1]]))
    np.testing.assert_allclose(combined[:, 0], wave1[:, 0]+7.5)
    np.testing.assert_allclose(combined[:, 1], wave1[:, 1]+2.5)
    np.testing.assert_allclose(combined[:, :, 1]-combined[:, :, 0], 3)


def test_combined_wave_weights_reject_empty_strata():
    with pytest.raises(ValueError, match="nonempty"):
        combine_waves([np.ones((4, 2, 3))]*2, np.array([[0, 1], [2, 1]]))


def test_exact_zero_gaps_are_not_counted_as_directional_matches():
    metrics = alignment_metrics(np.array([0., -1., 2., 3.]), np.array([0., -2., 1., -1.]))
    assert metrics["target_gap_sign_matches"] == 2
    assert metrics["targets_with_nonzero_gaps"] == 3
    assert metrics["model_minus_human_mean_gap"] == -1.5
