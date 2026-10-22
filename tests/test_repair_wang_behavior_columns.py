import pandas as pd
import pytest

from scripts.repair_wang_behavior_columns import corrected_frame, validate_audits, verified_source, ensure_not_mislabeled_source


def test_explicit_rotation_preserves_subjects_and_exact_allocation_scale():
    source = pd.DataFrame({"Dataset": [1, 3], "Nation": [0, 1], "SubNo": [3, 5],
                           "GroupNo": [1, 2], "Age": [27.0, 31.0],
                           "Warmth": [0, 65], "Competence": [17, 89], "Token": [83, 11]})
    original = source.copy(deep=True)
    corrected = corrected_frame(source)
    assert corrected.Token.tolist() == [0, 65]
    assert corrected.Warmth.tolist() == [17, 89]
    assert corrected.Competence.tolist() == [83, 11]
    assert (corrected.Token / 5).tolist() == [0, 13]
    cols = ["Dataset", "Nation", "SubNo", "GroupNo", "Age"]
    pd.testing.assert_frame_equal(source[cols], corrected[cols])
    pd.testing.assert_frame_equal(source, original)


def test_unknown_source_hash_is_rejected_even_with_plausible_values(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("Token,Warmth,Competence\n50,50,50\n")
    with pytest.raises(ValueError, match="SHA256"):
        verified_source(source)


def test_all_eight_complete_audits_are_required():
    with pytest.raises(ValueError, match="eight samples"):
        validate_audits([])


def test_wrong_allocation_lattice_is_rejected():
    with pytest.raises(ValueError, match="lattice"):
        corrected_frame(pd.DataFrame({"Token": [60], "Warmth": [63], "Competence": [49]}))


def test_loader_guard_blocks_confirmed_bad_hash(tmp_path, monkeypatch):
    import scripts.repair_wang_behavior_columns as repair_module
    path = tmp_path / "known_bad.csv"
    path.write_bytes(b"known bad")
    monkeypatch.setattr(repair_module, "KNOWN_CSV_SHA256", repair_module.sha256(b"known bad"))
    with pytest.raises(ValueError, match="competence"):
        ensure_not_mislabeled_source(path)
    path.write_bytes(b"other source")
    ensure_not_mislabeled_source(path)
