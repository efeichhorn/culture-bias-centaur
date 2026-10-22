import pandas as pd
import pytest

from scripts.dictator_centaur70b_item_holdout import (
    filter_training_rows,
    heldout_group_nos,
)
from scripts.dictator_item_holdout_transfer import ITEM_FOLDS


def synthetic_training_rows() -> pd.DataFrame:
    rows = []
    for participant in ["p0", "p1", "p2"]:
        for group_no in range(1, 26):
            rows.append(
                {
                    "participant_key": participant,
                    "GroupNo": group_no,
                    "item": f"item {group_no}",
                    "prompt": f"prompt {group_no}",
                    "response_value": group_no % 21,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("fold", sorted(ITEM_FOLDS))
def test_filter_training_rows_exactly_excludes_fixed_fold(fold: int) -> None:
    filtered = filter_training_rows(synthetic_training_rows(), fold)

    assert filtered["GroupNo"].nunique() == 20
    assert len(filtered) == 60
    assert not set(filtered["GroupNo"]).intersection(heldout_group_nos(fold))
    assert filtered.groupby("participant_key").size().eq(20).all()


def test_heldout_group_nos_rejects_unknown_fold() -> None:
    with pytest.raises(ValueError):
        heldout_group_nos(5)
