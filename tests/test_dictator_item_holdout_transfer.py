from __future__ import annotations

import pandas as pd

from scripts.dictator_item_holdout_transfer import (
    ITEM_FOLDS,
    item_fold_manifest,
    validate_item_folds,
)


def test_fixed_item_folds_cover_every_target_once() -> None:
    validate_item_folds()
    flattened = [group_no for fold in ITEM_FOLDS.values() for group_no in fold]
    assert sorted(flattened) == list(range(1, 26))
    assert all(len(fold) == 5 for fold in ITEM_FOLDS.values())


def test_item_fold_manifest_matches_source_items() -> None:
    items = pd.DataFrame(
        {
            "GroupNo": range(1, 26),
            "item": [f"item {value}" for value in range(1, 26)],
            "item_chinese": [f"zh {value}" for value in range(1, 26)],
        }
    )

    manifest = item_fold_manifest(items)

    assert manifest.groupby("fold").size().eq(5).all()
    assert not manifest["GroupNo"].duplicated().any()
    assert manifest["fold_rationale"].notna().all()
    assert set(manifest.loc[manifest["fold"].eq(0), "GroupNo"]) == {
        1,
        5,
        12,
        16,
        23,
    }
