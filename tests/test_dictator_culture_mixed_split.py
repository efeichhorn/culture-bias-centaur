import pandas as pd

from scripts.dictator_culture_mixed_split import (
    CONDITIONS,
    aggregate_training_counts,
    balanced_participant_split,
    heldout_rows,
    select_training_conditions,
    tie_aware_mode_metrics,
)


def synthetic_giving() -> pd.DataFrame:
    rows = []
    for dataset in [1, 2]:
        for nation, country, participants in [
            (0, "United States", 10),
            (1, "China", 8),
        ]:
            for sub_no in range(participants):
                for group_no in range(1, 26):
                    rows.append(
                        {
                            "Dataset": dataset,
                            "Nation": nation,
                            "Country": country,
                            "SubNo": sub_no,
                            "GroupNo": group_no,
                            "item": f"item {group_no}",
                            "TokenChoice": (group_no + sub_no) % 21,
                        }
                    )
    return pd.DataFrame(rows)


def test_balanced_split_is_participant_disjoint_and_wave_stratified() -> None:
    giving = synthetic_giving()
    manifest = balanced_participant_split(
        giving,
        seed=11,
        train_fraction=0.5,
        validation_fraction=0.25,
    )

    counts = manifest.groupby(["Dataset", "split", "Country"]).size().unstack()
    assert counts["United States"].eq(counts["China"]).all()
    assert manifest.groupby(["Dataset", "Country", "split"]).size().to_dict() == {
        (1, "China", "test"): 2,
        (1, "China", "train"): 4,
        (1, "China", "validation"): 2,
        (1, "United States", "test"): 2,
        (1, "United States", "train"): 4,
        (1, "United States", "validation"): 2,
        (2, "China", "test"): 2,
        (2, "China", "train"): 4,
        (2, "China", "validation"): 2,
        (2, "United States", "test"): 2,
        (2, "United States", "train"): 4,
        (2, "United States", "validation"): 2,
    }
    assert not manifest[["Dataset", "Nation", "SubNo"]].duplicated().any()


def test_condition_budgets_and_compact_counts() -> None:
    giving = synthetic_giving()
    split_manifest = balanced_participant_split(
        giving,
        seed=5,
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    condition_manifest = select_training_conditions(split_manifest, seed=5)

    assert condition_manifest.groupby("condition")["participant_key"].nunique().to_dict() == {
        "china": 8,
        "pooled": 8,
        "pooled_full": 16,
        "us": 8,
    }
    assert (
        condition_manifest.loc[condition_manifest["condition"].eq("pooled"), "Country"]
        .value_counts()
        .to_dict()
        == {"United States": 4, "China": 4}
    )
    for condition in CONDITIONS:
        counts = aggregate_training_counts(giving, condition_manifest, condition)
        participants = int(counts["participant_count"].iloc[0])
        assert int(counts["count"].sum()) == participants * 25
        assert counts["GroupNo"].nunique() == 25
        assert set(counts["condition"]) == {condition}


def test_test_rows_are_common_and_exclude_training_participants() -> None:
    giving = synthetic_giving()
    split_manifest = balanced_participant_split(
        giving,
        seed=17,
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    rows = heldout_rows(giving, split_manifest)
    test_keys = set(
        map(tuple, split_manifest.loc[split_manifest["split"].eq("test"), ["Dataset", "Nation", "SubNo"]].to_numpy())
    )
    row_keys = set(map(tuple, rows[["Dataset", "Nation", "SubNo"]].drop_duplicates().to_numpy()))
    train_keys = set(
        map(tuple, split_manifest.loc[split_manifest["split"].eq("train"), ["Dataset", "Nation", "SubNo"]].to_numpy())
    )
    assert row_keys == test_keys
    assert not row_keys.intersection(train_keys)
    assert rows.groupby(["Dataset", "Nation", "SubNo"]).size().eq(25).all()


def test_tie_aware_mode_metrics_accepts_either_human_mode() -> None:
    rows = pd.DataFrame(
        [
            {"GroupNo": group_no, "TokenChoice": response}
            for group_no in range(1, 26)
            for response in [4, 6]
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "GroupNo": group_no,
                "response_value": response,
                "conditional_probability": 1.0 if response == 6 else 0.0,
            }
            for group_no in range(1, 26)
            for response in range(21)
        ]
    )

    accuracy, distance = tie_aware_mode_metrics(rows, candidates)

    assert accuracy == 1.0
    assert distance == 0.0
