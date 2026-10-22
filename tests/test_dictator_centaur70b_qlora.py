from pathlib import Path

import pandas as pd

from scripts.dictator_centaur70b_qlora import (
    FullItemSetBatchSampler,
    TRAIN_CONDITIONS,
    aggregate_item_response_distributions,
    build_participant_rows,
    resolve_saved_adapter_dir,
)
from scripts.dictator_culture_mixed_split import (
    balanced_participant_split,
    select_training_conditions,
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
                            "Token": float((group_no + sub_no) * 5 % 101),
                            "TokenChoice": (group_no + sub_no) % 21,
                        }
                    )
    return pd.DataFrame(rows)


def test_centaur_corpora_use_equal_participant_response_budgets() -> None:
    giving = synthetic_giving()
    split = balanced_participant_split(
        giving,
        seed=9,
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    manifest = select_training_conditions(split, seed=9)
    manifest = manifest.loc[manifest["condition"].isin(TRAIN_CONDITIONS)]

    corpora = {
        condition: build_participant_rows(giving, manifest, condition, seed=9)
        for condition in TRAIN_CONDITIONS
    }
    assert {condition: len(rows) for condition, rows in corpora.items()} == {
        "us": 200,
        "china": 200,
        "pooled": 200,
    }
    assert {
        condition: rows["participant_key"].nunique()
        for condition, rows in corpora.items()
    } == {"us": 8, "china": 8, "pooled": 8}
    assert corpora["pooled"]["Country"].value_counts().to_dict() == {
        "United States": 100,
        "China": 100,
    }
    for rows in corpora.values():
        assert rows["prompt"].str.endswith("You enter ").all()
        assert rows["response_text"].str.fullmatch(
            r"<<(?:20|1[0-9]|[0-9])>>\."
        ).all()


def test_saved_adapter_directory_accepts_peft_named_subdirectory(tmp_path: Path) -> None:
    named = tmp_path / "culture"
    named.mkdir()
    (named / "adapter_config.json").write_text("{}", encoding="utf-8")
    (named / "adapter_model.safetensors").write_bytes(b"test")

    assert resolve_saved_adapter_dir(tmp_path) == named


def test_item_distribution_aggregation_preserves_empirical_counts() -> None:
    rows = pd.DataFrame(
        {
            "GroupNo": [1, 1, 1, 2, 2],
            "item": ["boy", "boy", "boy", "girl", "girl"],
            "prompt": ["boy prompt"] * 3 + ["girl prompt"] * 2,
            "response_value": [0, 0, 20, 10, 20],
        }
    )

    aggregated = aggregate_item_response_distributions(rows).set_index("GroupNo")

    assert aggregated.loc[1, "participant_count"] == 3
    assert aggregated.loc[2, "participant_count"] == 2
    boy = aggregated.loc[1, "target_probabilities"]
    girl = aggregated.loc[2, "target_probabilities"]
    assert boy[0] == 2 / 3
    assert boy[20] == 1 / 3
    assert girl[10] == 1 / 2
    assert girl[20] == 1 / 2
    assert sum(boy) == 1
    assert sum(girl) == 1


def test_full_item_sampler_keeps_every_update_complete() -> None:
    batches = list(FullItemSetBatchSampler(items_per_repeat=3, repeats=4, seed=7))

    assert len(batches) == 4
    assert sorted(index for batch in batches for index in batch) == list(range(12))
    assert all(len(batch) == 3 for batch in batches)
    assert all(index // 3 == batch[0] // 3 for batch in batches for index in batch)
