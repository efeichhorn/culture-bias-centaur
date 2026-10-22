"""Prevent training/evaluation on mislabeled inputs or stale cached targets."""
from argparse import Namespace
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import dictator_centaur70b_item_holdout as item_holdout
from scripts import dictator_centaur70b_qlora as qlora
from scripts import dictator_culture_mixed_split as mixed
from scripts import dictator_item_holdout_transfer as item_holdout_8b
from scripts import repair_wang_behavior_columns as repair
from scripts.evaluate_dictator_giving_base import load_giving_data
from scripts.evaluate_dictator_taking_transfer import load_taking_data


def synthetic_sources(tmp_path: Path) -> tuple[Path, Path]:
    rows = [
        dict(Dataset=dataset, Nation=nation, SubNo=person, GroupNo=group,
             Token=5 * ((person + group) % 21))
        for dataset in range(1, 5)
        for nation in [0, 1]
        for person in range(8)
        for group in range(1, 26)
    ]
    behavior = tmp_path / "behavior.csv"
    groups = tmp_path / "groups.csv"
    pd.DataFrame(rows).to_csv(behavior, index=False)
    pd.DataFrame({"No.": range(1, 26), "English": [f"item {n}" for n in range(1, 26)],
                  "Chinese": [f"zh {n}" for n in range(1, 26)]}).to_csv(groups, index=False)
    return behavior, groups


@pytest.mark.parametrize("loader", [load_giving_data, load_taking_data])
def test_known_bad_source_is_rejected_unless_archival_opt_in(tmp_path, monkeypatch, loader):
    behavior, groups = synthetic_sources(tmp_path)
    # Register these exact fixture bytes as the known affected release. The
    # production guard compares real CSV/XLSX release hashes, not field values.
    monkeypatch.setattr(repair, "KNOWN_CSV_SHA256", hashlib.sha256(behavior.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="Known mislabeled Wang source"):
        loader(behavior, groups)
    legacy, _ = loader(behavior, groups, allow_legacy_mislabeled=True)
    assert len(legacy) == 800


@pytest.mark.parametrize("module", [mixed, qlora])
def test_prepared_corpus_records_exact_behavior_bytes(tmp_path, module):
    behavior, groups = synthetic_sources(tmp_path)
    output = tmp_path / module.__name__.split(".")[-1]
    args = Namespace(behavior_path=str(behavior), group_path=str(groups), output_dir=str(output),
                     seed=2, train_fraction=0.5, validation_fraction=0.25)
    module.prepare_corpora(args)
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["source_behavior_sha256"] == hashlib.sha256(behavior.read_bytes()).hexdigest()
    assert protocol["source_behavior_path"] == str(behavior.resolve())
    assert mixed.require_corpus_behavior_source(output, behavior, prepare_script="prepare.py",
                                              group_path=groups, seed=2) == protocol


@pytest.mark.parametrize("protocol", [None, {}, {"source_behavior_sha256": "old_competence_targets"}])
def test_stale_or_untracked_corpus_requires_regeneration(tmp_path, protocol):
    behavior, groups = synthetic_sources(tmp_path)
    corpus = tmp_path / "old corpus"
    corpus.mkdir()
    if protocol is not None:
        (corpus / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="missing or stale source_behavior_sha256") as error:
        mixed.require_corpus_behavior_source(corpus, behavior,
            prepare_script="scripts/dictator_centaur70b_qlora.py", group_path=groups, seed=2)
    assert "--stage prepare" in str(error.value)
    assert "--seed 2" in str(error.value)
    assert "old corpus_corrected" in str(error.value)


@pytest.mark.parametrize("trainer", [mixed.train_condition, qlora.train, item_holdout.train_fold_condition,
                                     item_holdout_8b.train_fold_condition])
def test_training_rejects_stale_corpus_before_reading_targets_or_loading_model(tmp_path, trainer):
    behavior, groups = synthetic_sources(tmp_path)
    args = Namespace(condition="us", fold=0, corpus_dir=str(tmp_path / "old_cache"),
                     behavior_path=str(behavior), group_path=str(groups), seed=0)
    # No cached target file, model options, or GPU exists: provenance must fail first.
    with pytest.raises(ValueError, match="Regenerate from corrected behavior"):
        trainer(args)


def test_item_holdout_carries_verified_source_into_filtered_corpus(tmp_path, monkeypatch):
    behavior, groups = synthetic_sources(tmp_path)
    parent = tmp_path / "parent"
    parent.mkdir()
    source = mixed.behavior_source_provenance(behavior)
    (parent / "protocol.json").write_text(json.dumps(source))
    pd.DataFrame({"GroupNo": range(1, 26), "participant_key": ["1:0:1"] * 25,
                  "response_value": [5] * 25}).to_json(parent / "rows_us.jsonl", orient="records", lines=True)
    output = tmp_path / "trained"
    args = Namespace(condition="us", fold=0, corpus_dir=str(parent), behavior_path=str(behavior),
                     group_path=str(groups), output_dir=str(output), seed=0)
    visited = []

    def fake_train(training_args):
        protocol = mixed.require_corpus_behavior_source(training_args.corpus_dir, behavior,
            prepare_script="scripts/dictator_centaur70b_qlora.py", group_path=groups)
        assert protocol["source_behavior_sha256"] == source["source_behavior_sha256"]
        assert protocol["parent_corpus_dir"] == str(parent.resolve())
        assert protocol["heldout_group_nos"] == list(item_holdout.heldout_group_nos(0))
        assert protocol["training_items_per_fold"] == 20
        assert protocol["equal_response_budget"] == 20
        rows = pd.read_json(Path(training_args.corpus_dir) / "rows_us.jsonl", lines=True)
        assert len(rows) == 20
        assert not set(rows.GroupNo) & set(item_holdout.heldout_group_nos(0))
        visited.append(training_args.corpus_dir)

    monkeypatch.setattr(item_holdout, "train_centaur70b", fake_train)
    item_holdout.train_fold_condition(args)
    assert visited == [str(output / "training_corpus")]
    assert args.corpus_dir == str(parent)
    metadata = json.loads((output / "train_metadata.json").read_text())
    assert metadata["source_behavior_sha256"] == source["source_behavior_sha256"]


def test_evaluation_rejects_legacy_adapter_even_with_corrected_human_data(tmp_path):
    behavior, groups = synthetic_sources(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = mixed.behavior_source_provenance(behavior)
    (corpus / "protocol.json").write_text(json.dumps({**source, "seed": 0}))
    output = tmp_path / "adapter_run"
    output.mkdir()
    args = Namespace(condition="us", fold=0, seed=0, behavior_path=str(behavior),
                     group_path=str(groups), corpus_dir=str(corpus), output_dir=str(output))
    with pytest.raises(ValueError, match="lacks train_metadata"):
        item_holdout.evaluation_source_provenance(args)
    metadata = dict(source_behavior_sha256="old", seed=0, fold=0, condition="us",
                    heldout_group_nos=list(item_holdout.heldout_group_nos(0)))
    (output / "train_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="source_behavior_sha256"):
        item_holdout.evaluation_source_provenance(args)
    metadata.update(source)
    (output / "train_metadata.json").write_text(json.dumps(metadata))
    assert item_holdout.evaluation_source_provenance(args)["source_behavior_sha256"] == source["source_behavior_sha256"]
    args.seed = 1
    with pytest.raises(ValueError, match="corpus seed"):
        item_holdout.evaluation_source_provenance(args)
