import copy
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_dictator_corrected_adapters import ITEM_FOLDS,run,validate_evaluation,validate_fit_metadata


def fit():
    heldout=list(ITEM_FOLDS[0])
    meta=dict(source_behavior_sha256="corrected",seed=0,fold=0,condition="us",heldout_group_nos=heldout,training_group_nos=sorted(set(range(1,26))-set(heldout)),heldout_items_used_for_training=False,taking_responses_used_for_training_or_selection=False,training_participants=910,training_responses=18200)
    metrics=dict(optimizer_steps=25,planned_optimizer_steps=25,condition="us",seed=0,unique_items=20,training_rows=18200,training_objective="item_distribution",step0_equivalence={"max_abs_probability_difference":0})
    return meta,metrics


def test_corrected_fit_guard_rejects_legacy_smoke_and_fold_contamination():
    meta,metrics=fit()
    validate_fit_metadata(meta,metrics,"corrected",0,0,"us",25)
    bad=copy.deepcopy(meta);bad["source_behavior_sha256"]="old"
    with pytest.raises(ValueError,match="source_behavior_sha256"):
        validate_fit_metadata(bad,metrics,"corrected",0,0,"us",25)
    bad=copy.deepcopy(metrics);bad["optimizer_steps"]=1
    with pytest.raises(ValueError,match="smoke"):
        validate_fit_metadata(meta,bad,"corrected",0,0,"us",25)
    bad=copy.deepcopy(meta);bad["training_group_nos"].append(1)
    with pytest.raises(ValueError,match="target groups"):
        validate_fit_metadata(bad,metrics,"corrected",0,0,"us",25)


def test_evaluation_guard_checks_source_protocol_seed_and_exact_targets():
    meta=dict(source_behavior_sha256="corrected",corpus_protocol_sha256="protocol",training_seed=0,item_fold=0,condition="us",evaluation_task="Giving",evaluation_group_nos=list(ITEM_FOLDS[0]))
    candidates=pd.DataFrame([dict(GroupNo=item,response_value=response,full_vocab_log_probability=-4.,conditional_probability=1/21,training_seed=0,item_fold=0,condition="us",evaluation_task="Giving") for item in ITEM_FOLDS[0] for response in range(21)])
    validate_evaluation(meta,candidates,"corrected","protocol",0,0,"us","Giving")
    with pytest.raises(ValueError,match="corpus_protocol_sha256"):
        validate_evaluation(meta,candidates,"corrected","different",0,0,"us","Giving")
    bad=candidates.copy();bad.loc[0,"training_seed"]=2
    with pytest.raises(ValueError,match="Candidate table mismatch"):
        validate_evaluation(meta,bad,"corrected","protocol",0,0,"us","Giving")
    with pytest.raises(ValueError,match="Missing or duplicate"):
        validate_evaluation(meta,candidates.iloc[1:],"corrected","protocol",0,0,"us","Giving")


def test_complete_numeric_aggregation_requires_no_matplotlib(tmp_path,monkeypatch):
    behavior=tmp_path/"Behavior.csv"
    pd.DataFrame([dict(Dataset=dataset,Nation=nation,SubNo=person,GroupNo=item,Token=50) for dataset in range(1,5) for nation in range(2) for person in range(1,5) for item in range(1,26)]).to_csv(behavior,index=False)
    source_sha=hashlib.sha256(behavior.read_bytes()).hexdigest()
    behavior.with_suffix(".provenance.json").write_text(json.dumps({"output_sha256":source_sha}))
    corpus=tmp_path/"corpora/seed0"
    corpus.mkdir(parents=True)
    (corpus/"protocol.json").write_text(json.dumps(dict(seed=0,source_behavior_sha256=source_sha)))
    protocol_sha=hashlib.sha256((corpus/"protocol.json").read_bytes()).hexdigest()
    pd.DataFrame([dict(Dataset=dataset,Nation=nation,SubNo=person,split="test") for dataset in [1,2] for nation in range(2) for person in range(1,5)]).to_csv(corpus/"participant_split_manifest.csv",index=False)
    controls=tmp_path/"controls.csv"
    pd.DataFrame([dict(model_condition="centaur",task=task,prompt_variant="original",GroupNo=item,response_value=response,full_vocab_log_probability=np.log(.5/21),conditional_probability=1/21) for task in ["giving","taking"] for item in range(1,26) for response in range(21)]).to_csv(controls,index=False)
    result_root=tmp_path/"results"
    for fold in range(5):
        for condition in ["us","china","pooled"]:
            directory=result_root/f"seed0/fold{fold}/{condition}"
            directory.mkdir(parents=True)
            train_meta,train_metrics=fit()
            train_meta.update(source_behavior_sha256=source_sha,fold=fold,condition=condition,heldout_group_nos=list(ITEM_FOLDS[fold]),training_group_nos=sorted(set(range(1,26))-set(ITEM_FOLDS[fold])))
            train_metrics["condition"]=condition
            (directory/"train_metadata.json").write_text(json.dumps(train_meta))
            (directory/"train_metrics.json").write_text(json.dumps(train_metrics))
            for task in ["Giving","Taking"]:
                evaluation=directory/f"evaluations/{task.lower()}"
                evaluation.mkdir(parents=True)
                meta=dict(source_behavior_sha256=source_sha,corpus_protocol_sha256=protocol_sha,training_seed=0,item_fold=fold,condition=condition,evaluation_task=task,evaluation_group_nos=list(ITEM_FOLDS[fold]))
                (evaluation/"run_metadata.json").write_text(json.dumps(meta))
                pd.DataFrame([dict(GroupNo=item,response_value=response,full_vocab_log_probability=np.log(.75/21),conditional_probability=1/21,training_seed=0,item_fold=fold,condition=condition,evaluation_task=task) for item in ITEM_FOLDS[fold] for response in range(21)]).to_csv(evaluation/"candidate_probabilities.csv",index=False)
                pd.DataFrame([dict(sample="Combined",country=country,centaur_nll_nats=-np.log(.75/21),conditional_nll_nats=np.log(21)) for country in ["United States","China"]]).to_csv(evaluation/"model_metrics.csv",index=False)
    import scripts.analyze_dictator_corrected_adapters as module
    monkeypatch.setattr(module,"plot",lambda *args: pytest.fail("Plotting must not run under --no-plots"))
    args=argparse.Namespace(behavior_path=behavior,corpus_root=tmp_path/"corpora",results_root=result_root,output_dir=tmp_path/"aggregate",baseline_controls=controls,seeds=[0],expected_optimizer_steps=25,bootstrap_resamples=100,bootstrap_seed=11,coverage_only=False,no_plots=True)
    run(args)
    summary=pd.read_csv(args.output_dir/"contrast_summary.csv")
    gain=summary.loc[summary.metric.eq("raw_response_ll")&summary.contrast.eq("us_gain_vs_base"),"mean"]
    np.testing.assert_allclose(gain,np.log(1.5))
    np.testing.assert_allclose(summary.loc[summary.metric.eq("conditional_ll"),"mean"],0)
    assert json.loads((args.output_dir/"readiness.json").read_text())["status"]=="complete_validated"
    assert (args.output_dir/"run_summary.md").is_file()
