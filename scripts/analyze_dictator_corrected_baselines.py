#!/usr/bin/env python3
"""Evaluate frozen models on an explicitly identified allocation response field.

This script never reads existing culture-adapter outputs. It requires an explicit
behavior file, response column, and scale; it never silently reinterprets Token.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.audit_dictator_a1 import histogram_candidates, score_participants, shared_bootstrap
from scripts.repair_wang_behavior_columns import ensure_not_mislabeled_source

KEYS = ["Dataset", "Nation", "SubNo"]
METRICS = ["raw_response_ll", "conditional_ll"]
COUNTRIES = {"United States": "us", "China": "china"}


def load_explicit_behavior(path: Path, response_column: str, response_scale: str) -> pd.DataFrame:
    ensure_not_mislabeled_source(path)
    frame = pd.read_csv(path)
    required = set(KEYS + ["GroupNo", response_column])
    if required - set(frame):
        raise ValueError(f"Missing behavior columns: {sorted(required-set(frame))}")
    values = pd.to_numeric(frame[response_column], errors="raise").to_numpy(dtype=float)
    tokens = values / 5 if response_scale == "percent" else values
    if not np.isfinite(tokens).all() or (tokens < 0).any() or (tokens > 20).any():
        raise ValueError("Allocation responses must be finite and inside the specified 0..20 or 0..100 scale")
    if not np.allclose(tokens, np.rint(tokens), atol=1e-10, rtol=0):
        raise ValueError("Allocation responses must lie exactly on the 21-choice grid; refusing to bin another rating scale")
    frame["TokenChoice"] = np.rint(tokens).astype(int)
    frame["Country"] = frame.Nation.map({0: "United States", 1: "China"})
    if frame.Country.isna().any() or set(frame.Dataset.unique()) != {1, 2, 3, 4}:
        raise ValueError("Expected Nation 0/1 and Dataset 1..4")
    if frame.duplicated(KEYS + ["GroupNo"]).any():
        raise ValueError("Duplicate participant/item rows")
    sizes = frame.groupby(KEYS).GroupNo.agg(["size", "min", "max"])
    if not (sizes["size"].eq(25) & sizes["min"].eq(1) & sizes["max"].eq(25)).all():
        raise ValueError("Every participant must provide each of 25 items exactly once")
    return frame


def select_people(frame: pd.DataFrame, manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    ids = manifest.loc[manifest.split.eq(split), KEYS]
    if ids.duplicated().any():
        raise ValueError("Duplicate manifest participants")
    selected = frame.merge(ids, on=KEYS, how="inner", validate="many_to_one")
    if len(selected) != len(ids) * 25:
        raise ValueError("Behavior data are missing selected manifest participants")
    return selected


def summarize_contrasts(participants: pd.DataFrame, resamples: int, bootstrap_seed: int) -> tuple[pd.DataFrame,pd.DataFrame]:
    contrasts = []
    for (task,country), frame in participants.groupby(["task","Country"]):
        own = COUNTRIES[country]
        other = "china" if own == "us" else "us"
        for metric in METRICS:
            wide = frame.pivot(index=["seed"]+KEYS, columns="condition", values=metric)
            pairs = {"centaur_minus_llama_original": ("centaur_original","llama_original")}
            for model in ["llama","centaur"]:
                pairs[f"{model}_matched_minus_original"] = (f"{model}_{own}",f"{model}_original")
                pairs[f"{model}_matched_minus_opposite"] = (f"{model}_{own}",f"{model}_{other}")
                if task == "Giving":
                    pairs[f"histogram_own_minus_{model}_original"] = (f"histogram_{own}",f"{model}_original")
                    pairs[f"histogram_own_minus_{model}_matched"] = (f"histogram_{own}",f"{model}_{own}")
            if task == "Giving":
                pairs["histogram_own_minus_other"] = (f"histogram_{own}",f"histogram_{other}")
                pairs["histogram_own_minus_pooled"] = (f"histogram_{own}","histogram_pooled")
            for name,(left,right) in pairs.items():
                contrasts.append((wide[left]-wide[right]).rename("value").reset_index().assign(task=task,Country=country,metric=metric,contrast=name))
    values = pd.concat(contrasts,ignore_index=True)
    split_values = values.groupby(["task","Country","metric","contrast","seed"])["value"].mean().reset_index()
    rng = np.random.default_rng(bootstrap_seed)
    rows = []
    for (task,country),frame in values.groupby(["task","Country"]):
        frame = frame.copy()
        frame["outcome"] = frame.metric+":"+frame.contrast
        bootstrap,_,outcomes = shared_bootstrap(frame,resamples,rng)
        samples = bootstrap.mean(axis=1)
        observed = frame.groupby(["seed","outcome"])["value"].mean().groupby("outcome").mean()
        for k,outcome in enumerate(outcomes):
            metric,contrast = outcome.split(":")
            low,high = np.quantile(samples[:,k],[.025,.975])
            rows.append(dict(task=task,Country=country,metric=metric,contrast=contrast,mean=observed[outcome],ci_low=low,ci_high=high,method="shared_participant_fixed_evaluation_splits",resamples=resamples))
    return split_values,pd.DataFrame(rows)


def plot(summary: pd.DataFrame, output: Path, validation_status: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    figure,axes = plt.subplots(2,2,figsize=(10.5,7.1),sharey="row")
    ordered = ["centaur_minus_llama_original","llama_matched_minus_original","centaur_matched_minus_original"]
    labels = ["Centaur − Llama\noriginal prompt","Llama: matched\nlabel − original","Centaur: matched\nlabel − original"]
    for row,metric in enumerate(METRICS):
        for col,task in enumerate(["Giving","Taking"]):
            axis = axes[row,col]
            for i,(country,color) in enumerate([("United States","#2476A8"),("China","#C76B26")]):
                data = summary.loc[summary.task.eq(task)&summary.Country.eq(country)&summary.metric.eq(metric)].set_index("contrast").loc[ordered]
                point = data["mean"].to_numpy()
                axis.errorbar(np.arange(3)+(-.12 if i==0 else .12),point,yerr=np.vstack([point-data.ci_low,data.ci_high-point]),fmt="o" if i==0 else "s",color=color,capsize=3,label=country)
            axis.axhline(0,color="0.5",lw=.8)
            axis.grid(axis="y",alpha=.2)
            axis.set_xticks(np.arange(3),labels)
            axis.set_title(f"{task} · {'Raw response LL' if row==0 else 'Conditional LL'}",loc="left",fontweight="bold")
    for axis in axes[:,0]:
        axis.set_ylabel("LL difference (nats per allocation)")
    handles,labels = axes[0,0].get_legend_handles_labels()
    figure.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.062),ncol=2,frameon=False)
    figure.text(.5,.035,"Frozen 70B models only. Bars: 95% participant bootstrap; three fixed Giving evaluation splits, shared person weights.",ha="center",fontsize=8)
    figure.text(.5,.013,"Taking uses separately collected samples. Conditional LL normalizes over the 21 valid allocation responses.",ha="center",fontsize=8)
    if validation_status != "validated":
        figure.suptitle("PROVISIONAL — input allocation mapping has not been marked validated",color="#AA2222",fontsize=11)
    else:
        figure.suptitle("Allocation responses · frozen model and country-label controls",fontsize=12)
    figure.tight_layout(rect=(0,.13,1,.95))
    for suffix in ["png","pdf"]:
        figure.savefig(output/f"corrected_baseline_controls.{suffix}",dpi=220)
    plt.close(figure)


def run(args: argparse.Namespace) -> pd.DataFrame:
    args.output_dir.mkdir(parents=True,exist_ok=True)
    source = load_explicit_behavior(args.behavior_path,args.response_column,args.response_scale)
    candidates = pd.read_csv(args.candidates_path)
    expected = {(model,task,variant) for model in ["llama","centaur"] for task in ["giving","taking"] for variant in ["original","us","china"]}
    combinations = set(map(tuple,candidates[["model_condition","task","prompt_variant"]].drop_duplicates().to_numpy()))
    if combinations != expected or len(candidates) != 6300:
        raise ValueError("Expected complete frozen-model controls: 2 models x 2 tasks x 3 variants x 25 items x 21 responses")
    inputs = {args.behavior_path,args.candidates_path,Path(__file__).resolve()}
    provenance_sidecar=args.behavior_path.with_suffix(".provenance.json")
    if provenance_sidecar.exists():
        inputs.add(provenance_sidecar)
    repeats=[]
    for task in ["giving","taking"]:
        archived=args.root/f"model_results/dictator_centaur70b_item_holdout/seed0/base/evaluations/{task}/candidate_probabilities.csv"
        if archived.exists():
            inputs.add(archived)
            old=pd.read_csv(archived)
            current=candidates.loc[candidates.model_condition.eq("centaur")&candidates.task.eq(task)&candidates.prompt_variant.eq("original")]
            merged=current.merge(old,on=["GroupNo","response_value"],validate="one_to_one",suffixes=("_new","_old"))
            for field in ["full_vocab_log_probability","conditional_probability","valid_response_mass"]:
                difference=merged[f"{field}_new"]-merged[f"{field}_old"]
                repeats.append(dict(task=task,field=field,mean_difference=difference.mean(),mean_absolute_difference=difference.abs().mean(),maximum_absolute_difference=difference.abs().max()))
    pd.DataFrame(repeats).to_csv(args.output_dir/"centaur_probability_repeat_check.csv",index=False)
    participant_parts, histograms, coverage = [],[],[]
    for seed in args.seeds:
        corpus = args.root/f"data_cache/dictator_centaur70b/seed{seed}"
        split_path,train_path = corpus/"participant_split_manifest.csv",corpus/"training_participant_manifest.csv"
        inputs.update([split_path,train_path])
        split,train = pd.read_csv(split_path),pd.read_csv(train_path)
        giving = select_people(source.loc[source.Dataset.isin([1,2])],split,"test")
        for task,people in [("Giving",giving),("Taking",source.loc[source.Dataset.isin([3,4])])]:
            for (model,variant),frame in candidates.loc[candidates.task.eq(task.lower())].groupby(["model_condition","prompt_variant"]):
                condition = f"{model}_{variant}"
                scores = score_participants(people,frame).assign(seed=seed,task=task,condition=condition)
                participant_parts.append(scores)
                coverage.append(dict(seed=seed,task=task,condition=condition,participants=len(scores),items=25,responses=len(people)))
        test_ids = set(map(tuple,giving[KEYS].drop_duplicates().to_numpy()))
        for condition in ["us","china","pooled"]:
            ids = train.loc[train.condition.eq(condition),KEYS].drop_duplicates()
            if test_ids & set(map(tuple,ids.to_numpy())):
                raise ValueError("Train/test overlap in histogram manifest")
            training = source.loc[source.Dataset.isin([1,2])].merge(ids,on=KEYS,how="inner",validate="many_to_one")
            if len(training)!=len(ids)*25:
                raise ValueError("Missing histogram training participants")
            histogram = histogram_candidates(training)
            histograms.append(histogram.assign(seed=seed,condition=condition))
            participant_parts.append(score_participants(giving,histogram).assign(seed=seed,task="Giving",condition=f"histogram_{condition}"))
    participants = pd.concat(participant_parts,ignore_index=True)
    participants.to_csv(args.output_dir/"participant_log_likelihoods.csv",index=False)
    pd.concat(histograms,ignore_index=True).to_csv(args.output_dir/"training_histograms.csv",index=False)
    pd.DataFrame(coverage).to_csv(args.output_dir/"coverage.csv",index=False)
    split_metrics = participants.groupby(["seed","task","condition","Country"])[METRICS].mean().reset_index()
    split_metrics["invalid_mass_penalty_nats"] = split_metrics.conditional_ll-split_metrics.raw_response_ll
    split_metrics.to_csv(args.output_dir/"split_metrics.csv",index=False)
    split_metrics.groupby(["task","condition","Country"])[METRICS+["invalid_mass_penalty_nats"]].mean().reset_index().to_csv(args.output_dir/"metrics_summary.csv",index=False)
    split_contrasts,summary = summarize_contrasts(participants,args.bootstrap_resamples,args.bootstrap_seed)
    split_contrasts.to_csv(args.output_dir/"split_contrasts.csv",index=False)
    summary.to_csv(args.output_dir/"contrast_summary.csv",index=False)
    candidates.groupby(["model_condition","task","prompt_variant","GroupNo"])["valid_response_mass"].first().groupby(["model_condition","task","prompt_variant"]).agg(mean="mean",minimum="min",maximum="max").reset_index().to_csv(args.output_dir/"valid_mass.csv",index=False)
    human = source.groupby(["Dataset","Country"]).agg(participants=("SubNo","nunique"),responses=("TokenChoice","size"),mean_allocation_tokens=("TokenChoice","mean"))
    human.to_csv(args.output_dir/"human_allocation_summary.csv")
    pd.DataFrame([dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(inputs)]).to_csv(args.output_dir/"input_provenance.csv",index=False)
    metadata = dict(status=f"{args.validation_status}_explicit_allocation_baselines",behavior_path=str(args.behavior_path.resolve()),response_column=args.response_column,response_scale=args.response_scale,
        behavior_provenance_path=str(provenance_sidecar.resolve()) if provenance_sidecar.exists() else None,
        candidates_path=str(args.candidates_path.resolve()),evaluation_split_ids=args.seeds,bootstrap_seed=args.bootstrap_seed,bootstrap_resamples=args.bootstrap_resamples,
        uncertainty="Shared participant multiplicities across fixed evaluation splits, stratified by Dataset and Nation; 95% percentile intervals. No inference-seed resampling: saved model probabilities are deterministic reused predictions.",
        histogram="Add-one smoothed per-target allocation histogram over 21 choices; uses saved Giving training participant IDs only. Histogram fit kept fixed during evaluation bootstrap. Available only for Giving.",
        scope="Frozen plain Llama-70B, frozen Centaur-70B, original/US/China label prompts, and a Giving allocation histogram. No existing culture-adapter probabilities included.",
        limitations=["Validation status is explicit caller input; inspect independent source-column audit evidence before interpreting allocation outcomes.","Giving participants are held out relative to saved culture-training manifests, not necessarily upstream Centaur pretraining.","Taking samples were collected separately; no Taking histogram fitted.","Prompt comparisons are descriptive for this exact wording; no prompt-family generalization interval.","Raw LL includes probability assigned outside the 21 valid allocations; conditional LL removes that format penalty."])
    (args.output_dir/"analysis_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    plot(summary,args.output_dir,args.validation_status)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=REPO_ROOT.parent)
    parser.add_argument("--behavior-path",type=Path,required=True)
    parser.add_argument("--response-column",required=True)
    parser.add_argument("--response-scale",choices=["percent","tokens"],required=True)
    parser.add_argument("--candidates-path",type=Path,required=True)
    parser.add_argument("--validation-status",choices=["pending","validated"],default="pending")
    parser.add_argument("--output-dir",type=Path,default=REPO_ROOT.parent/"model_results/a1_corrected_baselines_20260930")
    parser.add_argument("--seeds",nargs="+",type=int,default=[0,1,2])
    parser.add_argument("--bootstrap-resamples",type=int,default=4000)
    parser.add_argument("--bootstrap-seed",type=int,default=20260930)
    args = parser.parse_args()
    summary = run(args)
    print(summary.loc[summary.contrast.eq("centaur_minus_llama_original")].to_string(index=False))
    print(f"Wrote {args.output_dir}")


if __name__=="__main__":
    main()
