#!/usr/bin/env python3
"""Strict aggregation for the corrected 70B item-holdout experiment.

Only explicit corrected source/corpus/results paths are accepted. Missing folds,
legacy provenance, one-step smoke fits and mismatched evaluation manifests fail
before any scientific summary is written. --coverage-only reports readiness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO_ROOT))
from scripts.analyze_dictator_corrected_baselines import load_explicit_behavior, select_people
from scripts.audit_dictator_a1 import check_candidates, score_participants, shared_bootstrap
from scripts.dictator_item_holdout_transfer import ITEM_FOLDS

KEYS=["Dataset","Nation","SubNo"]
METRICS=["raw_response_ll","conditional_ll"]
COUNTRIES={"United States":"us","China":"china"}
CONDITIONS=["us","china","pooled"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_fit_metadata(metadata: dict, metrics: dict, source_sha: str, seed: int, fold: int, condition: str, expected_steps: int) -> None:
    expected={"source_behavior_sha256":source_sha,"seed":seed,"fold":fold,"condition":condition}
    for field,value in expected.items():
        if metadata.get(field)!=value:
            raise ValueError(f"Fit metadata mismatch {field}: expected {value!r}, found {metadata.get(field)!r}")
    heldout=set(ITEM_FOLDS[fold])
    if set(metadata.get("heldout_group_nos",[]))!=heldout or set(metadata.get("training_group_nos",[]))!=set(range(1,26))-heldout:
        raise ValueError("Fit target groups do not match the fixed held-out fold")
    if metadata.get("heldout_items_used_for_training") is not False or metadata.get("taking_responses_used_for_training_or_selection") is not False:
        raise ValueError("Fit metadata does not certify held-out targets and unused Taking responses")
    if metrics.get("optimizer_steps")!=expected_steps or metrics.get("planned_optimizer_steps")!=expected_steps:
        raise ValueError("Incomplete fit or smoke output: optimizer-step budget mismatch")
    for field,value in [("condition",condition),("seed",seed),("unique_items",20),("training_rows",18200),("training_objective","item_distribution")]:
        if metrics.get(field)!=value:
            raise ValueError(f"Training metric mismatch {field}")
    if metadata.get("training_participants")!=910 or metadata.get("training_responses")!=18200:
        raise ValueError("Unequal or unexpected training participant/response budget")
    step0=metrics.get("step0_equivalence",{}).get("max_abs_probability_difference")
    if step0 is None or not np.isfinite(step0) or step0>1e-6:
        raise ValueError("Missing or failed frozen-model equivalence check before fitting")


def validate_evaluation(metadata: dict, candidates: pd.DataFrame, source_sha: str, corpus_sha: str, seed: int, fold: int, condition: str, task: str) -> None:
    expected={"source_behavior_sha256":source_sha,"corpus_protocol_sha256":corpus_sha,"training_seed":seed,"item_fold":fold,"condition":condition,"evaluation_task":task}
    for field,value in expected.items():
        if metadata.get(field)!=value:
            raise ValueError(f"Evaluation metadata mismatch {field}: expected {value!r}, found {metadata.get(field)!r}")
    groups=set(range(1,26)) if condition=="base" else set(ITEM_FOLDS[fold])
    if set(metadata.get("evaluation_group_nos",[]))!=groups or set(candidates.GroupNo.unique())!=groups:
        raise ValueError("Evaluation targets do not match the requested held-out fold")
    if len(candidates)!=len(groups)*21 or candidates.duplicated(["GroupNo","response_value"]).any():
        raise ValueError("Missing or duplicate evaluation item/response pairs")
    if not np.isfinite(candidates[["full_vocab_log_probability","conditional_probability"]]).all().all():
        raise ValueError("Nonfinite evaluation probabilities")
    if not np.allclose(candidates.groupby("GroupNo").conditional_probability.sum(),1,atol=1e-8):
        raise ValueError("Evaluation conditional probabilities do not sum to one")
    for field,value in [("training_seed",seed),("item_fold",fold),("condition",condition),("evaluation_task",task)]:
        if field not in candidates or not candidates[field].eq(value).all():
            raise ValueError(f"Candidate table mismatch {field}")


def coverage(args: argparse.Namespace) -> pd.DataFrame:
    rows=[]
    for seed in args.seeds:
        for fold in range(5):
            for condition in CONDITIONS:
                directory=args.results_root/f"seed{seed}/fold{fold}/{condition}"
                paths=[directory/"train_metadata.json",directory/"train_metrics.json"]
                for task in ["giving","taking"]:
                    evaluation=directory/f"evaluations/{task}"
                    required=paths+[evaluation/name for name in ["candidate_probabilities.csv","run_metadata.json","model_metrics.csv"]]
                    missing=[str(path.relative_to(args.results_root)) for path in required if not path.is_file()]
                    rows.append(dict(seed=seed,fold=fold,condition=condition,task=task,files_present=not missing,missing_files=";".join(missing)))
        if args.baseline_controls is None:
            for task in ["giving","taking"]:
                directory=args.results_root/f"seed{seed}/base/evaluations/{task}"
                required=[directory/"candidate_probabilities.csv",directory/"run_metadata.json"]
                missing=[str(path.relative_to(args.results_root)) for path in required if not path.is_file()]
                rows.append(dict(seed=seed,fold=-1,condition="base",task=task,files_present=not missing,missing_files=";".join(missing)))
    return pd.DataFrame(rows)


def bootstrap(contrasts: pd.DataFrame,resamples: int,rng_seed: int) -> pd.DataFrame:
    rng=np.random.default_rng(rng_seed)
    rows=[]
    for task,task_frame in contrasts.groupby("task"):
        nseeds=task_frame.seed.nunique()
        chosen=rng.integers(0,nseeds,size=(resamples,nseeds))
        for country,frame in task_frame.groupby("Country"):
            frame=frame.copy()
            frame["outcome"]=frame.metric+":"+frame.contrast
            samples,seeds,outcomes=shared_bootstrap(frame,resamples,rng)
            fixed=samples.mean(axis=1)
            combined=np.take_along_axis(samples,chosen[:,:,None],axis=1).mean(axis=1)
            point=frame.groupby(["seed","outcome"]).value.mean().groupby("outcome").mean()
            for j,outcome in enumerate(outcomes):
                metric,contrast=outcome.split(":")
                for method,array in [("shared_participant_fixed_seeds",fixed),("shared_participant_and_seed",combined)]:
                    low,high=np.quantile(array[:,j],[.025,.975])
                    rows.append(dict(task=task,Country=country,metric=metric,contrast=contrast,mean=point[outcome],ci_low=low,ci_high=high,method=method,seeds=len(seeds),resamples=resamples))
    return pd.DataFrame(rows)


def plot(intervals: pd.DataFrame,seed_contrasts: pd.DataFrame,output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    for metric in METRICS:
        fig,axes=plt.subplots(2,2,figsize=(10.5,7),sharey="row")
        selected=intervals.loc[intervals.metric.eq(metric)&intervals.method.eq("shared_participant_and_seed")]
        for column,task in enumerate(["Giving","Taking"]):
            for row,(contrasts,labels) in enumerate([(["us_gain_vs_base","pooled_gain_vs_base","china_gain_vs_base"],["US adapter","Pooled adapter","China adapter"]),(["own_minus_other","pooled_minus_other"],["Own − other country","Pooled − other country"])]):
                axis=axes[row,column]
                for index,(country,color) in enumerate([("United States","#2476A8"),("China","#C76B26")]):
                    data=selected.loc[selected.task.eq(task)&selected.Country.eq(country)].set_index("contrast").loc[contrasts]
                    x=np.arange(len(contrasts))+(-.12 if index==0 else .12)
                    y=data["mean"].to_numpy()
                    axis.errorbar(x,y,yerr=np.vstack([y-data.ci_low,data.ci_high-y]),fmt="o" if index==0 else "s",color=color,capsize=3,label=country)
                    for j,contrast in enumerate(contrasts):
                        points=seed_contrasts.loc[seed_contrasts.metric.eq(metric)&seed_contrasts.task.eq(task)&seed_contrasts.Country.eq(country)&seed_contrasts.contrast.eq(contrast),"value"].to_numpy()
                        axis.scatter(x[j]+np.linspace(-.035,.035,len(points)),points,s=11,color=color,alpha=.65)
                axis.axhline(0,color=".5",lw=.8)
                axis.grid(axis="y",alpha=.2)
                axis.set_xticks(range(len(labels)),labels)
                axis.set_title(f"{task}: {'gain over frozen Centaur' if row==0 else 'paired country contrasts'}",loc="left",fontweight="bold")
                if column==0:
                    axis.set_ylabel("LL difference (nats per allocation)")
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.055),ncol=2,frameon=False)
        nseeds=seed_contrasts.seed.nunique()
        fig.text(.5,.031,f"Points: equal mean of {nseeds} fitted seeds; dots: seeds; bars: shared-participant + seed bootstrap 95% intervals.",ha="center",fontsize=8)
        fig.text(.5,.01,"Five fixed folds provide one out-of-fold prediction per target and seed. Giving uses held-out participants; Taking uses separate samples.",ha="center",fontsize=7.7)
        fig.suptitle(f"Corrected allocation responses · {'raw response LL' if metric=='raw_response_ll' else 'conditional LL'}",fontsize=12)
        fig.tight_layout(rect=(0,.12,1,.95))
        for suffix in ["png","pdf"]:
            fig.savefig(output/f"corrected_adapters_{metric}.{suffix}",dpi=220)
        plt.close(fig)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True,exist_ok=True)
    frame=coverage(args)
    frame.to_csv(args.output_dir/"file_coverage.csv",index=False)
    source=load_explicit_behavior(args.behavior_path,"Token","percent")
    source_sha=sha(args.behavior_path)
    inputs={args.behavior_path,Path(__file__).resolve()}
    sidecar=args.behavior_path.with_suffix(".provenance.json")
    if not sidecar.is_file() or json.loads(sidecar.read_text()).get("output_sha256")!=source_sha:
        raise ValueError("Missing or mismatching corrected-behavior provenance sidecar")
    inputs.add(sidecar)
    corpus_data={}
    for seed in args.seeds:
        directory=args.corpus_root/f"seed{seed}"
        protocol_path=directory/"protocol.json"
        protocol=json.loads(protocol_path.read_text())
        if protocol.get("source_behavior_sha256")!=source_sha or protocol.get("seed")!=seed:
            raise ValueError("Corpus protocol does not match corrected behavior or seed")
        split_path=directory/"participant_split_manifest.csv"
        split=pd.read_csv(split_path)
        corpus_data[seed]=(sha(protocol_path),select_people(source.loc[source.Dataset.isin([1,2])],split,"test"))
        inputs.update([protocol_path,split_path])
    ready=bool(frame.files_present.all())
    (args.output_dir/"readiness.json").write_text(json.dumps(dict(status="files_present_unvalidated" if ready else "waiting_for_complete_outputs",seeds=args.seeds,expected_fit_count=len(args.seeds)*15,evaluation_files_present=int(frame.files_present.sum()),expected_evaluation_groups=len(frame),behavior_sha256=source_sha),indent=2)+"\n")
    if args.coverage_only:
        print(f"{int(frame.files_present.sum())}/{len(frame)} evaluation groups present; corrected corpus provenance verified")
        return
    if not ready:
        raise ValueError("Incomplete corrected experiment; inspect file_coverage.csv. No scientific summary written.")
    controls=None
    if args.baseline_controls is not None:
        controls=pd.read_csv(args.baseline_controls)
        inputs.add(args.baseline_controls)
    parts=[]
    validation=[]
    for seed in args.seeds:
        corpus_sha,giving=corpus_data[seed]
        for task,people in [("Giving",giving),("Taking",source.loc[source.Dataset.isin([3,4])])]:
            if controls is not None:
                base=controls.loc[controls.model_condition.eq("centaur")&controls.task.eq(task.lower())&controls.prompt_variant.eq("original")].copy()
            else:
                directory=args.results_root/f"seed{seed}/base/evaluations/{task.lower()}"
                path,meta_path=directory/"candidate_probabilities.csv",directory/"run_metadata.json"
                base=pd.read_csv(path)
                validate_evaluation(json.loads(meta_path.read_text()),base,source_sha,corpus_sha,seed,-1,"base",task)
                inputs.update([path,meta_path])
            check_candidates(base)
            parts.append(score_participants(people,base).assign(seed=seed,task=task,condition="base"))
            for condition in CONDITIONS:
                fold_parts=[]
                stored=[]
                for fold in range(5):
                    directory=args.results_root/f"seed{seed}/fold{fold}/{condition}"
                    train_meta,train_metrics=directory/"train_metadata.json",directory/"train_metrics.json"
                    validate_fit_metadata(json.loads(train_meta.read_text()),json.loads(train_metrics.read_text()),source_sha,seed,fold,condition,args.expected_optimizer_steps)
                    evaluation=directory/f"evaluations/{task.lower()}"
                    candidate_path,meta_path,metric_path=[evaluation/name for name in ["candidate_probabilities.csv","run_metadata.json","model_metrics.csv"]]
                    candidates=pd.read_csv(candidate_path)
                    validate_evaluation(json.loads(meta_path.read_text()),candidates,source_sha,corpus_sha,seed,fold,condition,task)
                    fold_parts.append(candidates)
                    stored.append(pd.read_csv(metric_path))
                    inputs.update([train_meta,train_metrics,candidate_path,meta_path,metric_path])
                scores=score_participants(people,pd.concat(fold_parts,ignore_index=True))
                previous=pd.concat(stored,ignore_index=True)
                expected=previous.loc[previous["sample"].eq("Combined")].groupby("country")[["centaur_nll_nats","conditional_nll_nats"]].mean()
                actual=scores.groupby("Country")[METRICS].mean()
                for country in COUNTRIES:
                    for metric,column in [("raw_response_ll","centaur_nll_nats"),("conditional_ll","conditional_nll_nats")]:
                        error=abs(actual.loc[country,metric]+expected.loc[country,column])
                        if error>1e-10:
                            raise ValueError("Saved metrics disagree with corrected source allocation recomputation")
                        validation.append(dict(seed=seed,condition=condition,task=task,Country=country,metric=metric,absolute_error=error))
                parts.append(scores.assign(seed=seed,task=task,condition=condition))
    participants=pd.concat(parts,ignore_index=True)
    participants.to_csv(args.output_dir/"participant_log_likelihoods.csv",index=False)
    participants.groupby(["seed","task","condition","Country"])[METRICS].mean().reset_index().to_csv(args.output_dir/"seed_metrics.csv",index=False)
    pd.DataFrame(validation).to_csv(args.output_dir/"saved_metric_validation.csv",index=False)
    contrast_parts=[]
    for (task,country),part in participants.groupby(["task","Country"]):
        own=COUNTRIES[country]
        other="china" if own=="us" else "us"
        for metric in METRICS:
            wide=part.pivot(index=["seed"]+KEYS,columns="condition",values=metric)
            pairs={f"{condition}_gain_vs_base":(condition,"base") for condition in CONDITIONS}
            pairs.update({"own_minus_other":(own,other),"pooled_minus_other":("pooled",other),"own_minus_pooled":(own,"pooled")})
            for name,(left,right) in pairs.items():
                contrast_parts.append((wide[left]-wide[right]).rename("value").reset_index().assign(task=task,Country=country,metric=metric,contrast=name))
    contrasts=pd.concat(contrast_parts,ignore_index=True)
    seed_contrasts=contrasts.groupby(["task","Country","metric","contrast","seed"]).value.mean().reset_index()
    seed_contrasts.to_csv(args.output_dir/"seed_contrasts.csv",index=False)
    seed_contrasts.groupby(["task","Country","metric","contrast"]).value.agg(mean="mean",seed_sd="std",seed_min="min",seed_max="max",seeds="count").reset_index().to_csv(args.output_dir/"contrast_summary.csv",index=False)
    intervals=bootstrap(contrasts,args.bootstrap_resamples,args.bootstrap_seed)
    intervals.to_csv(args.output_dir/"bootstrap_intervals.csv",index=False)
    if not args.no_plots:
        plot(intervals,seed_contrasts,args.output_dir)
    pd.DataFrame([dict(path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size) for path in sorted(inputs)]).to_csv(args.output_dir/"input_provenance.csv",index=False)
    metadata=dict(status="complete_validated_corrected_allocation_analysis",behavior_path=str(args.behavior_path.resolve()),source_behavior_sha256=source_sha,corpus_root=str(args.corpus_root.resolve()),results_root=str(args.results_root.resolve()),seeds=args.seeds,expected_optimizer_steps=args.expected_optimizer_steps,bootstrap_seed=args.bootstrap_seed,bootstrap_resamples=args.bootstrap_resamples,
        baseline="explicit frozen Centaur controls" if controls is not None else "fresh per-seed frozen Centaur evaluation",
        uncertainty="Shared participant multiplicities across overlapping seed test sets, stratified by Dataset/Nation; five fixed folds are stitched once per seed. Separate participant-only and empirical seed-plus-participant percentile intervals. Three seeds provide limited training-run uncertainty; target groups/folds fixed, not resampled.",
        contrasts={"own_minus_other":"Specialist advantage on each test population","pooled_minus_other":"Selection penalty: same-size pooled minus other-country-selected adapter on the outside population","own_minus_pooled":"Specialist advantage over same-size pooled"},
        scope="Corrected allocation responses only; 25 out-of-fold targets per seed. Giving manifest-held-out participants; all separately collected Taking participants.")
    (args.output_dir/"analysis_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    (args.output_dir/"readiness.json").write_text(json.dumps(dict(status="complete_validated",seeds=args.seeds,validated_fit_count=len(args.seeds)*15,behavior_sha256=source_sha),indent=2)+"\n")
    reported=intervals.loc[intervals.metric.eq("raw_response_ll")&intervals.method.eq("shared_participant_and_seed")&intervals.contrast.isin(["own_minus_other","pooled_minus_other"])]
    lines=["# Corrected allocation adapter results","",f"Validated {len(args.seeds)*15} fits across {len(args.seeds)} seeds, five fixed item folds, and three culture conditions.","","Positive specialist advantage means the own-country adapter predicts that population better than the other-country adapter. Positive selection penalty means the same-size pooled adapter predicts the outside population better than the selected other-country adapter.","","| Task | Population | Contrast | Raw LL difference | 95% interval |","|---|---|---|---:|---:|"]
    for row in reported.itertuples():
        lines.append(f"| {row.task} | {row.Country} | {row.contrast} | {row.mean:.6f} | [{row.ci_low:.6f}, {row.ci_high:.6f}] |")
    lines.extend(["","Units are nats per allocation response. Conditional LL results are reported separately in CSV files. Intervals resample participants with shared weights across seed overlaps, then resample the observed fitted seeds. Five fixed folds are stitched within each seed; they are not independent repetitions. Only three seeds limit inference about training variability.","","Giving uses manifest-held-out participants; Taking uses separately collected samples. No Taking responses or held-out target items were used to fit these adapters.","",f"Corrected behavior SHA256: `{source_sha}`."])
    (args.output_dir/"run_summary.md").write_text("\n".join(lines)+"\n")
    print(f"Validated and aggregated {len(args.seeds)*15} corrected fits into {args.output_dir}")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior-path",type=Path,required=True)
    parser.add_argument("--corpus-root",type=Path,required=True)
    parser.add_argument("--results-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--baseline-controls",type=Path)
    parser.add_argument("--seeds",nargs="+",type=int,default=[0,1,2])
    parser.add_argument("--expected-optimizer-steps",type=int,default=25)
    parser.add_argument("--bootstrap-resamples",type=int,default=4000)
    parser.add_argument("--bootstrap-seed",type=int,default=20260930)
    parser.add_argument("--coverage-only",action="store_true")
    parser.add_argument("--no-plots",action="store_true",help="Write numeric and text artifacts without importing matplotlib")
    run(parser.parse_args())


if __name__=="__main__":
    main()
