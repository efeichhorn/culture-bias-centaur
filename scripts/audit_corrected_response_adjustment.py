#!/usr/bin/env python3
"""Analyse corrected allocation predictions and fit the H4 response adjustment.

Requires the original participant splits and saved model predictions. The
ridge penalty is fixed at .001. Writes the results to --output.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument('--workspace',type=Path,required=True,help='Local data workspace containing data_cache/ and model_results/')
ap.add_argument('--output',type=Path,default=Path('/tmp/culture_audit_20261013'))
args=ap.parse_args()
root=args.workspace/'model_results/dictator_centaur70b_item_holdout_corrected_20260930'
repo=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo))
from scripts.audit_dictator_response_prior import fit_response_offsets, apply_offsets, training_counts, score
from scripts.dictator_item_holdout_transfer import ITEM_FOLDS
from scripts.analyze_dictator_corrected_adapters import bootstrap
args.output.mkdir(parents=True,exist_ok=True)
inputs=set()
def read(path):
    inputs.add(path)
    return pd.read_csv(path)
def save(df,name):
    if isinstance(df.columns,pd.MultiIndex):
        df=df.copy()
        df.columns=['_'.join(str(part) for part in col if str(part)) for col in df.columns]
    df.to_csv(args.output/name,index=False)

# Saved participant-level LLs provide independent existing wave data.
participants=read(root/'analysis_20261001/participant_log_likelihoods.csv')
participants['sample']=np.where(participants.Dataset.isin([1,3]),'Wave 1','Wave 2')
wave=participants.groupby(['task','Country','sample','seed','condition'])[['raw_response_ll','conditional_ll']].mean().reset_index()
save(wave,'wave_ll_by_seed.csv')
save(wave.groupby(['task','Country','sample','condition'])[['raw_response_ll','conditional_ll']].agg(['mean','std']).reset_index(),'wave_ll_summary.csv')
contrasts=[]
for (task,country,sample,seed),g in wave.groupby(['task','Country','sample','seed']):
    x=g.set_index('condition')
    own,other=('us','china') if country=='United States' else ('china','us')
    for metric in ['raw_response_ll','conditional_ll']:
        for name,a,b in [('own_minus_other',own,other),('pooled_minus_other','pooled',other),('own_minus_pooled',own,'pooled')]:
            contrasts.append(dict(task=task,country=country,sample=sample,seed=seed,metric=metric,contrast=name,value=x.loc[a,metric]-x.loc[b,metric]))
contrasts=pd.DataFrame(contrasts)
save(contrasts,'wave_contrasts_by_seed.csv')
save(contrasts.groupby(['task','country','sample','metric','contrast']).value.agg(mean='mean',seed_sd='std',seed_min='min',seed_max='max').reset_index(),'wave_contrasts_summary.csv')

frames=[]
for s in range(3):
    for c in ['us','china','pooled']:
        for f in range(5):
            for t in ['giving','taking']:
                frames.append(read(root/f'seed{s}/fold{f}/{c}/evaluations/{t}/human_item_comparison.csv').assign(seed=s,fold=f,task=t,condition=c))
d=pd.concat(frames,ignore_index=True)
wide=d.pivot(index=['seed','fold','task','sample','country','GroupNo'],columns='condition',values='centaur_nll_nats').reset_index()
wide['own_minus_other']=np.where(wide.country.eq('United States'),wide.china-wide.us,wide.us-wide.china)
wide['pooled_minus_other']=np.where(wide.country.eq('United States'),wide.china-wide.pooled,wide.us-wide.pooled)
save(wide,'target_wave_raw_ll_contrasts.csv')
rows=[]
for (t,c),sub in wide[wide['sample'].eq('Combined')].groupby(['task','country']):
    item=sub.groupby('GroupNo')[['own_minus_other','pooled_minus_other']].mean()
    folds=sub.groupby('fold')[['own_minus_other','pooled_minus_other']].mean()
    for col in item.columns:
        loo=(item[col].sum()-item[col])/24
        rows.append(dict(task=t,country=c,contrast=col,positive_targets=int((item[col]>0).sum()),targets=25,min_target=item[col].min(),max_target=item[col].max(),min_leave_one_target_out=loo.min(),max_leave_one_target_out=loo.max(),min_fold=folds[col].min(),max_fold=folds[col].max()))
save(pd.DataFrame(rows),'target_sensitivity.csv')
means=d.groupby(['seed','task','sample','country','condition'])[['human_expected_tokens','model_expected_tokens','expected_error_tokens','ordinal_wasserstein_tokens']].mean().reset_index()
save(means,'human_model_means_by_seed.csv')
save(means.groupby(['task','sample','country','condition'])[['human_expected_tokens','model_expected_tokens','expected_error_tokens','ordinal_wasserstein_tokens']].agg(['mean','std']).reset_index(),'human_model_means_summary.csv')
gap_rows=[]
for (seed,task),sub in d[d['sample'].eq('Combined')].groupby(['seed','task']):
    pred=sub[sub.country.eq('United States')].pivot(index='GroupNo',columns='condition',values='model_expected_tokens')
    human=sub[sub.condition.eq('us')].pivot(index='GroupNo',columns='country',values='human_expected_tokens')
    hg=human['United States']-human['China']
    pg=pred.us-pred.china
    gap_rows.append(dict(seed=seed,task=task,human_us_minus_china_mean=hg.mean(),adapter_us_minus_china_mean=pg.mean(),target_gap_correlation=hg.corr(pg),target_gap_sign_matches=int(((hg>0)==(pg>0)).sum()),target_gap_rmse=float(np.sqrt(np.square(pg-hg).mean())),human_gap_sd=hg.std(),adapter_gap_sd=pg.std()))
save(pd.DataFrame(gap_rows),'population_mean_gap_by_seed.csv')

# One shared response logit offset across all targets. Twenty free parameters.
source=args.workspace/'data_cache/dictator_Behavior_corrected_20260930.csv'
expected_sha='65f8913600fddeae3ca99196c2b0e2db76f2bf2021e1b32f4419f954645a5ecd'
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected_sha
behavior=read(source)
assert (behavior.Token % 5).eq(0).all()
behavior['response_value']=(behavior.Token/5).astype(int)
keys=['Dataset','Nation','SubNo']
fit_rows=[]; metric_rows=[]; offset_rows=[]; prior_scores=[]
for seed in range(3):
    corpus=args.workspace/f'data_cache/dictator_centaur70b_corrected_20260930/seed{seed}'
    split=read(corpus/'participant_split_manifest.csv')
    manifest=read(corpus/'training_participant_manifest.csv')
    test_ids=split.loc[split.split.eq('test'),keys]
    giving=behavior.loc[behavior.Dataset.isin([1,2])].merge(test_ids,on=keys,validate='many_to_one')
    taking=behavior.loc[behavior.Dataset.isin([3,4])]
    bases={t:read(root/f'seed{seed}/base/evaluations/{t.lower()}/candidate_probabilities.csv') for t in ['Giving','Taking']}
    for condition in ['us','china','pooled']:
        selected=manifest.loc[manifest.condition.eq(condition),keys]
        assert len(selected)==910 and not selected.duplicated().any()
        assert not len(selected.merge(test_ids,on=keys))
        out={'Giving':[],'Taking':[]}
        for fold,held in ITEM_FOLDS.items():
            groups=sorted(set(range(1,26))-set(held))
            counts,n=training_counts(behavior,selected,groups)
            base_matrix=bases['Giving'].pivot(index='GroupNo',columns='response_value',values='conditional_probability').loc[groups,range(21)]
            offsets,diagnostic=fit_response_offsets(np.log(base_matrix.to_numpy()),counts,ridge=.001)
            fit_rows.append(dict(seed=seed,condition=condition,fold=fold,**diagnostic))
            offset_rows.extend(dict(seed=seed,condition=condition,fold=fold,response_value=v,offset=float(offsets[v])) for v in range(21))
            for task in out:
                out[task].append(apply_offsets(bases[task].loc[bases[task].GroupNo.isin(held)],offsets))
        for task,humans in [('Giving',giving),('Taking',taking)]:
            prior=score(humans,pd.concat(out[task],ignore_index=True))
            base=score(humans,bases[task])
            prior_scores.append(prior.assign(seed=seed,task=task,condition=condition))
            for country in ['United States','China']:
                pp=prior[prior.country.eq(country)]
                bb=base[base.country.eq(country)]
                lora=participants[participants.seed.eq(seed)&participants.task.eq(task)&participants.condition.eq(condition)&participants.Country.eq(country)]
                for metric in ['conditional_ll','raw_response_ll']:
                    pv,bv,lv=pp[metric].mean(),bb[metric].mean(),lora[metric].mean()
                    metric_rows.append(dict(seed=seed,task=task,country=country,condition=condition,metric=metric,prior_ll=pv,base_ll=bv,lora_ll=lv,prior_gain=pv-bv,lora_gain=lv-bv,prior_minus_lora=pv-lv))
save(pd.DataFrame(fit_rows),'prior_fit_diagnostics.csv')
save(pd.DataFrame(offset_rows),'prior_offsets.csv')
metrics=pd.DataFrame(metric_rows)
save(metrics,'prior_metrics_by_seed.csv')
save(metrics.groupby(['task','country','condition','metric'])[['prior_ll','base_ll','lora_ll','prior_gain','lora_gain','prior_minus_lora']].agg(['mean','std']).reset_index(),'prior_metrics_summary.csv')
prior_people=pd.concat(prior_scores).rename(columns={'country':'Country'})
save(prior_people,'prior_participant_scores.csv')
paired_parts=[]
for (task,country,condition),pp in prior_people.groupby(['task','Country','condition']):
    ll=participants[participants.task.eq(task)&participants.Country.eq(country)&participants.condition.eq(condition)]
    bb=participants[participants.task.eq(task)&participants.Country.eq(country)&participants.condition.eq('base')]
    ix=['seed']+keys
    for metric in ['conditional_ll','raw_response_ll']:
        p=pp.set_index(ix)[metric]
        l=ll.set_index(ix)[metric]
        b=bb.set_index(ix)[metric]
        for name,difference in [('prior_minus_lora',p-l),('prior_gain',p-b)]:
            paired_parts.append(difference.rename('value').reset_index().assign(task=task,Country=country,metric=metric,contrast=condition+'_'+name))
paired=pd.concat(paired_parts,ignore_index=True)
assert paired.value.notna().all()
intervals=bootstrap(paired,4000,20261013)
save(intervals,'prior_paired_intervals.csv')
inputs.update([source,Path(__file__),repo/'scripts/audit_dictator_response_prior.py',repo/'scripts/dictator_item_holdout_transfer.py',repo/'scripts/analyze_dictator_corrected_adapters.py',repo/'scripts/audit_dictator_a1.py'])
save(pd.DataFrame([dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(inputs)]),'input_hashes.csv')
(args.output/'metadata.json').write_text(json.dumps(dict(status='exploratory_corrected_cpu_followup',source_sha=expected_sha,seeds=[0,1,2],folds=5,response_offset_free_parameters=20,ridge=.001,uncertainty='Wave/target summaries have seed SD only. Prior paired intervals reuse 4000 shared-participant + observed-seed bootstrap replicates; fixed target groups, fits, and splits. Bootstrap seed 20261013. No target-population generalization inferred.',raw_prior_construction='Preserves frozen Centaur valid-number mass, changes conditional distribution only.',target_expected_values='Conditional on 21 valid responses. Giving is tokens given; Taking is tokens taken.'),indent=2)+'\n')
print(metrics[metrics.metric.eq('conditional_ll')].groupby(['task','country','condition'])[['prior_gain','lora_gain','prior_minus_lora']].mean().round(5).to_string())
print('45 convex fits converged:',all(x['converged'] for x in fit_rows))
print('Saved exploratory audit diagnostics:',args.output)
