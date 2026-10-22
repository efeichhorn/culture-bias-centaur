"""Reproduce the prespecified Ruggeri country audit. Private inputs are not redistributed."""
from pathlib import Path
import argparse,json,re,zipfile,pickle,io,codecs,hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--test-file',type=Path,required=True)
ap.add_argument('--private-dir',type=Path,required=True,help='Local tokenizer.json and psych201-*.parquet metadata; participant scores stay here')
ap.add_argument('--score-dir',type=Path,required=True,help='Released centaur70b.pth and llama70b.pth')
ap.add_argument('--output-dir',type=Path,required=True)
ap.add_argument('--protocol',type=Path,default=R/'designs/ruggeri_protocol.json')
args=ap.parse_args()
from tokenizers import Tokenizer
P=args.private_dir; O=args.output_dir; O.mkdir(parents=True,exist_ok=True)
protocol=json.loads(args.protocol.read_text())
source=args.test_file
original=pd.DataFrame([x for l in source.open() if (x:=json.loads(l))['experiment'].startswith('ruggeri')])
# All public metadata are filtered before loading. Original test row order is retained.
import pyarrow.parquet as pq
meta=pd.concat([pq.read_table(p,filters=[('experiment','==','ruggeri2022globalizability/exp1.csv')],columns=['experiment','participant','text','nationality','is_psych101_test']).to_pandas() for p in sorted(P.glob('psych201-*.parquet'))])
d=original.merge(meta,on=['experiment','participant'],how='left',validate='one_to_one',sort=False,suffixes=('','_meta'))
assert d.participant.tolist()==original.participant.tolist()
matched=d.nationality.notna(); assert (d.loc[matched,'text']==d.loc[matched,'text_meta']).all()
assert d.loc[matched,'is_psych101_test'].all()
tok=Tokenizer.from_file(str(P/'tokenizer.json'))
left=tok.encode(' <<',add_special_tokens=False).ids;right=tok.encode('>>',add_special_tokens=False).ids
counts=[];lens=[]
for text in d.text:
 ids=tok.encode(text).ids;lens.append(len(ids));assert len(ids)<32768
 starts=[i+len(left) for i in range(len(ids)) if ids[i:i+len(left)]==left]
 stops=[i for i in range(len(ids)) if ids[i:i+len(right)]==right]
 answers=re.findall(r'<<([^<>]+)>>',text)
 assert len(starts)==len(stops)==len(answers)
 for a,b,answer in zip(starts,stops,answers):
  assert b-a==1 and tok.decode(ids[a:b])==answer,(a,b,answer)
 counts.append(len(answers))
assert sum(counts)==15421
class NumpyOnly(pickle.Unpickler):
 def find_class(self,m,n):
  allowed={('numpy.core.multiarray','_reconstruct'):np._core.multiarray._reconstruct,('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype,('_codecs','encode'):codecs.encode}
  if (m,n) not in allowed:raise ValueError((m,n))
  return allowed[m,n]
paths={}
for model in ['centaur','llama']:
 path=args.score_dir/f'{model}70b.pth';paths[model]=hashlib.sha256(path.read_bytes()).hexdigest()
 with zipfile.ZipFile(path) as z:obj=NumpyOnly(io.BytesIO(z.read(next(n for n in z.namelist() if n.endswith('/data.pkl'))))).load()
 a=np.asarray(obj['ruggeri2022globalizability']).ravel().astype(float)
 assert len(a)==sum(counts) and np.isfinite(a).all() and (a>=0).all()
 offsets=np.r_[0,np.cumsum(counts)];d[model]=[a[x:y].mean() for x,y in zip(offsets[:-1],offsets[1:])];d[model+'_first']=a[offsets[:-1]]
d['n_responses']=counts
coverage={'original_participants':len(d),'matched_participants':int(matched.sum()),'missing_participants':int((~matched).sum()),'matched_responses':int(d.loc[matched,'n_responses'].sum()),'all_responses':sum(counts),'countries':int(d.nationality.nunique()),'max_sequence_tokens':max(lens),'all_response_spans_single_token':True,'exact_prompt_matches':int(matched.sum()),'test_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'score_sha256':paths,'missing_vs_matched_means':d.assign(matched=matched).groupby('matched')[['centaur','llama']].mean().to_dict('index')}
d.drop(columns=['text','text_meta']).to_csv(P/'participant_scores.csv',index=False)
d=d.loc[matched].copy(); western=set(protocol['western_countries']); d['western']=d.nationality.isin(western)
B=10000
# Vectorized within-country resampling; each duplicated country draw gets independent participant resamples.
def estimate(frame,group_set,cols=('centaur','llama'),weighted=False,seed=20261021):
 rng=np.random.default_rng(seed); points=[];draws=[]; ns=[]
 for iswest in [True,False]:
  f=frame[frame.nationality.isin(group_set)==iswest];cs=[g[list(cols)].to_numpy() for _,g in f.groupby('nationality')]; k=len(cs)
  ns.append({'countries':k,'participants':len(f)})
  points.append(np.average([a.mean(0) for a in cs],axis=0,weights=[len(a) for a in cs] if weighted else None))
  if weighted:
   result=np.zeros((B,len(cols)))
   for a in cs:result+=a[rng.integers(len(a),size=(B,len(a)))].sum(1)
   result/=len(f)
  else:
   selections=rng.integers(k,size=(B,k));result=np.zeros((B,len(cols)))
   for j,a in enumerate(cs):
    row,col=np.where(selections==j)
    values=a[rng.integers(len(a),size=(len(row),len(a)))].mean(1)
    np.add.at(result,row,values)
   result/=k
  draws.append(result)
 points=np.array(points);draws=np.array(draws);delta=points[1]-points[0];dd=draws[1]-draws[0]
 gain=points[:,1]-points[:,0];gd=draws[:,:,1]-draws[:,:,0]
 return {'groups':ns,'means':points.tolist(),'mean_ci':np.quantile(draws,[.025,.975],axis=1).transpose(1,2,0).tolist(),'other_minus_west':delta.tolist(),'contrast_ci':np.quantile(dd,[.025,.975],axis=0).T.tolist(),'centaur_gain':gain.tolist(),'gain_ci':np.quantile(gd,[.025,.975],axis=1).T.tolist(),'western_minus_other_gain':float(gain[0]-gain[1]),'gain_difference_ci':np.quantile(gd[0]-gd[1],[.025,.975]).tolist()}
results={'primary':estimate(d,western),'participant_weighted':estimate(d,western,weighted=True),'including_eastern_europe':estimate(d,western|{'Belarus','Bulgaria','Czechia','Hungary','Poland','Moldova','Romania','Russia','Slovakia','Ukraine'}),'anglophone':estimate(d,{'United States','United Kingdom','Canada','Australia','New Zealand'}),'n_at_least_10':estimate(d[d.groupby('nationality').nationality.transform('size')>=10],western),'first_response':estimate(d,western,cols=('centaur_first','llama_first'))}
country=d.groupby('nationality').agg(n=('participant','size'),centaur=('centaur','mean'),llama=('llama','mean'),western=('western','first')).reset_index()
rng=np.random.default_rng(8401)
for name,g in d.groupby('nationality'):
 vals=g[['centaur','llama']].to_numpy();boot=vals[rng.integers(len(vals),size=(B,len(vals)))].mean(1);ci=np.quantile(boot,[.025,.975],axis=0)
 for j,m in enumerate(['centaur','llama']):
  country.loc[country.nationality==name,m+'_low']=ci[0,j];country.loc[country.nationality==name,m+'_high']=ci[1,j]
country.to_csv(O/'country_results.csv',index=False)
(O/'coverage.json').write_text(json.dumps(coverage,indent=2));(O/'results.json').write_text(json.dumps(results,indent=2))
print(json.dumps(coverage,indent=2)); print(json.dumps(results,indent=2))
