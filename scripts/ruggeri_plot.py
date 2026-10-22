from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tueplots import bundles
ap=argparse.ArgumentParser(description='Plot the aggregate Ruggeri country audit.')
ap.add_argument('--results-dir',type=Path,default=Path(__file__).resolve().parents[1]/'results/ruggeri')
ap.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'build/ruggeri')
args=ap.parse_args();O=args.output_dir;O.mkdir(parents=True,exist_ok=True)
s=json.loads((args.results_dir/'results.json').read_text());c=pd.read_csv(args.results_dir/'country_results.csv')
plt.rcParams.update(bundles.neurips2024(usetex=False));plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial'],'font.size':10,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#A6B0B8','text.color':'#253340','axes.labelcolor':'#253340','xtick.color':'#253340','ytick.color':'#253340','svg.fonttype':'none','pdf.fonttype':42})
colors=['#477FA0','#C98C55']
def save(fig,name):
 for ext in ['svg','pdf','png']:fig.savefig(O/f'{name}.{ext}',dpi=240,bbox_inches='tight',facecolor='white')
 plt.close(fig)
f,ax=plt.subplots(figsize=(6.2,2.8));r=s['primary'];means=np.array(r['means']);ci=np.array(r['mean_ci'])
for g in range(2):
 x=np.arange(2)+(g-.5)*.29;y=means[g];ax.bar(x,y,.27,color=colors[g],label=['Western reference','Other countries'][g]);ax.errorbar(x,y,yerr=np.array([y-ci[g,:,0],ci[g,:,1]-y]),fmt='none',color='#253340',lw=1,capsize=3)
 for xx,yy in zip(x,y):ax.text(xx,yy+.045,f'{yy:.3f}',ha='center',fontsize=10)
ax.set_xticks([0,1],['Frozen Centaur','Frozen Llama']);ax.set_ylim(0,.92);ax.set_yticks([0,.2,.4,.6,.8]);ax.set_ylabel('Prediction error (nats)');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.19),ncol=2,frameon=False);ax.tick_params(axis='x',length=0);save(f,'ruggeri_h1')
# A direct contrast avoids mistaking overlap between group intervals for a test.
f,ax=plt.subplots(figsize=(7.4,3.2));keys=['primary','participant_weighted','including_eastern_europe','anglophone','n_at_least_10','first_response'];labels=['Primary: equal country weight','Participant weight','Include Eastern Europe','Anglophone reference','Countries with ≥10 participants','First response only']
for i,k in enumerate(keys):
 r=s[k];v=r['other_minus_west'][0];lo,hi=r['contrast_ci'][0];ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color='#477FA0',capsize=3,ms=5,lw=1.2)
ax.axvline(0,color='#A6B0B8',lw=1);ax.set_yticks(range(len(keys)),labels);ax.invert_yaxis();ax.set_xlabel('Other − reference prediction error (nats)');ax.set_xlim(-.18,.04);ax.set_xticks([-.15,-.10,-.05,0]);ax.tick_params(axis='y',length=0);save(f,'ruggeri_sensitivity')
# Alphabetical country panels: no selection by model performance.
f,axs=plt.subplots(1,3,figsize=(13,7.5),sharex=True)
for ax,inds in zip(axs,np.array_split(np.arange(len(c)),3)):
 q=c.iloc[inds]
 for i,(_,row) in enumerate(q.iterrows()):
  color=colors[0] if row.western else colors[1];v=row.centaur;ax.errorbar(v,i,xerr=[[v-row.centaur_low],[row.centaur_high-v]],fmt='o',color=color,capsize=2,ms=4,lw=1)
 ax.set_yticks(range(len(q)),[f'{x.nationality} ({x.n})' for x in q.itertuples()],fontsize=9);ax.invert_yaxis();ax.set_xlabel('Centaur error (nats)');ax.tick_params(axis='y',length=0);ax.set_xlim(.12,.67)
save(f,'ruggeri_all_countries')
