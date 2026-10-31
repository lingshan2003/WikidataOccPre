#!/usr/bin/env python3
"""Compare GraphMask seeds with ten frozen per-window RGCN42s."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 groups=[('inherited','Inherited ties'),('intimate_partnership','Intimate partnership'),('education_mentorship','Education / mentorship'),('influence_succession','Influence'),('other_acquired','Other acquired ties')]
 rows=[];metrics=[];trends=[]
 for year in range(1900,1910):
  original=None;counts=None
  for seed in [42,43,44]:
   d=a.input_root/f'seed{seed}'/f'center_{year}'/'test_report';v=json.loads((d/'test_metrics.json').read_text());current=[l['message_observations'] for l in v['layers']]
   if original is None:original=v['original'];counts=current
   else:assert original==v['original'] and counts==current
   metrics.append(dict(year=year,seed=seed,original_f1=v['original']['macro_f1']*100,masked_f1=v['masked']['macro_f1']*100,agreement=v['prediction_agreement']*100,overall_keep=v['hard_retention_rate']*100,layer0_keep=v['layers'][0]['hard_retention_rate']*100))
   rel=[x for x in csv.DictReader((d/'relations_base.csv').open()) if x['layer']=='0'];assert len(rel)==5
   assert abs(sum(float(x['retained_edge_share']) for x in rel)-1)<1e-6
   for x in rel:rows.append(dict(year=year,seed=seed,group=x['relation'],share_pct=float(x['retained_edge_share'])*100))
 fig,axes=plt.subplots(3,2,figsize=(12,11));years=list(range(1900,1910))
 for ax,(g,label) in zip(axes.flat,groups):
  for seed,color,marker,style in [(42,'#2878ad','o','-'),(43,'#e78a25','s','--'),(44,'#25977e','^',':')]:
   vals=[next(x['share_pct'] for x in rows if x['year']==y and x['seed']==seed and x['group']==g) for y in years]
   ax.plot(years,vals,label=f'GraphMask seed {seed}',color=color,marker=marker,linestyle=style,markersize=4)
   trends.append(dict(group=g,seed=seed,mean_annual_absolute_change_pp=sum(abs(b-a) for a,b in zip(vals,vals[1:]))/9))
  ax.set(title=label,xlabel='Window center year',ylabel='Retained Layer 0 message share (%)',xticks=[1900,1902,1904,1906,1908,1909]);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=9)
 axes.flat[-1].axis('off');axes.flat[-1].text(.05,.65,'Ten frozen RGCN seed42 models\nOne shared GraphMask per seed\nFixed validation/report sampling\nRaw annual values; no smoothing\nEach panel uses its own y-axis scale',transform=axes.flat[-1].transAxes,fontsize=11)
 fig.suptitle('Freebase: shared GraphMask seed comparison — Layer 0',fontsize=15);fig.tight_layout(rect=(0,0,1,.96))
 for ext in ['png','pdf','svg']:fig.savefig(a.output_dir/f'shared_graphmask_seeds.{ext}',dpi=200,bbox_inches='tight')
 for name,items in [('annual_relation_shares',rows),('metrics',metrics),('annual_variation',trends)]:
  with (a.output_dir/f'{name}.tsv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(items[0]),delimiter='\t');w.writeheader();w.writerows(items)
 print('Validated 30 reports, 150 group observations')
if __name__=='__main__':main()
