#!/usr/bin/env python3
"""Compare shared versus independent probes over the same frozen RGCN42s."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
P=R/'artifacts/freebase_shared_graphmask_1900_1909'
O=R/'outputs/freebase_shared_graphmask_1900_1909';O.mkdir(exist_ok=True,parents=True)
groups=[('inherited','Inherited ties'),('intimate_partnership','Intimate partnership'),('education_mentorship','Education / mentorship'),('influence_succession','Influence'),('other_acquired','Other acquired ties')]
metrics=[];annual=[];trends=[]
for y in range(1900,1910):
 models={}
 for kind in ['independent','shared']:
  d=json.loads((P/f'{kind}/{y}/test_metrics.json').read_text());models[kind]=d
  metrics.append(dict(year=y,method=kind,original_f1=d['original']['macro_f1']*100,masked_f1=d['masked']['macro_f1']*100,relative_f1_drop=max(0,(d['original']['macro_f1']-d['masked']['macro_f1'])/d['original']['macro_f1'])*100,agreement=d['prediction_agreement']*100,overall_keep=d['hard_retention_rate']*100,layer0_keep=d['layers'][0]['hard_retention_rate']*100))
  rel=list(csv.DictReader((P/f'{kind}/{y}/relations_base.csv').open()))
  for row in rel:
   if row['layer']=='0':annual.append(dict(year=y,method=kind,group=row['relation'],share=float(row['retained_edge_share'])*100))
 assert models['independent']['original']==models['shared']['original']
 assert [x['message_observations'] for x in models['independent']['layers']]==[x['message_observations'] for x in models['shared']['layers']]
fig,axes=plt.subplots(3,2,figsize=(12,11));years=list(range(1900,1910))
for ax,(g,label) in zip(axes.flat,groups):
 result={}
 for method,color,marker in [('independent','#888888','o'),('shared','#2878ad','s')]:
  vals=[next(x['share'] for x in annual if x['year']==y and x['method']==method and x['group']==g) for y in years]
  ax.plot(years,vals,label=method.capitalize(),color=color,marker=marker,markersize=4)
  result[method]=sum(abs(b-a) for a,b in zip(vals,vals[1:]))/9
  trends.append(dict(group=g,method=method,mean_annual_absolute_change_pp=result[method]))
 ax.set(title=label,xlabel='Window center year',ylabel='Retained Layer 0 message share (%)',xticks=[1900,1902,1904,1906,1908,1909]);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
 ax.legend(frameon=False)
axes.flat[-1].axis('off');axes.flat[-1].text(.05,.65,'Ten frozen RGCN seed42 models\nShared GraphMask seed42 vs independent seed42\nRaw annual results; no smoothing\nEach panel uses its own y-axis scale',transform=axes.flat[-1].transAxes,fontsize=11)
fig.suptitle('Freebase: shared versus independent GraphMask — Layer 0',fontsize=15);fig.tight_layout(rect=(0,0,1,.96))
for ext in ['png','pdf','svg']:fig.savefig(O/f'shared_vs_independent.{ext}',dpi=200,bbox_inches='tight')
for name,rows in [('metrics',metrics),('annual_relation_shares',annual),('annual_variation',trends)]:
 with (O/f'{name}.tsv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
print('PER YEAR SHARED')
for x in metrics:
 if x['method']=='shared':print(x['year'],*[round(x[k],2) for k in ['original_f1','masked_f1','agreement','overall_keep','layer0_keep']])
for kind in ['independent','shared']:
 a=[x for x in metrics if x['method']==kind];print(kind,{k:round(sum(x[k] for x in a)/10,3) for k in ['original_f1','masked_f1','agreement','overall_keep','layer0_keep']},'worst_test_drop',max(x['relative_f1_drop'] for x in a))
print('VARIATION');print(trends)
