#!/usr/bin/env python3
"""Global101 shared GraphMask reports and comparisons to decadal/independent masks."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];P=R/'artifacts/freebase_global_shared_graphmask';O=R/'outputs/freebase_global_shared_graphmask';O.mkdir(parents=True,exist_ok=True)
def read_tsv(p):return list(csv.DictReader(p.open(),delimiter='\t'))
rows=[];metrics=[]
manifest=json.loads((P/'experiment_manifest.json').read_text());assert manifest['status']=='complete' and (P/'exit_code').read_text().strip()=='0'
h=json.loads((P/'training_history.json').read_text())['history'];selection=next(x for x in h if x['epoch']==manifest['selected_epoch']);assert selection['enabled_layers'][0]
decadal=read_tsv(R/'outputs/freebase_decadal_shared_graphmask/matrix_summary.tsv');dm={int(x['year']):x for x in decadal}
for year in range(1900,2001):
 d=P/f'center_{year}/test_report';v=json.loads((d/'test_metrics.json').read_text())
 assert abs(float(dm[year]['original_f1'])-100*v['original']['macro_f1'])<1e-8
 drop=max(0,(v['original']['macro_f1']-v['masked']['macro_f1'])/v['original']['macro_f1'])*100
 metrics.append(dict(year=year,original_f1=v['original']['macro_f1']*100,masked_f1=v['masked']['macro_f1']*100,relative_f1_drop_pct=drop,agreement_pct=v['prediction_agreement']*100,overall_keep_pct=v['hard_retention_rate']*100,layer0_keep_pct=v['layers'][0]['hard_retention_rate']*100))
 rel=[x for x in csv.DictReader((d/'relations_base.csv').open()) if x['layer']=='0'];assert len(rel)==5 and abs(sum(float(x['retained_edge_share']) for x in rel)-1)<1e-6
 for x in rel:rows.append(dict(year=year,relation=x['relation'],retained_share_pct=float(x['retained_edge_share'])*100,group_keep_pct=float(x['hard_retention_rate'])*100))
assert len(rows)==505 and len(metrics)==101
for name,values in [('matrix_summary',metrics),('relation_group_summary',rows)]:
 with (O/f'{name}.tsv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(values[0]),delimiter='\t');w.writeheader();w.writerows(values)
base=read_tsv(R/'outputs/l1_only_comparison_20261020/freebase/layer0_group_annual.tsv');dec=read_tsv(R/'outputs/freebase_decadal_shared_graphmask/relation_group_summary.tsv')
groups=[('inherited','Inherited ties'),('intimate_partnership','Intimate partnership'),('education_mentorship','Education / mentorship'),('influence_succession','Influence'),('other_acquired','Other acquired ties')]
stats=[]
for mode in ['global','comparison']:
 fig,axes=plt.subplots(3,2,figsize=(12,11))
 for ax,(g,label) in zip(axes.flat,groups):
  group_color={'inherited':'#2878ad','intimate_partnership':'#e78a25','education_mentorship':'#25977e','influence_succession':'#ca5968','other_acquired':'#69838f'}[g]
  gg=sorted([x for x in rows if x['relation']==g],key=lambda x:x['year']);vals=[x['retained_share_pct'] for x in gg]
  if mode=='comparison':
   bb=sorted([x for x in base if x['group']==g],key=lambda x:int(x['year']))
   ax.plot([int(x['year']) for x in bb],[float(x['retained_message_share_percent']) for x in bb],color='#aaaaaa',alpha=.7,linewidth=1,label='Independent')
   dd=sorted([x for x in dec if x['relation']==g],key=lambda x:int(x['year']))
   for i,group in enumerate(sorted({x['window_group'] for x in dd})):
    a=[x for x in dd if x['window_group']==group];ax.plot([int(x['year']) for x in a],[float(x['retained_share_pct']) for x in a],color='#e78a25',linewidth=1.2,label='Decadal shared' if i==0 else None)
  ax.plot([x['year'] for x in gg],vals,color=(group_color if mode=='global' else '#2878ad'),linewidth=1.6,label='Global shared');ax.set(title=label,xlabel='Window center year',ylabel='Retained Layer 0 message share (%)',xlim=(1900,2000));ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
  if mode=='comparison':ax.legend(frameon=False,fontsize=8)
  else:
   bb=sorted([x for x in base if x['group']==g],key=lambda x:int(x['year']));dd=sorted([x for x in dec if x['relation']==g],key=lambda x:int(x['year']))
   tv=lambda a:sum(abs(y-x) for x,y in zip(a,a[1:]))/100
   early=sum(x['retained_share_pct'] for x in gg if 1901<=x['year']<=1930)/30;late=sum(x['retained_share_pct'] for x in gg if 1971<=x['year']<=2000)/30
   stats.append(dict(relation=g,early_share_pct=early,late_share_pct=late,independent_mean_annual_change_pp=tv([float(x['retained_message_share_percent']) for x in bb]),decadal_mean_annual_change_pp=tv([float(x['retained_share_pct']) for x in dd]),global_mean_annual_change_pp=tv(vals)))
 axes.flat[-1].axis('off');axes.flat[-1].text(.05,.65,'101 frozen RGCN42 models\nGlobal GraphMask42; randomized order42\nFixed per-window random streams\nRaw annual values; no smoothing\nEach panel uses its own y-axis scale',transform=axes.flat[-1].transAxes,fontsize=11)
 fig.suptitle('Freebase: '+('global shared GraphMask — Layer 0 (101/101)' if mode=='global' else 'independent / decadal / global GraphMask — Layer 0'),fontsize=15);fig.tight_layout(rect=(0,0,1,.96))
 for ext in ['png']:fig.savefig(O/f'{mode}_layer0.{ext}',dpi=200,bbox_inches='tight')
 plt.close(fig)
with (O/'trend_comparison.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(stats[0]),delimiter='\t');w.writeheader();w.writerows(stats)
print('MEANS',{k:round(sum(x[k] for x in metrics)/101,3) for k in metrics[0] if k!='year'});print('WORST_TEST_DROP',max(x['relative_f1_drop_pct'] for x in metrics),'PASS5',sum(x['relative_f1_drop_pct']<=5 for x in metrics));print('STATS');print(json.dumps(stats,indent=2))
