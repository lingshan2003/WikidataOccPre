#!/usr/bin/env python3
"""Seed42 Freebase 1940–1950 and 1980–1990: inherited/education paired deletion."""
import csv,fcntl,json,os,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUT=ROOT/'runs/freebase_inherited_education_ablation_1940_1950_1980_1990_seed42_v1'
YEARS=list(range(1940,1951))+list(range(1980,1991))
GROUPS=['inherited','education_mentorship']
BASE=ROOT/'runs/freebase_grouped_sliding_1900_2000_pm20_step1_l1_only_alive2026_v4'
OUT.mkdir(parents=True,exist_ok=True)
lock=(OUT/'.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (OUT/'experiment_manifest.json').exists():raise RuntimeError('Existing experiment: inspect status before any recovery; refusing duplicate launch')
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2))
taxonomy=OUT/"collapsed_relation_taxonomy.json"
write(taxonomy,dict(name="freebase_collapsed_ablation_groups_v1",version=1,groups={"inherited":["inherited"],"education_mentorship":["education_mentorship"],"remaining":"all_remaining"}))
jobs=[]
for year in YEARS:
 b=BASE/f'center_{year}/multi_group/seed_42'
 record=read(b/'pipeline_train.json');assert record['status']=='complete'
 cmd=record['contract']['command']; assert cmd[cmd.index('--seed')+1]=='42'
 for group,condition,flag in [(g,prefix+g,flag) for g in GROUPS for prefix,flag in [('without_','--drop-relation-taxonomy-groups'),('random_matched_','--match-random-drop-to-relation-taxonomy-groups')]]:
  dest=OUT/f'center_{year}'/condition;dest.mkdir(parents=True,exist_ok=True)
  c=cmd.copy();c[0]=sys.executable;c[c.index('--output-dir')+1]=str(dest);c += ['--relation-taxonomy',str(taxonomy),flag,group]
  jobs.append(dict(year=year,condition=condition,command=c,output=str(dest),group=group,status='pending'))
manifest=dict(question='Retraining damage: inherited and education versus equal original/reverse edge-instance deletion',seed=42,split_seed=20260814,budget='Original max50 epochs, patience6, validation Macro-F1 checkpoint',extra_loss='targeted F1 minus random F1, percentage points',git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),jobs=jobs,status='running')
active=[]
def save():write(OUT/'experiment_manifest.json',manifest)
try:
 while any(j['status']=='pending' for j in jobs) or active:
  for item in active[:]:
   j,p,f=item;code=p.poll()
   if code is not None:
    f.close();j.update(status='complete' if code==0 else 'failed',exit_code=code);pathlib.Path(j['output'],'exit_code').write_text(str(code));active.remove(item);save()
  stats=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  for line in stats.splitlines():
   gpu,free,util=[int(x.strip()) for x in line.split(',')]
   if gpu==2 or free<8000 or util>=60 or gpu in [x[0]['gpu'] for x in active] or len(active)>=4:continue
   j=next((j for j in jobs if j['status']=='pending'),None)
   if j is None:break
   d=pathlib.Path(j['output']);f=(d/'train.log').open('a');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
   write(d/'command.json',j['command']);p=subprocess.Popen(j['command'],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   j.update(status='running',gpu=gpu,pid=p.pid);active.append((j,p,f));save();print('Started',j['year'],j['condition'],'GPU',gpu,flush=True)
  if active or any(j['status']=='pending' for j in jobs):time.sleep(10)
 if any(j['status']!='complete' for j in jobs):raise RuntimeError('Failed jobs: inspect train.log')
 rows=[]
 for y in YEARS:
  baseline=read(BASE/f'center_{y}/multi_group/seed_42/metrics.json')['test']['macro_f1']*100
  for group in GROUPS:
   ds=[read(OUT/f'center_{y}'/(prefix+group)/'metrics.json') for prefix in ['without_','random_matched_']]
   a,b=[d['relation_perturbation'] for d in ds]
   removed=a['edge_count_before']-a['edge_count_after_ablation']
   assert removed==b['edge_count_before']-b['edge_count_after_random_drop']==2*b['random_edge_instance_pairs']
   assert b['random_control_unit']=='original_edge_instance_plus_generated_reverse'
   vals=[d['test']['macro_f1']*100 for d in ds]
   rows.append(dict(year=y,group=group,full_macro_f1=baseline,without_group=vals[0],random_matched=vals[1],extra_loss_pp=vals[0]-vals[1],deleted_directed_edges=removed))
 with (OUT/'summary.tsv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
 manifest['status']='complete';save();(OUT/'exit_code').write_text('0')
except BaseException:
 manifest['status']='failed';save();(OUT/'exit_code').write_text('1');raise
