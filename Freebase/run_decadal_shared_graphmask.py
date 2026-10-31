#!/usr/bin/env python3
"""Schedule disjoint decadal shared GraphMasks over existing Freebase RGCN42s."""
import argparse,csv,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2))
def available_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError(f'Existing active probe owns {path}; refusing duplicate')

def summarize(jobs,out):
    rows=[];relations=[]
    for j in jobs:
        c=read(j['config']);p=ROOT/c['output_root'];m=read(p/'experiment_manifest.json')
        assert m['status']=='complete' and m['config']==c and (p/'exit_code').read_text().strip()=='0'
        h=read(p/'training_history.json')['history'];selected=next(x for x in h if x['epoch']==m['selected_epoch'])
        val=read(p/'validation.json');assert max(v['relative_macro_f1_drop'] for v in val.values())<=c['maximum_relative_f1_drop']
        for year in c['years']:
            d=p/f'center_{year}/test_report';v=read(d/'test_metrics.json')
            rows.append(dict(year=year,window_group=j['group'],selected_epoch=m['selected_epoch'],layer0_enabled=selected['enabled_layers'][0],original_f1=v['original']['macro_f1']*100,masked_f1=v['masked']['macro_f1']*100,prediction_agreement=v['prediction_agreement']*100,overall_keep_pct=v['hard_retention_rate']*100,layer0_keep_pct=v['layers'][0]['hard_retention_rate']*100,validation_relative_f1_drop_pct=val[str(year)]['relative_macro_f1_drop']*100))
            rel=[x for x in csv.DictReader((d/'relations_base.csv').open()) if x['layer']=='0']
            assert len(rel)==5 and abs(sum(float(x['retained_edge_share']) for x in rel)-1)<1e-6
            for x in rel:relations.append(dict(year=year,window_group=j['group'],layer0_enabled=selected['enabled_layers'][0],relation=x['relation'],retained_share_pct=float(x['retained_edge_share'])*100,group_keep_pct=float(x['hard_retention_rate'])*100))
    assert sorted(x['year'] for x in rows)==list(range(1900,2001)) and len(relations)==505
    for name,data in [('matrix_summary',rows),('relation_group_summary',relations)]:
        with (out/f'{name}.tsv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]),delimiter='\t');w.writeheader();w.writerows(data)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--summarize-only',action='store_true');args=parser.parse_args()
    mpath=ROOT/'config/freebase_decadal_shared_graphmask_1900_2000_rgcn42_gm42_order42_v1.json';matrix=read(mpath)
    out=ROOT/'runs_graphmask'/matrix['name'];out.mkdir(exist_ok=True,parents=True)
    lock=(out/'.matrix.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=[];seen=[]
    for config in matrix['configs']:
        c=read(config);base=read(ROOT/c['baseline_config']);years=c['years'];seen+=years
        assert c['model_seed']==c['graphmask_seed']==c['window_order_seed']==42 and c['rng_protocol']=='per_window_batch_v1'
        for year in years:
            model=ROOT/base['model_root']/f'center_{year}/multi_group/seed_42'
            source=read(model/'pipeline_train.json');assert source['status']=='complete'
            cmd=source['contract']['command']
            assert cmd[cmd.index('--seed')+1]=='42' and cmd[cmd.index('--occupation-feature-levels')+1]=='1' and cmd[cmd.index('--auxiliary-features')+1]=='none'
            assert (model/'best_model.pt').is_file() and (model/'metrics.json').is_file()
            for flag in ['--data','--tie-taxonomy']:assert Path(cmd[cmd.index(flag)+1]).is_file()
        p=ROOT/c['output_root'];p.mkdir(parents=True,exist_ok=True);status='pending'
        if (p/'experiment_manifest.json').exists():
            previous=read(p/'experiment_manifest.json');assert previous['config']==c
            if previous['status']=='complete':
                assert (p/'exit_code').read_text().strip()=='0'
                for y in years:assert (p/f'center_{y}/test_report/relations_base.csv').is_file() and (p/f'center_{y}/test_report/test_metrics.json').is_file()
                status='reused'
            else:available_lock(p/'.lock')
        jobs.append(dict(group=f'{years[0]}-{years[-1]}',config=config,output=str(p),status=status))
    assert sorted(seen)==list(range(1900,2001))
    if args.summarize_only:summarize(jobs,out);print('101 windows summarized');return
    manifest=dict(status='running',config=matrix,git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),hostname=os.uname().nodename,cwd=str(ROOT),python=sys.executable,jobs=jobs,started_at=time.time())
    active=[]
    def save():write(out/'experiment_manifest.json',manifest)
    save()
    try:
        while any(j['status']=='pending' for j in jobs) or active:
            for item in active[:]:
                j,p,f=item;code=p.poll()
                if code is not None:
                    f.close();j.update(status='complete' if code==0 else 'failed',exit_code=code,finished_at=time.time());active.remove(item);save();print('[finished]',j['group'],code,flush=True)
            text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            gpus=[tuple(map(lambda x:int(x.strip()),line.split(','))) for line in text.splitlines()]
            for gpu,free,util in sorted(gpus,key=lambda g:(g[2],-g[1])):
                if gpu not in matrix['candidate_gpus'] or free<matrix['minimum_free_mib'] or util>=matrix['maximum_gpu_utilization_exclusive'] or gpu in [x[0]['gpu'] for x in active] or len(active)>=matrix['maximum_parallel_jobs']:continue
                j=next((x for x in jobs if x['status']=='pending'),None)
                if j is None:break
                cmd=[sys.executable,'-u',str(ROOT/'Freebase/train_shared_graphmask.py'),'--config',j['config']]
                if (Path(j['output'])/'experiment_manifest.json').exists():cmd+=['--resume']
                f=(Path(j['output'])/'controller.log').open('a');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                j.update(status='running',gpu=gpu,pid=p.pid,command=cmd,started_at=time.time(),gpu_free_at_launch=free,gpu_util_at_launch=util);active.append((j,p,f));save();print('[started]',j['group'],'GPU',gpu,flush=True)
            if active or any(j['status']=='pending' for j in jobs):time.sleep(10)
        if any(j['status']=='failed' for j in jobs):raise RuntimeError('Some groups failed; preserve per-group logs and recovery state')
        summarize(jobs,out);manifest.update(status='complete',finished_at=time.time());save();(out/'exit_code').write_text('0');print('[summary] 101/101 reports; 505 relation rows',flush=True)
    except BaseException:
        manifest.update(status='failed',error=traceback.format_exc());save();(out/'exit_code').write_text('1');raise
if __name__=='__main__':main()
