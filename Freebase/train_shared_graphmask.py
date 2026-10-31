#!/usr/bin/env python3
"""One shared probe over frozen per-window RGCNs; separate fidelity multipliers."""
import argparse,copy,csv,fcntl,json,os,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from training.attention_common import root_indices
from training.graphmask.common import (load_run_context,make_loader,resolve_device,resolve_fanouts,set_seed,evaluate_probe,graphmask_kl,probe_payload,write_json)
from training.graphmask.core import GraphMaskProbe
from training.train import batch_features
from training.graphmask.window_schedule import window_order,batch_seed

def drop(v):
    a,b=v['original']['macro_f1'],v['masked']['macro_f1']
    if a is None or b is None:raise ValueError('Missing validation F1')
    return max(0.,(a-b)/max(abs(a),1e-12))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config/freebase_shared_graphmask_1900_1909_rgcn42_v1.json')
    p.add_argument('--device',default='cuda:0');p.add_argument('--resume',action='store_true')
    p.add_argument('--preflight-only',action='store_true');args=p.parse_args()
    os.chdir(ROOT);cfg=json.loads(Path(args.config).read_text());base=json.loads((ROOT/cfg['baseline_config']).read_text())
    out=ROOT/cfg['output_root'];out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not args.preflight_only and (out/'experiment_manifest.json').exists() and not args.resume:
        raise RuntimeError('Existing run: inspect before using --resume')
    if args.resume and (out/'exit_code').exists() and (out/'exit_code').read_text().strip()=='0':
        print('Already complete');return
    device=resolve_device(args.device);set_seed(cfg['graphmask_seed']);contexts=[];probe=None
    for year in cfg['years']:
        data_path=ROOT/base['relation_root']/f'center_{year}/multi_group/graph_data.pt'
        model_path=ROOT/base['model_root']/f'center_{year}/multi_group/seed_42/best_model.pt'
        metrics=json.loads((model_path.parent/'metrics.json').read_text())
        assert metrics['run_config']['seed']==cfg['model_seed']
        record=json.loads((model_path.parent/'pipeline_train.json').read_text());assert record['status']=='complete'
        command=record['contract']['command']
        assert command[command.index('--occupation-feature-levels')+1]=='1'
        assert command[command.index('--auxiliary-features')+1]=='none'
        data,meta,ckpt,restored=load_run_context(data_path,model_path,device)
        assert not any(x.requires_grad for x in restored.adapter.model.parameters())
        assert not any(x.requires_grad for x in restored.adapter.reference_model.parameters())
        fanouts=resolve_fanouts(model_path,ckpt,'auto');roots=root_indices(data,'train')
        loader=make_loader(data,roots,fanouts,cfg['batch_size'],shuffle=False,num_workers=0)
        batch=next(iter(loader)).to(device)
        features=batch_features(batch,restored.feature_schema,meta['occupation_unknown_ids'])
        with torch.no_grad():_,traces=restored.adapter.trace(features,batch.edge_index,batch.edge_type)
        candidate=GraphMaskProbe.from_traces(traces,temperature=1/3,location_bias=3.)
        if probe is None:probe=candidate.to(device)
        else:assert candidate.layer_specs==probe.layer_specs
        if contexts:assert meta['relation_to_id']==contexts[0]['meta']['relation_to_id']
        contexts.append(dict(year=year,data_path=data_path,model_path=model_path,data=data,meta=meta,ckpt=ckpt,restored=restored,fanouts=fanouts,roots=roots))
        print(f'[preflight] {year}: frozen RGCN42, roots={len(roots)}',flush=True)
    steps=min(len(c['roots'])//cfg['batch_size'] for c in contexts)
    assert steps>0
    print(f'[balance] {steps} full batches/window/epoch; {len(contexts)} windows',flush=True)
    if args.preflight_only:return
    manifest=dict(status='running',config=cfg,git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),hostname=os.uname().nodename,cwd=str(ROOT),python=sys.executable,steps_per_window=steps,sources=[dict(year=c['year'],data=str(c['data_path']),checkpoint=str(c['model_path'])) for c in contexts])
    write_json(out/'experiment_manifest.json',manifest)
    # A single Adam state for the shared probe; independent dual variables per window.
    optimizer=torch.optim.Adam(probe.parameters(),lr=cfg['learning_rate'])
    alpha=torch.full((len(contexts),),.55,device=device,requires_grad=True)
    dual=torch.optim.RMSprop([alpha],lr=cfg['multiplier_learning_rate'],centered=True)
    history=[];best=None;best_rank=None;start=0
    if args.resume:
        state=torch.load(out/'last_training_state.pt',map_location=device,weights_only=False)
        assert state['config']==cfg
        probe.load_state_dict(state['probe']);optimizer.load_state_dict(state['optimizer']);dual.load_state_dict(state['dual'])
        with torch.no_grad():alpha.copy_(state['alpha'])
        history=state['history'];best=state['best'];best_rank=state['best_rank'];start=state['epoch']
    def validate():
        values={}
        for c in contexts:
            set_seed(cfg['validation_seed'])
            loader=make_loader(c['data'],root_indices(c['data'],'val'),c['fanouts'],cfg['batch_size'],shuffle=False,num_workers=0)
            v=evaluate_probe(c['restored'].adapter,probe,loader,c['restored'].feature_schema,c['meta']['occupation_unknown_ids'],device)
            v['relative_macro_f1_drop']=drop(v);v['passes_5pct']=drop(v)<=.05;v['passes_10pct']=drop(v)<=.10
            values[str(c['year'])]=v
        return values
    try:
        for epoch in range(start,len(probe.gates)*cfg['epochs_per_layer']):
            layer=len(probe.gates)-1-epoch//cfg['epochs_per_layer']
            for enabled in range(len(probe.gates)-1,layer-1,-1):probe.enable_layer(enabled)
            probe.train();set_seed(cfg['graphmask_seed']+epoch+1)
            isolated=cfg.get('rng_protocol')=='per_window_batch_v1'
            loaders=[]
            for i,c in enumerate(contexts):
                if isolated:set_seed(batch_seed(cfg['graphmask_seed'],epoch,0,i))
                loaders.append(iter(make_loader(c['data'],c['roots'],c['fanouts'],cfg['batch_size'],shuffle=True,num_workers=0)))
            losses=[dict(kl=0.,penalty=0.,updates=0) for c in contexts]
            for step in range(steps):
                order=window_order(len(contexts),cfg.get('window_order_seed'),epoch,step)
                for i in order:
                    c=contexts[i]
                    if isolated:set_seed(batch_seed(cfg['graphmask_seed'],epoch,step,i))
                    batch=next(loaders[i]).to(device);n=int(batch.batch_size)
                    f=batch_features(batch,c['restored'].feature_schema,c['meta']['occupation_unknown_ids'])
                    with torch.no_grad():original,traces=c['restored'].adapter.trace(f,batch.edge_index,batch.edge_type)
                    gates,_,expected,count=probe(traces)
                    if not count:continue
                    masked=c['restored'].adapter.masked_forward(f,batch.edge_index,batch.edge_type,gates,probe.baselines)
                    kl=graphmask_kl(original[:n],masked[:n]).mean();penalty=expected/count
                    optimizer.zero_grad();dual.zero_grad()
                    loss=penalty+torch.nn.functional.softplus(alpha[i])*torch.relu(kl-cfg['beta'])
                    loss.backward();optimizer.step()
                    if alpha.grad is not None:alpha.grad.mul_(-1.)
                    dual.step()
                    with torch.no_grad():alpha.clamp_(-2.,30.)
                    losses[i]['kl']+=float(kl.detach());losses[i]['penalty']+=float(penalty.detach());losses[i]['updates']+=1
                if step%50==0:print(f'[train] epoch {epoch+1} step {step+1}/{steps}',flush=True)
            validation=validate();worst=max(drop(v) for v in validation.values())
            retention=sum(v['hard_retention_rate'] for v in validation.values())/len(contexts)
            tier=0 if worst<=cfg['strict_relative_f1_drop'] else 1
            rank=(tier,retention)
            record=dict(epoch=epoch+1,window_order_seed=cfg.get('window_order_seed'),first_round_order=[contexts[i]['year'] for i in window_order(len(contexts),cfg.get('window_order_seed'),epoch,0)],layer=layer,enabled_layers=probe.enabled_layers.tolist(),worst_relative_f1_drop=worst,mean_retention=retention,validation=validation,train=losses,multipliers=torch.nn.functional.softplus(alpha).detach().cpu().tolist())
            history.append(record);write_json(out/'training_history.json',dict(history=history));print(json.dumps(record),flush=True)
            if worst<=cfg['maximum_relative_f1_drop'] and (best_rank is None or rank<best_rank):
                best_rank=rank;best=dict(state=copy.deepcopy(probe.state_dict()),record=record)
            state=dict(config=cfg,probe=probe.state_dict(),optimizer=optimizer.state_dict(),dual=dual.state_dict(),alpha=alpha.detach(),epoch=epoch+1,history=history,best=best,best_rank=best_rank)
            torch.save(state,out/'last_training_state.pt')
        if best is None:raise RuntimeError('No common checkpoint passes all-window 10% validation bound; inspect history')
        probe.load_state_dict(best['state']);probe.enable_recorded_layers();probe.eval()
        selected_validation=validate()
        assert max(drop(v) for v in selected_validation.values())<=cfg['maximum_relative_f1_drop']
        torch.save(dict(state_dict=probe.state_dict(),probe_config=probe.config(),selected_checkpoint=best['record'],config=cfg),out/'shared_graphmask.pt')
        write_json(out/'validation.json',selected_validation)
        for c in contexts:
            d=out/f"center_{c['year']}";d.mkdir(exist_ok=True)
            # Identical shared state; per-source provenance envelopes let the native
            # report validate each original graph/checkpoint without bypasses.
            payload=probe_payload(probe,c['data_path'],c['model_path'],c['ckpt'],c['fanouts'],cfg['graphmask_seed'],selected_validation[str(c['year'])],cfg)
            payload['selected_checkpoint']={k:v for k,v in best['record'].items() if k!='validation'}
            torch.save(payload,d/'graphmask_probe.pt')
            cmd=[sys.executable,str(ROOT/'run.py'),'graphmask-report','--data',str(c['data_path']),'--checkpoint',str(c['model_path']),'--probe',str(d/'graphmask_probe.pt'),'--output-dir',str(d/'test_report'),'--seed',str(cfg['report_seed']),'--device',args.device]
            write_json(d/'report_command.json',dict(command=cmd))
            with (d/'report.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
            print(f"[report] {c['year']} complete",flush=True)
        manifest.update(status='complete',selected_epoch=best['record']['epoch'],selected_tier=best_rank[0],worst_validation_drop=max(drop(v) for v in selected_validation.values()))
        write_json(out/'experiment_manifest.json',manifest);(out/'exit_code').write_text('0')
    except BaseException:
        manifest.update(status='failed',error=traceback.format_exc());write_json(out/'experiment_manifest.json',manifest);(out/'exit_code').write_text('1');raise

if __name__=='__main__':main()
