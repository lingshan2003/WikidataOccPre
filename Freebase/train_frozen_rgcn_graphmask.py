#!/usr/bin/env python3
"""Native GraphMask algorithm with an explicitly fixed validation sampler.

This process-local adapter leaves the shared training module on disk unchanged,
so an already running full-pipeline experiment retains its original code.
"""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--validation-sampling-seed",type=int,default=100042)
    extra,rest=parser.parse_known_args()
    sys.argv=[sys.argv[0],*rest]
    from training import graphmask_train as native
    old_parse=native.parse_args
    old_validation=native._validation_loader
    old_load=native.load_run_context
    def parse():
        args=old_parse()
        args.validation_sampling_seed=extra.validation_sampling_seed
        args.frozen_model_seed=42
        return args
    def validation(data,args,fanouts):
        import copy
        fixed=copy.copy(args)
        fixed.seed=extra.validation_sampling_seed-100000
        return old_validation(data,fixed,fanouts)
    def load(data_path,checkpoint_path,device):
        import json
        metrics=json.loads((Path(checkpoint_path).parent/"metrics.json").read_text())
        if metrics["run_config"]["seed"]!=42 or set(metrics["model_feature_schema"])!={"occupation_level1"}:
            raise ValueError("Expected original L1-only seed42 RGCN")
        result=old_load(data_path,checkpoint_path,device)
        adapter=result[-1].adapter
        if any(p.requires_grad for model in (adapter.model,adapter.reference_model) for p in model.parameters()):
            raise ValueError("RGCN parameters were not frozen")
        return result
    native.parse_args=parse
    native._validation_loader=validation
    native.load_run_context=load
    native.main()


if __name__=="__main__": main()
