#!/usr/bin/env python3
"""Export completed frozen-RGCN GraphMask comparisons, excluding all weights."""
import argparse
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from DBpedia.grouped_pipeline import read_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.stdout==bool(args.output): raise ValueError("Choose --stdout or --output")
    cfg=read_json(ROOT/"config/freebase_frozen_rgcn42_gm_seeds43_44_1900_1910_20261030_v1.json")
    out=ROOT/cfg["output_root"]; manifest=read_json(out/"experiment_manifest.json")
    if manifest["status"]!="complete" or len(manifest["jobs"])!=22 or any(j.get("exit_code")!=0 for j in manifest["jobs"]):
        raise ValueError("The 22 new probes are not all complete")
    if not (out/"all_relations.tsv").is_file() or len(read_json(out/"comparison_audit.json"))!=33:
        raise ValueError("Comparison summary is missing/incomplete")
    permitted={"manifest.json","validation.json","training_history.json","pipeline_probe.json","pipeline_report.json","test_metrics.json","relations_base.csv","relations_directed.csv"}
    files=[]
    for p in out.rglob("*"):
        if not p.is_file(): continue
        rel=p.relative_to(out)
        if len(rel.parts)==1 and (p.suffix in (".json",".tsv",".log",".exit_code") or p.name=="exit_code"):
            files.append((p,rel))
        elif rel.parts[0] in ("configs","code_snapshot") or p.name in permitted:
            files.append((p,rel))
    if args.stdout:
        with tarfile.open(fileobj=sys.stdout.buffer,mode="w|gz") as archive:
            for p,rel in files: archive.add(p,arcname=str(rel),recursive=False)
    else:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("xb") as stream,tarfile.open(fileobj=stream,mode="w|gz") as archive:
            for p,rel in files: archive.add(p,arcname=str(rel),recursive=False)
    print(f"Packaged {len(files)} report/config/log files; no weights",file=sys.stderr)


if __name__=="__main__": main()
