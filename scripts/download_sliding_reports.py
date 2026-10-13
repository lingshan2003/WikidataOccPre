#!/usr/bin/env python3
"""Download lightweight reports over SSH; execute the exporter in memory remotely.

No server files are created or edited. Authentication stays in SSH, and an
existing user-authenticated ControlMaster connection can be reused.
"""

import argparse
import json
from pathlib import Path
import shlex
import subprocess

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--control-path", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-python", default="python3")
    parser.add_argument("--config", required=True, help="Actual resolved experiment config on the server")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_name(args.output.name + ".partial").exists():
        raise FileExistsError("Choose a new output path; existing downloads are preserved")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = (REPO / "DBpedia/package_sliding_results.py").read_bytes()
    remote_file = str(Path(args.remote_root) / "DBpedia/package_sliding_results.py")
    code = f"__file__={remote_file!r}; exec(compile(__import__('sys').stdin.read(), __file__, 'exec'))"
    config = str(Path(args.remote_root) / args.config)
    command = ["ssh", "-S", args.control_path, "-o", "BatchMode=yes", "-o", "ControlMaster=no",
               "-o", "ConnectTimeout=10", args.host,
               shlex.join([args.remote_python, "-c", code, "--config", config, "--stdout"])]
    partial = args.output.with_name(args.output.name + ".partial")
    with partial.open("xb") as stream:
        result = subprocess.run(command, input=source, stdout=stream, stderr=subprocess.PIPE)
    stderr = result.stderr.decode(errors="replace")
    print(stderr.strip())
    if result.returncode:
        partial.unlink()
        raise SystemExit(result.returncode)
    partial.rename(args.output)
    args.output.with_suffix(args.output.suffix + ".download.json").write_text(json.dumps({
        "ssh_host": args.host, "remote_root": args.remote_root, "resolved_config": args.config,
        "server_files_modified": False, "bytes": args.output.stat().st_size,
        "command": command, "export_log": stderr,
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"Downloaded {args.output.resolve()} ({args.output.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
