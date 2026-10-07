#!/usr/bin/env python3
"""Show/install PyG sampling wheels matching the active PyTorch build.

PyG's official wheel matrix uses major.minor.0 for PyTorch 2.x releases:
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
"""
import argparse
import re
import shlex
import subprocess
import sys


def wheel_url(torch_version, cuda_version, override=None):
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", torch_version)
    if not match:
        raise ValueError(f"Cannot parse torch version: {torch_version}")
    major, minor, _ = match.groups()
    version = override or f"{major}.{minor}.0"
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Wheel torch version must be numeric major.minor.patch")
    cuda = "cpu" if cuda_version is None else "cu" + cuda_version.replace(".", "")
    if not re.fullmatch(r"cpu|cu\d+", cuda):
        raise ValueError(f"Cannot parse torch CUDA build version: {cuda_version}")
    return f"https://data.pyg.org/whl/torch-{version}+{cuda}.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Install wheels; default only prints the command")
    parser.add_argument("--wheel-torch-version", help="Explicit wheel page version for a nonstandard PyTorch build")
    args = parser.parse_args()
    import torch
    url = wheel_url(torch.__version__, torch.version.cuda, args.wheel_torch_version)
    command = [sys.executable, "-m", "pip", "install", "--only-binary=:all:", "--no-index",
               "--find-links", url, "pyg_lib", "torch_scatter", "torch_sparse"]
    print(f"torch={torch.__version__}, torch CUDA build={torch.version.cuda}", flush=True)
    print(shlex.join(command), flush=True)
    if args.run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
