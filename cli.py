"""Single command-line entry point for all supported workflows."""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="run.py", description="Occupation GNN experiment runner")
    parser.add_argument("command", choices=["prepare", "train", "explain"], help="Workflow to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to that workflow")
    parsed = parser.parse_args()
    if parsed.command == "prepare":
        from data.prepare import main as command_main
    elif parsed.command == "train":
        from training.train import main as command_main
    else:
        from training.explain import main as command_main
    sys.argv = [f"run.py {parsed.command}", *parsed.args]
    command_main()
