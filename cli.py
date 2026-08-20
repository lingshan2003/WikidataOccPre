"""Single command-line entry point for all supported workflows."""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="run.py", description="Occupation GNN experiment runner")
    parser.add_argument(
        "command", choices=[
            "prepare", "collapse-ties", "collapse-relations", "occupation-embed", "train", "explain", "attention-report", "attention-edge-report",
            "attention-node-report", "attention-rollout-report", "message-contribution-report",
            "gradient-attribution-report", "relation-pair-ablation-report", "relation-pair-sweep-report",
            "attention-bootstrap", "graphmask-train", "graphmask-report", "graphmask-occupation-pair-report", "diagnose", "link-prepare", "link-train",
        ], help="Workflow to run"
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to that workflow")
    parsed = parser.parse_args()
    if parsed.command == "prepare":
        from data.prepare import main as command_main
    elif parsed.command == "collapse-ties":
        from data.collapse_ties import main as command_main
    elif parsed.command == "collapse-relations":
        from data.collapse_relations import main as command_main
    elif parsed.command == "occupation-embed":
        from data.occupation_semantics import main as command_main
    elif parsed.command == "train":
        from training.train import main as command_main
    elif parsed.command == "diagnose":
        from training.diagnose import main as command_main
    elif parsed.command in {"attention-report", "attention-edge-report"}:
        from training.attention_edge_report import main as command_main
    elif parsed.command == "attention-node-report":
        from training.attention_node_report import main as command_main
    elif parsed.command == "attention-rollout-report":
        from training.attention_rollout import main as command_main
    elif parsed.command == "message-contribution-report":
        from training.message_contribution import main as command_main
    elif parsed.command == "gradient-attribution-report":
        from training.gradient_attribution import main as command_main
    elif parsed.command == "relation-pair-ablation-report":
        from training.relation_pair_ablation import main as command_main
    elif parsed.command == "relation-pair-sweep-report":
        from training.relation_pair_sweep import main as command_main
    elif parsed.command == "attention-bootstrap":
        from training.attention_bootstrap import main as command_main
    elif parsed.command == "graphmask-train":
        from training.graphmask_train import main as command_main
    elif parsed.command == "graphmask-report":
        from training.graphmask_report import main as command_main
    elif parsed.command == "graphmask-occupation-pair-report":
        from training.graphmask_occupation_pair_report import main as command_main
    elif parsed.command == "link-prepare":
        from link_prediction.prepare import main as command_main
    elif parsed.command == "link-train":
        from link_prediction.train import main as command_main
    else:
        from training.explain import main as command_main
    sys.argv = [f"run.py {parsed.command}", *parsed.args]
    command_main()
