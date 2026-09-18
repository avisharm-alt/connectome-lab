from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load_edge_list
from .metrics import graph_metrics, node_metrics
from .perturb import lesion_benchmark
from .synthetic import modular_connectome


def _write_outputs(graph, out: Path, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    baseline = graph_metrics(graph)
    nodes = node_metrics(graph)
    perturbations = lesion_benchmark(graph, n_random=100, seed=seed)

    (out / "baseline_metrics.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8"
    )
    nodes.to_csv(out / "node_metrics.csv", index=False)
    perturbations.to_csv(out / "perturbations.csv", index=False)

    summary = {
        "seed": seed,
        "targeted_mean_delta": float(
            perturbations.loc[perturbations.condition == "targeted", "delta"].mean()
        ),
        "random_mean_delta": float(
            perturbations.loc[perturbations.condition == "random", "delta"].mean()
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="connectome-lab")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run on a synthetic modular connectome")
    demo.add_argument("--nodes", type=int, default=250)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--out", type=Path, default=Path("results/demo"))

    analyze = sub.add_parser("analyze", help="analyze a CSV edge list")
    analyze.add_argument("path", type=Path)
    analyze.add_argument("--source", default="source")
    analyze.add_argument("--target", default="target")
    analyze.add_argument("--weight", default="weight")
    analyze.add_argument("--seed", type=int, default=7)
    analyze.add_argument("--out", type=Path, default=Path("results/run"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "demo":
        graph = modular_connectome(n_nodes=args.nodes, seed=args.seed)
    else:
        graph = load_edge_list(
            args.path, source=args.source, target=args.target, weight=args.weight
        )
    _write_outputs(graph, args.out, args.seed)


if __name__ == "__main__":
    main()
