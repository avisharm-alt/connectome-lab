from __future__ import annotations

import argparse
import json
from pathlib import Path

from .flybrain_fm import (
    ANNOTATION_FILE,
    FlyBrainEmbedding,
    build_sparse_connectome,
    fit_connectome_embeddings,
    nearest_neurons,
    probe_annotations,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="flybrain-fm",
        description="Learn and inspect ML representations of the MaleCNS connectome.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build sparse graph and fit neuron embeddings")
    build.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/malecns-v1.0/raw"),
    )
    build.add_argument("--min-synapses", type=int, default=5)
    build.add_argument("--dimensions", type=int, default=64)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--max-edges", type=int)
    build.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/malecns_embeddings.npz"),
    )

    probe = sub.add_parser("probe", help="probe biological labels from embeddings")
    probe.add_argument("--embeddings", type=Path, required=True)
    probe.add_argument("--annotations", type=Path, required=True)
    probe.add_argument("--label", default="class")
    probe.add_argument("--min-class-count", type=int, default=20)
    probe.add_argument("--seed", type=int, default=0)

    nn = sub.add_parser("neighbors", help="find embedding-nearest neurons")
    nn.add_argument("--embeddings", type=Path, required=True)
    nn.add_argument("--body-id", type=int, required=True)
    nn.add_argument("--k", type=int, default=10)
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "build":
        graph, body_ids = build_sparse_connectome(
            args.data_dir,
            min_synapses=args.min_synapses,
            max_edges=args.max_edges,
        )
        print(
            f"graph: {graph.shape[0]:,} neurons, "
            f"{graph.nnz:,} retained directed edges"
        )
        rep = fit_connectome_embeddings(
            graph,
            body_ids,
            dimensions=args.dimensions,
            seed=args.seed,
            min_synapses=args.min_synapses,
        )
        rep.save(args.output)
        print(f"saved {rep.embeddings.shape} embeddings -> {args.output}")

    elif args.command == "probe":
        rep = FlyBrainEmbedding.load(args.embeddings)
        result = probe_annotations(
            rep,
            args.annotations,
            label_column=args.label,
            min_class_count=args.min_class_count,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2))

    elif args.command == "neighbors":
        rep = FlyBrainEmbedding.load(args.embeddings)
        print(nearest_neurons(rep, args.body_id, k=args.k).to_string(index=False))


if __name__ == "__main__":
    main()
