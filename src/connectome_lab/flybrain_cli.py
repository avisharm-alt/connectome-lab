from __future__ import annotations

import argparse
import json
from pathlib import Path

from .flybrain_fm import (
    FlyBrainEmbedding,
    build_sparse_connectome,
    fit_connectome_embeddings,
    nearest_neurons,
    probe_annotations,
)


def _graph_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/malecns-v1.0/raw"),
    )
    parser.add_argument("--min-synapses", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-edges", type=int)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="flybrain-fm",
        description="Learn and inspect ML representations of the MaleCNS connectome.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="fit fast sparse SVD connectome embeddings")
    _graph_args(build)
    build.add_argument("--dimensions", type=int, default=64)
    build.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/malecns_embeddings.npz"),
    )

    neural = sub.add_parser(
        "neural-build",
        help="train self-supervised neural embeddings from real vs negative edges",
    )
    _graph_args(neural)
    neural.add_argument("--dimensions", type=int, default=64)
    neural.add_argument("--epochs", type=int, default=5)
    neural.add_argument("--batch-size", type=int, default=65_536)
    neural.add_argument("--negatives", type=int, default=3)
    neural.add_argument("--training-edges", type=int, default=2_000_000)
    neural.add_argument("--learning-rate", type=float, default=1e-2)
    neural.add_argument("--device", default="auto")
    neural.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/malecns_neural_embeddings.npz"),
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

    if args.command in {"build", "neural-build"}:
        graph, body_ids = build_sparse_connectome(
            args.data_dir,
            min_synapses=args.min_synapses,
            max_edges=args.max_edges,
        )
        print(
            f"graph: {graph.shape[0]:,} neurons, "
            f"{graph.nnz:,} retained directed edges"
        )

        if args.command == "build":
            rep = fit_connectome_embeddings(
                graph,
                body_ids,
                dimensions=args.dimensions,
                seed=args.seed,
                min_synapses=args.min_synapses,
            )
        else:
            from .neural_embed import train_neural_edge_embeddings

            rep = train_neural_edge_embeddings(
                graph,
                body_ids,
                dimensions=args.dimensions,
                epochs=args.epochs,
                batch_size=args.batch_size,
                negatives=args.negatives,
                max_positive_edges=args.training_edges,
                learning_rate=args.learning_rate,
                seed=args.seed,
                min_synapses=args.min_synapses,
                device=args.device,
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
