from __future__ import annotations

import random

import networkx as nx
import numpy as np
import pandas as pd

from .metrics import graph_metrics


def _metric_delta(
    baseline: dict[str, float],
    perturbed: dict[str, float],
    metric: str,
) -> float:
    return float(perturbed[metric] - baseline[metric])


def lesion_benchmark(
    graph: nx.DiGraph,
    n_random: int = 100,
    top_k: int = 10,
    seed: int = 0,
    target_metric: str = "largest_weak_component_fraction",
) -> pd.DataFrame:
    """Compare targeted hub lesions with random single-node lesions.

    Targeted nodes are ranked by total degree. Random controls are sampled
    uniformly without replacement within each replicate.
    """
    if graph.number_of_nodes() < 2:
        raise ValueError("graph must contain at least two nodes")

    baseline = graph_metrics(graph)
    nodes = list(graph.nodes)
    top_k = max(1, min(top_k, len(nodes)))
    ranked = sorted(nodes, key=lambda n: graph.degree(n), reverse=True)[:top_k]

    rows: list[dict[str, object]] = []

    for rank, node in enumerate(ranked, start=1):
        g = graph.copy()
        g.remove_node(node)
        metrics = graph_metrics(g)
        rows.append(
            {
                "condition": "targeted",
                "replicate": rank,
                "node": node,
                "removed_degree": graph.degree(node),
                "target_metric": target_metric,
                "baseline": baseline[target_metric],
                "perturbed": metrics[target_metric],
                "delta": _metric_delta(baseline, metrics, target_metric),
            }
        )

    rng = random.Random(seed)
    for replicate in range(1, n_random + 1):
        node = rng.choice(nodes)
        g = graph.copy()
        g.remove_node(node)
        metrics = graph_metrics(g)
        rows.append(
            {
                "condition": "random",
                "replicate": replicate,
                "node": node,
                "removed_degree": graph.degree(node),
                "target_metric": target_metric,
                "baseline": baseline[target_metric],
                "perturbed": metrics[target_metric],
                "delta": _metric_delta(baseline, metrics, target_metric),
            }
        )

    return pd.DataFrame(rows)


def degree_matched_controls(
    graph: nx.DiGraph,
    target_nodes: list[str],
    tolerance: int = 1,
    n_per_target: int = 25,
    seed: int = 0,
) -> pd.DataFrame:
    """Sample degree-matched control nodes for a set of target neurons."""
    rng = random.Random(seed)
    nodes = list(graph.nodes)
    rows = []

    for target in target_nodes:
        if target not in graph:
            raise KeyError(f"Unknown target node: {target}")
        degree = graph.degree(target)
        pool = [
            n
            for n in nodes
            if n != target and abs(graph.degree(n) - degree) <= tolerance
        ]
        if not pool:
            continue
        for i in range(n_per_target):
            control = rng.choice(pool)
            rows.append(
                {
                    "target": target,
                    "target_degree": degree,
                    "control": control,
                    "control_degree": graph.degree(control),
                    "replicate": i + 1,
                }
            )
    return pd.DataFrame(rows)
