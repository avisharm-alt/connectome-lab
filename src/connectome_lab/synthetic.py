from __future__ import annotations

import networkx as nx
import numpy as np


def modular_connectome(
    n_nodes: int = 200,
    n_modules: int = 4,
    p_within: float = 0.08,
    p_between: float = 0.01,
    seed: int = 0,
) -> nx.DiGraph:
    """Generate a weighted directed modular graph for reproducible experiments."""
    if n_nodes < 2:
        raise ValueError("n_nodes must be >= 2")
    if n_modules < 1 or n_modules > n_nodes:
        raise ValueError("n_modules must be between 1 and n_nodes")

    rng = np.random.default_rng(seed)
    labels = np.arange(n_nodes) % n_modules
    rng.shuffle(labels)

    graph = nx.DiGraph()
    graph.add_nodes_from([f"n{i}" for i in range(n_nodes)])

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            p = p_within if labels[i] == labels[j] else p_between
            if rng.random() < p:
                weight = float(max(1, rng.poisson(3)))
                graph.add_edge(f"n{i}", f"n{j}", weight=weight)

    nx.set_node_attributes(
        graph, {f"n{i}": int(labels[i]) for i in range(n_nodes)}, "module"
    )
    return graph
