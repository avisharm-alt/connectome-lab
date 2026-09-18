from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def directed_spectral_embedding(
    graph: nx.DiGraph,
    dimensions: int = 16,
    weight: str = "weight",
) -> pd.DataFrame:
    """Compute source/target spectral embeddings for a directed weighted graph.

    The weighted adjacency matrix A is factorized with SVD:
        A ~= U S V^T

    Node representations concatenate U sqrt(S) and V sqrt(S), preserving
    outgoing and incoming structural roles. This is a deterministic
    representation-learning baseline for directed connectomes.
    """
    nodes = list(graph.nodes)
    if len(nodes) < 2:
        raise ValueError("graph must contain at least two nodes")
    if dimensions < 1:
        raise ValueError("dimensions must be >= 1")

    A = nx.to_numpy_array(graph, nodelist=nodes, weight=weight, dtype=float)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    k = min(dimensions, len(s))
    scale = np.sqrt(np.maximum(s[:k], 0.0))
    source = U[:, :k] * scale
    target = Vt[:k, :].T * scale
    Z = np.concatenate([source, target], axis=1)

    columns = [f"src_{i}" for i in range(k)] + [f"dst_{i}" for i in range(k)]
    out = pd.DataFrame(Z, columns=columns)
    out.insert(0, "node", nodes)
    return out
