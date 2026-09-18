from __future__ import annotations

import random

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .embeddings import directed_spectral_embedding


def _embedding_lookup(graph: nx.DiGraph, dimensions: int) -> dict[str, np.ndarray]:
    frame = directed_spectral_embedding(graph, dimensions=dimensions)
    features = frame.drop(columns=["node"]).to_numpy(dtype=float)
    return {
        str(node): features[i]
        for i, node in enumerate(frame["node"].astype(str).tolist())
    }


def node_classification_probe(
    graph: nx.DiGraph,
    labels: dict,
    dimensions: int = 16,
    test_fraction: float = 0.25,
    seed: int = 0,
) -> dict[str, float]:
    """Evaluate how well graph embeddings linearly recover node labels."""
    lookup = _embedding_lookup(graph, dimensions)
    nodes = [str(n) for n in graph.nodes if n in labels or str(n) in labels]
    if len(nodes) < 4:
        raise ValueError("need at least four labeled nodes")

    X = np.stack([lookup[n] for n in nodes])
    y = np.asarray([labels.get(n, labels.get(str(n))) for n in nodes])

    if len(np.unique(y)) < 2:
        raise ValueError("node labels must contain at least two classes")

    train_idx, test_idx = train_test_split(
        np.arange(len(nodes)),
        test_size=test_fraction,
        random_state=seed,
        stratify=y,
    )

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])

    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(y[test_idx], pred)
        ),
        "n_train": float(len(train_idx)),
        "n_test": float(len(test_idx)),
        "embedding_dimensions": float(X.shape[1]),
    }


def _negative_edges(
    graph: nx.DiGraph,
    n_samples: int,
    seed: int,
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    nodes = [str(n) for n in graph.nodes]
    existing = {(str(u), str(v)) for u, v in graph.edges}
    negatives: set[tuple[str, str]] = set()

    attempts = 0
    max_attempts = max(1000, n_samples * 50)
    while len(negatives) < n_samples and attempts < max_attempts:
        u, v = rng.sample(nodes, 2)
        if (u, v) not in existing:
            negatives.add((u, v))
        attempts += 1

    if len(negatives) < n_samples:
        raise RuntimeError("could not sample enough negative edges")
    return list(negatives)


def _pair_features(
    edges: list[tuple[str, str]],
    lookup: dict[str, np.ndarray],
) -> np.ndarray:
    """Hadamard + absolute-difference pair representation."""
    rows = []
    for u, v in edges:
        zu, zv = lookup[str(u)], lookup[str(v)]
        rows.append(np.concatenate([zu * zv, np.abs(zu - zv)]))
    return np.asarray(rows, dtype=float)


def link_prediction_benchmark(
    graph: nx.DiGraph,
    dimensions: int = 16,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, float]:
    """Predict held-out directed edges from graph embeddings.

    Positive test edges are removed before the spectral embedding is fit.
    Negatives are sampled from non-edges in the original graph.
    """
    positives = [(str(u), str(v)) for u, v in graph.edges]
    if len(positives) < 10:
        raise ValueError("need at least ten positive edges")

    train_pos, test_pos = train_test_split(
        positives,
        test_size=test_fraction,
        random_state=seed,
    )

    train_graph = nx.DiGraph()
    train_graph.add_nodes_from((str(n), data) for n, data in graph.nodes(data=True))
    for u, v in train_pos:
        data = graph.get_edge_data(u, v) or {}
        train_graph.add_edge(u, v, **data)

    lookup = _embedding_lookup(train_graph, dimensions)

    negatives = _negative_edges(graph, len(positives), seed=seed)
    train_neg, test_neg = train_test_split(
        negatives,
        test_size=test_fraction,
        random_state=seed,
    )

    train_edges = train_pos + train_neg
    test_edges = test_pos + test_neg
    y_train = np.asarray([1] * len(train_pos) + [0] * len(train_neg))
    y_test = np.asarray([1] * len(test_pos) + [0] * len(test_neg))

    X_train = _pair_features(train_edges, lookup)
    X_test = _pair_features(test_edges, lookup)

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "average_precision": float(average_precision_score(y_test, probability)),
        "n_train_edges": float(len(train_edges)),
        "n_test_edges": float(len(test_edges)),
    }
