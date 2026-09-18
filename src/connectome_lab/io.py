from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd


def load_edge_list(
    path: str | Path,
    source: str = "source",
    target: str = "target",
    weight: str = "weight",
) -> nx.DiGraph:
    """Load a weighted directed graph from a CSV edge list."""
    df = pd.read_csv(path)
    required = {source, target}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if weight not in df.columns:
        df = df.assign(**{weight: 1.0})

    graph = nx.DiGraph()
    for row in df[[source, target, weight]].itertuples(index=False, name=None):
        u, v, w = row
        graph.add_edge(str(u), str(v), weight=float(w))
    return graph


def save_edge_list(graph: nx.DiGraph, path: str | Path) -> None:
    """Write a weighted directed graph to CSV."""
    rows = [
        {"source": u, "target": v, "weight": data.get("weight", 1.0)}
        for u, v, data in graph.edges(data=True)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
