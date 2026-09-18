from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd


def graph_metrics(graph: nx.DiGraph) -> dict[str, float]:
    """Return stable summary metrics for a directed connectome."""
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    if n == 0:
        return {
            "n_nodes": 0.0,
            "n_edges": 0.0,
            "density": 0.0,
            "reciprocity": 0.0,
            "mean_in_degree": 0.0,
            "mean_out_degree": 0.0,
            "mean_total_strength": 0.0,
            "largest_weak_component_fraction": 0.0,
            "undirected_clustering": 0.0,
        }

    in_degrees = np.array([d for _, d in graph.in_degree()], dtype=float)
    out_degrees = np.array([d for _, d in graph.out_degree()], dtype=float)
    strengths = np.array(
        [
            graph.in_degree(node, weight="weight")
            + graph.out_degree(node, weight="weight")
            for node in graph.nodes
        ],
        dtype=float,
    )

    weak_components = list(nx.weakly_connected_components(graph))
    largest = max((len(c) for c in weak_components), default=0)
    reciprocity = nx.reciprocity(graph)
    if reciprocity is None or math.isnan(reciprocity):
        reciprocity = 0.0

    undirected = graph.to_undirected()
    clustering = nx.average_clustering(undirected, weight=None) if n > 1 else 0.0

    return {
        "n_nodes": float(n),
        "n_edges": float(m),
        "density": float(nx.density(graph)),
        "reciprocity": float(reciprocity),
        "mean_in_degree": float(in_degrees.mean()),
        "mean_out_degree": float(out_degrees.mean()),
        "mean_total_strength": float(strengths.mean()),
        "largest_weak_component_fraction": float(largest / n),
        "undirected_clustering": float(clustering),
    }


def node_metrics(graph: nx.DiGraph) -> pd.DataFrame:
    """Return one row per node with degree/strength and centrality features."""
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(
            columns=[
                "node",
                "in_degree",
                "out_degree",
                "in_strength",
                "out_strength",
                "pagerank",
                "betweenness",
            ]
        )

    pagerank = nx.pagerank(graph, weight="weight")
    betweenness = nx.betweenness_centrality(graph, weight=None, normalized=True)

    rows = []
    for node in graph.nodes:
        rows.append(
            {
                "node": node,
                "in_degree": graph.in_degree(node),
                "out_degree": graph.out_degree(node),
                "in_strength": graph.in_degree(node, weight="weight"),
                "out_strength": graph.out_degree(node, weight="weight"),
                "pagerank": pagerank[node],
                "betweenness": betweenness[node],
            }
        )
    return pd.DataFrame(rows).sort_values("pagerank", ascending=False).reset_index(drop=True)
