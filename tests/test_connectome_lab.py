import networkx as nx

from connectome_lab.metrics import graph_metrics, node_metrics
from connectome_lab.perturb import lesion_benchmark
from connectome_lab.synthetic import modular_connectome


def test_graph_metrics_on_small_graph():
    g = nx.DiGraph()
    g.add_edge("a", "b", weight=2)
    g.add_edge("b", "a", weight=1)
    metrics = graph_metrics(g)
    assert metrics["n_nodes"] == 2.0
    assert metrics["n_edges"] == 2.0
    assert metrics["reciprocity"] == 1.0
    assert metrics["largest_weak_component_fraction"] == 1.0


def test_synthetic_graph_is_reproducible():
    g1 = modular_connectome(n_nodes=40, seed=3)
    g2 = modular_connectome(n_nodes=40, seed=3)
    assert sorted(g1.edges(data="weight")) == sorted(g2.edges(data="weight"))


def test_node_metrics_has_one_row_per_node():
    g = modular_connectome(n_nodes=30, seed=1)
    assert len(node_metrics(g)) == 30


def test_lesion_benchmark_shapes():
    g = modular_connectome(n_nodes=50, seed=2)
    df = lesion_benchmark(g, n_random=20, top_k=5, seed=2)
    assert len(df) == 25
    assert set(df["condition"]) == {"targeted", "random"}
