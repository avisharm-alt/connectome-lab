from connectome_lab.ml import link_prediction_benchmark, node_classification_probe
from connectome_lab.synthetic import modular_connectome


def test_node_classification_probe_runs():
    g = modular_connectome(n_nodes=80, n_modules=4, seed=12)
    labels = {n: g.nodes[n]["module"] for n in g.nodes}
    result = node_classification_probe(g, labels, dimensions=6, seed=12)
    assert 0.0 <= result["balanced_accuracy"] <= 1.0


def test_link_prediction_benchmark_runs():
    g = modular_connectome(
        n_nodes=80,
        n_modules=4,
        p_within=0.14,
        p_between=0.03,
        seed=13,
    )
    result = link_prediction_benchmark(g, dimensions=6, seed=13)
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert 0.0 <= result["average_precision"] <= 1.0
