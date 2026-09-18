"""Graph ML and representation learning for neural connectomes."""

from .embeddings import directed_spectral_embedding
from .flybrain_fm import (
    build_sparse_connectome,
    fit_connectome_embeddings,
    nearest_neurons,
)
from .io import load_edge_list
from .metrics import graph_metrics, node_metrics
from .ml import link_prediction_benchmark, node_classification_probe
from .perturb import lesion_benchmark
from .synthetic import modular_connectome

__all__ = [
    "directed_spectral_embedding",
    "build_sparse_connectome",
    "fit_connectome_embeddings",
    "nearest_neurons",
    "load_edge_list",
    "graph_metrics",
    "node_metrics",
    "link_prediction_benchmark",
    "node_classification_probe",
    "lesion_benchmark",
    "modular_connectome",
]
__version__ = "0.3.0"
