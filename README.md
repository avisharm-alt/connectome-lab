# connectome-lab

**Graph machine learning for neural connectomes.**

`connectome-lab` is a compact AI/ML research toolkit for learning from brain connectivity graphs. It combines graph representation learning, link prediction, node classification, and perturbation-based robustness analysis in one reproducible package.

The project is designed around a core question: **can graph-derived representations learn biologically meaningful structure that generalizes beyond trivial degree and hub statistics?**

## ML tasks

- **Graph representation learning** — directed spectral node embeddings from weighted connectomes
- **Link prediction** — predict held-out synaptic/structural edges with learned graph representations
- **Node classification** — probe whether embeddings recover known node or cell-type labels
- **Robustness analysis** — measure how model-relevant graph structure changes under targeted and random lesions
- **Leakage-aware evaluation** — remove held-out positive edges before computing embeddings for link prediction
- **Synthetic benchmarks** — modular directed graphs with known ground-truth community labels

## Why AI/ML + connectomics?

Connectomes are naturally graph-structured datasets. Modern neuroscience increasingly asks the same questions as graph ML: how to represent nodes, predict missing relationships, transfer learned features, and determine whether a model is exploiting real organization or superficial graph statistics.

This repo is intended as a research sandbox for that intersection.

## Install

```bash
git clone https://github.com/avisharm-alt/connectome-lab
cd connectome-lab
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart: graph representation learning

```python
from connectome_lab.synthetic import modular_connectome
from connectome_lab.embeddings import directed_spectral_embedding
from connectome_lab.ml import node_classification_probe, link_prediction_benchmark

G = modular_connectome(n_nodes=250, seed=7)

Z = directed_spectral_embedding(G, dimensions=16)
print(Z.head())

node_result = node_classification_probe(
    G,
    labels={n: G.nodes[n]["module"] for n in G.nodes},
    dimensions=16,
    seed=7,
)
print(node_result)

link_result = link_prediction_benchmark(
    G,
    dimensions=16,
    test_fraction=0.2,
    seed=7,
)
print(link_result)
```

## Classical graph robustness

The original graph-analysis layer remains available:

```bash
connectome-lab demo --nodes 250 --seed 7 --out results/demo
```

It writes graph metrics, node features, and targeted-vs-random lesion experiments for the same network.

## Research directions

This codebase is structured to extend toward:

- Graph neural networks (GCN / GraphSAGE / GAT)
- Self-supervised graph representation learning
- Contrastive learning on connectomes
- Cell-type prediction
- Connectome completion / missing-edge prediction
- Cross-species graph transfer
- Perturbation-aware graph encoders
- Foundation models over neural graphs

## Reproducibility

All stochastic functions accept explicit seeds. Tests cover graph generation, embeddings, ML probes, link prediction, metrics, and perturbation logic.

## License

MIT
