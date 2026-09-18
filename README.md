# connectome-lab

A reproducible perturbation and robustness toolkit for connectomics.

This project is designed around a simple question: **which graph-level conclusions about a neural circuit survive realistic perturbations of nodes, edges, and sampling?** It provides a compact research workflow for sparse directed connectomes, with group-aware controls and machine-readable outputs.

## What it does

- Loads weighted directed connectomes from edge lists
- Computes degree, strength, reciprocity, density, clustering, and component statistics
- Runs targeted and random node-lesion experiments
- Runs degree-matched perturbation controls
- Quantifies metric deltas against an intact-network baseline
- Generates a synthetic modular connectome for smoke tests and demos
- Exposes the workflow through a CLI and a small Python API

## Why this exists

Connectomic analyses can look convincing while depending heavily on graph density, hub structure, or a few high-degree neurons. This repo makes those sensitivities explicit by treating perturbation analysis as a first-class evaluation step.

## Install

```bash
git clone https://github.com/avisharm-alt/connectome-lab
cd connectome-lab
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart

Generate a synthetic connectome and benchmark perturbations:

```bash
connectome-lab demo --nodes 250 --seed 7 --out results/demo
```

Run on an edge list:

```bash
connectome-lab analyze data/edges.csv --source source --target target --weight weight --out results/run
```

Expected input:

```csv
source,target,weight
n1,n2,3
n2,n3,1
n3,n1,2
```

## Output

The CLI writes:

- `baseline_metrics.json` — intact graph summary
- `node_metrics.csv` — per-node graph features
- `perturbations.csv` — lesion/control experiment results
- `summary.json` — experiment metadata and aggregate effects

## Python API

```python
from connectome_lab.synthetic import modular_connectome
from connectome_lab.metrics import graph_metrics
from connectome_lab.perturb import lesion_benchmark

G = modular_connectome(n_nodes=250, seed=7)
print(graph_metrics(G))

results = lesion_benchmark(G, n_random=100, seed=7)
print(results.head())
```

## Research directions

The package is intentionally small enough to extend into real connectomics work: cell-type-stratified perturbations, motif preservation, null graph models, subgraph embeddings, graph neural-network probes, and dataset adapters for public connectome releases.

## Reproducibility

All stochastic functions accept explicit seeds. Tests cover graph construction, metric calculation, perturbation behavior, and CLI-level assumptions.

## License

MIT
