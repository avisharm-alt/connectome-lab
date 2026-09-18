# Methods

## Graph representation

The package treats a connectome as a weighted directed graph (G=(V,E)). Nodes represent biological units such as neurons or regions; directed edges encode connectivity and may carry a scalar weight.

## Baseline descriptors

The intact graph is summarized using node/edge count, density, reciprocity, mean in/out degree, mean total strength, largest weakly connected component fraction, and undirected clustering. Per-node outputs include degree, weighted strength, PageRank, and betweenness centrality.

## Perturbation benchmark

The default lesion benchmark compares:

1. **Targeted lesions:** removal of the highest-total-degree nodes.
2. **Random lesions:** uniformly sampled single-node removals.

For each perturbation, the same graph metric is recomputed and expressed as a delta from the intact graph.

## Degree-matched controls

When a biological target list is supplied, control neurons can be sampled from a degree-matched pool within a configurable tolerance. This reduces the risk that a claimed target effect is simply a hub effect.

## Reproducibility

Random procedures use explicit seeds. The synthetic graph generator creates modular weighted directed graphs and is intended only for software validation, not biological inference.

## Extension points

Useful next analyses include cell-type-aware nulls, edge lesions, motif preservation, spatial constraints, rich-club perturbations, and adapters for specific public connectome datasets.
