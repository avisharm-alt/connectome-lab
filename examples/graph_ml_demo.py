from connectome_lab import (
    directed_spectral_embedding,
    link_prediction_benchmark,
    modular_connectome,
    node_classification_probe,
)

graph = modular_connectome(n_nodes=200, n_modules=4, seed=7)
labels = {node: graph.nodes[node]["module"] for node in graph.nodes}

embeddings = directed_spectral_embedding(graph, dimensions=16)
print(embeddings.head())

print(node_classification_probe(graph, labels, dimensions=16, seed=7))
print(link_prediction_benchmark(graph, dimensions=16, seed=7))
