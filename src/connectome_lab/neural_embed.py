from __future__ import annotations

import math

import numpy as np
from scipy import sparse

from .flybrain_fm import FlyBrainEmbedding


def _choose_device(requested: str = "auto") -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            'Neural embeddings require PyTorch. Install with: pip install -e ".[deep]"'
        ) from exc

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_neural_edge_embeddings(
    graph: sparse.csr_matrix,
    body_ids: np.ndarray,
    dimensions: int = 64,
    epochs: int = 5,
    batch_size: int = 65_536,
    negatives: int = 3,
    max_positive_edges: int = 2_000_000,
    learning_rate: float = 1e-2,
    seed: int = 0,
    min_synapses: int = 5,
    device: str = "auto",
    verbose: bool = True,
) -> FlyBrainEmbedding:
    """Learn self-supervised neuron embeddings from observed vs random edges.

    The model has separate source and target embedding tables for the directed
    connectome. Positive examples are released synaptic edges; negatives are
    randomly sampled neuron pairs. Training uses logistic negative sampling,
    analogous to shallow representation-learning objectives used in word2vec
    and network embedding.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError(
            'Neural embeddings require PyTorch. Install with: pip install -e ".[deep]"'
        ) from exc

    if dimensions < 2:
        raise ValueError("dimensions must be >= 2")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if negatives < 1:
        raise ValueError("negatives must be >= 1")

    device = _choose_device(device)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    coo = graph.tocoo(copy=False)
    src = coo.row.astype(np.int64, copy=False)
    dst = coo.col.astype(np.int64, copy=False)
    weight = coo.data.astype(np.float32, copy=False)

    if len(src) == 0:
        raise ValueError("graph contains no edges")

    if len(src) > max_positive_edges:
        chosen = rng.choice(len(src), size=max_positive_edges, replace=False)
        src, dst, weight = src[chosen], dst[chosen], weight[chosen]

    # log-weighted sparse graph already stores log1p(synapse count).
    # Normalize positive-example weights to mean 1 and cap outliers.
    sample_weight = weight / max(float(weight.mean()), 1e-6)
    sample_weight = np.clip(sample_weight, 0.25, 8.0).astype(np.float32)

    n_nodes = graph.shape[0]
    out_embedding = torch.nn.Embedding(n_nodes, dimensions).to(device)
    in_embedding = torch.nn.Embedding(n_nodes, dimensions).to(device)

    scale = 1.0 / math.sqrt(dimensions)
    torch.nn.init.normal_(out_embedding.weight, mean=0.0, std=scale)
    torch.nn.init.normal_(in_embedding.weight, mean=0.0, std=scale)

    optimizer = torch.optim.AdamW(
        list(out_embedding.parameters()) + list(in_embedding.parameters()),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    n = len(src)
    for epoch in range(epochs):
        order = rng.permutation(n)
        running = 0.0
        seen = 0

        for start in range(0, n, batch_size):
            batch_idx = order[start : start + batch_size]
            s = torch.as_tensor(src[batch_idx], dtype=torch.long, device=device)
            d = torch.as_tensor(dst[batch_idx], dtype=torch.long, device=device)
            w = torch.as_tensor(
                sample_weight[batch_idx], dtype=torch.float32, device=device
            )

            pos_score = (out_embedding(s) * in_embedding(d)).sum(dim=1)
            pos_loss = F.softplus(-pos_score) * w

            neg_dst = torch.randint(
                low=0,
                high=n_nodes,
                size=(len(batch_idx), negatives),
                device=device,
            )
            src_rep = out_embedding(s).unsqueeze(1)
            neg_score = (src_rep * in_embedding(neg_dst)).sum(dim=2)
            neg_loss = F.softplus(neg_score).mean(dim=1)

            loss = (pos_loss + neg_loss).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += float(loss.detach().cpu()) * len(batch_idx)
            seen += len(batch_idx)

        if verbose:
            print(
                f"epoch {epoch + 1}/{epochs} "
                f"loss={running / max(seen, 1):.4f} device={device}"
            )

    with torch.no_grad():
        z = 0.5 * (out_embedding.weight + in_embedding.weight)
        z = F.normalize(z, p=2, dim=1)
        embeddings = z.detach().cpu().numpy().astype(np.float32)

    return FlyBrainEmbedding(
        body_ids=np.asarray(body_ids, dtype=np.int64),
        embeddings=embeddings,
        min_synapses=min_synapses,
    )
