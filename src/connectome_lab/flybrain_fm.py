from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


ANNOTATION_FILE = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
EDGE_FILE = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"


@dataclass
class FlyBrainEmbedding:
    body_ids: np.ndarray
    embeddings: np.ndarray
    min_synapses: int

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            body_ids=self.body_ids,
            embeddings=self.embeddings.astype(np.float32),
            min_synapses=np.asarray([self.min_synapses], dtype=np.int64),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FlyBrainEmbedding":
        obj = np.load(path)
        return cls(
            body_ids=obj["body_ids"],
            embeddings=obj["embeddings"],
            min_synapses=int(obj["min_synapses"][0]),
        )


def _body_column(frame: pd.DataFrame) -> str:
    for candidate in ("body", "bodyId", "body_id", "bodyid"):
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        f"Could not find neuron body-id column. Columns: {list(frame.columns)}"
    )


def traced_body_ids(annotation_path: str | Path) -> np.ndarray:
    """Return sorted neuron IDs, preferring proofread/traced neurons when available."""
    annotations = pd.read_feather(annotation_path)
    body_col = _body_column(annotations)

    if "status" in annotations.columns:
        traced = annotations[
            annotations["status"].astype(str).str.lower().eq("traced")
        ]
        if len(traced):
            annotations = traced

    body_ids = (
        pd.to_numeric(annotations[body_col], errors="coerce")
        .dropna()
        .astype(np.int64)
        .unique()
    )
    return np.sort(body_ids)


def _map_ids(sorted_body_ids: np.ndarray, values: np.ndarray):
    pos = np.searchsorted(sorted_body_ids, values)
    valid = pos < len(sorted_body_ids)
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx):
        valid[valid_idx] &= sorted_body_ids[pos[valid_idx]] == values[valid_idx]
    return pos, valid


def build_sparse_connectome(
    data_dir: str | Path,
    min_synapses: int = 5,
    batch_size: int = 1_000_000,
    max_edges: int | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Build a sparse weighted MaleCNS adjacency matrix without pandas edge expansion.

    The 1.1 GB Feather edge table is memory-mapped with Arrow and processed
    in record batches. Only endpoints found in the traced annotation set are kept.
    """
    data_dir = Path(data_dir)
    annotation_path = data_dir / ANNOTATION_FILE
    edge_path = data_dir / EDGE_FILE

    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing {annotation_path}")
    if not edge_path.exists():
        raise FileNotFoundError(
            f"Missing {edge_path}. Run: flybrain-data download --graph"
        )
    if min_synapses < 1:
        raise ValueError("min_synapses must be >= 1")

    body_ids = traced_body_ids(annotation_path)
    table = feather.read_table(
        edge_path,
        columns=["body_pre", "body_post", "weight"],
        memory_map=True,
    )

    rows, cols, vals = [], [], []
    kept = 0

    for batch in table.to_batches(max_chunksize=batch_size):
        pre = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        post = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        weight = batch.column(2).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)

        mask = weight >= min_synapses
        if not np.any(mask):
            continue
        pre, post, weight = pre[mask], post[mask], weight[mask]

        pre_pos, pre_valid = _map_ids(body_ids, pre)
        post_pos, post_valid = _map_ids(body_ids, post)
        valid = pre_valid & post_valid

        if not np.any(valid):
            continue

        r = pre_pos[valid]
        c = post_pos[valid]
        w = np.log1p(weight[valid]).astype(np.float32)

        if max_edges is not None and kept + len(r) > max_edges:
            take = max_edges - kept
            if take <= 0:
                break
            r, c, w = r[:take], c[:take], w[:take]

        rows.append(r.astype(np.int32, copy=False))
        cols.append(c.astype(np.int32, copy=False))
        vals.append(w)
        kept += len(r)

        if max_edges is not None and kept >= max_edges:
            break

    if not rows:
        raise RuntimeError("No edges survived the requested filters")

    row = np.concatenate(rows)
    col = np.concatenate(cols)
    val = np.concatenate(vals)

    graph = sparse.coo_matrix(
        (val, (row, col)),
        shape=(len(body_ids), len(body_ids)),
        dtype=np.float32,
    ).tocsr()
    graph.sum_duplicates()
    graph.eliminate_zeros()
    return graph, body_ids


def fit_connectome_embeddings(
    graph: sparse.csr_matrix,
    body_ids: np.ndarray,
    dimensions: int = 64,
    seed: int = 0,
    min_synapses: int = 5,
) -> FlyBrainEmbedding:
    """Learn directed low-rank neuron embeddings from incoming/outgoing connectivity."""
    if dimensions < 4:
        raise ValueError("dimensions must be >= 4")
    half = max(2, dimensions // 2)

    outgoing = normalize(graph, norm="l2", axis=1, copy=True)
    incoming = normalize(graph.T.tocsr(), norm="l2", axis=1, copy=True)

    out_model = TruncatedSVD(n_components=half, n_iter=7, random_state=seed)
    in_model = TruncatedSVD(n_components=half, n_iter=7, random_state=seed + 1)

    z_out = out_model.fit_transform(outgoing)
    z_in = in_model.fit_transform(incoming)
    z = np.concatenate([z_out, z_in], axis=1).astype(np.float32)
    z = normalize(z, norm="l2", axis=1).astype(np.float32)

    return FlyBrainEmbedding(
        body_ids=np.asarray(body_ids, dtype=np.int64),
        embeddings=z,
        min_synapses=min_synapses,
    )


def nearest_neurons(
    representation: FlyBrainEmbedding,
    body_id: int,
    k: int = 10,
) -> pd.DataFrame:
    """Return cosine-nearest neurons in the learned connectome embedding."""
    matches = np.flatnonzero(representation.body_ids == int(body_id))
    if not len(matches):
        raise KeyError(f"Body ID {body_id} is not in this embedding")
    idx = int(matches[0])

    scores = representation.embeddings @ representation.embeddings[idx]
    order = np.argsort(-scores)
    order = order[order != idx][:k]

    return pd.DataFrame(
        {
            "body_id": representation.body_ids[order],
            "cosine_similarity": scores[order],
        }
    )


def probe_annotations(
    representation: FlyBrainEmbedding,
    annotation_path: str | Path,
    label_column: str = "class",
    min_class_count: int = 20,
    seed: int = 0,
) -> dict[str, float | int | str]:
    """Probe whether unsupervised wiring embeddings recover a categorical label."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    annotations = pd.read_feather(annotation_path)
    body_col = _body_column(annotations)
    if label_column not in annotations.columns:
        raise ValueError(
            f"Unknown label column {label_column!r}. "
            f"Available columns: {list(annotations.columns)}"
        )

    index = pd.DataFrame(
        {
            body_col: representation.body_ids,
            "_embedding_index": np.arange(len(representation.body_ids)),
        }
    )
    merged = index.merge(
        annotations[[body_col, label_column]],
        on=body_col,
        how="left",
    ).dropna(subset=[label_column])

    counts = merged[label_column].astype(str).value_counts()
    keep = counts[counts >= min_class_count].index
    merged = merged[merged[label_column].astype(str).isin(keep)]

    if merged[label_column].nunique() < 2:
        raise ValueError("Not enough sufficiently frequent label classes to probe")

    idx = merged["_embedding_index"].to_numpy(dtype=int)
    X = representation.embeddings[idx]
    y = merged[label_column].astype(str).to_numpy()

    train, test = train_test_split(
        np.arange(len(y)),
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(X[train], y[train])
    pred = model.predict(X[test])

    return {
        "label": label_column,
        "n_examples": int(len(y)),
        "n_classes": int(len(np.unique(y))),
        "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
        "macro_f1": float(f1_score(y[test], pred, average="macro")),
    }
