from pathlib import Path

import numpy as np
import pandas as pd

from connectome_lab.flybrain_fm import (
    FlyBrainEmbedding,
    build_sparse_connectome,
    nearest_neurons,
)


def test_build_sparse_malecns_fixture(tmp_path: Path):
    annotations = pd.DataFrame(
        {
            "body": [1, 2, 3, 4],
            "status": ["Traced", "Traced", "Traced", "Glia"],
            "class": ["a", "a", "b", "g"],
        }
    )
    edges = pd.DataFrame(
        {
            "body_pre": [1, 1, 2, 4],
            "body_post": [2, 3, 3, 1],
            "weight": [10, 2, 8, 20],
        }
    )
    annotations.to_feather(
        tmp_path / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    )
    edges.to_feather(
        tmp_path / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    )

    graph, body_ids = build_sparse_connectome(tmp_path, min_synapses=5)
    assert body_ids.tolist() == [1, 2, 3]
    assert graph.shape == (3, 3)
    assert graph.nnz == 2


def test_nearest_neurons():
    rep = FlyBrainEmbedding(
        body_ids=np.array([10, 20, 30]),
        embeddings=np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        min_synapses=5,
    )
    result = nearest_neurons(rep, 10, k=1)
    assert int(result.iloc[0]["body_id"]) == 20
