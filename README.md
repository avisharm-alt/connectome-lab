# FlyBrainFM / connectome-lab

**Machine learning on the complete Google Research + Janelia MaleCNS fruit-fly connectome.**

This repository turns the 2026 MaleCNS v1.0 release into a laptop-accessible graph-ML research project. The raw connectome stays in Janelia's public Google Cloud bucket; the repo contains reproducible download/query code, a local graph builder, unsupervised connectome embeddings, downstream probes, and neuron-similarity search.

MaleCNS v1.0 contains the complete male *Drosophila melanogaster* central nervous system: brain, optic lobes, neck connective, and ventral nerve cord.

## The ML idea

Treat each neuron as a node and each directed synaptic connection as a weighted edge.

FlyBrainFM then learns a compact representation of every neuron from the wiring diagram itself:

1. build a sparse directed connectome from the released neuron-to-neuron weights;
2. learn separate outgoing and incoming low-rank graph representations with randomized SVD;
3. concatenate and normalize them into one embedding per neuron;
4. test whether those unsupervised embeddings recover biological labels such as neuron class;
5. search for structurally similar neurons directly in embedding space.

The goal is not to claim a "fly foundation model" yet. It is to build the clean substrate for progressively stronger self-supervised graph models, GNNs, and eventually masked-connectome pretraining.

## Official dataset

The code points directly to MaleCNS v1.0 files released by HHMI Janelia / Cambridge / MRC LMB / Google Research:

| File | Approx. size | Purpose |
| --- | ---: | --- |
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 13 MB | neuron annotations |
| `body-neurotransmitters-male-cns-v1.0.feather` | 42 MB | neuron-level neurotransmitter predictions |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1.1 GB | complete directed connection graph |

The much larger synapse-point, synapse-partner, EM, segmentation, mesh, and skeleton resources remain accessible from the official MaleCNS release but are not required for the first ML experiments.

Raw data is **not committed to GitHub**. It is downloaded locally into `data/malecns-v1.0/raw/`, which is gitignored.

## Install

```bash
git clone https://github.com/avisharm-alt/connectome-lab.git
cd connectome-lab

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev,neuprint]"
```

## 1. Put the fly brain on your computer

Download annotations + neurotransmitters first (~55 MB):

```bash
flybrain-data download
```

Add the complete 1.1 GB neuron-to-neuron connection graph:

```bash
flybrain-data download --graph
```

Files will appear here:

```text
connectome-lab/
└── data/
    └── malecns-v1.0/
        └── raw/
            ├── body-annotations-male-cns-v1.0-minconf-0.5.feather
            ├── body-neurotransmitters-male-cns-v1.0.feather
            └── connectome-weights-male-cns-v1.0-minconf-0.5.feather
```

Inspect what you have locally:

```bash
flybrain-data status
```

## 2. Learn embeddings from the connectome

The laptop-friendly default keeps connections supported by at least 5 synapses before fitting the sparse representation model:

```bash
flybrain-fm build \
  --data-dir data/malecns-v1.0/raw \
  --min-synapses 5 \
  --dimensions 64 \
  --output artifacts/malecns_embeddings.npz
```

This produces a normalized embedding vector for every retained neuron.

To use every released connection, set:

```bash
flybrain-fm build --min-synapses 1 --dimensions 64
```

That is materially heavier in RAM and compute because the complete weight table contains an enormous number of directed edges.

## 3. Ask whether the wiring embeddings learned biology

Probe a biological annotation column:

```bash
flybrain-fm probe \
  --embeddings artifacts/malecns_embeddings.npz \
  --annotations data/malecns-v1.0/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather \
  --label class
```

The probe reports held-out balanced accuracy and macro-F1. You can substitute another categorical annotation column after inspecting the annotation table.

## 4. Find neurons with similar wiring

```bash
flybrain-fm neighbors \
  --embeddings artifacts/malecns_embeddings.npz \
  --body-id 12781 \
  --k 12
```

This searches the learned connectome space rather than neuron names.

## Query MaleCNS without downloading the graph

Janelia also hosts the dataset in neuPrint as `male-cns:v1.0`.

1. Create/log into a neuPrint account.
2. Copy your API token.
3. Export it locally:

```bash
export NEUPRINT_TOKEN="YOUR_TOKEN_HERE"
```

Then:

```bash
python examples/query_malecns_neuprint.py
```

Never commit the token; `.env` is ignored.

## View the actual 3D brain

The official Neuroglancer scene can display the EM volume, segmentation, synapses, neuropil compartments, and neuron meshes. See `docs/MALECNS.md` for the canonical link and data-access notes.

## Existing graph-ML toolkit

The repository still includes the generic connectome tooling:

- directed spectral embeddings
- held-out-edge link prediction
- node classification
- targeted/random lesion analysis
- synthetic modular connectomes

FlyBrainFM is the real-data layer that applies those ideas to the complete MaleCNS release.

## Roadmap

The next serious ML steps are:

- masked-edge pretraining;
- GraphSAGE/GAT baselines with neighborhood sampling;
- contrastive neuron embeddings;
- multimodal fusion of connectivity + morphology + neurotransmitter identity;
- male/female transfer using aligned connectomes;
- zero-shot cell-type retrieval;
- connectome-language models where neighborhoods become structured token sequences.

## Data attribution

MaleCNS v1.0 is released under CC-BY by the FlyEM/Janelia collaboration with Cambridge, MRC LMB, and Google Research. This repository does not redistribute the large raw files; it downloads them from the canonical public release.

## License

Code in this repository: MIT.
