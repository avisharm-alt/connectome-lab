# MaleCNS v1.0 data access

This repository uses the **MaleCNS v1.0** release of the complete male *Drosophila melanogaster* central nervous system connectome.

## Canonical resources

- Project: https://male-cns.janelia.org/
- Downloads: https://male-cns.janelia.org/download/
- neuPrint dataset: `male-cns:v1.0`
- Neuroglancer scene: https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.jso

## Bulk files used here

Base URL:

```text
https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/
```

Files:

```text
body-annotations-male-cns-v1.0-minconf-0.5.feather
body-neurotransmitters-male-cns-v1.0.feather
connectome-weights-male-cns-v1.0-minconf-0.5.feather
```

Janelia documents the connectivity file as the full segment-to-segment connection-strength graph. The release also provides synapse coordinates, synaptic partner pairs, per-T-bar neurotransmitter predictions, neuron skeletons, segmentation, EM volumes, and a downloadable neuPrint/Neo4j database.

## Why the raw brain is not committed to GitHub

The core graph alone is about 1.1 GB, while synapse-level resources are several gigabytes each and the EM volume is far larger. Git is the wrong transport for those immutable scientific assets.

The reproducible pattern used here is:

1. pin the dataset/version and canonical URLs in code;
2. download raw assets locally;
3. gitignore raw data;
4. commit the transformations, ML code, and derived small artifacts.

## Remote access with neuPrint

Install:

```bash
pip install -e ".[neuprint]"
```

After obtaining a token from neuPrint:

```bash
export NEUPRINT_TOKEN="..."
```

Python:

```python
from connectome_lab.malecns import neuprint_client
from neuprint import fetch_adjacencies, fetch_neurons

client = neuprint_client()

neurons, roi_distribution = fetch_neurons("DNge104")
outgoing, neuron_info = fetch_adjacencies("DNge104")
incoming, neuron_info2 = fetch_adjacencies(None, "DNge104")
```

## Direct EM access

Janelia documents the contrast-adjusted EM volume as a Neuroglancer precomputed dataset:

```python
from cloudvolume import CloudVolume

vol = CloudVolume(
    "precomputed://gs://flyem-male-cns/em/em-clahe-jpeg",
    use_https=True,
)
cutout = vol[40000:40500, 40000:40500, 20000, 0]
```

Install `cloud-volume` separately if you want voxel-level EM access.

## License / attribution

The MaleCNS dataset is released under CC-BY. Preserve the dataset citation and attribution in any derived work.
