# MaleCNS local data directory

Raw MaleCNS files are intentionally not committed to Git.

Run:

```bash
flybrain-data download
flybrain-data download --graph
```

The downloader fetches the canonical MaleCNS v1.0 assets directly from Janelia's public Google Cloud Storage bucket.

Expected local paths:

```text
raw/body-annotations-male-cns-v1.0-minconf-0.5.feather
raw/body-neurotransmitters-male-cns-v1.0.feather
raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather
```
