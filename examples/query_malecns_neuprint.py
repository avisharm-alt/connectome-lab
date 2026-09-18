import os

from connectome_lab.malecns import neuprint_client
from neuprint import fetch_adjacencies, fetch_neurons


if not os.environ.get("NEUPRINT_TOKEN"):
    raise SystemExit(
        "Set NEUPRINT_TOKEN to the token from your neuPrint account first."
    )

client = neuprint_client()

# Official MaleCNS docs use DNge104 as a simple example cell type.
neurons, roi_distribution = fetch_neurons("DNge104")
print("DNge104 neurons")
print(neurons)

outgoing, _ = fetch_adjacencies("DNge104")
print("\nTop outgoing connections")
print(outgoing.sort_values("weight", ascending=False).head(20))
