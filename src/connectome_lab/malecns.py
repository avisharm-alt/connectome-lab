from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests

BASE_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/"
    "connectome-data/flat-connectome"
)

FILES = {
    "annotations": {
        "name": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "approx": "13 MB",
    },
    "neurotransmitters": {
        "name": "body-neurotransmitters-male-cns-v1.0.feather",
        "approx": "42 MB",
    },
    "graph": {
        "name": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "approx": "1.1 GB",
    },
}

DEFAULT_DATA_DIR = Path("data/malecns-v1.0/raw")


def canonical_url(key: str) -> str:
    if key not in FILES:
        raise KeyError(f"Unknown MaleCNS asset: {key}")
    return f"{BASE_URL}/{FILES[key]['name']}"


def download_file(url: str, destination: Path, chunk_mb: int = 8) -> Path:
    """Download a public MaleCNS file with basic resume support."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0

    headers = {"Range": f"bytes={existing}-"} if existing else {}
    response = requests.get(url, stream=True, headers=headers, timeout=60)
    response.raise_for_status()

    resumed = existing > 0 and response.status_code == 206
    if existing and not resumed:
        existing = 0
        partial.unlink(missing_ok=True)

    mode = "ab" if resumed else "wb"
    total_header = response.headers.get("content-length")
    remaining = int(total_header) if total_header else None
    total = existing + remaining if remaining is not None else None

    downloaded = existing
    print(f"Downloading {destination.name}")
    if total:
        print(f"  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB")

    with partial.open(mode) as handle:
        for chunk in response.iter_content(chunk_size=chunk_mb * 1024 * 1024):
            if not chunk:
                continue
            handle.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100.0 * downloaded / total
                print(
                    f"\r  {downloaded / 1e6:,.1f} / {total / 1e6:,.1f} MB "
                    f"({pct:5.1f}%)",
                    end="",
                    flush=True,
                )
    if total:
        print()

    partial.replace(destination)
    return destination


def download_dataset(data_dir: Path = DEFAULT_DATA_DIR, include_graph: bool = False) -> None:
    keys = ["annotations", "neurotransmitters"]
    if include_graph:
        keys.append("graph")

    for key in keys:
        info = FILES[key]
        destination = data_dir / info["name"]
        if destination.exists() and destination.stat().st_size > 0:
            print(f"✓ {destination} already exists")
            continue
        print(f"{key}: {info['approx']} from the official MaleCNS v1.0 release")
        download_file(canonical_url(key), destination)


def print_status(data_dir: Path = DEFAULT_DATA_DIR) -> None:
    print(f"MaleCNS local data: {data_dir.resolve()}")
    for key, info in FILES.items():
        path = data_dir / info["name"]
        if path.exists():
            print(f"✓ {key:18s} {path.stat().st_size / 1e6:9.1f} MB  {path.name}")
        else:
            print(f"· {key:18s} {'not downloaded':>12s}  {path.name}")


def neuprint_client(token: str | None = None):
    """Create a neuPrint client for MaleCNS v1.0."""
    try:
        from neuprint import Client
    except ImportError as exc:
        raise RuntimeError(
            'Install neuPrint support with: pip install -e ".[neuprint]"'
        ) from exc

    token = token or os.environ.get("NEUPRINT_TOKEN")
    if not token:
        raise RuntimeError(
            "Set NEUPRINT_TOKEN to the token from your neuPrint account."
        )

    return Client(
        "https://neuprint.janelia.org",
        dataset="male-cns:v1.0",
        token=token,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flybrain-data",
        description="Download or inspect the official MaleCNS v1.0 dataset.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="download public MaleCNS files")
    dl.add_argument("--graph", action="store_true", help="also download full 1.1 GB graph")
    dl.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    status = sub.add_parser("status", help="show locally available MaleCNS files")
    status.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "download":
        download_dataset(args.data_dir, include_graph=args.graph)
    elif args.command == "status":
        print_status(args.data_dir)


if __name__ == "__main__":
    main()
