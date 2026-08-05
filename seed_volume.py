"""Populate the Network Volume with MinerU model weights.

Run ONCE per volume before serving real traffic:

    HF_HOME=/runpod-volume/huggingface-cache HF_HUB_OFFLINE=0 \
      python3 seed_volume.py

Both models MinerU needs are downloaded into the volume's HuggingFace
cache layout, so workers (which run with HF_HUB_OFFLINE=1 and
HF_HOME=/runpod-volume/huggingface-cache) load them locally without any
network pull or 27 GB image transfer.

Exit code 0 on success, non-zero on failure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MODELS = [
    "opendatalab/MinerU2.5-Pro-2605-1.2B",
    "opendatalab/PDF-Extract-Kit-1.0",
]


def main() -> int:
    cache_root = Path(
        os.environ.get(
            "HF_HOME",
            "/runpod-volume/huggingface-cache",
        )
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    print(f"cache root: {cache_root}", flush=True)

    from huggingface_hub import snapshot_download

    for repo_id in MODELS:
        print(f"downloading {repo_id} ...", flush=True)
        path = snapshot_download(repo_id=repo_id)
        print(f"  -> {path}", flush=True)

    print("seed complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
