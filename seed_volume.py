"""Populate the Network Volume with MinerU model weights.

Run once per volume before serving real traffic. From inside a worker this
is exposed as a job: POST {"input": {"seed_volume": true}}. The handler
imports `seed` and calls it with offline mode temporarily disabled, so the
download targets the volume despite the image-wide HF_HUB_OFFLINE=1.

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


def seed() -> dict:
    """Download all MODELS into HF_HOME on the volume.

    Temporarily forces HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE to "0" so the
    download works even though the image-wide ENV keeps them at "1".
    Returns a dict summarizing what was fetched and where.
    """
    cache_root = Path(
        os.environ.get("HF_HOME", "/runpod-volume/huggingface-cache")
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    old_offline = os.environ.get("HF_HUB_OFFLINE")
    old_transformers = os.environ.get("TRANSFORMERS_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    try:
        from huggingface_hub import snapshot_download

        for repo_id in MODELS:
            print(f"downloading {repo_id} ...", flush=True)
            path = snapshot_download(repo_id=repo_id)
            print(f"  -> {path}", flush=True)
            results.append({"model": repo_id, "path": str(path)})
    finally:
        if old_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_offline
        if old_transformers is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = old_transformers

    return {"cache_root": str(cache_root), "models": results}


def main() -> int:
    result = seed()
    print("seed complete:", result, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
