# MinerU on RunPod Serverless — generic PDF parsing worker.
# MinerU 3.2.x runtime, MinerU2.5-Pro-2605-1.2B VLM as the default model.
#
# Base image: vllm/vllm-openai (recommended by MinerU upstream — bundles CUDA
# + a working vLLM that the VLM backend depends on).
#
# At runtime: handler.py listens for RunPod jobs, downloads/decodes the input
# PDF, calls MinerU's async parse, and returns the result as a base64 tarball.
#
# Model weights are NOT baked into the image and NOT stored on a Network
# Volume. The image stays ~5 GB so fresh hosts pull it quickly and reliably.
# At boot, warmup.ensure_models() downloads both MinerU models into the
# container's HF cache (Container Disk) when they are missing, then warms up
# the VLM. HF_HUB_OFFLINE=1 keeps normal runtime offline; ensure_models()
# temporarily disables it for the one-time fetch. No manual volume seeding
# or region matching is required.
#
# Model selection: MinerU 3.2.x's library default is
# `opendatalab/MinerU2.5-Pro-2605-1.2B` for the VLM backend; pipeline
# backend uses `opendatalab/PDF-Extract-Kit-1.0`. Both are fetched at boot.
# Note: MinerU bumps the VLM default on minor-version releases (3.1→3.2
# bumped 2604→2605); the requirements.txt pin is minor-locked to keep
# the fetched model in sync with the library default.

ARG VLLM_VERSION=v0.11.2
FROM vllm/vllm-openai:${VLLM_VERSION}

# Model weights are NOT baked into the image — the image stays ~5 GB so
# fresh hosts pull it quickly and reliably. At boot, warmup.ensure_models()
# downloads both MinerU models into the container's HF cache (Container Disk,
# default 50 GB) when they are missing, then warms up the VLM. Offline mode is
# NOT set globally — huggingface_hub caches the flag at import time, so a global
# HF_HUB_OFFLINE=1 would break ensure_models()' one-time fetch. Normal runtime
# reads from the local cache; no per-job network model access happens.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_CACHE=/root/.cache/huggingface/hub

# vllm-openai inherits an entrypoint that launches the OpenAI server. Override
# it so our handler can be the process.
ENTRYPOINT []

# System deps. The base image already has CUDA + Python; we only need the
# things mineru/pdf processing want at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        poppler-utils \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /worker

# Install uv (10x+ faster than pip on resolution-heavy installs like
# mineru[core,vllm], which churns through pydantic / opencv / numpy
# version conflicts with the base image). Negligible image size (~10 MB)
# in exchange for a meaningful build-time win.
# hadolint ignore=DL3013
RUN pip install --no-cache-dir uv

# Install MinerU + RunPod worker SDK. mineru[core,vllm] pulls the VLM-engine
# dependencies that match the vllm version in the base image.
COPY requirements.txt /worker/requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the worker code last so iterating on it doesn't bust the pip layer.
# handler.py is the entry point; the worker/ package holds the modules it
# imports (schema, io, parse, package, debug, logging). Both must land at
# /worker/ so `from worker import ...` resolves from the script's directory.
# Model weights are intentionally NOT copied — they live on the Network
# Volume (see HF_HOME above).
COPY handler.py /worker/handler.py
COPY seed_volume.py /worker/seed_volume.py
COPY http_server.py /worker/http_server.py
COPY worker /worker/worker

# Tiny fixture PDF used by local smoke input and optional Hub tests. It is
# copied into /worker/test-fixture.pdf so validations can round-trip a real
# document without adding meaningful image size.
COPY .runpod/test-fixture.pdf /worker/test-fixture.pdf

# Pod entry point: boot the HTTP server (downloads models into the Container
# Disk cache if missing, then serves /health + /parse). `python3` is what
# vllm/vllm-openai ships on PATH; `python` is not always aliased.
CMD ["python3", "-u", "http_server.py", "--port", "8000"]
