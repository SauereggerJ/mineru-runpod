# MinerU on RunPod Serverless — generic PDF parsing worker.
# MinerU 3.2.x runtime, MinerU2.5-Pro-2605-1.2B VLM as the default model.
#
# Base image: vllm/vllm-openai (recommended by MinerU upstream — bundles CUDA
# + a working vLLM that the VLM backend depends on).
#
# At runtime: handler.py listens for RunPod jobs, downloads/decodes the input
# PDF, calls MinerU's async parse, and returns the result as a base64 tarball.
#
# Model weights live on the attached Network Volume at
# /runpod-volume/huggingface-cache (HF's default cache layout), NOT in the
# image. This keeps the image at ~5 GB so fresh hosts pull it quickly and
# reliably; the model cache is written once to the volume and reused by every
# worker on every host. HF_HUB_OFFLINE=1 forces the libs to read from the
# cache only — fail-fast against a misconfigured/missing volume.
#
# Populate the volume once before serving real traffic:
#   HF_HOME=/runpod-volume/huggingface-cache HF_HUB_OFFLINE=0 python3 -c \
#     "from huggingface_hub import snapshot_download; \
#      snapshot_download(repo_id='opendatalab/MinerU2.5-Pro-2605-1.2B'); \
#      snapshot_download(repo_id='opendatalab/PDF-Extract-Kit-1.0')"
#
# Model selection: MinerU 3.2.x's library default is
# `opendatalab/MinerU2.5-Pro-2605-1.2B` for the VLM backend; pipeline
# backend uses `opendatalab/PDF-Extract-Kit-1.0`. Both live on the volume.
# Note: MinerU bumps the VLM default on minor-version releases (3.1→3.2
# bumped 2604→2605); the requirements.txt pin is minor-locked to keep
# the baked model in sync with the library default.

ARG VLLM_VERSION=v0.11.2
FROM vllm/vllm-openai:${VLLM_VERSION}

# HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1 force the HuggingFace libs to
# read from cache only. The cache lives on the Network Volume, which is
# mounted at /runpod-volume. Offline mode prevents accidental downloads and
# fail-fast against a misconfigured/missing volume.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/runpod-volume/huggingface-cache \
    HF_HUB_CACHE=/runpod-volume/huggingface-cache/hub \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

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
COPY worker /worker/worker

# Tiny fixture PDF used by local smoke input and optional Hub tests. It is
# copied into /worker/test-fixture.pdf so validations can round-trip a real
# document without adding meaningful image size.
COPY .runpod/test-fixture.pdf /worker/test-fixture.pdf

# RunPod's serverless runtime invokes Python directly. `python3` is what
# vllm/vllm-openai ships on PATH; `python` is not always aliased.
CMD ["python3", "-u", "handler.py"]
