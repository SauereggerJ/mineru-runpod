"""Pod HTTP server for MinerU parse requests.

Runs the same MinerU parsing logic as the RunPod serverless handler, but over
a plain HTTP API so the worker can be deployed as a persistent RunPod Pod and
driven programmatically from the Go app. Use this as the Pod's CMD:

    python3 -u http_server.py --port 8000

Endpoints:
    GET  /health         -> {"ok": true, "mineru_available": bool}
    POST /parse          -> same request/response contract as the serverless
                            handler's `input` field, minus RunPod wrappers.
                            Starts the conversion async and returns a job id.
    GET  /status/{id}    -> polls a running /parse job's finished result.
    POST /seed           -> download both MinerU models into the HF cache
                            (Container Disk) if missing.

The Go side starts the Pod, waits for /health, submits the batched
conversions via /parse, then stops the Pod.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import handler as _handler
from worker import debug as _debug
from worker import logging as _logging
from worker import schema as _schema

# Async parse jobs. /parse starts a conversion in the background and returns a
# job id immediately (so the request stays well under the Cloudflare proxy's
# ~100s ceiling); /status/{id} polls the finished result.
_PARSE_JOBS: dict[str, dict[str, Any]] = {}


def _bootstrap_models() -> None:
    """Download MinerU models into the Container Disk cache if missing."""
    from worker import warmup as _warmup  # noqa: PLC0415

    _warmup.check_volume()
    try:
        _warmup.ensure_models()
    except Exception as exc:  # noqa: BLE001
        _logging.error("model cache bootstrap failed", error=repr(exc))
        _logging.error("jobs will fail until models are present on the container disk")


async def _handle_parse_request(body: dict) -> dict:
    started = time.monotonic()
    phase_ms: dict[str, int] = {}
    gpu_info = _debug.collect_gpu_info()
    raw_input = body.get("input") or body
    if raw_input.get("seed_volume") is True:
        from seed_volume import seed  # noqa: PLC0415
        result = seed()
        return {"ok": True, "seed_volume": result, "elapsed_seconds": round(time.monotonic() - started, 2)}
    if raw_input.get("probe") is True:
        return await _handler._handle_probe(started, gpu_info, phase_ms)
    cleaned = _schema.validate_input(raw_input)
    return await _handler._handle_parse({"id": raw_input.get("id") or "<http>"}, cleaned, started, gpu_info, phase_ms)


async def _run_parse_job(job_id: str, body: dict) -> None:
    """Execute one /parse request in the background and stash its outcome."""
    try:
        result = await _handle_parse_request(body)
        _PARSE_JOBS[job_id] = {"done": True, "result": result}
    except Exception as exc:  # noqa: BLE001
        _logging.error("async parse failed", job_id=job_id, error=repr(exc))
        _PARSE_JOBS[job_id] = {
            "done": True,
            "result": {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": __import__("traceback").format_exc(limit=8),
            },
        }


async def _run_server(port: int) -> None:
    from aiohttp import web  # noqa: PLC0415

    _bootstrap_models()

    async def health(_: web.Request) -> web.Response:
        return web.json_response({
            "ok": True,
            "mineru_available": _handler._MINERU_AVAILABLE,
            "mineru_version": _handler.MINERU_VERSION,
            "model_dir": _debug.find_model_dir(),
        })

    async def parse(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"ok": False, "error": f"invalid JSON: {exc}"}, status=400)
        try:
            _schema.validate_input(body.get("input") or body)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=400)
        job_id = f"{int(time.time() * 1000)}-{os.urandom(4).hex()}"
        _PARSE_JOBS[job_id] = {"done": False, "result": None}
        asyncio.get_running_loop().create_task(_run_parse_job(job_id, body))
        return web.json_response({"ok": True, "job_id": job_id, "status": "queued"})

    async def status(request: web.Request) -> web.Response:
        job_id = request.match_info["id"]
        entry = _PARSE_JOBS.get(job_id)
        if entry is None:
            return web.json_response({"ok": False, "error": f"unknown job {job_id}"}, status=404)
        if not entry["done"]:
            return web.json_response({"ok": True, "job_id": job_id, "status": "running"})
        return web.json_response({"ok": True, "job_id": job_id, "status": "done", "result": entry["result"]})

    async def seed(_: web.Request) -> web.Response:
        from seed_volume import seed  # noqa: PLC0415
        try:
            result = seed()
            return web.json_response({"ok": True, "seed_volume": result})
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"ok": False, "error": repr(exc)}, status=500)

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/parse", parse)
    app.router.add_get("/status/{id}", status)
    app.router.add_post("/seed", seed)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=port)
    await site.start()
    _logging.info(f"http server listening on 0.0.0.0:{port}")
    while True:
        await asyncio.sleep(3600)


def main() -> int:
    parser = argparse.ArgumentParser(description="MinerU Pod HTTP server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("MINERU_HTTP_PORT", "8000")))
    args = parser.parse_args()
    try:
        asyncio.run(_run_server(args.port))
    except KeyboardInterrupt:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
