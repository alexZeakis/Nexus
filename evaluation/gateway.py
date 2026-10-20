"""
OpenAI logging gateway.

Run:   uvicorn gateway:app --port 8000        (one worker, async handles concurrency)
Use:   OPENAI_BASE_URL=http://localhost:8000/<method>/<stage>/v1
       e.g. http://localhost:8000/Nexus_Mix/retrieval/v1

Each client sends its own API key; the proxy forwards it untouched and only
logs a short hash of it. One JSON line per upstream call is appended to LOG_FILE.
"""
import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

UPSTREAM = os.getenv("UPSTREAM", "https://api.openai.com")
LOG_PREFIX = os.getenv("LOG_PREFIX", "logs")  # log file: <prefix>_<port>.jsonl

client = httpx.AsyncClient(
    base_url=UPSTREAM,
    timeout=httpx.Timeout(900, connect=30),
    limits=httpx.Limits(max_connections=500, max_keepalive_connections=100),
)
app = FastAPI()
log_lock = asyncio.Lock()

SKIP_REQ = {"host", "content-length", "connection", "keep-alive",
            "transfer-encoding", "accept-encoding", "upgrade"}
SKIP_RESP = {"content-length", "content-encoding", "transfer-encoding", "connection"}


def norm_usage(u):
    if not u:
        return {}
    d_in = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
    d_out = u.get("completion_tokens_details") or u.get("output_tokens_details") or {}
    return {
        "prompt_tokens": u.get("prompt_tokens", u.get("input_tokens")),
        "completion_tokens": u.get("completion_tokens", u.get("output_tokens")),
        "total_tokens": u.get("total_tokens"),
        "cached_tokens": d_in.get("cached_tokens"),
        "reasoning_tokens": d_out.get("reasoning_tokens"),
    }


def usage_of(obj):
    """Usage block from a chat/embeddings body, a chat stream chunk or a Responses event."""
    if not isinstance(obj, dict):
        return None, None
    if obj.get("usage"):
        return obj["usage"], obj.get("model")
    r = obj.get("response")
    if isinstance(r, dict) and r.get("usage"):
        return r["usage"], r.get("model")
    return None, obj.get("model")


async def write_log(log_path, rec):
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    async with log_lock:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)


@app.api_route("/{method}/{stage}/v1/{path:path}", methods=["GET", "POST", "DELETE"])
async def proxy(method: str, stage: str, path: str, request: Request):
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in SKIP_REQ}
    key_id = hashlib.sha256(headers.get("authorization", "").encode()).hexdigest()[:8]
    log_path = f"{LOG_PREFIX}_{request.scope['server'][1]}.jsonl"

    payload, model, stream = None, None, False
    if body and "json" in request.headers.get("content-type", ""):
        try:
            payload = json.loads(body)
        except ValueError:
            payload = None
    if isinstance(payload, dict):
        model = payload.get("model")
        stream = bool(payload.get("stream"))
        if stream and path.startswith("chat/completions"):
            so = payload.get("stream_options") or {}
            if "include_usage" not in so:
                so["include_usage"] = True
                payload["stream_options"] = so
                body = json.dumps(payload).encode()

    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "method": method, "stage": stage, "endpoint": "/v1/" + path,
        "model": model, "stream": stream, "key_id": key_id,
        "body_sha1": hashlib.sha1(body).hexdigest()[:12],
    }
    t0 = time.perf_counter()
    req = client.build_request(request.method, "/v1/" + path, headers=headers,
                               content=body, params=request.query_params)
    try:
        resp = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        rec.update(status=0, error=repr(e), latency_s=round(time.perf_counter() - t0, 4))
        await write_log(log_path, rec)
        return Response(content=json.dumps({"error": {"message": repr(e)}}),
                        status_code=502, media_type="application/json")

    rec["status"] = resp.status_code
    rec["request_id"] = resp.headers.get("x-request-id")
    out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in SKIP_RESP}

    if "text/event-stream" in resp.headers.get("content-type", ""):
        async def gen():
            usage, m, buf, ttfb = None, None, b"", None
            try:
                async for chunk in resp.aiter_bytes():
                    if ttfb is None:
                        ttfb = time.perf_counter() - t0
                    yield chunk
                    buf += chunk
                    *lines, buf = buf.split(b"\n")
                    for ln in lines:
                        if ln.startswith(b"data:") and b"[DONE]" not in ln:
                            try:
                                u, mm = usage_of(json.loads(ln[5:]))
                            except ValueError:
                                continue
                            usage = u or usage
                            m = mm or m
            finally:
                await resp.aclose()
                rec.update(norm_usage(usage))
                rec["model"] = m or rec["model"]
                rec["ttfb_s"] = round(ttfb, 4) if ttfb else None
                rec["latency_s"] = round(time.perf_counter() - t0, 4)
                await asyncio.shield(write_log(log_path, rec))

        return StreamingResponse(gen(), status_code=resp.status_code, headers=out_headers)

    data = await resp.aread()
    await resp.aclose()
    try:
        usage, m = usage_of(json.loads(data))
    except ValueError:
        usage, m = None, None
    rec.update(norm_usage(usage))
    rec["model"] = m or rec["model"]
    rec["latency_s"] = round(time.perf_counter() - t0, 4)
    await write_log(log_path, rec)
    return Response(content=data, status_code=resp.status_code, headers=out_headers)
