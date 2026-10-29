"""Proxy for Esri's WorldElevation3D/Terrain3D ImageServer that returns WGS84
ellipsoidal heights instead of EGM96 orthometric heights.

Only the tile endpoint is converted (that's what SceneView's ElevationLayer uses);
the service JSON is patched, and everything else is passed through untouched.
"""

from __future__ import annotations

import asyncio
import os
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from . import lerc1
from .geoid import GeoidGrid

UPSTREAM = os.environ.get(
    "UPSTREAM_URL",
    "https://elevation3d.arcgis.com/arcgis/rest/services/WorldElevation3D/Terrain3D/ImageServer",
).rstrip("/")
SERVICE_PATH = os.environ.get(
    "SERVICE_PATH", "/arcgis/rest/services/WorldElevation3D/Terrain3D/ImageServer"
).rstrip("/")
GEOID_CACHE = Path(os.environ.get("GEOID_CACHE", Path.home() / ".cache/terrain-wgs84-proxy/egm96_0p25.npy"))
TILE_CACHE_SIZE = int(os.environ.get("TILE_CACHE_SIZE", "2000"))
# Quantization error for re-encoded tiles; defaults to the upstream tile's own value.
MAX_Z_ERROR = float(os.environ["MAX_Z_ERROR"]) if "MAX_Z_ERROR" in os.environ else None

# EGM96 undulation is within roughly [-107, +86] m worldwide.
GEOID_MIN, GEOID_MAX = -107.0, 86.0

_PASSTHROUGH_HEADERS = ("content-type", "cache-control", "etag", "last-modified", "expires")


class TileCache:
    def __init__(self, size: int):
        self.size = size
        self._items: OrderedDict[tuple[int, int, int], bytes] = OrderedDict()

    def get(self, key):
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        return None

    def put(self, key, value: bytes):
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.size:
            self._items.popitem(last=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.geoid = await asyncio.to_thread(GeoidGrid.load, GEOID_CACHE)
    app.state.client = httpx.AsyncClient(timeout=30, follow_redirects=True)
    app.state.tiles = TileCache(TILE_CACHE_SIZE)
    yield
    await app.state.client.aclose()


app = FastAPI(title="Terrain3D WGS84 ellipsoidal-height proxy", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "HEAD", "OPTIONS"], allow_headers=["*"])


def convert_tile(data: bytes, geoid: GeoidGrid, level: int, row: int, col: int) -> bytes:
    image = lerc1.decode(data)
    height, width = image.values.shape
    if height != width:
        raise ValueError(f"unexpected tile shape {image.values.shape}")
    image.values = image.values + geoid.for_tile(level, row, col, width)
    if MAX_Z_ERROR is not None:
        image.max_z_error = MAX_Z_ERROR
    return lerc1.encode(image)


def _forward_headers(resp: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in resp.headers.items() if k.lower() in _PASSTHROUGH_HEADERS}


@app.get(SERVICE_PATH + "/tile/{level}/{row}/{col}")
async def tile(level: int, row: int, col: int, request: Request) -> Response:
    key = (level, row, col)
    cached = request.app.state.tiles.get(key)
    if cached is not None:
        return Response(cached, media_type="application/octet-stream", headers={"Cache-Control": "max-age=86400"})

    upstream = await request.app.state.client.get(
        f"{UPSTREAM}/tile/{level}/{row}/{col}", params=dict(request.query_params)
    )
    if upstream.status_code != 200 or not upstream.content.startswith(b"CntZImage"):
        # Missing tiles, errors, or anything unexpected: hand back exactly what Esri sent.
        return Response(upstream.content, status_code=upstream.status_code, headers=_forward_headers(upstream))

    body = await asyncio.to_thread(convert_tile, upstream.content, request.app.state.geoid, level, row, col)
    request.app.state.tiles.put(key, body)
    headers = _forward_headers(upstream)
    headers.pop("etag", None)
    headers["content-type"] = "application/octet-stream"
    return Response(body, headers=headers)


@app.get(SERVICE_PATH)
async def service_info(request: Request) -> Response:
    upstream = await request.app.state.client.get(UPSTREAM, params=dict(request.query_params))
    if request.query_params.get("f", "").lower() not in ("json", "pjson") or upstream.status_code != 200:
        return Response(upstream.content, status_code=upstream.status_code, headers=_forward_headers(upstream))

    info = upstream.json()
    if "minValues" in info:
        info["minValues"] = [v + GEOID_MIN for v in info["minValues"]]
    if "maxValues" in info:
        info["maxValues"] = [v + GEOID_MAX for v in info["maxValues"]]
    note = (
        "<p><b>Proxied:</b> tile values are WGS84 ellipsoidal heights "
        "(orthometric EGM96 heights + geoid undulation).</p>"
    )
    for field in ("serviceDescription", "description"):
        if field in info:
            info[field] = note + (info[field] or "")
    return Response(_json_bytes(info), media_type="application/json")


@app.api_route(SERVICE_PATH + "/{rest:path}", methods=["GET", "HEAD"])
async def passthrough(rest: str, request: Request) -> Response:
    upstream = await request.app.state.client.request(
        request.method, f"{UPSTREAM}/{rest}", params=dict(request.query_params)
    )
    return Response(upstream.content, status_code=upstream.status_code, headers=_forward_headers(upstream))


@app.get("/healthz")
async def healthz(request: Request) -> dict:
    n = float(request.app.state.geoid.undulation(np.array(0.0), np.array(0.0)))
    return {"ok": True, "geoid_N_at_0_0": n}


def _json_bytes(obj) -> bytes:
    import json

    return json.dumps(obj, separators=(",", ":")).encode()
