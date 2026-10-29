"""Proxy for Esri's WorldElevation3D/Terrain3D ImageServer that returns WGS84
ellipsoidal heights instead of EGM96 orthometric heights.

Only the tile endpoint is converted (that's what SceneView's ElevationLayer uses);
the service JSON is patched, and everything else is passed through untouched.

Also exposes `/altitude`, which converts point altitudes between WGS84 ellipsoidal,
orthometric (EGM96, ~MSL) and above-ground-level heights using the same terrain.
"""

from __future__ import annotations

import asyncio
import os
from collections import OrderedDict
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import lerc1
from .geoid import GeoidGrid
from .terrain import TerrainSampler

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
# Finest Terrain3D tile level (~2.4 m at the equator) used for AGL lookups.
TERRAIN_LEVEL = int(os.environ.get("TERRAIN_LEVEL", "16"))
TERRAIN_CACHE_SIZE = int(os.environ.get("TERRAIN_CACHE_SIZE", "500"))
ALTITUDE_MAX_POINTS = int(os.environ.get("ALTITUDE_MAX_POINTS", "1000"))

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
    app.state.terrain = TerrainSampler(_tile_fetcher(app.state.client), TERRAIN_CACHE_SIZE)
    yield
    await app.state.client.aclose()


app = FastAPI(title="Terrain3D WGS84 ellipsoidal-height proxy", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "HEAD", "POST", "OPTIONS"], allow_headers=["*"])


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


def _tile_fetcher(client: httpx.AsyncClient):
    limit = asyncio.Semaphore(16)

    async def fetch(level: int, row: int, col: int) -> bytes | None:
        async with limit:
            resp = await client.get(f"{UPSTREAM}/tile/{level}/{row}/{col}")
        if resp.status_code == 404 or (resp.status_code == 200 and not resp.content.startswith(b"CntZImage")):
            return None
        resp.raise_for_status()
        return resp.content

    return fetch


class AltitudeRef(str, Enum):
    wgs84 = "wgs84"  # WGS84 ellipsoidal height
    msl = "msl"  # orthometric height, EGM96
    agl = "agl"  # height above the Terrain3D ground surface


class AltitudePoint(BaseModel):
    lon: float = Field(ge=-180, le=180)
    lat: float = Field(ge=-90, le=90)
    alt: float


class AltitudeRequest(BaseModel):
    ref: AltitudeRef = AltitudeRef.wgs84
    level: int = Field(TERRAIN_LEVEL, ge=0, le=TERRAIN_LEVEL)
    points: list[AltitudePoint] = Field(min_length=1, max_length=ALTITUDE_MAX_POINTS)


class Terrain(BaseModel):
    msl: float
    wgs84: float
    level: int  # tile level the ground height was sampled from (lower = coarser)


class AltitudeResult(BaseModel):
    lon: float
    lat: float
    wgs84: float
    msl: float
    agl: float
    geoid_undulation: float  # N: wgs84 = msl + N
    terrain: Terrain


class AltitudeResponse(BaseModel):
    results: list[AltitudeResult]


async def _convert_altitude(request: Request, p: AltitudePoint, ref: AltitudeRef, level: int) -> AltitudeResult:
    n = float(request.app.state.geoid.undulation(np.array(p.lon), np.array(p.lat)))
    try:
        ground, ground_level = await request.app.state.terrain.height(p.lon, p.lat, level)
    except LookupError as exc:
        raise HTTPException(422, str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"terrain tile fetch failed: {exc}") from exc
    if ref is AltitudeRef.wgs84:
        msl = p.alt - n
    elif ref is AltitudeRef.msl:
        msl = p.alt
    else:
        msl = p.alt + ground
    return AltitudeResult(
        lon=p.lon,
        lat=p.lat,
        wgs84=msl + n,
        msl=msl,
        agl=msl - ground,
        geoid_undulation=n,
        terrain=Terrain(msl=ground, wgs84=ground + n, level=ground_level),
    )


@app.get("/altitude", response_model=AltitudeResult)
async def altitude(
    request: Request,
    lon: float = Query(ge=-180, le=180),
    lat: float = Query(ge=-90, le=90),
    alt: float = Query(),
    ref: AltitudeRef = AltitudeRef.wgs84,
    level: int = Query(TERRAIN_LEVEL, ge=0, le=TERRAIN_LEVEL),
) -> AltitudeResult:
    """Convert one altitude (given relative to `ref`) to WGS84 ellipsoidal, MSL and AGL heights."""
    return await _convert_altitude(request, AltitudePoint(lon=lon, lat=lat, alt=alt), ref, level)


@app.post("/altitude", response_model=AltitudeResponse)
async def altitude_batch(body: AltitudeRequest, request: Request) -> AltitudeResponse:
    """Batch version of `GET /altitude`; results are in the same order as `points`."""
    results = await asyncio.gather(*(_convert_altitude(request, p, body.ref, body.level) for p in body.points))
    return AltitudeResponse(results=list(results))


@app.get("/healthz")
async def healthz(request: Request) -> dict:
    n = float(request.app.state.geoid.undulation(np.array(0.0), np.array(0.0)))
    return {"ok": True, "geoid_N_at_0_0": n}


def _json_bytes(obj) -> bytes:
    import json

    return json.dumps(obj, separators=(",", ":")).encode()
