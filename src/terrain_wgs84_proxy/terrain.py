"""Point sampling of Terrain3D orthometric (EGM96) ground heights.

Upstream `getSamples` ignores `pixelSize` and samples the coarsest level, so we
sample the elevation tiles ourselves: find the tile containing the point at the
requested level, decode it, and bilinearly interpolate between its vertex samples.
If the tile is missing or the point is masked, we fall back to coarser levels.
"""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from collections.abc import Awaitable, Callable

import numpy as np

from . import lerc1
from .geoid import EARTH_RADIUS, ORIGIN

TileKey = tuple[int, int, int]
# Returns the raw upstream tile bytes, or None if the tile doesn't exist.
TileFetcher = Callable[[int, int, int], Awaitable[bytes | None]]


def locate(lon: float, lat: float, level: int) -> tuple[int, int, float, float]:
    """Tile (row, col) containing the point, and its (y, x) position within that tile as fractions in [0, 1]."""
    lat = max(min(lat, 85.0511287798), -85.0511287798)
    x = math.radians(((lon + 180) % 360) - 180) * EARTH_RADIUS
    y = math.asinh(math.tan(math.radians(lat))) * EARTH_RADIUS
    n = 1 << level
    span = 2 * ORIGIN / n
    fc = (x + ORIGIN) / span
    fr = (ORIGIN - y) / span
    col = min(max(int(math.floor(fc)), 0), n - 1)
    row = min(max(int(math.floor(fr)), 0), n - 1)
    return row, col, fr - row, fc - col


def bilinear(values: np.ndarray, fy: float, fx: float) -> float:
    """Bilinear interpolation over valid (non-NaN) neighbours; NaN if all four are masked."""
    h, w = values.shape
    y0 = min(max(int(math.floor(fy)), 0), h - 2)
    x0 = min(max(int(math.floor(fx)), 0), w - 2)
    dy, dx = fy - y0, fx - x0
    v = values[y0 : y0 + 2, x0 : x0 + 2].astype(np.float64)
    wts = np.array([[(1 - dy) * (1 - dx), (1 - dy) * dx], [dy * (1 - dx), dy * dx]])
    ok = ~np.isnan(v)
    if not ok.any():
        return math.nan
    total = wts[ok].sum()
    if total == 0:  # all the weight is on masked samples; use the nearest valid one
        dist = np.hypot(np.array([[dy, dy], [1 - dy, 1 - dy]]), np.array([[dx, 1 - dx], [dx, 1 - dx]]))
        return float(v[ok][np.argmin(dist[ok])])
    return float((v[ok] * wts[ok]).sum() / total)


class TerrainSampler:
    def __init__(self, fetch: TileFetcher, cache_size: int):
        self._fetch = fetch
        self.cache_size = cache_size
        # Decoded orthometric tiles (None = tile doesn't exist upstream).
        self._tiles: OrderedDict[TileKey, np.ndarray | None] = OrderedDict()
        self._pending: dict[TileKey, asyncio.Future] = {}

    async def tile(self, key: TileKey) -> np.ndarray | None:
        if key in self._tiles:
            self._tiles.move_to_end(key)
            return self._tiles[key]
        if key in self._pending:
            return await self._pending[key]
        fut = asyncio.get_running_loop().create_future()
        self._pending[key] = fut
        try:
            data = await self._fetch(*key)
            values = None if data is None else (await asyncio.to_thread(lerc1.decode, data)).values
        except BaseException as exc:
            fut.set_exception(exc)
            fut.exception()  # mark retrieved so waiters-less failures don't warn
            raise
        finally:
            del self._pending[key]
        self._tiles[key] = values
        while len(self._tiles) > self.cache_size:
            self._tiles.popitem(last=False)
        fut.set_result(values)
        return values

    async def height(self, lon: float, lat: float, level: int) -> tuple[float, int]:
        """Orthometric ground height at the point and the tile level it came from."""
        for lvl in range(level, -1, -1):
            row, col, ty, tx = locate(lon, lat, lvl)
            values = await self.tile((lvl, row, col))
            if values is None:
                continue
            # Samples sit on the grid vertices, so the first/last rows and cols lie on the tile edges.
            h = bilinear(values, ty * (values.shape[0] - 1), tx * (values.shape[1] - 1))
            if not math.isnan(h):
                return h, lvl
        raise LookupError(f"no terrain data at lon={lon}, lat={lat}")
