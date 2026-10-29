"""EGM96 geoid undulation lookup for Web Mercator elevation tiles.

Ellipsoidal height (WGS84) = orthometric height (EGM96) + N, where N is the geoid
undulation. We sample N once on a global 0.25 degree grid via PROJ (EGM96 15' grid,
fetched from the PROJ CDN), cache it to disk, and bilinearly interpolate per tile.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

EARTH_RADIUS = 6378137.0
ORIGIN = 20037508.342787  # Web Mercator half-extent, matches the service tileInfo origin
STEP_DEG = 0.25


class GeoidGrid:
    def __init__(self, grid: np.ndarray):
        # grid[i, j] = N at lat = -90 + i*STEP, lon = -180 + j*STEP
        self.grid = grid

    @classmethod
    def load(cls, cache_path: Path) -> "GeoidGrid":
        if cache_path.exists():
            return cls(np.load(cache_path))
        grid = _build_grid()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, grid)
        return cls(grid)

    def undulation(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        fi = (np.clip(lat, -90, 90) + 90) / STEP_DEG
        fj = (((lon + 180) % 360)) / STEP_DEG
        i0 = np.clip(np.floor(fi).astype(int), 0, self.grid.shape[0] - 2)
        j0 = np.clip(np.floor(fj).astype(int), 0, self.grid.shape[1] - 2)
        di = fi - i0
        dj = fj - j0
        g = self.grid
        return (
            g[i0, j0] * (1 - di) * (1 - dj)
            + g[i0, j0 + 1] * (1 - di) * dj
            + g[i0 + 1, j0] * di * (1 - dj)
            + g[i0 + 1, j0 + 1] * di * dj
        )

    def for_tile(self, level: int, row: int, col: int, size: int) -> np.ndarray:
        """N for each sample of a (size x size) tile.

        Esri elevation tiles are 257x257 for a 256-pixel tile: samples sit on the
        grid vertices, so the first and last rows/cols lie on the tile edges.
        """
        span = 2 * ORIGIN / (1 << level)
        step = span / (size - 1)
        x = -ORIGIN + col * span + np.arange(size) * step
        y = ORIGIN - row * span - np.arange(size) * step
        lon = np.degrees(x / EARTH_RADIUS)
        lat = np.degrees(np.arctan(np.sinh(y / EARTH_RADIUS)))
        lon2d, lat2d = np.meshgrid(lon, lat)
        return self.undulation(lon2d, lat2d).astype(np.float32)


def _build_grid() -> np.ndarray:
    import pyproj
    from pyproj import Transformer

    pyproj.network.set_network_enabled(True)
    transformer = Transformer.from_crs("EPSG:4326+5773", "EPSG:4979", always_xy=True)
    lats = -90 + np.arange(int(180 / STEP_DEG) + 1) * STEP_DEG
    # One extra column past 180 so interpolation wraps across the antimeridian.
    lons = -180 + np.arange(int(360 / STEP_DEG) + 2) * STEP_DEG
    lon2d, lat2d = np.meshgrid(lons, lats)
    _, _, h = transformer.transform(lon2d.ravel(), lat2d.ravel(), np.zeros(lon2d.size))
    grid = np.asarray(h, dtype=np.float64).reshape(lat2d.shape)

    # Fail loudly if PROJ silently fell back to a no-op (grid unavailable).
    n_null_island = grid[int(90 / STEP_DEG), int(180 / STEP_DEG)]
    if not math.isfinite(n_null_island) or abs(n_null_island - 17.16) > 0.1:
        raise RuntimeError(
            "EGM96 geoid grid not applied by PROJ (got N=%r at 0,0; expected ~17.16). "
            "Ensure network access to cdn.proj.org or install us_nga_egm96_15.tif "
            "with `projsync --file us_nga_egm96_15.tif`." % n_null_island
        )
    return grid.astype(np.float32)
