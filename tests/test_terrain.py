import asyncio
import struct

import numpy as np
import pytest

from terrain_wgs84_proxy import lerc1
from terrain_wgs84_proxy.terrain import TerrainSampler, bilinear, locate

NO_MASK = struct.pack("<IIIf", 0, 0, 0, 1.0)


def _tile_bytes(values: np.ndarray) -> bytes:
    return lerc1.encode(lerc1.Lerc1Image(values=values.astype(np.float32), max_z_error=0.001, mask_section=NO_MASK))


def test_locate():
    assert locate(0.0, 0.0, 0) == (0, 0, pytest.approx(0.5), pytest.approx(0.5))
    # Just north-east of (0, 0) at level 1 is the top-right tile's bottom-left corner.
    row, col, ty, tx = locate(1e-9, 1e-9, 1)
    assert (row, col) == (0, 1)
    assert ty == pytest.approx(1.0) and tx == pytest.approx(0.0, abs=1e-9)
    # Antimeridian and poles stay inside the tile grid.
    assert locate(180.0, 0.0, 3)[1] == 0
    assert locate(0.0, 90.0, 3)[0] == 0
    assert locate(0.0, -90.0, 3)[0] == 7


def test_bilinear():
    values = np.array([[0.0, 10.0], [20.0, 30.0]])
    assert bilinear(values, 0.5, 0.5) == pytest.approx(15.0)
    assert bilinear(values, 0.0, 1.0) == pytest.approx(10.0)
    values[1, 1] = np.nan
    assert bilinear(values, 0.5, 0.5) == pytest.approx(10.0)  # mean of the three valid corners
    values[0, 1] = np.nan
    assert bilinear(values, 1.0, 1.0) == pytest.approx(20.0)  # all weight on a masked sample: nearest valid
    assert np.isnan(bilinear(np.full((2, 2), np.nan), 0.5, 0.5))


def test_sampler_falls_back_to_coarser_level():
    # Level 0: a plane rising west to east; level 1: nothing exists.
    x = np.linspace(0, 256, 257)
    plane = np.tile(x, (257, 1))
    fetched = []

    async def fetch(level, row, col):
        fetched.append((level, row, col))
        return _tile_bytes(plane) if level == 0 else None

    async def run():
        sampler = TerrainSampler(fetch, cache_size=10)
        return await asyncio.gather(*(sampler.height(0.0, 0.0, 1) for _ in range(5)))

    results = asyncio.run(run())
    assert all(lvl == 0 and h == pytest.approx(128.0, abs=0.01) for h, lvl in results)
    assert sorted(fetched) == [(0, 0, 0), (1, 1, 1)]  # each tile fetched once despite concurrency


def test_sampler_raises_when_no_data():
    async def fetch(level, row, col):
        return None

    with pytest.raises(LookupError):
        asyncio.run(TerrainSampler(fetch, cache_size=10).height(10.0, 10.0, 2))
