import struct

import lerc
import numpy as np

from terrain_wgs84_proxy import lerc1

NO_MASK = struct.pack("<IIIf", 0, 0, 0, 1.0)


def _masked_section(valid: np.ndarray) -> bytes:
    bitset = np.packbits(valid.ravel().astype(np.uint8)).tobytes()
    rle = struct.pack("<h", len(bitset)) + bitset + struct.pack("<h", -32768)
    return struct.pack("<IIIf", 1, 1, len(rle), 1.0) + rle


def test_roundtrip_within_max_z_error():
    rng = np.random.default_rng(0)
    values = (rng.normal(500, 300, (257, 257))).astype(np.float32)
    img = lerc1.Lerc1Image(values=values, max_z_error=0.1, mask_section=NO_MASK)
    blob = lerc1.encode(img)
    ours = lerc1.decode(blob).values
    esri = lerc.decode(blob)[1]  # Esri's reference decoder
    assert np.abs(ours - values).max() <= 0.1 + 1e-4
    assert np.abs(esri - values).max() <= 0.1 + 1e-4


def test_mask_is_preserved():
    rng = np.random.default_rng(1)
    valid = rng.random((257, 257)) > 0.3
    valid[:40, :40] = False  # fully masked blocks
    values = np.where(valid, rng.uniform(-50, 50, valid.shape), np.nan).astype(np.float32)
    img = lerc1.Lerc1Image(values=values, max_z_error=0.01, mask_section=_masked_section(valid))
    out = lerc1.decode(lerc1.encode(img))
    assert np.array_equal(out.valid, valid)
    assert np.nanmax(np.abs(out.values - values)) <= 0.01 + 1e-4


def test_constant_tile():
    values = np.full((257, 257), 17.25, dtype=np.float32)
    img = lerc1.Lerc1Image(values=values, max_z_error=0.1, mask_section=NO_MASK)
    assert np.array_equal(lerc1.decode(lerc1.encode(img)).values, values)
