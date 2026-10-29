"""Minimal LERC1 ("CntZImage") codec for single-band float32 elevation tiles.

Esri's elevation tile caches serve LERC1, and the ArcGIS Maps SDK's elevation
decoder expects it, so we decode, modify and re-encode in the same format.
Format reference: LercDecode.js (Lerc1Decoder) in https://github.com/Esri/lerc.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

_FILE_ID = b"CntZImage "
_FILE_VERSION = 11
_IMAGE_TYPE = 8  # CNT_Z


@dataclass
class Lerc1Image:
    values: np.ndarray  # float32 (height, width); masked pixels are NaN
    max_z_error: float
    # The raw mask section (header + RLE bytes), passed through verbatim on encode.
    mask_section: bytes

    @property
    def valid(self) -> np.ndarray:
        return ~np.isnan(self.values)


def _decode_mask(buf: bytes, pos: int, width: int, height: int) -> tuple[np.ndarray | None, int]:
    num_blocks_y, _num_blocks_x, num_bytes, max_value = struct.unpack_from("<IIIf", buf, pos)
    pos += 16
    num_pixels = width * height
    if num_bytes == 0:
        if num_blocks_y == 0 and max_value == 0:
            return np.zeros(num_pixels, dtype=bool), pos  # all nodata
        return None, pos  # all valid

    bitset = bytearray()
    end = pos + num_bytes
    while True:
        (cnt,) = struct.unpack_from("<h", buf, pos)
        pos += 2
        if cnt == -32768:
            break
        if cnt > 0:
            bitset += buf[pos : pos + cnt]
            pos += cnt
        else:
            bitset += bytes([buf[pos]]) * (-cnt)
            pos += 1
        if pos >= end:
            raise ValueError("unterminated LERC1 mask RLE")
    bits = np.unpackbits(np.frombuffer(bytes(bitset), dtype=np.uint8))[:num_pixels]
    if bits.size < num_pixels:
        raise ValueError("LERC1 mask too short")
    return bits.astype(bool), end


def _block_bounds(size: int, num_blocks: int) -> list[tuple[int, int]]:
    step = size // num_blocks
    bounds = [(i * step, (i + 1) * step) for i in range(num_blocks)]
    if size % num_blocks:
        bounds.append((num_blocks * step, size))
    return bounds


def _unstuff(data: bytes, bits_per_pixel: int, count: int) -> np.ndarray:
    # Values are packed MSB-first into little-endian uint32 words; the final word
    # is truncated to just the bytes needed, holding its high-order bytes.
    num_bytes = len(data)
    padded = data + b"\0" * (-num_bytes % 4)
    words = bytearray(padded)
    tail = num_bytes % 4
    if tail:
        last = int.from_bytes(words[-4:], "little") << (8 * (4 - tail))
        words[-4:] = (last & 0xFFFFFFFF).to_bytes(4, "little")
    be = np.frombuffer(bytes(words), dtype="<u4").astype(">u4").tobytes()
    bits = np.unpackbits(np.frombuffer(be, dtype=np.uint8))[: count * bits_per_pixel]
    bits = bits.reshape(count, bits_per_pixel)
    weights = (1 << np.arange(bits_per_pixel - 1, -1, -1, dtype=np.uint64))
    return (bits.astype(np.uint64) @ weights).astype(np.uint32)


def _stuff(values: np.ndarray, bits_per_pixel: int) -> bytes:
    shifts = np.arange(bits_per_pixel - 1, -1, -1, dtype=np.uint32)
    bits = ((values[:, None] >> shifts) & 1).astype(np.uint8).ravel()
    be = np.packbits(bits).tobytes()  # MSB-first byte stream == big-endian words
    num_bytes = len(be)
    full = num_bytes - num_bytes % 4
    out = np.frombuffer(be[:full], dtype=">u4").astype("<u4").tobytes()
    # The partial last word is stored with its meaningful (high) bytes moved down to
    # the low end, written little-endian: i.e. those bytes reversed.
    return out + be[full:][::-1]


def decode(buf: bytes) -> Lerc1Image:
    if buf[:10] != _FILE_ID:
        raise ValueError("not a LERC1 (CntZImage) blob")
    _version, _type, height, width, max_z_error = struct.unpack_from("<iiIId", buf, 10)
    pos = 34
    mask_start = pos
    mask, pos = _decode_mask(buf, pos, width, height)
    mask_section = buf[mask_start:pos]

    num_blocks_y, num_blocks_x, num_bytes, max_value = struct.unpack_from("<IIIf", buf, pos)
    pos += 16
    scale = 2 * max_z_error
    out = np.full((height, width), np.nan, dtype=np.float32)
    valid = np.ones((height, width), dtype=bool) if mask is None else mask.reshape(height, width)

    for y0, y1 in _block_bounds(height, num_blocks_y):
        for x0, x1 in _block_bounds(width, num_blocks_x):
            block_valid = valid[y0:y1, x0:x1]
            header = buf[pos]
            pos += 1
            encoding = header & 63
            if encoding == 2:
                out[y0:y1, x0:x1][block_valid] = 0.0
                continue
            if encoding == 0 and header == 0:
                # Uncompressed block: the reference decoder only supports this as the
                # sole block, sized by the pixel section's byte count.
                n = (num_bytes - 1) // 4
                raw = np.frombuffer(buf, dtype="<f4", count=n, offset=pos)
                pos += 4 * n
                out[y0:y1, x0:x1][block_valid] = raw[: int(block_valid.sum())]
                continue
            offset_type = header >> 6
            if offset_type == 2:
                (offset,) = struct.unpack_from("<b", buf, pos)
                pos += 1
            elif offset_type == 1:
                (offset,) = struct.unpack_from("<h", buf, pos)
                pos += 2
            elif offset_type == 0:
                (offset,) = struct.unpack_from("<f", buf, pos)
                pos += 4
            else:
                raise ValueError("invalid LERC1 block offset type")
            if encoding == 3:
                out[y0:y1, x0:x1][block_valid] = offset
                continue
            if encoding != 1:
                raise ValueError(f"invalid LERC1 block encoding {encoding}")
            header2 = buf[pos]
            pos += 1
            bits_per_pixel = header2 & 63
            count_type = header2 >> 6
            fmt, size = {2: ("<B", 1), 1: ("<H", 2), 0: ("<I", 4)}[count_type]
            (count,) = struct.unpack_from(fmt, buf, pos)
            pos += size
            nbytes = (count * bits_per_pixel + 7) // 8
            n = _unstuff(buf[pos : pos + nbytes], bits_per_pixel, count)
            pos += nbytes
            nmax = np.ceil((max_value - offset) / scale)
            vals = np.where(n < nmax, offset + n * scale, max_value).astype(np.float32)
            out[y0:y1, x0:x1][block_valid] = vals

    return Lerc1Image(values=out, max_z_error=max_z_error, mask_section=mask_section)


def encode(image: Lerc1Image, block_size: int = 16) -> bytes:
    values = image.values
    height, width = values.shape
    valid = image.valid
    scale = 2 * image.max_z_error
    max_value = np.float32(np.nanmax(values)) if valid.any() else np.float32(0)

    num_blocks_y = max(1, height // block_size)
    num_blocks_x = max(1, width // block_size)
    blocks = bytearray()
    for y0, y1 in _block_bounds(height, num_blocks_y):
        for x0, x1 in _block_bounds(width, num_blocks_x):
            v = values[y0:y1, x0:x1][valid[y0:y1, x0:x1]]
            if v.size == 0:
                blocks.append(2)  # nothing valid; encoding "all zero" emits nothing
                continue
            offset = np.float32(v.min())
            n = np.rint((v.astype(np.float64) - float(offset)) / scale).astype(np.uint32)
            max_n = int(n.max())
            if max_n == 0:
                blocks += struct.pack("<Bf", 3, offset)
                continue
            bits_per_pixel = max_n.bit_length()
            blocks += struct.pack("<BfBI", 1, offset, bits_per_pixel, v.size)
            blocks += _stuff(n, bits_per_pixel)

    return b"".join(
        [
            _FILE_ID,
            struct.pack("<iiIId", _FILE_VERSION, _IMAGE_TYPE, height, width, image.max_z_error),
            image.mask_section,
            struct.pack("<IIIf", num_blocks_y, num_blocks_x, len(blocks), max_value),
            bytes(blocks),
        ]
    )
