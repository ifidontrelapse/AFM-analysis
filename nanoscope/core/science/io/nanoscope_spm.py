"""Nanoscope SPM parsing: header fields, calibration, and the Z array.

Moved verbatim from `src/afm_io.py` in M2-T04. This is the module the 22 unit
tests from M1-T06 exercise against a synthetic byte stream — it is the
best-covered code in the project, and the move changed none of it.

It still takes a path and opens the file twice (header, then a seek to the data
offset). That is legacy shape, not a design: a genuinely pure parser would take
bytes. Rewriting it means rewriting how the header is found, which is exactly the
kind of change the golden cannot see — `afm_io` has no phantom — and the unit
tests would then be validating the new shape rather than the old behaviour.
The `ImageLoader` port in M2-T08 is where that boundary gets drawn properly.
"""

from __future__ import annotations

import re

import numpy as np


def _read_nanoscope_z(file_path: str) -> np.ndarray:
    HEADER_READ_BYTES = 65536

    with open(file_path, "rb") as f:
        raw = f.read(HEADER_READ_BYTES)

    header = raw.split(b"\x1a")[0].decode("latin-1", errors="ignore")

    blocks = header.split("\\*Ciao image list")
    if len(blocks) < 2:
        raise ValueError("Ciao image list blocks not found")

    # Ищем блок Height явно, не просто первый блок
    blk = None
    for b in blocks[1:]:
        if '"Height"' in b:
            blk = b
            break
    if blk is None:
        blk = blocks[1]

    def find_int(pattern: str):
        m = re.search(pattern, blk)
        return int(m.group(1)) if m else None

    data_offset = find_int(r"Data offset\s*:\s*(\d+)")
    data_length = find_int(r"Data length\s*:\s*(\d+)")
    samps = find_int(r"Samps/line\s*:\s*(\d+)")
    lines = find_int(r"Number of lines\s*:\s*(\d+)")
    bpp = find_int(r"Bytes/pixel\s*:\s*(\d+)")

    if None in (data_offset, data_length, samps, lines, bpp):
        raise ValueError("Header fields missing in SPM file")

    # Число ПОСЛЕ скобок = реальный Z диапазон скана в вольтах
    zscale_match = re.search(r"@2:Z scale:[^\n]*\([^)]+\)\s*([\d.eE+-]+)\s*V", blk)
    if not zscale_match:
        raise ValueError("Z scale voltage not found")
    z_scale_v = float(zscale_match.group(1))  # 9.238140 V

    # Zsens — точный паттерн, не поймает ZsensSens
    zsens_match = re.search(r"@Sens\.\s*Zsens\s*:\s*V\s+([\d.eE+-]+)\s*nm/V", header)
    if not zsens_match:
        raise ValueError("Zsens nm/V not found")
    nm_per_v = float(zsens_match.group(1))  # 11.42934 nm/V

    z_scale = z_scale_v * nm_per_v / 65536

    dtype = np.int16 if bpp == 2 else np.int32

    with open(file_path, "rb") as f:
        f.seek(data_offset)
        raw_data = np.frombuffer(f.read(data_length), dtype=dtype)

    z = raw_data[: lines * samps].reshape((lines, samps)).astype(np.float32)
    z *= z_scale

    # В блоке изображения (blk) после нахождения Height
    scan_match = re.search(r"Scan Size:\s*([\d.]+)\s*([\d.]+)\s*(~m|nm|um|µm)", blk)
    if scan_match:
        scan_size = float(scan_match.group(1))
        unit = scan_match.group(3)
        if unit in ("~m", "um", "µm"):
            scan_size_nm = scan_size * 1000  # µm → нм
        else:
            scan_size_nm = scan_size  # уже в нм
    else:
        scan_size_nm = None

    pixel_size_nm = scan_size_nm / samps  # нм/пиксель

    return scan_size_nm, pixel_size_nm, z
