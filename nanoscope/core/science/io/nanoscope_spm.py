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

from nanoscope.core.errors import DataFormatError


def _read_nanoscope_z(file_path: str) -> tuple[float | None, float | None, np.ndarray]:
    """Decode a Bruker Nanoscope file into a calibrated height map.

    Args:
        file_path: path to the `.spm` / `.000`-style file

    Returns:
        `(scan_size_nm, pixel_size_nm, z)`. The first two are **`None` together**
        when the header states no `Scan Size` — an unknown scale is a state, not
        a crash and not a substitute value (ADR-0019, ADR-0025, ADR-0026). `z` is
        always a calibrated `float32` array in nanometres.

    Raises:
        ValueError: if the Ciao blocks, any required header field, the Z scale or
            the Z sensitivity is missing; if `Samps/line` is not positive; if the
            header states a non-positive `Scan Size`; or if the payload is
            shorter than `lines * samps`.
    """
    HEADER_READ_BYTES = 65536

    with open(file_path, "rb") as f:
        raw = f.read(HEADER_READ_BYTES)

    header = raw.split(b"\x1a")[0].decode("latin-1", errors="ignore")

    blocks = header.split("\\*Ciao image list")
    if len(blocks) < 2:
        raise DataFormatError("Ciao image list blocks not found")

    # Look for the Height block explicitly, not just the first one
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
        raise DataFormatError("Header fields missing in SPM file")

    # `samps` is the divisor two dozen lines below, and the reshape's row width.
    # Zero is a malformed header, and it used to surface as a ZeroDivisionError
    # from the same expression this task fixes (ADR-0026).
    if not samps > 0:
        raise DataFormatError(f"header states a non-positive Samps/line: {samps}")

    # The number AFTER the parentheses is the real Z range of the scan, in volts
    zscale_match = re.search(r"@2:Z scale:[^\n]*\([^)]+\)\s*([\d.eE+-]+)\s*V", blk)
    if not zscale_match:
        raise DataFormatError("Z scale voltage not found")
    z_scale_v = float(zscale_match.group(1))  # 9.238140 V

    # Zsens — an exact pattern, so it does not match ZsensSens
    zsens_match = re.search(r"@Sens\.\s*Zsens\s*:\s*V\s+([\d.eE+-]+)\s*nm/V", header)
    if not zsens_match:
        raise DataFormatError("Zsens nm/V not found")
    nm_per_v = float(zsens_match.group(1))  # 11.42934 nm/V

    z_scale = z_scale_v * nm_per_v / 65536

    dtype = np.int16 if bpp == 2 else np.int32

    with open(file_path, "rb") as f:
        f.seek(data_offset)
        raw_data = np.frombuffer(f.read(data_length), dtype=dtype)

    z = raw_data[: lines * samps].reshape((lines, samps)).astype(np.float32)
    z *= z_scale

    # In the image block (blk), after Height has been located
    scan_match = re.search(r"Scan Size:\s*([\d.]+)\s*([\d.]+)\s*(~m|nm|um|µm)", blk)
    if scan_match:
        scan_size = float(scan_match.group(1))
        unit = scan_match.group(3)
        if unit in ("~m", "um", "µm"):
            scan_size_nm = scan_size * 1000  # µm -> nm
        else:
            scan_size_nm = scan_size  # already in nm
        # Stated and impossible is not the same as absent: `Scan Size: 0 0 nm`
        # would make every physical value zero rather than unknown (ADR-0026).
        if not scan_size_nm > 0:
            raise DataFormatError(f"header states a non-positive Scan Size: {scan_size_nm} nm")
    else:
        scan_size_nm = None

    # The scale is unknown, which is a state the whole pipeline now carries
    # (ADR-0025). This line used to divide `None` by `samps` — the fallback
    # crashed on the branch it had just taken.
    pixel_size_nm = None if scan_size_nm is None else scan_size_nm / samps

    return scan_size_nm, pixel_size_nm, z
