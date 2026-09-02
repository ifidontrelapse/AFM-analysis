"""A synthetic Bruker Nanoscope file, built once for every test that needs one.

Lifted out of `unit/test_afm_io.py`, which built it for M1-T06 and was its only
user until an import, a preview pane and a thumbnail all needed a file with a
**real header** — one that states a scan size, because the scale it states is the
thing under test (ADR-0083). Two builders would have been two headers, and a test
proving the import reads the scale off a header nobody else writes proves less
than it looks like.

No binary fixture enters git (PROJECT_RULES §7): this writes the bytes.

Derived from a real header — `data/pvp8k/2-6-dmfa-pvp.039`, 512x512, 3 um,
Zsens 11.43219 nm/V — read locally and never committed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# The real header puts the payload at 40960. Any fixed offset larger than the
# synthetic header works; the parser seeks to whatever the header declares.
DATA_OFFSET = 4096

# Taken verbatim from the reference file's Height block.
Z_SCALE_V = 6.498924
NM_PER_V = 11.43219
LSB_TO_NM = Z_SCALE_V * NM_PER_V / 65536  # the parser's own arithmetic

# Deliberately non-square: a transposed read cannot survive `reshape`, and a
# lines/samps mix-up in the calibration shows up as a wrong pixel size.
LINES = 6
SAMPS = 4


def height_block(
    *,
    samps: int = SAMPS,
    lines: int = LINES,
    data_length: int,
    bytes_per_pixel: int = 2,
    scan_size_line: str | None = r"\Scan Size: 3 3 ~m",
    z_scale_line: str | None = (rf"\@2:Z scale: V [Sens. Zsens] (0.006713765 V/LSB) {Z_SCALE_V} V"),
    samps_line: bool = True,
) -> str:
    lines_out = [
        r"\*Ciao image list",
        rf"\Data offset: {DATA_OFFSET}",
        rf"\Data length: {data_length}",
        rf"\Bytes/pixel: {bytes_per_pixel}",
        r"\Data type: AFM",
        rf"\Number of lines: {lines}",
        r"\Aspect Ratio: 1:1",
        r"\@2:Image Data: S [Height] " + '"Height"',
    ]
    if samps_line:
        lines_out.insert(5, rf"\Samps/line: {samps}")
    if scan_size_line is not None:
        lines_out.append(scan_size_line)
    if z_scale_line is not None:
        lines_out.append(z_scale_line)
    return "\n".join(lines_out)


def spm_bytes(
    z_lsb: np.ndarray,
    *,
    bytes_per_pixel: int = 2,
    nm_per_v_line: str | None = rf"\@Sens. Zsens: V {NM_PER_V} nm/V",
    truncate_payload: int = 0,
    ciao_blocks: bool = True,
    **block_kwargs: object,
) -> bytes:
    """Build a minimal Nanoscope SPM file around a known integer Z field.

    The parser reads the header as everything before the first ``0x1A`` byte,
    then seeks to the declared data offset. The gap between the two is padded, as
    it is in a real file.

    Args:
        z_lsb: raw integer Z values, shape ``(lines, samps)``, index order
            ``[y, x]``.
        bytes_per_pixel: 2 writes ``int16``, anything else writes ``int32`` —
            the parser's own rule.
        nm_per_v_line: the ``@Sens. Zsens`` line, or None to omit it.
        truncate_payload: drop this many bytes from the end of the payload while
            leaving ``Data length`` claiming the full size.
        ciao_blocks: False strips the image-list markers entirely.
        **block_kwargs: forwarded to `_height_block`.

    Returns:
        The complete file contents.
    """
    dtype = "<i2" if bytes_per_pixel == 2 else "<i4"
    payload = np.asarray(z_lsb, dtype=dtype).tobytes()

    # A decoy block first: if block selection ever stops looking for "Height",
    # the shape and the values it reads change, and these tests go red.
    decoy = "\n".join(
        [
            r"\*Ciao image list",
            r"\Data offset: 999999",
            r"\Data length: 8",
            r"\Bytes/pixel: 2",
            r"\Samps/line: 2",
            r"\Number of lines: 2",
            r"\Scan Size: 1 1 nm",
            r"\@2:Image Data: S [ZSensor] " + '"Deflection Error"',
            r"\@2:Z scale: V [Sens. Zsens] (0.1 V/LSB) 1.0 V",
        ]
    )
    block = height_block(data_length=len(payload), bytes_per_pixel=bytes_per_pixel, **block_kwargs)

    preamble = [r"\*File list", r"\Version: 0x09400202"]
    if nm_per_v_line is not None:
        preamble.append(nm_per_v_line)
    # Always present, exactly as in a real header: a second sensitivity 30x the
    # first, one character away from matching the Zsens pattern. It stays in the
    # file even when the real Zsens line is dropped — that is the case where a
    # loosened regex would silently substitute it.
    preamble.append(r"\@Sens. ZsensSens: V 351.8693 nm/V")

    header = "\n".join([*preamble, decoy, block, r"\*File list end", ""])
    if not ciao_blocks:
        header = header.replace(r"\*Ciao image list", r"\*Some other list")

    raw = header.encode("latin-1") + b"\x1a"
    assert len(raw) < DATA_OFFSET, "synthetic header outgrew the declared data offset"
    return raw + b"\x00" * (DATA_OFFSET - len(raw)) + payload[: len(payload) - truncate_payload]


def write_spm(directory: Path, z_lsb: np.ndarray, *, name: str = "synthetic.spm", **kwargs) -> Path:
    """Write one into `directory` and hand back the path.

    Args:
        directory: where to put it.
        z_lsb: raw integer Z values, shape ``(lines, samps)``.
        name: the file name — `scan.000` is what the instrument actually writes,
            and a test about the numbered extensions needs to say so.
        **kwargs: forwarded to `spm_bytes`.

    Returns:
        The path written.
    """
    path = directory / name
    path.write_bytes(spm_bytes(z_lsb, **kwargs))
    return path


def z_field(lines: int = LINES, samps: int = SAMPS) -> np.ndarray:
    """A field whose every element is unique, so orientation is observable."""
    return np.arange(lines * samps, dtype=np.int32).reshape(lines, samps) * 100 - 500
