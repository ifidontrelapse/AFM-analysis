"""Characterization tests for the SPM parser and the image loader.

The Nanoscope reader is hand-written binary parsing, and it owns the
`scan_size_nm -> pixel_size_nm` calibration that every `_nm` value in the project
is derived from. It had no test at all before M1-T06.

**These tests record what the parser does today, not what it should do.** Where
today's behaviour is a known defect the assertion says so and names the task that
fixes it; the fix then flips a documented assertion instead of breaking a
surprise. Nothing here edits `src/`.

The fixture is a synthetic SPM byte stream built by `_spm_bytes`, derived from a
real Bruker Nanoscope header (`data/pvp8k/2-6-dmfa-pvp.039`, 512x512, 3 um,
Zsens 11.43219 nm/V) — read locally, not committed. No binary fixture enters git
(PROJECT_RULES §7) and no test touches `data/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nanoscope.application.use_cases.preprocessing import afm_format
from nanoscope.core.errors import UnsupportedRequestError
from nanoscope.infrastructure.storage import load_afm, load_microscopy_image

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


def _height_block(
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


def _spm_bytes(
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
    block = _height_block(data_length=len(payload), bytes_per_pixel=bytes_per_pixel, **block_kwargs)

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


def _write_spm(tmp_path, z_lsb: np.ndarray, **kwargs: object):
    path = tmp_path / "synthetic.spm"
    path.write_bytes(_spm_bytes(z_lsb, **kwargs))
    return path


@pytest.fixture
def z_lsb() -> np.ndarray:
    """A field whose every element is unique, so orientation is observable."""
    return np.arange(LINES * SAMPS, dtype=np.int32).reshape(LINES, SAMPS) * 100 - 500


# ── the round trip ────────────────────────────────────────────────────────────


def test_spm_round_trip_preserves_shape_dtype_and_orientation(tmp_path, z_lsb) -> None:
    data = load_afm(str(_write_spm(tmp_path, z_lsb)), fmt="spm")

    assert data.z_raw.shape == (LINES, SAMPS)  # [y, x] — PROJECT_RULES §3
    assert data.z_raw.dtype == np.float32
    np.testing.assert_allclose(data.z_raw, z_lsb * LSB_TO_NM, rtol=1e-6)


def test_spm_z_scaling_uses_z_scale_volts_times_zsens_over_65536(tmp_path) -> None:
    """The volt->nanometre chain, pinned end to end on a single known pixel."""
    z_lsb = np.full((LINES, SAMPS), 1000, dtype=np.int32)

    data = load_afm(str(_write_spm(tmp_path, z_lsb)), fmt="spm")

    expected_nm = 1000 * Z_SCALE_V * NM_PER_V / 65536  # 1.1338 nm
    assert data.z_raw[0, 0] == pytest.approx(expected_nm, rel=1e-6)


def test_spm_reads_the_height_block_not_the_first_block(tmp_path, z_lsb) -> None:
    """The file's first image block is a decoy with a different geometry."""
    data = load_afm(str(_write_spm(tmp_path, z_lsb)), fmt="spm")

    assert data.z_raw.shape == (LINES, SAMPS)
    assert data.scan_size_nm == 3000.0  # the decoy claims 1 nm


def test_spm_reads_int32_when_bytes_per_pixel_is_not_2(tmp_path) -> None:
    z_lsb = np.arange(LINES * SAMPS, dtype=np.int32).reshape(LINES, SAMPS) * 70000

    data = load_afm(str(_write_spm(tmp_path, z_lsb, bytes_per_pixel=4)), fmt="spm")

    np.testing.assert_allclose(data.z_raw, z_lsb * LSB_TO_NM, rtol=1e-6)


# ── calibration ───────────────────────────────────────────────────────────────


def test_spm_pixel_size_is_scan_size_divided_by_samps_per_line(tmp_path, z_lsb) -> None:
    """Every physical unit in the project descends from this one division."""
    data = load_afm(str(_write_spm(tmp_path, z_lsb)), fmt="spm")

    assert data.scan_size_nm == 3000.0
    assert data.pixel_size_nm == pytest.approx(3000.0 / SAMPS)
    assert data.pixel_size_nm == pytest.approx(data.scan_size_nm / SAMPS)


@pytest.mark.parametrize(
    ("scan_size_line", "expected_nm"),
    [
        (r"\Scan Size: 3 3 ~m", 3000.0),  # '~m' is how Nanoscope writes 'µm'
        (r"\Scan Size: 3 3 um", 3000.0),
        (r"\Scan Size: 3 3 µm", 3000.0),
        (r"\Scan Size: 500 500 nm", 500.0),
    ],
)
def test_spm_scan_size_units_are_converted_to_nanometres(
    tmp_path, z_lsb, scan_size_line: str, expected_nm: float
) -> None:
    data = load_afm(str(_write_spm(tmp_path, z_lsb, scan_size_line=scan_size_line)), fmt="spm")

    assert data.scan_size_nm == pytest.approx(expected_nm)
    assert data.pixel_size_nm == pytest.approx(expected_nm / SAMPS)


# ── failure modes ─────────────────────────────────────────────────────────────


def test_spm_without_scan_size_reports_an_unknown_scale(tmp_path, z_lsb) -> None:
    """**M3-T17 / ADR-0026** — the defect this test used to pin.

    The `else` branch exists to handle a header with no `Scan Size:` field. It
    set `scan_size_nm = None` and the very next line divided by `samps`, so the
    fallback crashed on the branch it had just taken. PROJECT_RULES §3 says an
    unknown scale is `None`, never a crash — and since ADR-0025 that state has a
    meaning everywhere downstream, so the height map is still usable.
    """
    path = _write_spm(tmp_path, z_lsb, scan_size_line=None)

    data = load_afm(str(path), fmt="spm")

    assert data.scan_size_nm is None
    assert data.pixel_size_nm is None
    assert data.z_raw.shape == (LINES, SAMPS)  # the array is unaffected
    assert data.z_raw.dtype == np.float32


def test_spm_with_a_zero_samps_per_line_is_rejected(tmp_path, z_lsb) -> None:
    """The divisor on the same line. Zero is a malformed header rather than an
    unknown scale, and it used to be a `ZeroDivisionError` out of the expression
    M3-T17 fixes — an error that names nothing (ADR-0026, PROJECT_RULES §3)."""
    path = _write_spm(tmp_path, z_lsb, samps=0)

    with pytest.raises(ValueError, match="non-positive Samps/line"):
        load_afm(str(path), fmt="spm")


def test_spm_with_a_stated_scan_size_of_zero_is_rejected(tmp_path, z_lsb) -> None:
    """Stated and impossible is not the same as absent: `Scan Size: 0 0 nm`
    would make every physical value zero rather than unknown. Same rule as the
    npy loader's (ADR-0025), so the two loaders agree."""
    path = _write_spm(tmp_path, z_lsb, scan_size_line=r"\Scan Size: 0 0 nm")

    with pytest.raises(ValueError, match="non-positive Scan Size"):
        load_afm(str(path), fmt="spm")


def test_spm_missing_file_raises_filenotfound(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_afm(str(tmp_path / "absent.spm"), fmt="spm")


def test_spm_without_ciao_blocks_is_rejected(tmp_path, z_lsb) -> None:
    path = _write_spm(tmp_path, z_lsb, ciao_blocks=False)

    with pytest.raises(ValueError, match="Ciao image list blocks not found"):
        load_afm(str(path), fmt="spm")


def test_spm_with_a_missing_header_field_is_rejected(tmp_path, z_lsb) -> None:
    path = _write_spm(tmp_path, z_lsb, samps_line=False)

    with pytest.raises(ValueError, match="Header fields missing"):
        load_afm(str(path), fmt="spm")


def test_spm_without_z_scale_is_rejected(tmp_path, z_lsb) -> None:
    path = _write_spm(tmp_path, z_lsb, z_scale_line=None)

    with pytest.raises(ValueError, match="Z scale voltage not found"):
        load_afm(str(path), fmt="spm")


def test_spm_without_zsens_is_rejected_and_zsenssens_is_not_a_substitute(tmp_path, z_lsb) -> None:
    """The fixture still carries `@Sens. ZsensSens: V 351.8693 nm/V`.

    It is 30x the real sensitivity, so accepting it would scale every height in
    the scan by 30 and nothing would crash — a wrong number, delivered
    confidently. Refusing to load is the correct answer, and this is the
    assertion that keeps the Zsens pattern narrow.
    """
    path = _write_spm(tmp_path, z_lsb, nm_per_v_line=None)

    with pytest.raises(ValueError, match="Zsens nm/V not found"):
        load_afm(str(path), fmt="spm")


def test_spm_truncated_payload_fails_instead_of_returning_a_short_map(tmp_path, z_lsb) -> None:
    """A half-written scan must not silently become a smaller image.

    The parser slices `[:lines * samps]` before reshaping, so a short payload
    reaches `reshape` and raises there. The message is NumPy's, not the
    project's — that is D-15, fixed by the error taxonomy in M3-T13.
    """
    path = _write_spm(tmp_path, z_lsb, truncate_payload=8)

    with pytest.raises(ValueError, match="reshape"):
        load_afm(str(path), fmt="spm")


def test_unsupported_format_is_rejected_by_name(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported format: gwy"):
        load_afm(str(tmp_path / "whatever.gwy"), fmt="gwy")


# ── the npy path ──────────────────────────────────────────────────────────────


def test_npy_uses_the_metadata_it_is_given(tmp_path) -> None:
    z = np.arange(12, dtype=np.float64).reshape(3, 4)
    path = tmp_path / "z.npy"
    np.save(path, z)

    data = load_afm(str(path), fmt="npy", pixel_size_nm=2.5, scan_size_nm=10.0)

    assert data.z_raw.dtype == np.float32  # cast, even from float64
    np.testing.assert_allclose(data.z_raw, z)
    assert data.pixel_size_nm == 2.5
    assert data.scan_size_nm == 10.0


def test_npy_without_metadata_reports_an_unknown_scale(tmp_path) -> None:
    """**M3-T20 / ADR-0025** — the defect this test used to characterize.

    It fabricated `1.0` nm/px and a scan size equal to the **row count**, so
    every downstream `_nm` became a pixel count wearing nanometre units and no
    consumer could tell. PROJECT_RULES §3 and D-07 both say an unknown physical
    scale is `None`, never a substitute value.
    """
    z = np.zeros((7, 3), dtype=np.float32)
    path = tmp_path / "z.npy"
    np.save(path, z)

    data = load_afm(str(path), fmt="npy")

    assert data.pixel_size_nm is None
    assert data.scan_size_nm is None
    assert data.z_raw.shape == (7, 3)  # the array still loads; only the metadata is absent


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_npy_refuses_a_scale_that_is_not_a_size(tmp_path, bad: float) -> None:
    """The other half of M3-T20: `or` swallowed an explicit `0.0` the same way
    it swallowed `None`, so a caller who meant zero was silently given 1.0.
    Zero is not "unknown" — it is a caller error, and `PixelScale` has said so
    since M2-T02. `nan` fails the same `not value > 0`, deliberately."""
    path = tmp_path / "z.npy"
    np.save(path, np.zeros((4, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="pixel_size_nm must be positive"):
        load_afm(str(path), fmt="npy", pixel_size_nm=bad)
    with pytest.raises(ValueError, match="scan_size_nm must be positive"):
        load_afm(str(path), fmt="npy", scan_size_nm=bad)


def test_npy_keeps_a_known_scale_and_an_unknown_one_apart(tmp_path) -> None:
    """One of the two may be known — an operator who knows the pixel size need
    not also know the scan size. Nothing is derived from the other: see ADR-0025
    on why `pixel_size_nm * z.shape[0]` is not filled in here."""
    path = tmp_path / "z.npy"
    np.save(path, np.zeros((4, 6), dtype=np.float32))

    data = load_afm(str(path), fmt="npy", pixel_size_nm=2.5)

    assert data.pixel_size_nm == 2.5
    assert data.scan_size_nm is None


# ── SEM / TEM ─────────────────────────────────────────────────────────────────


def test_microscopy_image_loads_as_greyscale(tmp_path) -> None:
    cv2 = pytest.importorskip("cv2")
    image = np.arange(24, dtype=np.uint8).reshape(4, 6) * 10
    path = tmp_path / "grain.png"
    cv2.imwrite(str(path), image)

    data = load_microscopy_image(str(path), modality="tem", nm_per_pixel=0.5)

    assert data.image.shape == (4, 6)  # [y, x]
    assert data.image.dtype == np.uint8
    np.testing.assert_array_equal(data.image, image)
    assert data.modality == "tem"
    assert data.nm_per_pixel == 0.5


def test_microscopy_image_scale_stays_none_when_unknown(tmp_path) -> None:
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "grain.png"
    cv2.imwrite(str(path), np.zeros((4, 4), dtype=np.uint8))

    data = load_microscopy_image(str(path), modality="sem")

    assert data.nm_per_pixel is None  # §3: unknown scale is None, never 0


def test_microscopy_image_missing_file_raises_filenotfound(tmp_path) -> None:
    """cv2.imread returns None rather than raising; the wrapper must translate."""
    with pytest.raises(FileNotFoundError, match="Could not read image"):
        load_microscopy_image(str(tmp_path / "absent.png"), modality="sem")


# ── Which files are AFM files (2026-08-30) ───────────────────────────────────
#
# A Bruker Nanoscope does not write `.spm`. It writes `scan.000`, `scan.001`,
# `scan.002` — the number is the acquisition, not the format — and this parser
# has read them since it was written; the docstring of `_read_nanoscope_z` says
# "`.spm` / `.000`-style" in as many words.
#
# The extension map in front of it did not, and it was written **twice**: once
# in `use_cases/preprocessing.py` and once, copied, in `use_cases/display.py`.
# So a folder straight off the microscope imported fine, appeared in the project
# explorer, and refused to open. Found by an operator, not by 1400 tests.


@pytest.mark.parametrize(
    ("name", "fmt"),
    [
        ("scan.spm", "spm"),
        ("scan.SPM", "spm"),
        ("scan.npy", "npy"),
        ("scan.000", "spm"),
        ("scan.001", "spm"),
        ("scan.017", "spm"),
        ("scan.3", "spm"),
        # The instrument's own naming: dots in the stem, the number last.
        ("2-2-dmfa-citr-temp.014", "spm"),
    ],
)
def test_an_afm_extension_names_its_reader(name: str, fmt: str) -> None:
    assert afm_format(Path(name)) == fmt


@pytest.mark.parametrize(
    "name",
    [
        "scan.jpg",
        "scan.tif",
        "scan.png",
        # A JPEG *export* of an acquisition — the number is in the stem and the
        # extension is what it is. Refusing this is the whole point of looking.
        "2-2-dmfa-citr-temp.000.jpg",
        "scan",
        "scan.",
        # `str.isdigit` is true of Arabic-Indic digits; this is not a Nanoscope
        # acquisition number. Escaped rather than written, because ruff is right
        # that the literal is unreadable (RUF001).
        "scan.\u0660\u0660\u0661",
    ],
)
def test_a_file_with_no_afm_reader_is_refused_by_name(name: str) -> None:
    with pytest.raises(UnsupportedRequestError) as refusal:
        afm_format(Path(name))

    # The message has to name the numbered form, or an operator holding `.003`
    # reads "supported extensions are .npy, .spm" as "yours is not supported".
    assert ".000" in str(refusal.value)


def test_a_numbered_nanoscope_file_actually_reads(tmp_path, z_lsb) -> None:
    """The half a name check cannot make: the bytes go through the parser.

    Same synthetic file as every test above, under the name the instrument
    would have given it.
    """
    path = tmp_path / "2-2-dmfa-citr-temp.001"
    path.write_bytes(_spm_bytes(z_lsb))

    raw = load_afm(str(path), fmt=afm_format(path))

    assert raw.z_raw.shape == (LINES, SAMPS)
    assert raw.pixel_size_nm is not None
