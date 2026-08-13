"""Load a raw AFM file and level it into arrays `run_pipeline` can use.

Moved from `src/preprocessing_pipeline.py` in M2-T15. `application`, for the same
reason as its neighbour: it sequences a loader from `infrastructure.storage` and
three steps from `core.science`, and owns none of them.

It had no caller in the repository, which is why the audit called it dead. It is
the documented preprocessing entry point in `README` and `Development.md`, and
M2-T13 kept it deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nanoscope.core.entities import ImageRecord, PreprocessingResult
from nanoscope.core.errors import UnsupportedRequestError
from nanoscope.core.ports import ProjectRepository
from nanoscope.core.science.preprocessing import (
    DEFAULT_OPENING_SCALE,
    build_substrate_map,
    flatten_lines,
    flatten_plane,
)
from nanoscope.core.values import Modality
from nanoscope.infrastructure.storage import load_afm

#: The defaults this module passes on, named so a caller — a panel, in M6-T01 —
#: can show the value it will actually get instead of repeating the number. The
#: rule for the whole of M6 is that *the UI must not introduce its own defaults*,
#: and a literal typed into a spin box is exactly that.
#: `DEFAULT_MIN_SIZE_NM` mirrors `build_substrate_map`'s own default, which is a
#: bare literal in a science signature this task does not rewrite
#: (PROJECT_RULES §4.1); a test asserts the two agree.
DEFAULT_MIN_SIZE_NM = 5.0


@dataclass(frozen=True)
class PreprocessingParams:
    """The three numbers the substrate step takes, as one thing to pass around.

    One object rather than three keyword arguments threaded through three layers
    (M6-T02): the detection run and the preview must use the *same* numbers, and
    the way they stay the same is that there is one value to hand over.
    """

    min_size_nm: float = DEFAULT_MIN_SIZE_NM
    manual_radius_px: float | None = None
    opening_scale: float = DEFAULT_OPENING_SCALE


#: What each AFM extension is called on the way into `load_afm`. Here rather
#: than in `analysis.py`, which owned it until M6-T01: this is the module that
#: loads AFM files, and a second caller was about to copy the mapping.
AFM_FORMATS = {".spm": "spm", ".npy": "npy"}


def run_preprocessing(
    file_path: str | Path,
    fmt: str = "spm",
    pixel_size_nm: float | None = None,
    *,
    min_size_nm: float = DEFAULT_MIN_SIZE_NM,
    manual_radius_px: float | None = None,
    opening_scale: float = DEFAULT_OPENING_SCALE,
) -> PreprocessingResult:
    """
    Load and preprocess a raw AFM file.

    Steps:
        1. load_afm            — read file, extract z and pixel_size_nm
        2. flatten_plane       — remove global tilt (least-squares plane)
        3. flatten_lines       — row-by-row linear detrending
        4. build_substrate_map — morphological opening to estimate substrate,
                                 compute z_result = z_flat - substrate,
                                 estimate particle radii via Otsu

    All parameters use library defaults — no manual tuning required. The three
    added in M6-T01 are **pass-through, with the values this function already
    used**: a panel that offers them must not become a second place where a
    default lives (`docs/Roadmap.md`, M6: *the UI must not introduce its own
    defaults*), and the golden proves the defaults did not move.

    Args:
        file_path: path to the AFM file
        fmt:       file format passed to load_afm ("spm" or "npy")
        pixel_size_nm: nm/pixel for an "npy", whose file carries no metadata at
                   all. `None` leaves the scale unknown, which is a state and not
                   a reason to invent 1.0 (ADR-0025). **Ignored for "spm"**,
                   where the header is the source — added in M4-T05, when a
                   project that *knew* an npy's scale was found analysing it as
                   though it did not, producing `radius_nm=None` for every
                   particle and skipping the physical minimum-size filter
        min_size_nm: the smallest particle radius that counts, in nanometres —
                   a physical size at both of its sites since ADR-0024
        manual_radius_px: the opening radius to use, in pixels. When given it
                   **is** the radius, and the estimate is not consulted at all
                   (ADR-0014)
        opening_scale: multiplier on the Otsu typical radius. Measured in
                   ADR-0037: smaller finds more particles in a dense field,
                   larger measures radii better

    Returns:
        PreprocessingResult with all arrays and metadata
    """
    raw = load_afm(str(file_path), fmt=fmt, pixel_size_nm=pixel_size_nm)

    z_plane = flatten_plane(raw.z_raw)
    z_flat = flatten_lines(z_plane)

    substrate, z_result, opening_radius, sizes = build_substrate_map(
        z_flat,
        raw.pixel_size_nm,
        min_size_nm=min_size_nm,
        manual_radius_px=manual_radius_px,
        opening_scale=opening_scale,
    )

    return PreprocessingResult(
        z_raw=raw.z_raw,
        z_flat=z_flat,
        z_result=z_result,
        substrate=substrate,
        pixel_size_nm=raw.pixel_size_nm,
        scan_size_nm=raw.scan_size_nm,
        sizes=sizes,
        opening_radius=opening_radius,
    )


def afm_format(path: Path) -> str:
    """The `fmt` string `load_afm` expects, from the file's own extension.

    Raises:
        UnsupportedRequestError: an extension with no AFM reader. Nothing about
            the request is malformed — this version has no path for that file.
    """
    fmt = AFM_FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise UnsupportedRequestError(
            f"no AFM reader for {path.name}; supported extensions are "
            f"{', '.join(sorted(AFM_FORMATS))}"
        )
    return fmt


def preprocess_image(
    repository: ProjectRepository,
    image_id: int,
    params: PreprocessingParams | None = None,
) -> PreprocessingResult:
    """Preprocess one of a project's images, by id (M6-T01).

    The same resolution `run_analysis` does — the row, the path through the
    repository, and **the scale the project recorded** — so a panel hands over an
    id rather than assembling a path, a format and a scale of its own. That
    assembly is where M4-T05 found the D-07 family of defect reintroduced one
    layer up.

    Nothing is stored: a preview is a look at intermediate arrays, and a run is
    what `run_analysis` records (ADR-0042, ADR-0061).

    Raises:
        InvalidParameterError: no image has that id.
        UnsupportedRequestError: the row is not AFM, or its extension has no
            reader. SEM and TEM have no substrate to build — they are analysed
            as they are (ADR-0031).
        MissingFileError: the file is gone (ADR-0040, from the other side).
    """
    params = params or PreprocessingParams()
    record: ImageRecord = repository.get_image(image_id)
    if record.modality is not Modality.AFM:
        raise UnsupportedRequestError(
            f"{record.display_name} is {record.modality}; preprocessing levels an AFM height "
            "map, and an SEM or TEM image is analysed as it is"
        )

    path = repository.path_of(record)
    return run_preprocessing(
        path,
        fmt=afm_format(path),
        pixel_size_nm=record.pixel_size_nm,
        min_size_nm=params.min_size_nm,
        manual_radius_px=params.manual_radius_px,
        opening_scale=params.opening_scale,
    )
