"""Turning a stored image into something a widget can draw (M5-T05, ADR-0056).

Two steps, deliberately separate: **loading** an image (which touches a disk) and
**rendering** it (which needs a colormap). Both live here rather than in `gui/`
for the same reason: the widget may not import `infrastructure`, and deciding
how a value becomes a colour is not a widget's decision anyway.

**What is loaded is what is in the file.** A tilted AFM map is harder to read
than a flattened one, and every SPM tool flattens for display — but flattening is
an *analysis*, `flatten_plane` has an ADR, and its output is what a run records.
A viewer that silently flattens shows something that is not in the file, and an
operator comparing it against a measurement compares two different arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

import numpy as np

from nanoscope.application.use_cases.preprocessing import afm_format
from nanoscope.core.entities import PreprocessingResult
from nanoscope.core.errors import InvalidParameterError, UnsupportedRequestError
from nanoscope.core.ports import ProjectRepository
from nanoscope.core.values import Modality
from nanoscope.infrastructure.imaging.colormap import afm_to_rgb
from nanoscope.infrastructure.storage import load_afm, load_microscopy_image

#: What a viewer may offer. Matplotlib knows hundreds; these are the ones an SPM
#: operator expects, and a list of hundreds is a list nobody reads. `afmhot` is
#: first because it is the convention for height maps.
COLORMAPS: tuple[str, ...] = ("afmhot", "gray", "viridis", "magma", "cividis", "bone")

#: The default window: a single hot pixel otherwise flattens the whole image to
#: grey, and percentiles are what every SPM tool does.
DEFAULT_PERCENTILES = (2.0, 98.0)


@dataclass(frozen=True)
class DisplayImage:
    """One image, ready to be looked at — and the numbers that make it data.

    `pixel_size_nm` is `None` when the scale is unknown, and every consumer has
    to say so rather than invent one. ADR-0025 spent a milestone on
    absent-not-fabricated, and a viewer writing "1 nm/px" would undo it in a
    line.
    """

    name: str
    #: The array as it is in the file: 2-D, floating point for AFM, greyscale
    #: for SEM/TEM. Never flattened, never rescaled (ADR-0056).
    data: np.ndarray
    modality: Modality
    pixel_size_nm: float | None

    @property
    def size_px(self) -> tuple[int, int]:
        """`(width, height)`, in the order a widget wants them."""
        height, width = self.data.shape[:2]
        return width, height

    @property
    def size_nm(self) -> tuple[float, float] | None:
        """The scan's physical size, or `None` when the scale is unknown."""
        if self.pixel_size_nm is None:
            return None
        width, height = self.size_px
        return width * self.pixel_size_nm, height * self.pixel_size_nm


def load_for_display(repository: ProjectRepository, image_id: int) -> DisplayImage:
    """Read one of the project's images off the disk, as it is.

    Raises:
        InvalidParameterError: no image has that id.
        UnsupportedRequestError: its extension has no AFM reader. **Through
            `afm_format`, not a copy of it** — this module kept its own map of
            extensions until an operator imported a folder of `scan.000` files
            and watched the viewer refuse every one, while the analysis path
            beside it would have read them. Two lists of the same thing are one
            list and one bug.
        MissingFileError: the file is gone — the dangling row `check_integrity`
            reports (ADR-0040), met from the viewer's side.
    """
    record = repository.get_image(image_id)
    return load_file_for_display(
        repository.path_of(record),
        modality=record.modality,
        #: The project's recorded scale, not the file's — an `.npy` has none,
        #: and M4-T05 was the task that learned to pass it through. An SPM
        #: header wins over it inside `load_afm`, which is the whole of
        #: ADR-0083 seen from this side.
        pixel_size_nm=record.pixel_size_nm,
        name=record.display_name,
    )


def load_file_for_display(
    path: Path | str,
    *,
    modality: Modality | None = None,
    pixel_size_nm: float | None = None,
    name: str | None = None,
) -> DisplayImage:
    """Read *any* file on disk the way the viewer reads a project's image.

    Split out of `load_for_display` so a file that is **not in a project yet**
    can be looked at — the import preview, which is the one surface where an
    operator has a path and no row.

    Args:
        path: the file, anywhere on disk.
        modality: which reader to use. `None` means *nobody has said yet*, and
            then the **extension decides**: what `afm_format` accepts is read as
            AFM, everything else as an SEM/TEM greyscale. That guess is
            legitimate for a picture and for nothing else — what a project
            **records** is what the operator stated in the import dialog, and
            this function never writes anything.
        pixel_size_nm: the scale to use where the file carries none. Ignored for
            an SPM, whose header states its own (ADR-0083).
        name: what to call it; the file's own name by default.

    Returns:
        The array, its modality and its scale — `None` when nothing states one.

    Raises:
        UnsupportedRequestError: `modality` is AFM and the extension has no AFM
            reader.
        MissingFileError: there is no readable file there.
    """
    file = Path(path)
    display_name = name or file.name
    if modality is None:
        modality = Modality.AFM if _has_afm_reader(file) else Modality.SEM

    if modality is Modality.AFM:
        raw = load_afm(str(file), fmt=afm_format(file), pixel_size_nm=pixel_size_nm)
        return DisplayImage(
            name=display_name,
            data=raw.z_raw,
            modality=modality,
            pixel_size_nm=raw.pixel_size_nm,
        )

    image = load_microscopy_image(
        str(file),
        modality=modality.value,  # type: ignore[arg-type]  # M2-T10 adopts the enum
        nm_per_pixel=pixel_size_nm,
    )
    return DisplayImage(
        name=display_name,
        data=image.image,
        modality=modality,
        pixel_size_nm=image.nm_per_pixel,
    )


def _has_afm_reader(path: Path) -> bool:
    """Whether `afm_format` would dispatch this extension to an AFM reader.

    Asking the one function that owns the map, rather than keeping a second
    copy of it — the mistake this module already made once and paid for
    (`load_for_display`'s docstring says how).
    """
    try:
        afm_format(path)
    except UnsupportedRequestError:
        return False
    return True


def value_range(image: DisplayImage, *, full: bool = False) -> tuple[float, float]:
    """The window to map colours over: percentiles, or everything.

    Percentiles by default because one hot pixel otherwise flattens the image to
    grey; `full=True` because *"what am I clipping?"* is a question an operator
    must be able to answer.
    """
    finite = image.data[np.isfinite(image.data)]
    if finite.size == 0:  # pragma: no cover — a map with no finite value at all
        return 0.0, 1.0
    if full:
        return float(finite.min()), float(finite.max())
    low, high = np.percentile(finite, DEFAULT_PERCENTILES)
    return (float(low), float(high)) if high > low else (float(finite.min()), float(finite.max()))


def render(
    image: DisplayImage,
    colormap: str = COLORMAPS[0],
    limits: tuple[float, float] | None = None,
) -> np.ndarray:
    """The image as `uint8` RGB, ready for a widget to wrap in a `QImage`.

    The colormap is applied here, not in the widget: it lives in
    `infrastructure.imaging` (matplotlib), which `gui/` may not import — and how
    a value becomes a colour is not a widget's decision.

    Args:
        image: what to draw.
        colormap: one of `COLORMAPS`.
        limits: the value window; `value_range(image)` when omitted.

    Returns:
        `(height, width, 3)` of `uint8`.

    Raises:
        InvalidParameterError: a colormap this application does not offer. A
            typo in a settings file must produce a message, not a matplotlib
            traceback.
    """
    if colormap not in COLORMAPS:
        raise InvalidParameterError(
            f"unknown colormap {colormap!r}; this version offers {', '.join(COLORMAPS)}"
        )

    low, high = limits if limits is not None else value_range(image)
    clipped = np.clip(np.nan_to_num(image.data, nan=low), low, high)
    #: `afm_to_rgb` percentile-clips on its own, and this window is already the
    #: decision — so it is handed an array that is exactly its own range.
    return afm_to_rgb(clipped.astype(np.float64), colormap=colormap, clip_percentile=100.0)


def thumbnail(
    image: DisplayImage, *, size_px: int = 64, colormap: str = COLORMAPS[0]
) -> np.ndarray:
    """The same picture, small enough for a list row.

    **Subsampled before it is coloured**, not after: a 4096 x 4096 map is 16 M
    values, and mapping all of them to RGB to then throw 99.99% of them away is
    a second of matplotlib per row. Striding is the cheapest possible reduction
    and the honest one for a 48-pixel icon — it is a *look*, not a measurement,
    and nothing is derived from it.

    The value window is computed on the subsample, so the thumbnail is contrasted
    the way the full image is (`value_range`'s percentiles), not flattened to
    grey by one hot pixel it happened to keep.

    Args:
        image: what to shrink.
        size_px: the longer side of the result, at most.
        colormap: one of `COLORMAPS`.

    Returns:
        `(height, width, 3)` of `uint8`, no side longer than `size_px`.

    Raises:
        InvalidParameterError: `size_px` is not positive, or a colormap this
            application does not offer.
    """
    if size_px <= 0:
        raise InvalidParameterError(f"size_px must be positive, got {size_px!r}")

    data = image.data
    if data.ndim > 2:  # pragma: no cover — every reader returns 2-D today
        data = data[..., 0]
    step = max(1, int(np.ceil(max(data.shape) / size_px)))
    return render(replace(image, data=data[::step, ::step]), colormap=colormap)


class Stage(StrEnum):
    """Which array of the pipeline a viewer is showing (M6-T01, ADR-0061).

    `RAW` is the file, which is what ADR-0056 made the default and the only
    thing available before a preview exists. The other three are what
    `PreprocessingResult` carries; the plane-only intermediate is deliberately
    not here, because the result object does not keep it and adding a field to
    an entity for a preview is a change to what a *run* records.
    """

    RAW = "raw"
    FLATTENED = "flattened"
    SUBSTRATE = "substrate"
    RESULT = "result"


#: What each stage is called on screen, and what it means. The viewer shows the
#: name beside the scan: ADR-0056's rule was never "show the file and nothing
#: else", it was **never show something the file does not contain without saying
#: so**, and this is how that promise is kept once there is something else to
#: show.
STAGE_LABELS: dict[Stage, str] = {
    Stage.RAW: "raw (the file)",
    Stage.FLATTENED: "flattened (plane + lines)",
    Stage.SUBSTRATE: "substrate (estimated)",
    Stage.RESULT: "result (flattened minus substrate)",
}


def stage_image(
    stage: Stage,
    image: DisplayImage,
    preview: PreprocessingResult | None,
) -> DisplayImage:
    """The array for a stage, wrapped so the viewer draws it the way it draws
    everything else — same colormap, same value window, same scale bar.

    Falls back to the file when there is no preview, rather than raising: the
    panels only offer a stage while a preview exists, and the honest answer to
    "show me the substrate of a scan nobody has preprocessed" is the scan.
    """
    if stage is Stage.RAW or preview is None:
        return image

    arrays = {
        Stage.FLATTENED: preview.z_flat,
        Stage.SUBSTRATE: preview.substrate,
        Stage.RESULT: preview.z_result,
    }
    return replace(image, name=f"{image.name} — {stage}", data=arrays[stage])
