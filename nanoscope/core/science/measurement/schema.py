"""What a measurement table holds, declared once for all four producers
(D-16, D-17, ADR-0031).

Before this module there were four schemas. `measure_all_baseline` and the two
SAM2 producers each emitted their own column set, the SEM/TEM path emitted a
fourth, and `run_pipeline`'s detect mode emitted a table with no columns at all.
Three separate faults lived in that:

- **one quantity under two names** — `score` / `sam_score`, `area_px` /
  `mask_area_px`, from two functions that were copy-pasted and then drifted;
- **two quantities under one name** — `radius_nm` meant the *detector's* blob
  radius in one table and the *measured mask's* equivalent radius in another, so
  concatenating them produced one column holding two different measurements;
- **columns that varied per row**, because both SAM2 producers assembled records
  with `if k in res`.

The shape that replaced it is a **core plus blocks**. Every row of every table
carries `CORE_COLUMNS`. A block is present in full or absent in full, and
`method` says which producer wrote the row and therefore which blocks to expect.

Not one wide table with NaN where a producer cannot fill a column: that would say
SEM/TEM *has* heights and they are all missing. It has no heights — the modality
does not produce one — and this milestone has spent six ADRs on the difference
between absent and substituted.
"""

from __future__ import annotations

import pandas as pd

#: Every row of every measurement table, whatever produced it.
CORE_COLUMNS: dict[str, str] = {
    "particle_id": "int64",
    #: The centre, in pixels. **float64, not int64**: three of the four
    #: producers already emitted a subpixel centre, and rounding it in the
    #: fourth was the only reason the column could be an integer (ADR-0031).
    "x_px": "float64",
    "y_px": "float64",
    #: Pixels in the particle's mask — circular for the baseline producer, the
    #: real segmentation for SAM2. Was `mask_area_px` in the SAM2 tables.
    "area_px": "int64",
    #: Which producer wrote the row: "baseline_circle" or "sam2".
    "method": "str",
}

#: What the detector that *prompted* the measurement had to say. Separate from
#: `GEOMETRY_COLUMNS`'s `radius_nm` on purpose: one is where we looked, the other
#: is what we found.
DETECTOR_COLUMNS: dict[str, str] = {
    "sigma_px": "float64",
    "detector_radius_nm": "float64",
}

#: AFM only: needs a height map. Was emitted with `peak_nm` by SAM2 and with
#: `mean_nm` by the baseline producer; both are declared here and both are
#: emitted by both, because they are cheap and a block is all-or-nothing.
HEIGHT_COLUMNS: dict[str, str] = {
    "height_nm": "float64",
    "baseline_nm": "float64",
    "peak_nm": "float64",
    "mean_nm": "float64",
    "baseline_source": "str",
    "ring_px": "int64",
}

#: Shape of the mask that was measured. Requires a real segmentation, so the
#: circular-mask producer does not emit it: the geometry of a circle drawn from
#: a radius is that radius, and recording it as a measurement would be circular
#: in both senses.
GEOMETRY_COLUMNS: dict[str, str] = {
    "radius_px": "float64",
    "radius_nm": "float64",
    "area_nm2": "float64",
    "circularity": "float64",
    "aspect_ratio": "float64",
}

#: The segmenter's own opinion of the mask it produced — SAM2's predicted IoU.
#: **`mask_score`, not `score`**: this project has two scores since ADR-0028, and
#: `Detection.confidence` is the detector's.
SEGMENTATION_COLUMNS: dict[str, str] = {
    "mask_score": "float64",
}


def measurement_columns(
    *,
    detector: bool = False,
    height: bool = False,
    geometry: bool = False,
    segmentation: bool = False,
) -> dict[str, str]:
    """The columns a producer emits, in a fixed order: core first, then blocks.

    Args:
        detector:     the measurement was prompted by a detection whose radius
                      and sigma are known
        height:       AFM — a height map was available
        geometry:     a real mask was measured, so its shape is a result
        segmentation: a segmenter scored the mask it produced

    Returns:
        `{column: dtype}`, ordered. The order is part of the declaration so that
        two tables of the same kind compare and concatenate without reordering.
    """
    columns = dict(CORE_COLUMNS)
    if detector:
        columns.update(DETECTOR_COLUMNS)
    if height:
        columns.update(HEIGHT_COLUMNS)
    if geometry:
        columns.update(GEOMETRY_COLUMNS)
    if segmentation:
        columns.update(SEGMENTATION_COLUMNS)
    return columns


def empty_measurement_table(**blocks: bool) -> pd.DataFrame:
    """A table with the right columns, the right dtypes and no rows.

    The value a producer returns when nothing was measured, and the value
    `run_pipeline` returns in detect mode, where nothing was measured *by
    design* — the modality-dependent case ADR-0027 left to this task.

    Args:
        **blocks: the flags `measurement_columns` takes.
    """
    return pd.DataFrame(
        {name: pd.Series(dtype=dtype) for name, dtype in measurement_columns(**blocks).items()}
    )


def blocks_for(modality: str, *, has_height_map: bool | None = None) -> dict[str, bool]:
    """Which blocks a measurement of this modality can carry.

    AFM measures heights; SEM and TEM measure shapes. That is the whole rule,
    and it lives here rather than in `run_pipeline` because it is a statement
    about what the data contains, not about how the run is orchestrated.

    Args:
        modality:       "afm", "sem" or "tem"
        has_height_map: override for a caller that has an AFM image but no
                        `z_flat` — the SAM2 producers accept exactly that
    """
    afm = modality == "afm" if has_height_map is None else has_height_map
    return {"height": afm, "geometry": not afm}
