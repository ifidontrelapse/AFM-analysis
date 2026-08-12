"""Measurements as a file the operator can take somewhere else (M4-T11).

ADR-0042 already writes each run's measurement table to
`results/run_<id>/measurements.csv`, so this could have been a copy. It is not,
and the difference is the whole task: **storage is what the application needs to
reopen its own work; an export is what a person opens in a spreadsheet three
months later** (ADR-0048).

What that difference costs is three things the stored table does not have:

- **which image each row came from.** The stored table is filed *under* its run,
  so it does not repeat the fact; a CSV on a desktop has nothing around it, and
  a column of heights with no scan name is a column of numbers
- **more than one run at a time.** Statistics across a dataset is the reason the
  measurements exist, and it is not something an operator should assemble by
  hand from twelve files
- **a name that says what it is**, in `exports/`, which is the directory
  ADR-0003 set aside for exactly this
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

import pandas as pd

from nanoscope.core.entities import AnalysisRun
from nanoscope.core.errors import AnalysisFailedError
from nanoscope.core.ports import ProjectRepository

#: Columns prepended to every exported row, in this order. They answer "which
#: scan, which analysis" — the two questions a CSV cannot answer by sitting in
#: the right directory.
PROVENANCE_COLUMNS = ("image", "image_id", "run_id", "detector", "mode", "pixel_size_nm")


def export_measurements(
    repository: ProjectRepository,
    runs: Iterable[AnalysisRun] | None = None,
    *,
    file_name: str | None = None,
) -> str:
    """Write the measurements of one or more runs to a CSV under `exports/`.

    Args:
        repository: an open project.
        runs: which analyses to export. `None` means *every run of every image*,
            which is the "give me the dataset" case and the reason this takes a
            collection rather than one run.
        file_name: what to call it. Defaults to a timestamped name, because two
            exports on the same day are the normal case and silently replacing
            the first would lose work the operator thought they had.

    Returns:
        The path written, relative to the project root — relative, like every
        other path this application stores (ADR-0003).

    Raises:
        AnalysisFailedError: none of the runs produced a measurement table. A
            file with headers and no rows is indistinguishable from "we measured
            and found nothing", which is a different statement.
        MissingFileError: a run's stored table is gone. Loud, because an export
            silently missing one scan of twelve is a wrong dataset that looks
            right (ADR-0042 made the same call).
    """
    selected = list(runs) if runs is not None else _every_run(repository)
    frames = [
        _with_provenance(repository, run) for run in selected if run.measurements_path is not None
    ]
    if not frames:
        raise AnalysisFailedError(
            "nothing to export: none of the selected runs produced a measurement table "
            "(a detect-only run measures nothing)"
        )

    table = pd.concat(frames, ignore_index=True)
    return repository.write_export(file_name or _default_name(selected), table)


def _every_run(repository: ProjectRepository) -> list[AnalysisRun]:
    return [run for image in repository.list_images() for run in repository.runs_for(image.id)]


def _with_provenance(repository: ProjectRepository, run: AnalysisRun) -> pd.DataFrame:
    """One run's table, with the columns that say where it came from in front.

    In front, not appended: a spreadsheet opens on column A, and the first
    question is always which scan this is.
    """
    table = repository.measurements_for(run)
    image = repository.get_image(run.image_id)
    provenance = {
        "image": image.display_name,
        "image_id": run.image_id,
        "run_id": run.id,
        "detector": run.detector,
        "mode": run.mode,
        #: Empty rather than 0 when the scale is unknown — a spreadsheet reading
        #: a 0 here would compute nanometres from a pixel count (ADR-0025).
        "pixel_size_nm": run.pixel_size_nm,
    }
    return table.assign(**provenance)[list(PROVENANCE_COLUMNS) + list(table.columns)]


def _default_name(runs: Sequence[AnalysisRun]) -> str:
    """`measurements_<what>_<when>.csv`, safe on any filesystem.

    The timestamp is not decoration: an export is a snapshot, and two of them on
    one day are the normal case. Replacing the first silently would lose work
    the operator believes they have.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    what = f"run{runs[0].id}" if len(runs) == 1 else f"{len(runs)}runs"
    return f"measurements_{what}_{stamp}.csv"
