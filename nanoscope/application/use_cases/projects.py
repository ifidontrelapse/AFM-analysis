"""Opening a project, and filling it with images (M4-T04, ADR-0041).

The first module in `application` written for this milestone rather than moved
into it, and the first thing anywhere that talks to a `ProjectRepository`
through its port instead of by its class name.

**Two use cases, where the task listed five.** `close_project` and
`list_images` would be `repo.close()` and `repo.list_images()` under new names —
a function that forwards one call to one object is not a layer, and writing it
is how a layer becomes ceremony. They arrive when closing or listing starts to
mean something more than closing or listing (autosave is M4-T09). This
repository has made the same call before, on purpose: M2-T08 specified seven
ports and wrote one.

What the two here have that those do not is policy:

- `open_project` **reads the integrity report** rather than leaving it to be
  asked for, which is where ADR-0040's closing obligation is discharged
- `import_images` **does not abort the batch** on a bad file, because thirty-nine
  good scans should not be lost to the fortieth

Creating a project is not here either: it is `mkdir`, a manifest and SQLite from
end to end, and this layer may import none of them (Architecture §3.2). The
composition root constructs it — `SqliteProjectRepository.create`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from nanoscope.core.entities.project import (
    ImageRecord,
    ImportFailure,
    ImportReport,
    OpenedProject,
)
from nanoscope.core.errors import NanoscopeError
from nanoscope.core.ports import ProjectRepository
from nanoscope.core.values import Modality


def open_project(repository: ProjectRepository) -> OpenedProject:
    """Everything a caller needs to show an opened project, report included.

    The integrity check runs here, once, and its result travels with the images
    instead of waiting to be requested. ADR-0040 decided that the check reports
    and changes nothing; the obligation that came with it was that *something
    has to read the report*, and this is that something.

    Args:
        repository: an open project.

    Returns:
        Its name, its images, and where the index and the filesystem disagree.
    """
    return OpenedProject(
        name=repository.name,
        images=tuple(repository.list_images()),
        integrity=repository.check_integrity(),
    )


def import_images(
    repository: ProjectRepository,
    sources: Iterable[Path | str],
    *,
    modality: Modality,
    pixel_size_nm: float | None = None,
) -> ImportReport:
    """Copy files into the project and record them, one by one.

    **A failure is collected, not raised.** An operator importing a folder of
    forty scans should not lose the thirty-nine that were fine because the
    fortieth was a partial download — so each file is attempted on its own and
    the caller receives both lists. What to do about a partial success is a
    decision above this layer; refusing to make it here is the point.

    Only `NanoscopeError` is caught. A file this library rejects is data, and
    goes in the report; anything else is a bug in this application, and a bug
    that keeps going for another thirty-nine files is a worse bug.

    Args:
        repository: an open project.
        sources: files to import, anywhere on disk. Each is copied into the
            project before it is recorded.
        modality: which instrument produced them. One value for the batch,
            because a folder of scans comes off one instrument; a mixed import
            is two calls.
        pixel_size_nm: nm per pixel, or `None` when the scale is unknown — which
            is a state, not a reason to invent 1.0 (ADR-0025).

    Returns:
        What was imported, and what was not, with a reason for each failure.
    """
    imported: list[ImageRecord] = []
    failed: list[ImportFailure] = []

    for source in sources:
        try:
            imported.append(
                repository.import_image(
                    source,
                    modality=modality,
                    pixel_size_nm=pixel_size_nm,
                )
            )
        except NanoscopeError as exc:
            failed.append(ImportFailure(source=str(source), reason=str(exc)))

    return ImportReport(imported=tuple(imported), failed=tuple(failed))
