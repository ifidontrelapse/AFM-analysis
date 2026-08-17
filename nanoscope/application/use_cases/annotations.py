"""Annotations out, and labels back in, in the format the trainer reads (M7-T09).

M7's fourth exit criterion: *"annotations export to a format the M8 dataset
builder consumes"*. The format is **YOLO's** — `class cx cy w h`, normalised to
the image, one line per box, one file per scan, with a `classes.txt` whose line
numbers are the class indices. It is what M8-T03 will train with and what
labelImg, CVAT and Roboflow read and write, which is what makes export and
import one decision instead of two (ADR-0078).

**The export is lossy, and the lossy part is the point.** A row here carries an
outline (ADR-0072), a painted mask (ADR-0073), a note, an id, two timestamps and
a `source`; a label file carries a class and four numbers. So:

- a polygon exports as its bounding box, which is **already** what the row
  stores beside the outline — nothing is lost that the row did not lose first;
- `source` cannot survive the trip, so the *caller* chooses which sources go in
  (ADR-0044: a model trained on its own output is confirming itself);
- what an import cannot know, it is **told** rather than guessing, and the file
  it came from is written into the annotation's note.

`data.yaml` and the train/val split are **not** written here. A split is a
dataset decision — how much to hold out, stratified by what — and it belongs to
M8-T02 rather than to the task that happened to write the labels first.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from nanoscope.application.use_cases.display import load_for_display
from nanoscope.core.entities.project import Annotation, AnnotationSource, ImageRecord
from nanoscope.core.errors import AnalysisFailedError, InvalidParameterError
from nanoscope.core.ports import ProjectRepository

#: The file whose line numbers are the class indices. `classes.txt` rather than
#: `data.yaml`: this one *is* the mapping, and the other is a dataset (§2).
CLASSES_FILE = "classes.txt"

#: Where the label files go inside the export, and where an import looks for
#: them if it is handed the directory above them. YOLO's own layout.
LABELS_DIRECTORY = "labels"

#: Decimal places on a normalised coordinate. Six is what every tool in this
#: ecosystem writes, and at 1e-6 of a 2048-pixel scan the rounding is 2 nm at a
#: 1 nm scale — below the pixel a box was drawn on.
PRECISION = 6


@dataclass(frozen=True)
class AnnotationExport:
    """What an export wrote, so a caller can say where it went."""

    #: The export directory, relative to the project root.
    directory: str
    #: The class names, in the order their indices refer to.
    classes: tuple[str, ...]
    #: How many boxes were written, and over how many scans.
    boxes: int
    images: int


def export_annotations(
    repository: ProjectRepository,
    *,
    sources: Iterable[AnnotationSource] | None = None,
    directory_name: str | None = None,
) -> AnnotationExport:
    """Write every annotation in the project as YOLO labels under `exports/`.

    Args:
        repository: an open project.
        sources: which kinds of annotation to include. `None` means all of them;
            `(AnnotationSource.MANUAL,)` is the training set that cannot confirm
            itself (ADR-0044). The caller names the scope — this does not pick a
            default that hides one.
        directory_name: what to call the export directory. Defaults to a
            timestamped name, because two exports on one day are the normal case
            and replacing the first loses work (ADR-0048).

    Returns:
        Where it went, the class list, and how much was written.

    Raises:
        AnalysisFailedError: the project has no annotation of those kinds. An
            empty label set is indistinguishable from *"nothing was drawn"*,
            which is a different statement (ADR-0048's rule, second site).
        MissingFileError: a scan's file is gone. Loud, because normalising needs
            its size and a dataset silently missing one scan of twelve is wrong
            in a way that looks right.
    """
    wanted = None if sources is None else set(sources)
    per_image = [
        (record, kept)
        for record in repository.list_images()
        if (kept := _selected(repository.annotations_for(record.id), wanted))
    ]
    if not per_image:
        raise AnalysisFailedError(
            "nothing to export: no annotation of that kind exists in this project"
        )

    classes = tuple(sorted({one.label for _record, kept in per_image for one in kept}))
    index_of = {name: index for index, name in enumerate(classes)}
    root = directory_name or _default_name(wanted)

    boxes = 0
    for record, kept in per_image:
        height, width = load_for_display(repository, record.id).data.shape[:2]
        lines = [_to_label(one, index_of[one.label], width=width, height=height) for one in kept]
        repository.write_export_text(
            f"{root}/{LABELS_DIRECTORY}/{Path(record.relative_path).stem}.txt",
            "\n".join(lines) + "\n",
        )
        boxes += len(lines)

    #: The directory is read back off the path the adapter chose, rather than
    #: assembled here: where an export lands is the adapter's decision (M4-T11).
    written = repository.write_export_text(f"{root}/{CLASSES_FILE}", "\n".join(classes) + "\n")
    return AnnotationExport(
        directory=PurePosixPath(written).parent.as_posix(),
        classes=classes,
        boxes=boxes,
        images=len(per_image),
    )


def read_labels(
    directory: Path | str, images: Sequence[ImageRecord]
) -> tuple[list[tuple[ImageRecord, str]], list[tuple[str, str]], tuple[str, ...]]:
    """Match the label files in a directory to the project's images, by stem.

    Reading and matching only — nothing is stored here, so the caller can put
    every box on the command stack as **one** edit (ADR-0077 §3).

    Returns:
        `(matched, skipped, classes)`: the label files that name an image of
        this project paired with it, the ones that do not with the reason, and
        the class names their indices refer to.

    Raises:
        InvalidParameterError: the directory does not exist, or has no
            `classes.txt` — without which an index is a number and not a label.
    """
    where = Path(directory)
    labels = where / LABELS_DIRECTORY if (where / LABELS_DIRECTORY).is_dir() else where
    if not labels.is_dir():
        raise InvalidParameterError(f"no such directory to import from: {where}")

    classes_file = _classes_file(where, labels)
    if classes_file is None:
        raise InvalidParameterError(
            f"no {CLASSES_FILE} beside those labels: a class index without it is a number, "
            "not a label"
        )
    classes = tuple(
        line.strip()
        for line in classes_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not classes:
        raise InvalidParameterError(f"{classes_file.name} is empty: it must name the classes")

    by_stem = {PurePosixPath(record.relative_path).stem: record for record in images}
    matched: list[tuple[ImageRecord, str]] = []
    skipped: list[tuple[str, str]] = []
    for path in sorted(labels.glob("*.txt")):
        if path.name == CLASSES_FILE:
            continue
        record = by_stem.get(path.stem)
        if record is None:
            skipped.append((path.name, "no image of that name in this project"))
            continue
        matched.append((record, path.read_text(encoding="utf-8")))
    return matched, skipped, classes


def parse_labels(
    text: str, classes: Sequence[str], *, width: int, height: int
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """One label file, as `(label, box)` pairs in pixels.

    The inverse of `_to_label`, and deliberately strict about the two things a
    wrong file gets wrong: an index no class list has, and a coordinate outside
    `[0, 1]`. Both mean the file is not describing this image, and importing it
    anyway would put boxes in the wrong places with nothing saying so.

    Raises:
        InvalidParameterError: a line that is not five numbers, a class index
            out of range, or a coordinate outside `[0, 1]`.
    """
    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise InvalidParameterError(
                f"line {number}: a YOLO label is 'class cx cy w h', got {len(parts)} field(s)"
            )
        try:
            index = int(parts[0])
            centre_x, centre_y, box_width, box_height = (float(value) for value in parts[1:])
        except ValueError as bad:
            raise InvalidParameterError(f"line {number}: {bad}") from bad

        if not 0 <= index < len(classes):
            raise InvalidParameterError(
                f"line {number}: class {index} is not in a list of {len(classes)}"
            )
        for name, value in (
            ("cx", centre_x),
            ("cy", centre_y),
            ("w", box_width),
            ("h", box_height),
        ):
            if not 0.0 <= value <= 1.0:
                raise InvalidParameterError(
                    f"line {number}: {name}={value} is outside [0, 1]; a YOLO label is "
                    "normalised to its image"
                )
        if box_width <= 0.0 or box_height <= 0.0:
            raise InvalidParameterError(f"line {number}: a box with no area is not a box")

        half_w, half_h = box_width * width / 2.0, box_height * height / 2.0
        centre_px_x, centre_px_y = centre_x * width, centre_y * height
        boxes.append(
            (
                classes[index],
                (
                    centre_px_x - half_w,
                    centre_px_y - half_h,
                    centre_px_x + half_w,
                    centre_px_y + half_h,
                ),
            )
        )
    return boxes


def _default_name(wanted: set[AnnotationSource] | None) -> str:
    """`annotations_<scope>_<timestamp>`, and **the scope is in the name.**

    Found by a test: two exports of different scopes a second apart landed in one
    directory, so a *hand-drawn* label set and an *everything* one were mixed in
    the file an operator would then have trained on. The timestamp keeps
    ADR-0048's promise across time; the scope keeps it across the two menu items.
    """
    scope = "all" if wanted is None else "_".join(sorted(one.value for one in wanted))
    return f"annotations_{scope}_{datetime.now(UTC):%Y%m%dT%H%M%S}"


def _selected(
    annotations: Iterable[Annotation], wanted: set[AnnotationSource] | None
) -> tuple[Annotation, ...]:
    return tuple(one for one in annotations if wanted is None or one.source in wanted)


def _classes_file(where: Path, labels: Path) -> Path | None:
    """`classes.txt`, beside the labels or one directory above them."""
    return next(
        (path for path in (labels / CLASSES_FILE, where / CLASSES_FILE) if path.is_file()), None
    )


def _to_label(annotation: Annotation, index: int, *, width: int, height: int) -> str:
    """One annotation as a YOLO line, normalised and clamped to the image.

    Clamped rather than refused: a drag that ran off the edge of the scan is an
    ordinary thing an operator does, and the part of the box that is on the
    image is what they meant (the part that is not was never measurable).
    """
    x1, y1, x2, y2 = annotation.box
    x1, x2 = sorted((_clamp(x1 / width), _clamp(x2 / width)))
    y1, y2 = sorted((_clamp(y1 / height), _clamp(y2 / height)))
    values = ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)
    return " ".join([str(index), *(f"{value:.{PRECISION}f}" for value in values)])


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))
