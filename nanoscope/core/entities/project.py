"""What a project knows about the files in it (M4-T03).

These are what the repository hands back — never a `sqlite3.Row`, which is
untyped and would put the database's vocabulary into every layer above it. The
port that returns them is `core.ports.ProjectRepository`; the implementation is
`infrastructure.storage.SqliteProjectRepository`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from nanoscope.core.entities.detection import Detection
from nanoscope.core.values import Modality


@dataclass(frozen=True)
class ImageRecord:
    """One image in a project: where its file is, and what is known about it.

    `relative_path` is relative to the **project root**, with `/` separators, so
    the directory stays movable (ADR-0003) and a project written on one machine
    reads on another.
    """

    id: int
    relative_path: str
    display_name: str
    modality: Modality
    #: SHA-256 of the file as it was when it was recorded. Computed by the
    #: repository from the file itself, never accepted from a caller.
    sha256: str
    #: `None` when the scale is unknown, which an npy file always is unless the
    #: caller says otherwise. Never a fabricated 1.0 (ADR-0019, ADR-0025).
    pixel_size_nm: float | None
    imported_utc: str


@dataclass(frozen=True)
class IntegrityReport:
    """Where the index and the filesystem disagree — and nothing more.

    ADR-0003 named the disagreement as the price of two sources of truth. This
    is the collection of it: it *reports*, in both directions, and resolving
    what it found is a decision with an operator behind it (ADR-0040).

    Existence only. Comparing checksums would mean reading every scan in the
    project on every open; the checksum is in the row for the question that
    actually needs it.
    """

    #: Rows whose file is not where the row says it is. Not deleted: the file
    #: may be on an unmounted drive, and the row carries measurements it does
    #: not.
    missing_files: tuple[ImageRecord, ...] = ()
    #: Paths under `images/` that no row claims, relative to the project root.
    #: Not imported: nothing here knows they were meant to be in the project.
    untracked_files: tuple[str, ...] = ()

    @property
    def is_clean(self) -> bool:
        """True when the index and the filesystem agree about every file."""
        return not self.missing_files and not self.untracked_files


@dataclass(frozen=True)
class OpenedProject:
    """A project as `open_project` hands it over: what is in it, and what is off.

    The integrity report travels *with* the images rather than being available
    on request, which is how ADR-0040's closing obligation is discharged — a
    report nobody reads is a report that did nothing.
    """

    name: str
    images: tuple[ImageRecord, ...]
    integrity: IntegrityReport


@dataclass(frozen=True)
class AnalysisRun:
    """One analysis of one image: what was asked, and where the answer went.

    The *index* of a result, not the result. Its detections are rows in the
    database; its measurement table is the file at `measurements_path`, because
    that table is variable by construction (ADR-0031) and this one is not
    (ADR-0042).
    """

    id: int
    image_id: int
    #: Which detector produced the detections: "log" or "yolo".
    detector: str
    #: Which pipeline mode ran: "detect", "baseline" or "segment".
    mode: str
    modality: Modality
    #: `None` when the image has no known scale — a state, not a fabricated 1.0.
    pixel_size_nm: float | None
    #: Where the measurement table was written, relative to the project root.
    #: `None` in `detect` mode, which measures nothing: an empty table with the
    #: right columns is not a measurement, and writing one would claim it was.
    measurements_path: str | None
    created_utc: str
    #: What was found, in the order the detector returned it.
    detections: tuple[Detection, ...] = ()
    #: The masks a segmentation produced — **in memory only**. Empty on every
    #: run the repository hands back, filled on the one just computed. ADR-0042
    #: did not persist them, because the weights that produce them are outside
    #: the gate and the format would have been written blind; an overlay that
    #: showed them anyway would be showing something the project cannot restore
    #: (M6-T04, ADR-0064).
    masks: tuple[dict[str, Any], ...] = ()


class AnnotationSource(StrEnum):
    """Where an annotation's box came from.

    The distinction M8 has to be able to make: a model trained on boxes copied
    from its own output is confirming itself, and a training set that cannot
    tell the two apart cannot avoid it (ADR-0044).
    """

    #: Drawn by a person.
    MANUAL = "manual"
    #: Adopted from a detector's output, with or without correction afterwards.
    FROM_DETECTION = "from_detection"


@dataclass(frozen=True)
class Annotation:
    """One box an operator drew on an image, and what they called it.

    The only thing in a project that cannot be recomputed. Coordinates are
    **floats** in pixels: a drag is not on the pixel grid, and rounding is a
    decision the trainer makes with the whole box in hand, not the database.
    `Detection.bbox` stays integer — a detector's output and a person's
    judgement are two different things that happen to have four numbers.
    """

    id: int
    image_id: int
    label: str
    #: `(x1, y1, x2, y2)` in pixels, `x2 > x1` and `y2 > y1` — the convention
    #: every box in this project uses (PROJECT_RULES §3).
    box: tuple[float, float, float, float]
    source: AnnotationSource
    #: Whatever the operator wanted to say about this particle. Free text, and
    #: absent when they said nothing.
    note: str | None
    created_utc: str
    updated_utc: str
    #: The outline, when the operator drew one — `((x, y), …)` in pixels, at
    #: least three vertices. `None` means *a box, drawn as a box*, which is what
    #: every row written before M7-T03 is.
    #:
    #: `box` stays authoritative for everything that wants a box and is the
    #: **bounding box** of these points, derived by the repository so the two
    #: cannot disagree (ADR-0072).
    points: tuple[tuple[float, float], ...] | None = None
    #: Where a painted mask lives, relative to the project root — `None` unless
    #: the operator painted one. **Not the mask itself**: an array the size of a
    #: scan is a file (PROJECT_RULES §5), and the row keeps the path the way an
    #: analysis run keeps the path to its measurement table (ADR-0042,
    #: ADR-0073).
    mask_path: str | None = None


class RulerKind(StrEnum):
    """What a hand-drawn line is for.

    Two tools share one geometry: a distance is read as a length, a profile as
    the heights underneath it (M7-T05, M7-T06). One table, one migration.
    """

    DISTANCE = "distance"
    PROFILE = "profile"


@dataclass(frozen=True)
class Ruler:
    """A line an operator drew, and what they called it.

    **Not an annotation.** A line has no area, and ADR-0044's shapes are refused
    without one — twice. It is also not a *measurement* in this project's sense:
    `measurements.csv` is what an analysis run produces (ADR-0031, ADR-0042),
    derived and re-runnable, and a hand-drawn distance is neither. Hence the
    word `ruler` (ADR-0074).

    The length is **not** stored: it is `distance_px(start, end)` and cannot
    disagree with the points unless somebody stores it twice.
    """

    id: int
    image_id: int
    kind: RulerKind
    #: `(x, y)` in pixels, both ends.
    start: tuple[float, float]
    end: tuple[float, float]
    label: str
    created_utc: str


@dataclass(frozen=True)
class ImportFailure:
    """One file that did not make it in, and why.

    Collected rather than raised: a forty-file import must not lose thirty-seven
    good scans to one bad one (ADR-0041).
    """

    #: The path as the caller gave it, so the message can be matched to the input.
    source: str
    reason: str


@dataclass(frozen=True)
class ImportReport:
    """The outcome of importing a batch: both halves of it, always."""

    imported: tuple[ImageRecord, ...] = ()
    failed: tuple[ImportFailure, ...] = ()

    @property
    def is_complete(self) -> bool:
        """True when every file the caller offered was imported."""
        return not self.failed
