"""The images in a project, stored in its database (M4-T03).

The first thing in this repository that reads and writes rows. It satisfies
`core.ports.ProjectRepository` structurally, without importing it — the arrow
points inward and a Protocol needs no base class to stand under.

Two rules live here rather than in the callers, because a rule enforced by every
caller is a rule already broken somewhere:

- **a stored path is relative to the project root**, and this is the one funnel
  every writer passes through. M4-T02's `CHECK` is the backstop for a writer who
  does not, and this is the one that produces a message worth reading
- **a checksum describes the file the row points at**, because it is computed
  here from that file and never accepted as an argument

`create` and `import_image` joined them in M4-T04, and they are here for the same
reason the rest is: making a directory and copying a file are filesystem work,
and `application` may import neither the filesystem nor `sqlite3`
(Architecture §3.2).

What is *not* here: acting on what `check_integrity` reports — that is a decision
with an operator behind it (ADR-0040) — and any policy about a *batch* of
imports, which is `application/use_cases/projects.py` (ADR-0041).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import threading
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Any, Self, cast

import numpy as np
import pandas as pd

from nanoscope.core.entities import Detection, PipelineResult
from nanoscope.core.entities.device import Device
from nanoscope.core.entities.model import ModelDescriptor, ModelFramework, ModelTask
from nanoscope.core.entities.project import (
    AnalysisRun,
    Annotation,
    AnnotationSource,
    ImageRecord,
    IntegrityReport,
    Ruler,
    RulerKind,
)
from nanoscope.core.entities.training import (
    DatasetSpec,
    EpochMetrics,
    TrainingConfig,
    TrainingRun,
    TrainingStatus,
)
from nanoscope.core.errors import InvalidParameterError, MissingFileError
from nanoscope.core.values import DeviceKind, Modality
from nanoscope.infrastructure.storage.database import open_database
from nanoscope.infrastructure.storage.masks import read_mask, write_mask
from nanoscope.infrastructure.storage.project_format import (
    CACHE_DIRECTORY,
    DIRECTORIES,
    ProjectManifest,
    new_manifest,
    open_manifest,
    write_manifest,
)

_IMAGES_DIRECTORY = "images"
_RESULTS_DIRECTORY = "results"
_EXPORTS_DIRECTORY = "exports"
_ANNOTATIONS_DIRECTORY = "annotations"

#: 1 MiB. Large enough that the loop is not the cost, small enough that a
#: multi-gigabyte scan does not arrive in memory to be hashed.
_HASH_CHUNK_BYTES = 1024 * 1024


def _serialised[M: Callable[..., Any]](method: M) -> M:
    """Let one thread at a time inside the repository.

    Jobs run on worker threads (M4-T06) while the project was opened on another,
    and the two share one connection. SQLite's own library is serialized, so a
    *statement* is safe; a **sequence** of them is not — `save_analysis` writes a
    run, its detections and a path in three statements, and a second thread
    committing between them would commit half of it.

    One lock for the whole repository, not one per table: this is a single-user
    desktop application, contention is a project with two jobs writing at once,
    and the moment it is measurable the answer is a connection per thread rather
    than a finer lock.
    """

    @wraps(method)
    def guarded(self: SqliteProjectRepository, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return cast("M", guarded)


class SqliteProjectRepository:
    """One open project: its manifest, its database, and the files it indexes.

    Use it as a context manager, or call `close`. Opening is `open()`, which
    refuses a directory that is not a project and migrates the database forward
    if it is an older one.

    **Usable from more than one thread**, which the connection alone is not:
    `connect` passes `check_same_thread=False` and every method here is
    serialised by one reentrant lock.
    """

    def __init__(self, root: Path, manifest: ProjectManifest, conn: sqlite3.Connection) -> None:
        self._root = root
        self._manifest = manifest
        self._conn = conn
        self._lock = threading.RLock()

    @classmethod
    def open(cls, project_dir: Path | str) -> Self:
        """Open the project at `project_dir`.

        Raises:
            ProjectFormatError: the directory is not a project, or declares a
                format or schema version this application does not understand
                (ADR-0038's matrix, M4-T02's half of it).
        """
        root = Path(project_dir)
        manifest = open_manifest(root)
        conn = open_database(root)
        return cls(root, manifest, conn)

    @classmethod
    def create(cls, project_dir: Path | str, name: str) -> Self:
        """Make a new project at `project_dir` and open it (M4-T04).

        The layout of `docs/ProjectFormat.md` §1: the subdirectories, the
        manifest that makes the directory a project, and a database migrated to
        the current schema. A directory that does not exist yet is created.

        This lives in `infrastructure` rather than in a use case because it is
        `mkdir`, a manifest and SQLite from end to end, and `application` may
        import none of those (Architecture §3.2). The composition root is what
        constructs adapters (PROJECT_RULES §2.7).

        Args:
            project_dir: where to put it. Must not exist, or must be empty.
            name: the project's display name, which the directory's name does
                not have to match — the operator may rename either.

        Raises:
            InvalidParameterError: there is already something in that directory.
                Writing a manifest into a directory that has files in it turns
                somebody else's folder into a project.
        """
        root = Path(project_dir)
        if root.exists() and any(root.iterdir()):
            raise InvalidParameterError(
                f"{root} is not empty; a project is created in a new or empty directory"
            )

        for directory in DIRECTORIES:
            (root / directory).mkdir(parents=True, exist_ok=True)
        manifest = new_manifest(name)
        write_manifest(root, manifest)
        return cls(root, manifest, open_database(root))

    @property
    def root(self) -> Path:
        """The project directory. Every stored path is relative to it."""
        return self._root

    @property
    def name(self) -> str:
        """The project's display name, from the manifest — which is authoritative
        for identity, and readable when the database is not (ADR-0038)."""
        return self._manifest.name

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @_serialised
    def add_image(
        self,
        relative_path: str,
        *,
        modality: Modality,
        display_name: str | None = None,
        pixel_size_nm: float | None = None,
    ) -> ImageRecord:
        """Record a file that is already inside the project.

        Args:
            relative_path: where the file is, relative to the project root. An
                absolute path inside the project is accepted and stored relative;
                one outside it is refused, since the row would not survive the
                directory being moved.
            modality: which instrument produced it.
            display_name: what to call it. Defaults to the file's own name.
            pixel_size_nm: nm per pixel, or `None` when the scale is unknown —
                which is a state, not a reason to invent 1.0 (ADR-0025).

        Returns:
            The stored row, with the id the database assigned.

        Raises:
            InvalidParameterError: the path points outside the project.
            MissingFileError: there is no file there. A row whose file does not
                exist is the dangling row `check_integrity` reports, and there is
                no reason to create one on purpose.
        """
        stored = self._relative(relative_path)
        absolute = self._root / stored
        if not absolute.is_file():
            raise MissingFileError(
                f"no file at {absolute}: an image must be inside the project before it is recorded"
            )

        cursor = self._conn.execute(
            "INSERT INTO images "
            "(relative_path, display_name, modality, sha256, pixel_size_nm, imported_utc) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                stored,
                display_name or absolute.name,
                str(modality),
                sha256_of(absolute),
                pixel_size_nm,
                datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )
        self._conn.commit()
        return self.get_image(int(cursor.lastrowid or 0))

    @_serialised
    def import_image(
        self,
        source: Path | str,
        *,
        modality: Modality,
        display_name: str | None = None,
        pixel_size_nm: float | None = None,
    ) -> ImageRecord:
        """Copy a file into the project's `images/` and record it (M4-T04).

        The copy is `infrastructure`'s work and has to happen first: `add_image`
        records a file that is already inside the project, and a row pointing at
        a file that was never copied is the dangling row `check_integrity`
        exists to report.

        A name that is already taken is **disambiguated**, not refused —
        `scan.spm`, then `scan_1.spm`. Two different scans called `scan.spm`
        living in two folders is the ordinary shape of this work.

        The same file imported twice becomes a second copy with its own row.
        Deduplicating by checksum needs a `UNIQUE` index, a migration, and an
        answer to "are two identical scans ever legitimate?", which is an
        operator's question (ADR-0041).

        Args:
            source: the file to import, anywhere on disk.
            modality: which instrument produced it.
            display_name: what to call it. Defaults to the source's own name,
                *before* any disambiguating suffix — what the operator
                recognises is the name they gave the file.
            pixel_size_nm: nm per pixel, or `None` when the scale is unknown.

        Returns:
            The stored row.

        Raises:
            MissingFileError: there is no readable file at `source`.
        """
        origin = Path(source)
        if not origin.is_file():
            raise MissingFileError(f"no file to import at {origin}")

        destination = self._free_name(origin.name)
        shutil.copy2(origin, destination)
        return self.add_image(
            destination.relative_to(self._root).as_posix(),
            modality=modality,
            display_name=display_name or origin.name,
            pixel_size_nm=pixel_size_nm,
        )

    @_serialised
    def get_image(self, image_id: int) -> ImageRecord:
        """The row with this id.

        Raises:
            InvalidParameterError: no image has that id.
        """
        row = self._conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        if row is None:
            raise InvalidParameterError(f"no image with id {image_id} in {self._root}")
        return _record(row)

    @_serialised
    def path_of(self, image: ImageRecord) -> Path:
        """Where that image's file is. Relative paths are stored; absolute
        ones are what a reader needs, and this is the only place that joins
        the two (ADR-0038's compliance section)."""
        return self._root / image.relative_path

    @_serialised
    def list_images(self) -> list[ImageRecord]:
        """Every image in the project, in the order they were imported."""
        rows = self._conn.execute("SELECT * FROM images ORDER BY id")
        return [_record(row) for row in rows]

    @_serialised
    def remove_image(self, image_id: int) -> None:
        """Forget the row. The file stays on disk.

        Deleting the operator's scan is a different decision from forgetting it,
        and this layer does not get to make it — the file becomes an untracked
        file, which is exactly what `check_integrity` will then report.

        Raises:
            InvalidParameterError: no image has that id. Silence would make a
                typo look like a successful removal.
        """
        cursor = self._conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
        self._conn.commit()
        if cursor.rowcount == 0:
            raise InvalidParameterError(f"no image with id {image_id} in {self._root}")

    @_serialised
    def save_analysis(
        self, image_id: int, result: PipelineResult, *, model_id: str | None = None
    ) -> AnalysisRun:
        """Store what an analysis found, and return its index entry (M4-T05).

        The split ADR-0042 decided: the run and its detections become rows; the
        measurement **table** becomes `results/run_<id>/measurements.csv`,
        because ADR-0031 made that table variable by construction and this
        schema is not. Nothing is written for `detect` mode, which measures
        nothing — an empty table with the right columns is not a measurement,
        and storing one would claim otherwise.

        Masks are **not** persisted (ADR-0042 §3): SAM2 weights are outside this
        repository and outside the gate, so a format for them would be one
        nothing under test can produce.

        Args:
            image_id: the image this ran on. Its row must exist — the foreign
                key says so, and `PRAGMA foreign_keys` makes the database say it
                rather than this method.
            result: what `run_pipeline` returned.
            model_id: the registered model that produced the detections, or
                `None` for a detector that used none — every `log` run, and
                every run stored before M8-T06. Named by the caller: the result
                carries the *path* the detector loaded, and this project's
                answer to *which model?* is a name (ADR-0050, ADR-0086).

        Returns:
            The stored run, with its id and its detections.

        Raises:
            InvalidParameterError: no image has that id.
        """
        self.get_image(image_id)

        cursor = self._conn.execute(
            "INSERT INTO analysis_runs "
            "(image_id, detector, mode, modality, pixel_size_nm, measurements_path, "
            "created_utc, model_id) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?, ?)",
            (
                image_id,
                result.detector_name,
                result.mode,
                str(result.modality),
                result.pixel_size_nm,
                datetime.now(UTC).isoformat(timespec="seconds"),
                model_id,
            ),
        )
        run_id = int(cursor.lastrowid or 0)

        self._conn.executemany(
            "INSERT INTO detections "
            "(run_id, ordinal, x_px, y_px, radius_px, radius_nm, confidence, "
            "bbox_x1, bbox_y1, bbox_x2, bbox_y2) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    run_id,
                    ordinal,
                    detection.x_px,
                    detection.y_px,
                    detection.radius_px,
                    detection.radius_nm,
                    detection.confidence,
                    *(detection.bbox or (None, None, None, None)),
                )
                for ordinal, detection in enumerate(result.detections)
            ],
        )

        measurements_path = self._write_measurements(run_id, result)
        if measurements_path is not None:
            self._conn.execute(
                "UPDATE analysis_runs SET measurements_path = ? WHERE id = ?",
                (measurements_path, run_id),
            )
        self._conn.commit()
        return self.get_run(run_id)

    @_serialised
    def get_run(self, run_id: int) -> AnalysisRun:
        """One analysis, with its detections.

        Raises:
            InvalidParameterError: no run has that id.
        """
        row = self._conn.execute("SELECT * FROM analysis_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise InvalidParameterError(f"no analysis run with id {run_id} in {self._root}")
        return _run(row, self._detections_of(run_id))

    @_serialised
    def runs_for(self, image_id: int) -> list[AnalysisRun]:
        """Every analysis of this image, oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM analysis_runs WHERE image_id = ? ORDER BY id", (image_id,)
        ).fetchall()
        return [_run(row, self._detections_of(int(row["id"]))) for row in rows]

    @_serialised
    def measurements_for(self, run: AnalysisRun) -> pd.DataFrame:
        """The measurement table this run produced, read back from its file.

        Returns:
            The table as it was written. An **empty** table with no columns for a
            run that measured nothing, which is what `detect` mode produces —
            the columns of a measurement that did not happen are not this
            module's to invent (ADR-0031 owns them).

        Raises:
            MissingFileError: the run has a measurement file and it is not
                there. Derived data, so it is a re-run rather than a loss — but
                silently returning an empty table would report "no particles".
        """
        if run.measurements_path is None:
            return pd.DataFrame()

        path = self._root / run.measurements_path
        if not path.is_file():
            raise MissingFileError(
                f"the measurement table for run {run.id} is missing from {path}; "
                "re-run the analysis to produce it again"
            )
        return pd.read_csv(path)

    @_serialised
    def add_annotation(
        self,
        image_id: int,
        box: tuple[float, float, float, float],
        *,
        label: str,
        source: AnnotationSource = AnnotationSource.MANUAL,
        note: str | None = None,
        points: Sequence[tuple[float, float]] | None = None,
        mask: np.ndarray | None = None,
    ) -> Annotation:
        """Record a box the operator drew (M4-T07), with whatever shape it has.

        Args:
            image_id: which image it is on.
            box: `(x1, y1, x2, y2)` in pixels, with `x2 > x1` and `y2 > y1`.
            label: what the operator called it. Free text until a dataset needs
                a vocabulary (M8).
            source: hand-drawn, or adopted from a detector. The distinction M8
                must be able to make (ADR-0044).
            note: anything else they wanted to say.
            points: the outline, when the operator drew one — at least three
                vertices, in pixels. **`box` is then ignored and derived from
                them**, so a polygon and its bounding box cannot disagree
                (M7-T03, ADR-0072).
            mask: a painted mask, the shape of the scan. **Written to a file**
                under `annotations/` and the row keeps its path, because an
                array the size of a scan is not a database column
                (PROJECT_RULES §5, ADR-0073); `box` is derived from the painted
                pixels, by the same rule as `points`.

        Returns:
            The stored annotation, with its id.

        Raises:
            InvalidParameterError: no image has that id, the box has no area — a
                zero-area box is a mis-drag, and as a training example it is a
                picture of nothing — or an outline has fewer than three
                vertices, which is a line or the point ADR-0071 declined.
        """
        self.get_image(image_id)
        painted = None if mask is None else np.asarray(mask, dtype=bool)
        if painted is not None:
            if not painted.any():
                raise InvalidParameterError(
                    "a painted mask with no pixels in it has no shape; nothing was painted"
                )
            box = _mask_bounds(painted)

        outline = None if points is None else tuple((float(x), float(y)) for x, y in points)
        if outline is not None:
            if len(outline) < 3:
                raise InvalidParameterError(
                    f"an outline needs at least three vertices; got {len(outline)}"
                )
            box = _bounds(outline)

        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            raise InvalidParameterError(
                f"annotation box {box} has no area; expected x2 > x1 and y2 > y1"
            )

        now = datetime.now(UTC).isoformat(timespec="seconds")
        cursor = self._conn.execute(
            "INSERT INTO annotations "
            "(image_id, label, x1, y1, x2, y2, source, note, created_utc, updated_utc, points) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, label, x1, y1, x2, y2, str(source), note, now, now, _points_json(outline)),
        )
        annotation_id = int(cursor.lastrowid or 0)
        if painted is not None:
            #: Written **after** the row, because the id is the file's name —
            #: `save_analysis`'s own sequence (M4-T05).
            relative = f"{_ANNOTATIONS_DIRECTORY}/mask_{annotation_id}.png"
            write_mask(self._root / relative, painted)
            self._conn.execute(
                "UPDATE annotations SET mask_path = ? WHERE id = ?", (relative, annotation_id)
            )
        self._conn.commit()
        return self.get_annotation(annotation_id)

    @_serialised
    def restore_annotation(self, annotation: Annotation) -> Annotation:
        """Put a deleted annotation back **as itself**, id and timestamps intact.

        Not `add_annotation` with an extra argument: creating a box and undoing
        its deletion are different acts, and only one of them may choose an id.
        Undo needs this one, because everything else on the stack refers to the
        annotation by id — restoring it as a *new* row makes every command above
        it point at nothing (M4-T08, ADR-0045).

        Safe under a LIFO undo stack: anything created after the deletion is
        undone before this runs, so the id it reclaims is free. Outside that
        discipline the database's `UNIQUE` id will refuse a collision, which is
        the correct answer to restoring something twice.

        Raises:
            InvalidParameterError: an annotation with that id already exists, or
                its image is gone.
        """
        self.get_image(annotation.image_id)
        try:
            self._conn.execute(
                "INSERT INTO annotations "
                "(id, image_id, label, x1, y1, x2, y2, source, note, created_utc, updated_utc, "
                "points, mask_path) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    annotation.id,
                    annotation.image_id,
                    annotation.label,
                    *annotation.box,
                    str(annotation.source),
                    annotation.note,
                    annotation.created_utc,
                    annotation.updated_utc,
                    #: The outline comes back with it: an undo that restored the
                    #: box and dropped the polygon would silently redraw the
                    #: operator's work as a rectangle (M7-T03).
                    _points_json(annotation.points),
                    #: The path comes back too, pointing at the file the undo
                    #: left alone (ADR-0040's rule, third application).
                    annotation.mask_path,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise InvalidParameterError(
                f"cannot restore annotation {annotation.id}: {exc}"
            ) from exc
        self._conn.commit()
        return self.get_annotation(annotation.id)

    @_serialised
    def get_annotation(self, annotation_id: int) -> Annotation:
        """One annotation.

        Raises:
            InvalidParameterError: no annotation has that id.
        """
        row = self._conn.execute(
            "SELECT * FROM annotations WHERE id = ?", (annotation_id,)
        ).fetchone()
        if row is None:
            raise InvalidParameterError(f"no annotation with id {annotation_id} in {self._root}")
        return _annotation(row)

    @_serialised
    def add_ruler(
        self,
        image_id: int,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        kind: RulerKind = RulerKind.DISTANCE,
        label: str,
    ) -> Ruler:
        """Record a line an operator drew (M7-T05).

        The **length is not stored**: it is arithmetic over the endpoints
        (`core.science.metrology`), and a stored copy is a second answer waiting
        to disagree with the first.

        Raises:
            InvalidParameterError: no image has that id, or the two ends are the
                same point — a line of zero length measures nothing, which is
                the same refusal a zero-area box gets (ADR-0044 §5).
        """
        self.get_image(image_id)
        if start == end:
            raise InvalidParameterError(
                f"a ruler needs two different points; both ends are {start}"
            )

        now = datetime.now(UTC).isoformat(timespec="seconds")
        cursor = self._conn.execute(
            "INSERT INTO rulers (image_id, kind, x1, y1, x2, y2, label, created_utc) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, str(kind), *start, *end, label, now),
        )
        self._conn.commit()
        return self.get_ruler(int(cursor.lastrowid or 0))

    @_serialised
    def get_ruler(self, ruler_id: int) -> Ruler:
        """One ruler.

        Raises:
            InvalidParameterError: no ruler has that id.
        """
        row = self._conn.execute("SELECT * FROM rulers WHERE id = ?", (ruler_id,)).fetchone()
        if row is None:
            raise InvalidParameterError(f"no ruler with id {ruler_id} in {self._root}")
        return _ruler(row)

    @_serialised
    def rulers_for(self, image_id: int) -> list[Ruler]:
        """Every line drawn on this image, oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM rulers WHERE image_id = ? ORDER BY id", (image_id,)
        ).fetchall()
        return [_ruler(row) for row in rows]

    @_serialised
    def remove_ruler(self, ruler_id: int) -> None:
        """Forget a line. What an undo of drawing one does."""
        self.get_ruler(ruler_id)
        self._conn.execute("DELETE FROM rulers WHERE id = ?", (ruler_id,))
        self._conn.commit()

    @_serialised
    def restore_ruler(self, ruler: Ruler) -> Ruler:
        """Put one back **as itself**, id intact — the rule M4-T08 set for undo.

        Raises:
            InvalidParameterError: that id is taken, or its image is gone.
        """
        self.get_image(ruler.image_id)
        try:
            self._conn.execute(
                "INSERT INTO rulers (id, image_id, kind, x1, y1, x2, y2, label, created_utc) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ruler.id,
                    ruler.image_id,
                    str(ruler.kind),
                    *ruler.start,
                    *ruler.end,
                    ruler.label,
                    ruler.created_utc,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise InvalidParameterError(f"cannot restore ruler {ruler.id}: {exc}") from exc
        self._conn.commit()
        return self.get_ruler(ruler.id)

    @_serialised
    def mask_of(self, annotation: Annotation) -> np.ndarray | None:
        """The painted mask this annotation points at, or `None` if it has none.

        Raises:
            MissingFileError: the row points at a file that is not there — the
                dangling half of `check_integrity`'s report, met from the side
                that wanted to draw it (ADR-0040).
        """
        if annotation.mask_path is None:
            return None
        return read_mask(self._root / annotation.mask_path)

    @_serialised
    def annotations_for(self, image_id: int) -> list[Annotation]:
        """Every annotation on this image, oldest first.

        Also what a confirmation dialog counts with before `remove_image`
        discards them: they cascade, and they are the one thing in a project
        that cannot be recomputed (ADR-0044).
        """
        rows = self._conn.execute(
            "SELECT * FROM annotations WHERE image_id = ? ORDER BY id", (image_id,)
        )
        return [_annotation(row) for row in rows]

    @_serialised
    def update_annotation(
        self,
        annotation_id: int,
        *,
        box: tuple[float, float, float, float] | None = None,
        label: str | None = None,
        note: str | None = None,
    ) -> Annotation:
        """Change what an annotation says, keeping its identity.

        An edit, not a delete-and-add: the id survives, which is what undo
        (M4-T08) and any future reference to this annotation need. Only the
        fields given are changed; `created_utc` never moves and `updated_utc`
        always does.

        Raises:
            InvalidParameterError: no annotation has that id, or the new box has
                no area.
        """
        current = self.get_annotation(annotation_id)
        new_box = box if box is not None else current.box
        if new_box[2] <= new_box[0] or new_box[3] <= new_box[1]:
            raise InvalidParameterError(
                f"annotation box {new_box} has no area; expected x2 > x1 and y2 > y1"
            )

        self._conn.execute(
            "UPDATE annotations SET label = ?, x1 = ?, y1 = ?, x2 = ?, y2 = ?, "
            "note = ?, updated_utc = ? WHERE id = ?",
            (
                label if label is not None else current.label,
                *new_box,
                note if note is not None else current.note,
                datetime.now(UTC).isoformat(timespec="seconds"),
                annotation_id,
            ),
        )
        self._conn.commit()
        return self.get_annotation(annotation_id)

    @_serialised
    def remove_annotation(self, annotation_id: int) -> None:
        """Delete one annotation.

        Raises:
            InvalidParameterError: no annotation has that id. Silence would make
                a typo look like a successful deletion — and this is hand work.
        """
        cursor = self._conn.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
        self._conn.commit()
        if cursor.rowcount == 0:
            raise InvalidParameterError(f"no annotation with id {annotation_id} in {self._root}")

    @_serialised
    def save_training_run(self, run: TrainingRun) -> None:
        """Store a run and its epochs, replacing what this project knew (M8-T04).

        The whole snapshot every time, because that is what a provider publishes
        — a frozen `TrainingRun` complete in itself (ADR-0080 §3) — and because
        the alternative is this layer deciding which half of it moved.

        The epochs are rewritten rather than appended: a run reports one entry
        per completed epoch and never a sparse list, so the rows this replaces
        are the rows it already wrote. Hundreds of them, once an epoch.
        """
        device = run.device
        self._conn.execute(
            "INSERT INTO training_runs (run_id, status, dataset_root, classes, train_images, "
            "val_images, base_model, epochs, image_size_px, batch_size, requested_device, seed, "
            "output_directory, weights_path, device, started_utc, finished_utc, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET status = excluded.status, "
            "weights_path = excluded.weights_path, device = excluded.device, "
            "finished_utc = excluded.finished_utc, error = excluded.error",
            (
                run.run_id,
                str(run.status),
                run.dataset.root,
                json.dumps(list(run.dataset.classes)),
                run.dataset.train_images,
                run.dataset.val_images,
                run.config.base_model,
                run.config.epochs,
                run.config.image_size_px,
                run.config.batch_size,
                None if run.config.device is None else str(run.config.device),
                run.config.seed,
                run.config.output_directory,
                run.weights_path,
                None if device is None else json.dumps(_device_json(device)),
                run.started_utc,
                run.finished_utc,
                run.error,
            ),
        )
        self._conn.executemany(
            "INSERT INTO training_epochs (run_id, epoch, metrics) VALUES (?, ?, ?) "
            "ON CONFLICT(run_id, epoch) DO UPDATE SET metrics = excluded.metrics",
            [(run.run_id, one.epoch, json.dumps(dict(one.values))) for one in run.metrics],
        )
        self._conn.commit()

    @_serialised
    def get_training_run(self, run_id: str) -> TrainingRun:
        """One stored run, with its epochs.

        Raises:
            InvalidParameterError: this project has no run by that id. A
                different question from `TrainingProvider.status`, which knows
                only the runs this process started (ADR-0084).
        """
        row = self._conn.execute(
            "SELECT * FROM training_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise InvalidParameterError(f"no training run {run_id!r} recorded in {self._root}")
        return _training_run(row, self._epochs_of(run_id))

    @_serialised
    def list_training_runs(self) -> list[TrainingRun]:
        """Every recorded run, oldest first — the order they were started in."""
        rows = self._conn.execute(
            "SELECT * FROM training_runs ORDER BY started_utc, run_id"
        ).fetchall()
        return [_training_run(row, self._epochs_of(row["run_id"])) for row in rows]

    @_serialised
    def register_model(self, descriptor: ModelDescriptor) -> ModelDescriptor:
        """Record a model this project can use, replacing one with the same id.

        Args:
            descriptor: what the model is. `path` is stored relative when the
                weights are inside the project and left absolute when they are
                not (ADR-0050) — the conversion happens here, so a caller may
                hand over either.

        Returns:
            The stored record, with `registered_utc` filled in — and with
            `sha256` computed when the caller gave none and the weights are
            there. ADR-0050 left it `None` *"if nobody computed it"*, and the
            rule for who computes one is this module's oldest: **a checksum
            describes the file the row points at**, because it is taken here
            from that file rather than accepted as an argument (ADR-0040).
            A caller who passed one keeps it; nothing re-reads 137 MB to
            second-guess them.
        """
        path = descriptor.path
        if Path(path).is_absolute():
            # Inside the project it becomes relative like every other path; a
            # shared checkpoint elsewhere stays as it is (ADR-0050).
            with suppress(ValueError):
                path = Path(path).relative_to(self._root).as_posix()
        weights = Path(path) if Path(path).is_absolute() else self._root / path
        stored = replace(
            descriptor,
            path=path,
            sha256=descriptor.sha256 or (sha256_of(weights) if weights.is_file() else None),
            registered_utc=descriptor.registered_utc
            or datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self._conn.execute(
            "INSERT INTO models (model_id, task, framework, path, input_size_px, class_map, "
            "provenance, sha256, registered_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(model_id) DO UPDATE SET task = excluded.task, "
            "framework = excluded.framework, path = excluded.path, "
            "input_size_px = excluded.input_size_px, class_map = excluded.class_map, "
            "provenance = excluded.provenance, sha256 = excluded.sha256, "
            "registered_utc = excluded.registered_utc",
            (
                stored.model_id,
                str(stored.task),
                str(stored.framework),
                stored.path,
                stored.input_size_px,
                json.dumps({str(k): v for k, v in stored.class_map.items()}),
                stored.provenance,
                stored.sha256,
                stored.registered_utc,
            ),
        )
        self._conn.commit()
        return stored

    @_serialised
    def get_model(self, model_id: str) -> ModelDescriptor:
        """One registered model.

        Raises:
            InvalidParameterError: no model has that id — named, because the id
                came from a configuration and a typo there is the likely cause.
        """
        row = self._conn.execute("SELECT * FROM models WHERE model_id = ?", (model_id,)).fetchone()
        if row is None:
            raise InvalidParameterError(f"no model registered as {model_id!r} in {self._root}")
        return _model(row)

    @_serialised
    def list_models(self) -> list[ModelDescriptor]:
        """Every model this project knows about, by id."""
        rows = self._conn.execute("SELECT * FROM models ORDER BY model_id")
        return [_model(row) for row in rows]

    @_serialised
    def path_of_model(self, descriptor: ModelDescriptor) -> Path:
        """Where that model's weights are, whether inside the project or not."""
        return (
            Path(descriptor.path)
            if Path(descriptor.path).is_absolute()
            else self._root / descriptor.path
        )

    @_serialised
    def write_export(self, file_name: str, table: pd.DataFrame) -> str:
        """Write a table into `exports/` and return its path, relative to the root.

        The filesystem half of M4-T11: the use case decides what an export
        *contains*, and this decides where it lands and what it is called. A
        name from an operator's text field is reduced to something a filesystem
        accepts — a `/` in it would otherwise write outside the project
        (ADR-0048).
        """
        safe = re.sub(r"[^\w.\- ]+", "_", file_name).strip() or "export"
        if not safe.lower().endswith(".csv"):
            safe = f"{safe}.csv"

        directory = self._root / _EXPORTS_DIRECTORY
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / safe
        table.to_csv(path, index=False)
        return path.relative_to(self._root).as_posix()

    @_serialised
    def write_export_text(self, relative_name: str, text: str) -> str:
        """Write one text file under `exports/`, and return its path from the root.

        Same division as `write_export` (M4-T11): the use case decides what goes
        in the file, this decides where it may land. Each component of the name
        is reduced the same way, so a `..` or an absolute path from a caller
        cannot write outside the project — the check `write_export` makes by
        flattening the name, made once per component because a YOLO export is a
        directory (M7-T09).
        """
        parts = [
            re.sub(r"[^\w.\- ]+", "_", part).strip(" .") or "_"
            for part in PurePosixPath(relative_name).parts
            if part not in ("", ".", "/")
        ]
        if not parts:
            parts = ["export.txt"]

        path = self._root / _EXPORTS_DIRECTORY / PurePosixPath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path.relative_to(self._root).as_posix()

    def write_cache_text(self, relative_name: str, text: str) -> str:
        """Write one text file under `cache/`, and return its path from the root.

        `cache/` because what goes there is re-creatable and therefore safely
        deletable (PROJECT_RULES §5) — a built training dataset is derived from
        annotations that are still in the database. `exports/` is what an
        operator takes away.

        Not `@_serialised`: this writes a file and touches neither the
        connection nor a row, and holding the repository lock across a directory
        of a hundred images would make the dataset builder the one thing in the
        project that stops every other job.
        """
        path = self._cache_path(relative_name, "dataset.txt")
        path.write_text(text, encoding="utf-8")
        return path.relative_to(self._root).as_posix()

    def write_cache_image(self, relative_name: str, image: np.ndarray) -> str:
        """Write one `uint8` image under `cache/`, and return its path from the root.

        PNG: lossless, because a JPEG artefact on a 5 nm height range is a
        feature the model would learn (ADR-0015's argument about grey levels,
        one step later in the same pipeline).
        """
        import cv2

        path = self._cache_path(relative_name, "image.png")
        if not cv2.imwrite(str(path), image):
            raise InvalidParameterError(f"could not write {path}")
        return path.relative_to(self._root).as_posix()

    def _cache_path(self, relative_name: str, fallback: str) -> Path:
        """A writable path inside `cache/`, with its directory made.

        The same per-component flattening `write_export_text` does, for the same
        reason: a `..` or an absolute path from a caller must not write outside
        the project, and a dataset is a directory rather than a file.
        """
        parts = [
            re.sub(r"[^\w.\- ]+", "_", part).strip(" .") or "_"
            for part in PurePosixPath(relative_name).parts
            if part not in ("", ".", "/")
        ]
        path = self._root / CACHE_DIRECTORY / PurePosixPath(*(parts or [fallback]))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @_serialised
    def get_setting(self, key: str, default: object = None) -> object:
        """A preference this project states, or `default` if it states none.

        Satisfies `core.ports.SettingsStore` (M4-T10), so the merged view in
        `application.settings` can hold a project and a JSON file behind one
        type.
        """
        row = self._conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    @_serialised
    def set_setting(self, key: str, value: object) -> None:
        """State a preference for this project, replacing any earlier one.

        JSON, not `str()`: a store that returns everything as text makes every
        reader parse it back, and one of them gets it wrong (ADR-0047).
        """
        self._conn.execute(
            "INSERT INTO settings (key, value, updated_utc) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
            "updated_utc = excluded.updated_utc",
            (key, json.dumps(value), datetime.now(UTC).isoformat(timespec="seconds")),
        )
        self._conn.commit()

    @_serialised
    def all_settings(self) -> dict[str, object]:
        """Everything this project states, for a settings dialog to show at once."""
        rows = self._conn.execute("SELECT key, value FROM settings ORDER BY key")
        return {row["key"]: json.loads(row["value"]) for row in rows}

    @_serialised
    def check_integrity(self) -> IntegrityReport:
        """Compare the index against the filesystem, in both directions.

        The collection of ADR-0003's stated debt: *"deleting a file behind the
        application's back produces a dangling row; the repository layer must
        reconcile"*. It reconciles by **saying so** — nothing is deleted and
        nothing is imported (ADR-0040).

        Existence, not contents: comparing checksums here would read every scan
        in the project on every open.
        """
        recorded = self.list_images()
        missing = tuple(
            image for image in recorded if not (self._root / image.relative_path).is_file()
        )

        images_dir = self._root / _IMAGES_DIRECTORY
        claimed = {image.relative_path for image in recorded}
        untracked = tuple(
            sorted(
                path.relative_to(self._root).as_posix()
                for path in images_dir.rglob("*")
                if path.is_file() and path.relative_to(self._root).as_posix() not in claimed
            )
        )
        return IntegrityReport(missing_files=missing, untracked_files=untracked)

    def _detections_of(self, run_id: int) -> tuple[Detection, ...]:
        rows = self._conn.execute(
            "SELECT * FROM detections WHERE run_id = ? ORDER BY ordinal", (run_id,)
        )
        return tuple(_detection(row) for row in rows)

    def _epochs_of(self, run_id: str) -> tuple[EpochMetrics, ...]:
        """One entry per completed epoch, in order — the port's promise, read back.

        `EpochMetrics` validates in its constructor, so a row naming a metric
        this application does not know fails here rather than becoming a chart
        (ADR-0080 §4).
        """
        rows = self._conn.execute(
            "SELECT epoch, metrics FROM training_epochs WHERE run_id = ? ORDER BY epoch",
            (run_id,),
        ).fetchall()
        return tuple(
            EpochMetrics(epoch=int(row["epoch"]), values=json.loads(row["metrics"])) for row in rows
        )

    def _write_measurements(self, run_id: int, result: PipelineResult) -> str | None:
        """The measurement table as a file under `results/`, or `None`.

        CSV, because it is what the operator can open and what M4-T11 will
        export — the difference between the two being that an export is a
        decision about *their* file, and this is storage.
        """
        if result.measurements.empty:
            return None

        directory = self._root / _RESULTS_DIRECTORY / f"run_{run_id:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "measurements.csv"
        result.measurements.to_csv(path, index=False)
        return path.relative_to(self._root).as_posix()

    def _free_name(self, file_name: str) -> Path:
        """A path under `images/` that nothing occupies: `a.spm`, then `a_1.spm`.

        Checks the filesystem rather than the index, because an untracked file
        is still a file — copying over one would destroy data the project does
        not even claim to own (ADR-0040 §1 in the other direction).
        """
        images = self._root / _IMAGES_DIRECTORY
        images.mkdir(parents=True, exist_ok=True)
        stem, suffix = Path(file_name).stem, Path(file_name).suffix

        candidate = images / file_name
        attempt = 0
        while candidate.exists():
            attempt += 1
            candidate = images / f"{stem}_{attempt}{suffix}"
        return candidate

    def _relative(self, path: Path | str) -> str:
        """`path` as the database stores it: relative to the root, POSIX separators.

        An absolute path inside the project is accepted — the caller often has
        one, and rejecting it would only move this conversion into every caller.
        One outside it is refused: it would not survive the directory being
        moved, which is the whole reason ADR-0003 requires relative paths.
        """
        candidate = Path(path)
        if candidate.is_absolute():
            try:
                candidate = candidate.relative_to(self._root)
            except ValueError as exc:
                raise InvalidParameterError(
                    f"path {path} is outside the project at {self._root}; "
                    "every stored path is relative to the project root"
                ) from exc
        if ".." in candidate.parts:
            raise InvalidParameterError(
                f"path {path} escapes the project at {self._root}; "
                "every stored path is relative to the project root"
            )
        return candidate.as_posix()


def sha256_of(path: Path) -> str:
    """The file's SHA-256, read in chunks so a large scan is never held whole.

    Public because it is what "the checksum in the row" *means*: a caller
    verifying a file computes it the same way, or the comparison is between two
    different definitions.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _run(row: sqlite3.Row, detections: tuple[Detection, ...]) -> AnalysisRun:
    return AnalysisRun(
        id=row["id"],
        image_id=row["image_id"],
        detector=row["detector"],
        mode=row["mode"],
        modality=Modality(row["modality"]),
        pixel_size_nm=row["pixel_size_nm"],
        measurements_path=row["measurements_path"],
        created_utc=row["created_utc"],
        model_id=row["model_id"],
        detections=detections,
    )


def _model(row: sqlite3.Row) -> ModelDescriptor:
    return ModelDescriptor(
        model_id=row["model_id"],
        task=ModelTask(row["task"]),
        framework=ModelFramework(row["framework"]),
        path=row["path"],
        input_size_px=row["input_size_px"],
        class_map={int(k): v for k, v in json.loads(row["class_map"]).items()},
        provenance=row["provenance"],
        sha256=row["sha256"],
        registered_utc=row["registered_utc"],
    )


def _device_json(device: Device) -> dict[str, str]:
    """A resolved device as one JSON object.

    Three fields that are absent together: a run that never started ran nowhere,
    and three nullable columns can disagree about that (ADR-0084).
    """
    return {"kind": str(device.kind), "name": device.name, "torch_name": device.torch_name}


def _training_run(row: sqlite3.Row, metrics: tuple[EpochMetrics, ...]) -> TrainingRun:
    device = json.loads(row["device"]) if row["device"] else None
    return TrainingRun(
        run_id=row["run_id"],
        status=TrainingStatus(row["status"]),
        dataset=DatasetSpec(
            root=row["dataset_root"],
            classes=tuple(json.loads(row["classes"])),
            train_images=row["train_images"],
            val_images=row["val_images"],
        ),
        config=TrainingConfig(
            base_model=row["base_model"],
            epochs=row["epochs"],
            image_size_px=row["image_size_px"],
            batch_size=row["batch_size"],
            device=None if row["requested_device"] is None else DeviceKind(row["requested_device"]),
            seed=row["seed"],
            output_directory=row["output_directory"],
        ),
        metrics=metrics,
        weights_path=row["weights_path"],
        device=None
        if device is None
        else Device(
            kind=DeviceKind(device["kind"]),
            name=device["name"],
            torch_name=device["torch_name"],
        ),
        started_utc=row["started_utc"],
        finished_utc=row["finished_utc"],
        error=row["error"],
    )


def _bounds(points: tuple[tuple[float, float], ...]) -> tuple[float, float, float, float]:
    """The outline's bounding box — derived, never typed in (ADR-0072)."""
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return min(xs), min(ys), max(xs), max(ys)


def _mask_bounds(mask: np.ndarray) -> tuple[float, float, float, float]:
    """The painted pixels' bounding box — derived, like an outline's (ADR-0072).

    Inclusive on the far edge by one pixel, so a single painted pixel is a box
    with area rather than the zero-area one the `CHECK` refuses.
    """
    ys, xs = np.nonzero(mask)
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def _points_json(points: tuple[tuple[float, float], ...] | None) -> str | None:
    """The outline as text, or `None` for a box drawn as a box.

    JSON rather than a second table: an outline is read and written whole,
    always, and never queried by vertex.
    """
    return None if points is None else json.dumps([[x, y] for x, y in points])


def _annotation(row: sqlite3.Row) -> Annotation:
    return Annotation(
        id=row["id"],
        image_id=row["image_id"],
        label=row["label"],
        box=(row["x1"], row["y1"], row["x2"], row["y2"]),
        source=AnnotationSource(row["source"]),
        note=row["note"],
        created_utc=row["created_utc"],
        updated_utc=row["updated_utc"],
        points=(
            None
            if row["points"] is None
            else tuple((float(x), float(y)) for x, y in json.loads(row["points"]))
        ),
        mask_path=row["mask_path"],
    )


def _ruler(row: sqlite3.Row) -> Ruler:
    return Ruler(
        id=row["id"],
        image_id=row["image_id"],
        kind=RulerKind(row["kind"]),
        start=(row["x1"], row["y1"]),
        end=(row["x2"], row["y2"]),
        label=row["label"],
        created_utc=row["created_utc"],
    )


def _detection(row: sqlite3.Row) -> Detection:
    """One stored detection, back as the entity the science speaks in.

    A box that was absent stays absent: `bbox` is `None` rather than four
    `None`s wearing a tuple, which is ADR-0031's rule about the LoG path arriving
    at the database intact.
    """
    corners = (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"])
    return Detection(
        x_px=row["x_px"],
        y_px=row["y_px"],
        radius_px=row["radius_px"],
        radius_nm=row["radius_nm"],
        confidence=row["confidence"],
        bbox=None if corners[0] is None else corners,
    )


def _record(row: sqlite3.Row) -> ImageRecord:
    """One database row as the entity the layers above speak in."""
    return ImageRecord(
        id=row["id"],
        relative_path=row["relative_path"],
        display_name=row["display_name"],
        modality=Modality(row["modality"]),
        sha256=row["sha256"],
        pixel_size_nm=row["pixel_size_nm"],
        imported_utc=row["imported_utc"],
    )
