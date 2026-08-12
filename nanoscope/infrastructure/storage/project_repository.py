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
import shutil
import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from types import TracebackType
from typing import Any, Self, cast

import pandas as pd

from nanoscope.core.entities import Detection, PipelineResult
from nanoscope.core.entities.project import AnalysisRun, ImageRecord, IntegrityReport
from nanoscope.core.errors import InvalidParameterError, MissingFileError
from nanoscope.core.values import Modality
from nanoscope.infrastructure.storage.database import open_database
from nanoscope.infrastructure.storage.project_format import (
    DIRECTORIES,
    ProjectManifest,
    new_manifest,
    open_manifest,
    write_manifest,
)

_IMAGES_DIRECTORY = "images"
_RESULTS_DIRECTORY = "results"

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
    def save_analysis(self, image_id: int, result: PipelineResult) -> AnalysisRun:
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

        Returns:
            The stored run, with its id and its detections.

        Raises:
            InvalidParameterError: no image has that id.
        """
        self.get_image(image_id)

        cursor = self._conn.execute(
            "INSERT INTO analysis_runs "
            "(image_id, detector, mode, modality, pixel_size_nm, measurements_path, created_utc) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?)",
            (
                image_id,
                result.detector_name,
                result.mode,
                str(result.modality),
                result.pixel_size_nm,
                datetime.now(UTC).isoformat(timespec="seconds"),
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
        detections=detections,
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
