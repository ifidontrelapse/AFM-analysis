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

What is *not* here: creating the directory and copying an imported file into it
(M4-T04), and acting on what `check_integrity` reports — that is a decision with
an operator behind it (ADR-0040).
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from nanoscope.core.entities.project import ImageRecord, IntegrityReport
from nanoscope.core.errors import InvalidParameterError, MissingFileError
from nanoscope.core.values import Modality
from nanoscope.infrastructure.storage.database import open_database
from nanoscope.infrastructure.storage.project_format import (
    ProjectManifest,
    open_manifest,
)

_IMAGES_DIRECTORY = "images"

#: 1 MiB. Large enough that the loop is not the cost, small enough that a
#: multi-gigabyte scan does not arrive in memory to be hashed.
_HASH_CHUNK_BYTES = 1024 * 1024


class SqliteProjectRepository:
    """One open project: its manifest, its database, and the files it indexes.

    Use it as a context manager, or call `close`. Opening is `open()`, which
    refuses a directory that is not a project and migrates the database forward
    if it is an older one.
    """

    def __init__(self, root: Path, manifest: ProjectManifest, conn: sqlite3.Connection) -> None:
        self._root = root
        self._manifest = manifest
        self._conn = conn

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

    def get_image(self, image_id: int) -> ImageRecord:
        """The row with this id.

        Raises:
            InvalidParameterError: no image has that id.
        """
        row = self._conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        if row is None:
            raise InvalidParameterError(f"no image with id {image_id} in {self._root}")
        return _record(row)

    def list_images(self) -> list[ImageRecord]:
        """Every image in the project, in the order they were imported."""
        rows = self._conn.execute("SELECT * FROM images ORDER BY id")
        return [_record(row) for row in rows]

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
