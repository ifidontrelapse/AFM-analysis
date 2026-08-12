"""What a project knows about the files in it (M4-T03).

These are what the repository hands back — never a `sqlite3.Row`, which is
untyped and would put the database's vocabulary into every layer above it. The
port that returns them is `core.ports.ProjectRepository`; the implementation is
`infrastructure.storage.SqliteProjectRepository`.
"""

from __future__ import annotations

from dataclasses import dataclass

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
