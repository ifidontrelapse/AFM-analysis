"""What the application needs from a project's index (M4-T03).

The second port in this package, and it arrives under the rule
`core/ports/__init__.py` wrote for itself: *the rest ship with their first
adapter*. `SqliteProjectRepository` is that adapter.

It is not decoration. M4-T04's use cases live in `application/`, which may
import `core` and nothing else (Architecture §3.2) — typing a use case against
the SQLite class would put `infrastructure` on the application's import list and
`sqlite3` in its vocabulary.

What a `Protocol` buys over an ABC: the implementation does not import this
module, so the arrow still points inward from `infrastructure` to `core` without
a base class in the middle, and mypy checks the shape structurally.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from nanoscope.core.entities import PipelineResult
from nanoscope.core.entities.project import (
    AnalysisRun,
    Annotation,
    AnnotationSource,
    ImageRecord,
    IntegrityReport,
)
from nanoscope.core.values import Modality

if TYPE_CHECKING:  # pandas is heavy, and importing the domain must stay cheap (M2-T09).
    import pandas as pd


class ProjectRepository(Protocol):
    """The images in one open project, and their agreement with the disk."""

    @property
    def name(self) -> str:
        """The project's display name, from wherever the adapter keeps identity
        — the manifest, in the one that exists (ADR-0038)."""
        ...

    def add_image(
        self,
        relative_path: str,
        *,
        modality: Modality,
        display_name: str | None = None,
        pixel_size_nm: float | None = None,
    ) -> ImageRecord:
        """Record a file that is **already inside the project** and return its row.

        The checksum is computed here, from the file, and is not a parameter: a
        checksum a caller passes in can describe a different file, and then the
        only thing it proves is that two callers agreed (ADR-0040).
        """
        ...

    def import_image(
        self,
        source: Path | str,
        *,
        modality: Modality,
        display_name: str | None = None,
        pixel_size_nm: float | None = None,
    ) -> ImageRecord:
        """Copy a file into the project and record it, returning its row.

        The copy is the adapter's, because `application` may not touch the
        filesystem (Architecture §3.2) — which is why this is a port method and
        not something `import_images` does for itself.
        """
        ...

    def get_image(self, image_id: int) -> ImageRecord:
        """The row with this id."""
        ...

    def path_of(self, image: ImageRecord) -> Path:
        """Where that image's file actually is, right now.

        The adapter resolves it, because a project path assembled anywhere
        else is a project path assembled outside `infrastructure/storage`,
        which ADR-0038's compliance section rules out by name."""
        ...

    def list_images(self) -> list[ImageRecord]:
        """Every image in the project, in the order they were imported."""
        ...

    def remove_image(self, image_id: int) -> None:
        """Forget the row. The file is not touched — deleting it is a separate
        decision, and one this layer is not allowed to make on its own."""
        ...

    def save_analysis(self, image_id: int, result: PipelineResult) -> AnalysisRun:
        """Store what an analysis found, and return its index entry.

        Where each half of it goes is the adapter's business (ADR-0042): this
        layer knows only that a run is stored whole and comes back whole.
        """
        ...

    def get_run(self, run_id: int) -> AnalysisRun:
        """One stored analysis, with its detections."""
        ...

    def runs_for(self, image_id: int) -> list[AnalysisRun]:
        """Every analysis of this image, oldest first."""
        ...

    def measurements_for(self, run: AnalysisRun) -> pd.DataFrame:
        """The measurement table this run produced."""
        ...

    def add_annotation(
        self,
        image_id: int,
        box: tuple[float, float, float, float],
        *,
        label: str,
        source: AnnotationSource = AnnotationSource.MANUAL,
        note: str | None = None,
    ) -> Annotation:
        """Record a box the operator drew, and return it with its id."""
        ...

    def restore_annotation(self, annotation: Annotation) -> Annotation:
        """Put a deleted annotation back as itself, id intact.

        What undo needs and creation must not have: everything else on the
        stack refers to an annotation by id, so restoring it as a new row makes
        every command above it point at nothing (ADR-0045).
        """
        ...

    def get_annotation(self, annotation_id: int) -> Annotation:
        """One annotation."""
        ...

    def annotations_for(self, image_id: int) -> list[Annotation]:
        """Every annotation on this image, oldest first."""
        ...

    def update_annotation(
        self,
        annotation_id: int,
        *,
        box: tuple[float, float, float, float] | None = None,
        label: str | None = None,
        note: str | None = None,
    ) -> Annotation:
        """Change what an annotation says, keeping its id."""
        ...

    def remove_annotation(self, annotation_id: int) -> None:
        """Delete one annotation. The operator's hand work — never silent."""
        ...

    def check_integrity(self) -> IntegrityReport:
        """Where the index and the filesystem disagree. Reports; changes nothing."""
        ...

    def close(self) -> None:
        """Release the database. The project directory stays where it is."""
        ...
