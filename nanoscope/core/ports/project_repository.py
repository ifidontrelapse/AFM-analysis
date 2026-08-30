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

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from nanoscope.core.entities import PipelineResult
from nanoscope.core.entities.model import ModelDescriptor
from nanoscope.core.entities.project import (
    AnalysisRun,
    Annotation,
    AnnotationSource,
    ImageRecord,
    IntegrityReport,
    Ruler,
    RulerKind,
)
from nanoscope.core.values import Modality

if TYPE_CHECKING:
    import numpy as np  # pandas is heavy, and importing the domain must stay cheap (M2-T09).
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
        points: Sequence[tuple[float, float]] | None = None,
        mask: np.ndarray | None = None,
    ) -> Annotation:
        """Record a box the operator drew, and return it with its id.

        `points` is the outline when they drew one and `mask` is a painted one;
        `box` is derived from whichever was given, so the shape and its bounding
        box cannot disagree (ADR-0072, ADR-0073). A mask is **written to a
        file** and the row keeps its path (PROJECT_RULES §5).
        """
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

    def add_ruler(
        self,
        image_id: int,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        kind: RulerKind = RulerKind.DISTANCE,
        label: str,
    ) -> Ruler:
        """Record a line an operator drew. Its length is not stored (ADR-0074)."""
        ...

    def get_ruler(self, ruler_id: int) -> Ruler: ...

    def rulers_for(self, image_id: int) -> list[Ruler]:
        """Every line drawn on this image, oldest first."""
        ...

    def remove_ruler(self, ruler_id: int) -> None: ...

    def restore_ruler(self, ruler: Ruler) -> Ruler:
        """Put one back as itself, id intact — undo's rule (ADR-0045)."""
        ...

    def mask_of(self, annotation: Annotation) -> np.ndarray | None:
        """The painted mask it points at, or `None` when it has none."""
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

    def write_export(self, file_name: str, table: pd.DataFrame) -> str:
        """Write a table into `exports/`, returning its path relative to the root.

        The use case decides what an export contains; the adapter decides where
        it lands, because `application` may not touch the filesystem.
        """
        ...

    def write_export_text(self, relative_name: str, text: str) -> str:
        """Write one text file into `exports/`, returning its path from the root.

        Beside `write_export` rather than inside it: that one is a `DataFrame`
        and a `.csv` name, and a YOLO export is a directory of small text files
        with one class list beside them (M7-T09). `relative_name` may name
        subdirectories with `/`; the adapter is what decides they stay inside
        `exports/`.
        """
        ...

    def write_cache_text(self, relative_name: str, text: str) -> str:
        """Write one text file into `cache/`, returning its path from the root.

        `cache/` rather than `exports/`, and the difference is the rule
        PROJECT_RULES §5 states: *anything under `cache/` must be safely
        deletable at any time without data loss.* An export is what an operator
        takes away (ADR-0067); a built training dataset is derived from
        annotations that are still in the database, so it is re-creatable by
        definition — which is what `cache/` means (M8-T02).
        """
        ...

    def write_cache_image(self, relative_name: str, image: np.ndarray) -> str:
        """Write one `uint8` image into `cache/`, returning its path from the root.

        Beside `write_cache_text` for the reason `write_export_text` sits beside
        `write_export`: a dataset is a directory of pictures *and* a directory of
        small text files, and `application` may encode neither — writing a PNG is
        `cv2` (Architecture §3.2, and the division ADR-0073 already made for a
        painted mask).
        """
        ...

    def register_model(self, descriptor: ModelDescriptor) -> ModelDescriptor:
        """Record a model this project can use."""
        ...

    def get_model(self, model_id: str) -> ModelDescriptor:
        """One registered model, by the id a configuration names."""
        ...

    def list_models(self) -> list[ModelDescriptor]:
        """Every model this project knows about."""
        ...

    def path_of_model(self, descriptor: ModelDescriptor) -> Path:
        """Where that model's weights are, inside the project or not."""
        ...

    def get_setting(self, key: str, default: object = None) -> object:
        """A preference this project states. Satisfies `SettingsStore`."""
        ...

    def set_setting(self, key: str, value: object) -> None:
        """State a preference for this project."""
        ...

    def all_settings(self) -> dict[str, object]:
        """Everything this project states."""
        ...

    def check_integrity(self) -> IntegrityReport:
        """Where the index and the filesystem disagree. Reports; changes nothing."""
        ...

    def close(self) -> None:
        """Release the database. The project directory stays where it is."""
        ...
