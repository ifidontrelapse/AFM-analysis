"""Analyse an image in a project, and keep what was found (M4-T05, ADR-0042).

The first use case in this milestone that calls the scientific core rather than
arranging files around it — and therefore the first place a golden number could
move. It must not: `run_pipeline` is called, not modified, and everything here
happens on either side of it.

**One use case, where the task named three.** `RunDetection`,
`RunSegmentation` and `MeasureParticles` are `run_pipeline` with `mode` set to
`"detect"`, `"segment"` and `"baseline"` — a value `PipelineConfig` already
carries and `capabilities.py` already validates before anything runs. Three
functions differing by a string literal is ADR-0041's case one task later.

The loaders are imported by name here, as `use_cases/preprocessing.py` and
`use_cases/pipeline.py` already do. An `ImageLoader` port is the right answer
and it is not this task's: `core/ports/__init__.py` dates it M2-T10 / M6, and
introducing it here means rewriting two existing call sites in the same commit
as the first persistence code. Debt, taken on purpose, with its trigger written
down (ADR-0042 §5).
"""

from __future__ import annotations

from pathlib import Path

from nanoscope.application.use_cases.pipeline import run_pipeline
from nanoscope.application.use_cases.preprocessing import run_preprocessing
from nanoscope.core.entities import (
    AnalysisRun,
    MicroscopyData,
    PipelineConfig,
    PreprocessingResult,
)
from nanoscope.core.errors import UnsupportedRequestError
from nanoscope.core.ports import ProjectRepository
from nanoscope.core.values import Modality
from nanoscope.infrastructure.storage import load_microscopy_image

#: What each modality's file needs before the pipeline can see it. AFM arrives
#: as a height map that must be levelled and given a substrate; SEM and TEM
#: arrive as images and are analysed as they are (ADR-0031's `blocks_for` says
#: the same thing about what they can produce).
_AFM_FORMATS = {".spm": "spm", ".npy": "npy"}


def run_analysis(
    repository: ProjectRepository,
    image_id: int,
    config: PipelineConfig,
    predictor: object | None = None,
) -> AnalysisRun:
    """Run the pipeline over one of the project's images and store the result.

    Loads the file the row points at, preprocesses it if it is AFM, runs
    `run_pipeline`, and hands what came back to the repository — which puts the
    run and its detections in the database and the measurement table in
    `results/` (ADR-0042).

    Args:
        repository: an open project.
        image_id: which of its images to analyse.
        config: the detector, the mode, and their parameters. The mode is what
            makes this "detect", "measure" or "segment"; nothing here branches
            on it.
        predictor: an initialised SAM2 predictor, required when
            `config.mode == "segment"` and rejected before any file is read.

    Returns:
        The stored run, with its detections and the path to its measurements.

    Raises:
        InvalidParameterError: no image has that id.
        UnsupportedRequestError: the file's extension is one this application
            cannot analyse, or the (modality, detector, mode) combination does
            not exist — the second checked by `run_pipeline` before inference.
        MissingFileError: the image's file is gone. That is the dangling row
            `check_integrity` reports (ADR-0040), met from the other side.
    """
    record = repository.get_image(image_id)
    #: The repository resolves it. Joining `root` and a relative path here would
    #: be a project path built outside `infrastructure/storage`, which ADR-0038's
    #: compliance section rules out by name.
    path = repository.path_of(record)

    data: PreprocessingResult | MicroscopyData
    if record.modality is Modality.AFM:
        # The scale the project recorded is the scale the analysis uses. An npy
        # carries no metadata, so without this an image imported *with* a known
        # scale is analysed as though it had none — every `radius_nm` comes back
        # `None` and the physical minimum-size filter is skipped, which is the
        # D-07 family of defect M3 spent a milestone removing, reintroduced one
        # layer up. An SPM's header wins, because `load_afm` ignores the
        # argument there.
        data = run_preprocessing(path, fmt=_afm_format(path), pixel_size_nm=record.pixel_size_nm)
    else:
        # `Literal["sem", "tem"]` where the record carries a `Modality`. The
        # value is the same string — `Modality` is a `StrEnum` — and adopting
        # the enum through the loaders is M2-T10's remaining half, not this
        # task's.
        data = load_microscopy_image(
            str(path),
            modality=record.modality.value,  # type: ignore[arg-type]
            nm_per_pixel=record.pixel_size_nm,
        )

    result = run_pipeline(data, config, predictor=predictor)
    return repository.save_analysis(image_id, result)


def _afm_format(path: Path) -> str:
    """The `fmt` string `load_afm` expects, from the file's own extension.

    Raises:
        UnsupportedRequestError: an extension with no AFM reader. Nothing about
            the request is malformed — this version has no path for that file.
    """
    fmt = _AFM_FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise UnsupportedRequestError(
            f"no AFM reader for {path.name}; supported extensions are "
            f"{', '.join(sorted(_AFM_FORMATS))}"
        )
    return fmt
