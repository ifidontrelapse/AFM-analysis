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

from dataclasses import replace

from nanoscope.application.use_cases.pipeline import run_pipeline
from nanoscope.application.use_cases.preprocessing import (
    PreprocessingParams,
    afm_format,
    run_preprocessing,
)
from nanoscope.core.entities import (
    AnalysisRun,
    MicroscopyData,
    PipelineConfig,
    PreprocessingResult,
)
from nanoscope.core.errors import MissingFileError
from nanoscope.core.ports import ProjectRepository
from nanoscope.core.values import Modality
from nanoscope.infrastructure.storage import load_microscopy_image


def run_analysis(
    repository: ProjectRepository,
    image_id: int,
    config: PipelineConfig,
    predictor: object | None = None,
    preprocessing: PreprocessingParams | None = None,
    model_id: str | None = None,
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
        preprocessing: the levelling and substrate parameters for an AFM scan.
            Its defaults are `run_preprocessing`'s own, so omitting it is what
            this function always did — and passing it is what stops a scan
            *previewed* at one opening scale being *analysed* at another
            without anything saying so (M6-T02).
        model_id: which registered model the detector should load, by the id its
            operator gave it (ADR-0050). Resolved to a path **here**, because
            `repository.path_of_model` is the only thing allowed to join a
            project root to a relative path (ADR-0038), and recorded on the run
            so that *which model found these particles?* stays answerable once a
            project has two (ADR-0086).

            An argument rather than a field on `PipelineConfig`: the golden
            records that dataclass's field **names**, and the Roadmap's third
            sequencing rule keeps a golden update out of a use case's commit.
            `predictor` and `preprocessing` travel the same way for the same
            practical reason.

    Returns:
        The stored run, with its detections and the path to its measurements.

    Raises:
        InvalidParameterError: no image has that id.
        UnsupportedRequestError: the file's extension is one this application
            cannot analyse, or the (modality, detector, mode) combination does
            not exist — the second checked by `run_pipeline` before inference.
            Also a detector that needs weights when `model_id` names none.
        MissingFileError: the image's file is gone. That is the dangling row
            `check_integrity` reports (ADR-0040), met from the other side.
    """
    #: Resolved before the file is read, so a run naming a model this project
    #: does not have costs nothing — the same placement `validate_request` gets
    #: for the same reason (D-14, M2-T10).
    if model_id is not None:
        descriptor = repository.get_model(model_id)
        if not repository.path_of_model(descriptor).is_file():
            #: The row is real and the file is not: ADR-0040's dangling-row
            #: report, met from the model side. Refused here rather than left to
            #: a framework's `FileNotFoundError` halfway through a run, which is
            #: what this application did until M8-T06 (PROJECT_RULES §3).
            raise MissingFileError(
                f"the weights for model {model_id!r} are not at "
                f"{repository.path_of_model(descriptor)}. A model registered on another "
                "machine, or a file that has been moved"
            )
        config = replace(config, yolo_model_path=str(repository.path_of_model(descriptor)))

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
        params = preprocessing or PreprocessingParams()
        data = run_preprocessing(
            path,
            fmt=afm_format(path),
            pixel_size_nm=record.pixel_size_nm,
            min_size_nm=params.min_size_nm,
            manual_radius_px=params.manual_radius_px,
            opening_scale=params.opening_scale,
        )
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
    run = repository.save_analysis(image_id, result, model_id=model_id)
    #: The masks travel back with the run they came from, and **only** with that
    #: one: the repository stores none, so a run read back later has an empty
    #: tuple and an overlay drawn from it would be showing something the project
    #: cannot restore (ADR-0042, ADR-0064).
    return replace(run, masks=tuple(result.masks))
