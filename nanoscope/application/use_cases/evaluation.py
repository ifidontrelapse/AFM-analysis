"""Whether the new model is better, from what the project already kept (M8-T08, ADR-0088).

The Roadmap states M8's risk against the milestone rather than any one task:
*"new models change detections by design. Model comparison is reported through
the M3 evaluation harness."* Until something reports it, that sentence is a
licence rather than a control — M8-T05 produces models, M8-T06 lets a project
choose between them, and nothing says which one is right.

**And the harness has been waiting since M3-T15.** ADR-0032 put it in
`core/science/` and said why it was not in `tests/`: *"M4's annotation flow and
M8's training loop need it."* It closed on a sentence this is the first thing
able to answer — **a phantom is not a sample**. M7 built the sample.

**This runs no model.** M8-T06 added `analysis_runs.model_id` one task ago, so
the project already holds every detection each model made in it, beside the
annotations that are the truth. Re-running inference here would need ultralytics,
would put the gate behind a GPU (PROJECT_RULES §6), and — the part that matters
— would score a *different* run from the one the operator looked at.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nanoscope.core.entities.project import AnalysisRun, Annotation, AnnotationSource
from nanoscope.core.entities.training import TrainingRun, TrainingStatus
from nanoscope.core.ports import ProjectRepository
from nanoscope.core.science.evaluation import DetectionMetrics, evaluate_detections

logger = logging.getLogger(__name__)

#: What a scan was to a model. `UNKNOWN` is not a failure to look — it is the
#: measured consequence of ADR-0081 putting datasets in `cache/`: the *counts*
#: of a split live on `DatasetSpec` for ever, and the *membership* lives in a
#: directory an operator is told they may delete.
UNSEEN, TRAINED_ON, UNKNOWN = "unseen", "trained-on", "unknown"


@dataclass(frozen=True)
class ImageScore:
    """One model, scored on one scan, against what a person drew there."""

    image_id: int
    display_name: str
    #: The stored run whose detections were scored — so a reader can go and look
    #: at the very boxes this number came from.
    run_id: int
    #: `UNSEEN`, `TRAINED_ON` or `UNKNOWN`. **Never guessed**, and the third is
    #: an answer rather than an absence of one.
    exposure: str
    metrics: DetectionMetrics

    @property
    def counts_as_generalisation(self) -> bool:
        """Whether this number says anything about scans the model has not seen."""
        return self.exposure == UNSEEN


@dataclass(frozen=True)
class ModelScore:
    """What one model scored, and on what.

    Two totals, because they answer different questions and only one of them is
    about generalisation. Both are `None` when there is nothing to total: an
    aggregate that reports 0.0 for *no data* is the substitute value ADR-0032
    deleted, coming back through a sum.
    """

    model_id: str
    images: tuple[ImageScore, ...]
    #: Over the scans this model was **not** trained on. The number to compare.
    unseen: DetectionMetrics | None
    #: Over every scan it was scored on, whatever the model saw in training.
    overall: DetectionMetrics | None

    @property
    def unseen_images(self) -> int:
        return sum(1 for one in self.images if one.counts_as_generalisation)

    @property
    def exposure_is_known(self) -> bool:
        """Whether every scan could be placed on one side of the split.

        `False` means a dataset directory is gone, so *"the model never saw
        this"* cannot be said about at least one scan (ADR-0081's stated cost).
        """
        return all(one.exposure != UNKNOWN for one in self.images)


@dataclass(frozen=True)
class EvaluationReport:
    """Every model this project has detections from, over the same scans."""

    models: tuple[ModelScore, ...]
    #: The images every score was computed over, in the order they were read.
    #: Two models scored on different scans produce two numbers that cannot be
    #: subtracted, so what they were scored on is part of the report.
    image_ids: tuple[int, ...]
    #: Which annotations counted as truth.
    sources: tuple[AnnotationSource, ...]
    match_factor: float


def evaluate_models(
    repository: ProjectRepository,
    *,
    sources: Iterable[AnnotationSource] | None = (AnnotationSource.MANUAL,),
    match_factor: float = 1.0,
) -> EvaluationReport:
    """Score every model this project has stored detections from.

    Args:
        repository: an open project.
        sources: which annotations count as **truth**. `(MANUAL,)` by default and
            the caller may widen it, which is ADR-0044's rule at its third site:
            scoring a model against boxes adopted from a detector is scoring it
            against a detector. `None` means every annotation, said out loud.
        match_factor: multiplier on a particle's own radius, passed through to
            the harness. `1.0` means the detection's centre must land inside the
            particle — scale-free, which is why M3-T15 chose it over a pixel
            distance.

    Returns:
        One `ModelScore` per model, each carrying its per-image rows. Models with
        no stored detections do not appear: a model this project has never run is
        not a model that scored badly.
    """
    scope = None if sources is None else tuple(sources)
    truth = _truth_by_image(repository, scope)
    exposure = _exposure_by_model(repository)

    by_model: dict[str, list[ImageScore]] = {}
    scored_images: list[int] = []
    for record in repository.list_images():
        if record.id not in truth:
            continue
        scored_images.append(record.id)
        for run in _latest_run_per_model(repository.runs_for(record.id)):
            centres, radii = truth[record.id]
            by_model.setdefault(str(run.model_id), []).append(
                ImageScore(
                    image_id=record.id,
                    display_name=record.display_name,
                    run_id=run.id,
                    exposure=_exposure_of(exposure, str(run.model_id), record.id),
                    metrics=evaluate_detections(
                        run.detections,
                        centres,
                        radii,
                        match_factor=match_factor,
                        pixel_size_nm=record.pixel_size_nm,
                    ),
                )
            )

    models = tuple(
        ModelScore(
            model_id=model_id,
            images=tuple(rows),
            unseen=_total([one for one in rows if one.counts_as_generalisation], match_factor),
            overall=_total(rows, match_factor),
        )
        for model_id, rows in sorted(by_model.items())
    )
    logger.info("evaluated %d model(s) over %d annotated scan(s)", len(models), len(scored_images))
    return EvaluationReport(
        models=models,
        image_ids=tuple(scored_images),
        sources=scope or tuple(AnnotationSource),
        match_factor=match_factor,
    )


def truth_of(annotations: Sequence[Annotation]) -> tuple[np.ndarray, np.ndarray]:
    """Boxes as the centres and radii the harness scores against.

    `min(w, h) / 2`, which is **what a detector's own boxes already become**
    (`infrastructure/models/yolo.py`). The same rule on both sides is not a
    tidiness preference: the harness matches *a centre inside the particle*, so
    the truth radius **is** the tolerance, and a truth circumscribed while the
    detections are inscribed would compare two different circles and report the
    difference as a localisation error.

    Returns:
        `(N, 2)` centres as `[y, x]` — the project's array convention — and
        `(N,)` radii in pixels.
    """
    centres = np.array(
        [[(y1 + y2) / 2, (x1 + x2) / 2] for x1, y1, x2, y2 in (one.box for one in annotations)],
        dtype=np.float64,
    ).reshape(-1, 2)
    radii = np.array(
        [min(x2 - x1, y2 - y1) / 2 for x1, y1, x2, y2 in (one.box for one in annotations)],
        dtype=np.float64,
    )
    return centres, radii


def _truth_by_image(
    repository: ProjectRepository, sources: tuple[AnnotationSource, ...] | None
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """The ground truth of every scan that has any, by image id."""
    truth: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for record in repository.list_images():
        kept = [
            one
            for one in repository.annotations_for(record.id)
            if sources is None or one.source in sources
        ]
        if kept:
            truth[record.id] = truth_of(kept)
    return truth


def _latest_run_per_model(runs: Sequence[AnalysisRun]) -> list[AnalysisRun]:
    """One run per model on this scan — the newest, and only ones naming a model.

    Newest because re-running a model on a scan is what an operator does after
    changing something, and scoring both would count one model twice on one
    image. A run with no `model_id` used no registered model (every `log` run,
    and every run stored before M8-T06), and there is nothing to attribute it to.
    """
    latest: dict[str, AnalysisRun] = {}
    for run in runs:
        if run.model_id:
            latest[run.model_id] = run
    return list(latest.values())


def _exposure_by_model(repository: ProjectRepository) -> dict[str, set[int] | None]:
    """Which image ids each model was **trained on**, or `None` when unknowable.

    Two joins, neither of which needed new storage:

    1. **model → training run**, on the weights path. M8-T04 registers a model
       with `path = run.weights_path`, so the two are the same string.
    2. **run → the scans it saw**, from the dataset directory's `images/train`
       stems, which map back to image records by filename.

    `None` when the run cannot be found or its dataset is gone — and it usually
    will be gone eventually, because ADR-0081 put datasets in `cache/` precisely
    so an operator could delete them. Measured: the *counts* of a split live on
    `DatasetSpec` for ever and the *membership* lives only on disk.
    """
    stems = {Path(record.relative_path).stem: record.id for record in repository.list_images()}
    runs = {
        run.weights_path: run
        for run in repository.list_training_runs()
        if run.status is TrainingStatus.SUCCEEDED and run.weights_path
    }

    exposure: dict[str, set[int] | None] = {}
    for model in repository.list_models():
        run = runs.get(model.path)
        exposure[model.model_id] = None if run is None else _trained_on(repository, run, stems)
    return exposure


def _trained_on(
    repository: ProjectRepository, run: TrainingRun, stems: dict[str, int]
) -> set[int] | None:
    """The image ids in this run's training split, or `None` if it is not there."""
    directory = Path(repository.root) / run.dataset.root / "images" / "train"
    if not directory.is_dir():
        logger.info(
            "cannot say what run %s trained on: %s is gone (cache is deletable, ADR-0081)",
            run.run_id,
            run.dataset.root,
        )
        return None
    return {stems[path.stem] for path in directory.glob("*") if path.stem in stems}


def _exposure_of(exposure: dict[str, set[int] | None], model_id: str, image_id: int) -> str:
    """What this scan was to this model. `UNKNOWN` is an answer, not a gap."""
    trained_on = exposure.get(model_id)
    if trained_on is None:
        return UNKNOWN
    return TRAINED_ON if image_id in trained_on else UNSEEN


def _total(rows: Sequence[ImageScore], match_factor: float) -> DetectionMetrics | None:
    """Sum the counts, then recompute the ratios. **Never average the ratios.**

    A mean of per-image precisions weights a scan with two particles the same as
    one with two hundred, and it has no denominator to be honest about — so a
    scan the harness scored `None` would have to become a number to be included.
    Summing the counts keeps the ratio's meaning and keeps `None` where the
    denominator really is zero (ADR-0032's rule, one layer up).

    The localisation errors are averaged **weighted by true positives**, which is
    what a mean over all matched pairs is.
    """
    if not rows:
        return None

    truth = sum(one.metrics.n_truth for one in rows)
    detected = sum(one.metrics.n_detected for one in rows)
    tp = sum(one.metrics.true_positives for one in rows)
    precision = tp / detected if detected else None
    recall = tp / truth if truth else None
    return DetectionMetrics(
        n_truth=truth,
        n_detected=detected,
        true_positives=tp,
        false_positives=detected - tp,
        false_negatives=truth - tp,
        precision=precision,
        recall=recall,
        f1=(
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0
            else None
        ),
        mean_localisation_error_px=_weighted(rows, "mean_localisation_error_px"),
        median_localisation_error_px=None,
        mean_localisation_error_nm=_weighted(rows, "mean_localisation_error_nm"),
        mean_radius_error_px=_weighted(rows, "mean_radius_error_px"),
        median_radius_error_px=None,
        mean_signed_radius_error_px=_weighted(rows, "mean_signed_radius_error_px"),
        match_factor=match_factor,
    )


def _weighted(rows: Sequence[ImageScore], field: str) -> float | None:
    """A mean over matched pairs, not over images.

    A scan with one match and a scan with fifty do not carry equal weight in a
    mean localisation error; weighting by `true_positives` makes this the number
    a single call over all pairs would have produced.

    `None` when nothing was matched anywhere, and a row whose own value is
    `None` contributes nothing rather than a zero.
    """
    pairs = [
        (getattr(one.metrics, field), one.metrics.true_positives)
        for one in rows
        if getattr(one.metrics, field) is not None and one.metrics.true_positives
    ]
    total = sum(weight for _value, weight in pairs)
    if not total:
        return None
    return sum(value * weight for value, weight in pairs) / total


#: The **median** of a set of medians is not a median, so the aggregates above
#: report `None` for both median fields rather than a number that would read as
#: one. The per-image rows still carry theirs, which is where a median of a
#: single scan means what it says (ADR-0088).
