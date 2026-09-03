"""Whether the new model is better, scored from what the project kept (M8-T08, ADR-0088).

Integration rather than unit: the subject is a real project directory holding
annotations, stored analysis runs and a training run — the three things M7, M6
and M8 each put there for other reasons, joined here for the first time.

Four things carry the task:

- the score comes from **stored runs**, so no weights are loaded and the number
  describes the run an operator actually looked at;
- a scan the model **trained on** is labelled, and the unseen total is separate,
  because only one of those two numbers is about generalisation;
- a model whose dataset directory is gone reads **`unknown`**, never `unseen` —
  ADR-0081 made `cache/` deletable and this is what that costs;
- an absent ratio stays absent, in a row and in a total.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from nanoscope.application.use_cases.evaluation import (
    TRAINED_ON,
    UNKNOWN,
    UNSEEN,
    evaluate_models,
    truth_of,
)
from nanoscope.core.entities import Detection, PipelineResult
from nanoscope.core.entities.model import ModelDescriptor, ModelFramework, ModelTask
from nanoscope.core.entities.project import AnnotationSource
from nanoscope.core.entities.training import (
    DatasetSpec,
    TrainingConfig,
    TrainingRun,
    TrainingStatus,
)
from nanoscope.core.values import Modality
from nanoscope.infrastructure.storage import SqliteProjectRepository

SCANS = 4

#: Where the particles are, in every scan: two boxes an operator drew. Same
#: places everywhere, so a test can say what a perfect detector would find.
BOXES = ((10.0, 10.0, 20.0, 20.0), (30.0, 30.0, 40.0, 40.0))


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[SqliteProjectRepository]:
    root = tmp_path / "P"
    with SqliteProjectRepository.create(root, "P") as repository:
        for index in range(SCANS):
            source = tmp_path / f"scan{index}.npy"
            np.save(source, np.zeros((48, 48), dtype=np.float32))
            record = repository.import_image(source, modality=Modality.AFM, pixel_size_nm=2.0)
            for box in BOXES:
                repository.add_annotation(
                    record.id, label="particle", box=box, source=AnnotationSource.MANUAL
                )
        yield repository


def detections_at(*boxes: tuple[float, float, float, float]) -> list[Detection]:
    """Detections centred on the boxes given, the way a detector reports them."""
    found = []
    for x1, y1, x2, y2 in boxes:
        found.append(
            Detection(
                x_px=(x1 + x2) / 2,
                y_px=(y1 + y2) / 2,
                radius_px=min(x2 - x1, y2 - y1) / 2,
                radius_nm=None,
                confidence=0.9,
            )
        )
    return found


def store_run(
    repo: SqliteProjectRepository, image_id: int, detections: list[Detection], model_id: str
) -> int:
    """A stored detect run attributed to a model — schema v10's column, used."""
    import pandas as pd

    run = repo.save_analysis(
        image_id,
        PipelineResult(
            detections=detections,
            masks=[],
            measurements=pd.DataFrame(),
            pixel_size_nm=2.0,
            detector_name="yolo",
            mode="detect",
            modality="afm",
        ),
        model_id=model_id,
    )
    return run.id


def register(repo: SqliteProjectRepository, model_id: str, weights: str) -> ModelDescriptor:
    return repo.register_model(
        ModelDescriptor(
            model_id=model_id,
            task=ModelTask.DETECT,
            framework=ModelFramework.ULTRALYTICS,
            path=weights,
        )
    )


def a_training_run(repo: SqliteProjectRepository, weights: str, *, dataset_root: str) -> None:
    """A succeeded training run whose weights are the model's path.

    The join M8-T04 made possible without meaning to: it registers a model with
    `path = run.weights_path`, so the two are the same string and nothing new had
    to be stored to answer *which run trained this model?*
    """
    repo.save_training_run(
        TrainingRun(
            run_id="the-run",
            status=TrainingStatus.SUCCEEDED,
            dataset=DatasetSpec(
                root=dataset_root, classes=("particle",), train_images=2, val_images=2
            ),
            config=TrainingConfig(base_model="n.pt", epochs=3, image_size_px=32),
            weights_path=weights,
            started_utc="2026-09-03T10:00:00+00:00",
            finished_utc="2026-09-03T11:00:00+00:00",
        )
    )


def trained_on(repo: SqliteProjectRepository, dataset_root: str, image_ids: list[int]) -> None:
    """Write the training split the way `build_dataset` leaves it on disk.

    The stems are what map a dataset back to image records — measured, and the
    only place the *membership* of a split is recorded at all.
    """
    directory = Path(repo.root) / dataset_root / "images" / "train"
    directory.mkdir(parents=True, exist_ok=True)
    by_id = {record.id: record for record in repo.list_images()}
    for image_id in image_ids:
        (directory / f"{Path(by_id[image_id].relative_path).stem}.png").write_bytes(b"png")


class TestScoringWhatTheProjectAlreadyKept:
    def test_a_perfect_detector_scores_one(self, repo: SqliteProjectRepository) -> None:
        """No weights are loaded anywhere in this test, which is the design."""
        for record in repo.list_images():
            store_run(repo, record.id, detections_at(*BOXES), "particles-v1")

        report = evaluate_models(repo)

        assert [one.model_id for one in report.models] == ["particles-v1"]
        overall = report.models[0].overall
        assert overall is not None
        assert overall.n_truth == SCANS * len(BOXES)
        assert overall.precision == 1.0
        assert overall.recall == 1.0

    def test_a_miss_and_an_invention_are_counted_as_what_they_are(
        self, repo: SqliteProjectRepository
    ) -> None:
        first = repo.list_images()[0]
        #: One of the two particles found, and one box where there is nothing.
        store_run(repo, first.id, detections_at(BOXES[0], (0.0, 0.0, 6.0, 6.0)), "m")

        report = evaluate_models(repo)
        metrics = report.models[0].overall

        assert metrics is not None
        assert (metrics.true_positives, metrics.false_positives, metrics.false_negatives) == (
            1,
            1,
            1,
        )
        assert metrics.precision == 0.5
        assert metrics.recall == 0.5

    def test_two_models_are_scored_over_the_same_scans(self, repo: SqliteProjectRepository) -> None:
        """Two numbers over two different sets of scans cannot be subtracted, so
        what each was scored on is part of the report."""
        for record in repo.list_images():
            store_run(repo, record.id, detections_at(*BOXES), "good")
            store_run(repo, record.id, detections_at(BOXES[0]), "half")

        report = evaluate_models(repo)
        scores = {one.model_id: one.overall for one in report.models}

        assert set(scores) == {"good", "half"}
        assert scores["good"].recall == 1.0  # type: ignore[union-attr]
        assert scores["half"].recall == 0.5  # type: ignore[union-attr]
        assert len(report.image_ids) == SCANS

    def test_a_run_that_named_no_model_is_not_attributed_to_one(
        self, repo: SqliteProjectRepository
    ) -> None:
        """Every `log` run, and every run stored before M8-T06. `NULL` there is
        honest, and there is nothing to attribute it to."""
        first = repo.list_images()[0]
        store_run(repo, first.id, detections_at(*BOXES), "")

        assert evaluate_models(repo).models == ()

    def test_re_running_a_model_on_a_scan_counts_once(self, repo: SqliteProjectRepository) -> None:
        """The newest run, because that is what an operator meant by running it
        again — and counting both would count one model twice on one image."""
        first = repo.list_images()[0]
        store_run(repo, first.id, detections_at(BOXES[0]), "m")
        store_run(repo, first.id, detections_at(*BOXES), "m")

        metrics = evaluate_models(repo).models[0].overall

        assert metrics is not None
        assert metrics.n_detected == 2
        assert metrics.recall == 1.0


class TestWhatTheModelHadAlreadySeen:
    def test_a_scan_it_trained_on_is_labelled_and_kept_out_of_the_unseen_total(
        self, repo: SqliteProjectRepository
    ) -> None:
        """The distinction the whole report turns on: only one of the two totals
        says anything about scans the model has not seen."""
        images = repo.list_images()
        register(repo, "m", "models/run/best.pt")
        a_training_run(repo, "models/run/best.pt", dataset_root="cache/ds")
        trained_on(repo, "cache/ds", [images[0].id, images[1].id])

        #: Perfect where it trained, half-blind where it did not.
        for record in images[:2]:
            store_run(repo, record.id, detections_at(*BOXES), "m")
        for record in images[2:]:
            store_run(repo, record.id, detections_at(BOXES[0]), "m")

        score = evaluate_models(repo).models[0]

        assert {one.image_id: one.exposure for one in score.images} == {
            images[0].id: TRAINED_ON,
            images[1].id: TRAINED_ON,
            images[2].id: UNSEEN,
            images[3].id: UNSEEN,
        }
        assert score.unseen_images == 2
        assert score.exposure_is_known
        #: The unseen total is the honest one, and it is worse. That gap is the
        #: whole reason the two are reported apart.
        assert score.unseen.recall == 0.5  # type: ignore[union-attr]
        assert score.overall.recall == 0.75  # type: ignore[union-attr]

    def test_a_model_whose_dataset_is_gone_reads_unknown_and_never_unseen(
        self, repo: SqliteProjectRepository
    ) -> None:
        """ADR-0081 made `cache/` deletable on purpose, and measured: the
        *counts* of a split live on the spec for ever, the *membership* lives
        only in that directory. Calling those scans `unseen` would be inventing
        the one fact this report exists to be careful about."""
        register(repo, "m", "models/run/best.pt")
        a_training_run(repo, "models/run/best.pt", dataset_root="cache/deleted")
        for record in repo.list_images():
            store_run(repo, record.id, detections_at(*BOXES), "m")

        score = evaluate_models(repo).models[0]

        assert {one.exposure for one in score.images} == {UNKNOWN}
        assert not score.exposure_is_known
        assert score.unseen is None
        assert score.overall is not None

    def test_a_model_with_no_training_run_here_reads_unknown_too(
        self, repo: SqliteProjectRepository
    ) -> None:
        """An imported checkpoint (M8-T06): this project never trained it and
        cannot say what it saw."""
        register(repo, "imported", "/elsewhere/best.pt")
        for record in repo.list_images():
            store_run(repo, record.id, detections_at(*BOXES), "imported")

        score = evaluate_models(repo).models[0]

        assert {one.exposure for one in score.images} == {UNKNOWN}
        assert score.unseen is None


class TestWhatCountsAsTruth:
    def test_adopted_boxes_are_not_truth_by_default(self, repo: SqliteProjectRepository) -> None:
        """ADR-0044's rule at its third site: scoring a model against boxes
        adopted from a detector is scoring it against a detector."""
        extra = repo.list_images()[0]
        repo.add_annotation(
            extra.id,
            label="particle",
            box=(0.0, 0.0, 6.0, 6.0),
            source=AnnotationSource.FROM_DETECTION,
        )
        store_run(repo, extra.id, detections_at(*BOXES, (0.0, 0.0, 6.0, 6.0)), "m")

        default = evaluate_models(repo).models[0].overall
        widened = evaluate_models(repo, sources=None).models[0].overall

        assert default is not None and widened is not None
        #: The adopted box is a false positive by default and a true positive
        #: once the caller says to count it — the same three boxes either way.
        assert (default.n_truth, default.true_positives) == (2, 2)
        assert (widened.n_truth, widened.true_positives) == (3, 3)

    def test_a_scan_with_no_annotations_is_not_scored(
        self, repo: SqliteProjectRepository, tmp_path: Path
    ) -> None:
        """It has no truth, so a detection on it is neither right nor wrong."""
        source = tmp_path / "unlabelled.npy"
        np.save(source, np.zeros((48, 48), dtype=np.float32))
        blank = repo.import_image(source, modality=Modality.AFM, pixel_size_nm=2.0)
        store_run(repo, blank.id, detections_at(*BOXES), "m")

        assert evaluate_models(repo).models == ()

    def test_a_box_becomes_the_circle_a_detector_would_have_reported(self) -> None:
        """`min(w, h) / 2`, the same rule `infrastructure/models/yolo.py` uses on
        its own boxes. The harness matches *a centre inside the particle*, so
        the truth radius **is** the tolerance: a circumscribed truth against
        inscribed detections would compare two different circles."""

        class Box:
            box = (10.0, 20.0, 30.0, 26.0)

        centres, radii = truth_of([Box()])  # type: ignore[list-item]

        #: `[y, x]`, the project's array convention (PROJECT_RULES §3).
        assert centres.tolist() == [[23.0, 20.0]]
        assert radii.tolist() == [3.0]


class TestAnAbsentRatioStaysAbsent:
    def test_a_model_that_reported_nothing_has_no_precision(
        self, repo: SqliteProjectRepository
    ) -> None:
        """ADR-0032 deleted the seventh substitute value in this project for
        exactly this: a detector that found nothing has no precision, and 0.0
        would be a measurement it never made."""
        first = repo.list_images()[0]
        store_run(repo, first.id, [], "silent")

        metrics = evaluate_models(repo).models[0].overall

        assert metrics is not None
        assert metrics.precision is None
        assert metrics.recall == 0.0
        assert metrics.f1 is None

    def test_the_totals_sum_counts_rather_than_averaging_ratios(
        self, repo: SqliteProjectRepository
    ) -> None:
        """A mean of per-image precisions weights a scan with two particles the
        same as one with two hundred — and has no denominator to be honest
        about. Here the counts are summed and the ratio recomputed."""
        images = repo.list_images()
        #: Two found on one scan, none on the other three.
        store_run(repo, images[0].id, detections_at(*BOXES), "m")
        for record in images[1:]:
            store_run(repo, record.id, [], "m")

        metrics = evaluate_models(repo).models[0].overall

        assert metrics is not None
        assert metrics.n_truth == SCANS * len(BOXES)
        assert metrics.recall == pytest.approx(2 / 8)
        #: Averaging the four per-image recalls would have given 0.25 as well —
        #: but averaging the *precisions* would have to skip three `None`s, and
        #: this reports the one that exists.
        assert metrics.precision == 1.0

    def test_a_localisation_error_is_a_mean_over_pairs_not_over_scans(
        self, repo: SqliteProjectRepository
    ) -> None:
        """A scan with one match and a scan with two do not carry equal weight."""
        images = repo.list_images()
        #: Both particles found, dead centre, on one scan.
        store_run(repo, images[0].id, detections_at(*BOXES), "m")
        #: One particle found, two pixels off, on another.
        off = (BOXES[0][0] + 2, BOXES[0][1], BOXES[0][2] + 2, BOXES[0][3])
        store_run(repo, images[1].id, detections_at(off), "m")

        metrics = evaluate_models(repo).models[0].overall

        assert metrics is not None
        #: Three pairs: 0, 0 and 2 px. Over scans it would have been 1.0.
        assert metrics.mean_localisation_error_px == pytest.approx(2 / 3)

    def test_a_median_of_medians_is_not_reported_as_a_median(
        self, repo: SqliteProjectRepository
    ) -> None:
        """It is not one, so the aggregate says nothing rather than a number
        that would read as one. The per-image rows still carry theirs."""
        for record in repo.list_images():
            store_run(repo, record.id, detections_at(*BOXES), "m")

        score = evaluate_models(repo).models[0]

        assert score.overall is not None
        assert score.overall.median_localisation_error_px is None
        assert score.images[0].metrics.median_localisation_error_px is not None
