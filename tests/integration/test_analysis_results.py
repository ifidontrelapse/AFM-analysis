"""What the analysis found, written down and read back (M4-T05, ADR-0042).

M4's second exit criterion:

> *Detection and measurement results round-trip through SQLite and the
> filesystem.*

Both halves, on a real project: the run and its detections are rows, the
measurement table is a file under `results/`, and a phantom from the
characterization suite is what goes through the pipeline — a real height map
with known particles in it, run by the LoG detector, which is the one that works
without weights and therefore the one CI can execute (PROJECT_RULES §6).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from schema_history import revert_to

from nanoscope.application.use_cases import run_analysis
from nanoscope.core.entities import Detection, PipelineConfig, PipelineResult
from nanoscope.core.errors import InvalidParameterError, MissingFileError
from nanoscope.core.science.measurement.schema import blocks_for, empty_measurement_table
from nanoscope.core.values import Modality
from nanoscope.infrastructure.storage import (
    SCHEMA_VERSION,
    SqliteProjectRepository,
    schema_version,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from characterization.phantoms import afm_flat_monodisperse


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project with one AFM phantom imported into it, scale and all."""
    phantom = afm_flat_monodisperse()
    source = tmp_path / "phantom.npy"
    np.save(source, phantom.image)

    root = tmp_path / "P"
    with SqliteProjectRepository.create(root, "P") as repo:
        repo.import_image(source, modality=Modality.AFM, pixel_size_nm=phantom.pixel_size_nm)
    return root


def test_the_second_exit_criterion(project: Path) -> None:
    """Detect, measure, store, close — then reopen and find all of it."""
    with SqliteProjectRepository.open(project) as repo:
        image = repo.list_images()[0]
        run = run_analysis(repo, image.id, PipelineConfig(detector="log", mode="baseline"))
        run_id = run.id

    with SqliteProjectRepository.open(project) as repo:
        stored = repo.get_run(run_id)
        measurements = repo.measurements_for(stored)

    assert stored.detections
    assert len(measurements) == len(stored.detections)
    assert stored.mode == "baseline"
    assert stored.modality is Modality.AFM


class TestWhatIsStoredWhere:
    def test_the_detections_are_rows(self, project: Path) -> None:
        with SqliteProjectRepository.open(project) as repo:
            image = repo.list_images()[0]
            run = run_analysis(repo, image.id, PipelineConfig(detector="log", mode="detect"))

            assert repo.get_run(run.id).detections == run.detections

    def test_a_detection_comes_back_as_it_went_in(self, project: Path) -> None:
        with SqliteProjectRepository.open(project) as repo:
            image = repo.list_images()[0]
            run = run_analysis(repo, image.id, PipelineConfig(detector="log", mode="detect"))

            first = repo.get_run(run.id).detections[0]
            assert first == run.detections[0]
            assert first.bbox == run.detections[0].bbox

    def test_an_absent_box_stays_absent(self, project: Path) -> None:
        """Four `None`s wearing a tuple would be D-16 again on the way out of the
        database (ADR-0031), so the absence has to survive storage. Built by
        hand: every detector in the gate emits a box, LoG's being synthesised
        from the radius, so the `None` case has no producer to reach it with."""
        with SqliteProjectRepository.open(project) as repo:
            image = repo.list_images()[0]
            boxless = PipelineResult(
                detections=[Detection(x_px=1.0, y_px=2.0, radius_px=3.0, radius_nm=None)],
                masks=[],
                measurements=empty_measurement_table(**blocks_for("afm")),
                pixel_size_nm=None,
                detector_name="log",
                mode="detect",
                modality="afm",
            )

            run = repo.save_analysis(image.id, boxless)

            assert repo.get_run(run.id).detections[0].bbox is None

    def test_the_scale_the_project_recorded_is_the_scale_that_is_used(self, project: Path) -> None:
        """An npy carries no metadata. Without the record's scale reaching the
        loader, an image imported *with* a known scale is analysed as though it
        had none — `radius_nm` is `None` for every particle and the physical
        minimum-size filter is skipped. That is the D-07 family of defect M3
        spent a milestone removing, one layer up. Found by this test."""
        with SqliteProjectRepository.open(project) as repo:
            image = repo.list_images()[0]
            assert image.pixel_size_nm == 2.0

            run = run_analysis(repo, image.id, PipelineConfig(detector="log", mode="detect"))

            assert run.pixel_size_nm == 2.0
            assert all(detection.radius_nm is not None for detection in run.detections)

    def test_the_measurement_table_is_a_file_under_results(self, project: Path) -> None:
        """ADR-0031 made that table variable by construction; a relational shape
        for it is wide-with-NULLs or an EAV pivot. It is a file, and the run row
        points at it with a **relative** path like everything else."""
        with SqliteProjectRepository.open(project) as repo:
            image = repo.list_images()[0]
            run = run_analysis(repo, image.id, PipelineConfig(detector="log", mode="baseline"))

            assert run.measurements_path is not None
            assert run.measurements_path.startswith("results/")
            assert (repo.root / run.measurements_path).is_file()

    def test_the_table_read_back_is_the_table_that_was_written(self, project: Path) -> None:
        with SqliteProjectRepository.open(project) as repo:
            image = repo.list_images()[0]
            run = run_analysis(repo, image.id, PipelineConfig(detector="log", mode="baseline"))

            table = repo.measurements_for(run)

            assert "height_nm" in table.columns
            assert "method" in table.columns
            assert set(table["method"]) == {"baseline_circle"}

    def test_detect_mode_writes_no_table(self, project: Path) -> None:
        """It measured nothing. An empty table with the right columns is not a
        measurement, and storing one would claim it was."""
        with SqliteProjectRepository.open(project) as repo:
            image = repo.list_images()[0]
            run = run_analysis(repo, image.id, PipelineConfig(detector="log", mode="detect"))

            assert run.measurements_path is None
            assert repo.measurements_for(run).empty

    def test_a_deleted_table_is_reported_not_answered_with_nothing(self, project: Path) -> None:
        """Silently returning an empty table would report "no particles" for a
        run that found some."""
        with SqliteProjectRepository.open(project) as repo:
            image = repo.list_images()[0]
            run = run_analysis(repo, image.id, PipelineConfig(detector="log", mode="baseline"))
            assert run.measurements_path is not None
            (repo.root / run.measurements_path).unlink()

            with pytest.raises(MissingFileError, match="re-run"):
                repo.measurements_for(run)


class TestRunsBelongToTheirImage:
    def test_every_run_of_an_image_is_listed_oldest_first(self, project: Path) -> None:
        with SqliteProjectRepository.open(project) as repo:
            image = repo.list_images()[0]
            first = run_analysis(repo, image.id, PipelineConfig(detector="log", mode="detect"))
            second = run_analysis(repo, image.id, PipelineConfig(detector="log", mode="baseline"))

            assert [run.id for run in repo.runs_for(image.id)] == [first.id, second.id]

    def test_an_analysis_of_an_image_that_does_not_exist_is_refused(self, project: Path) -> None:
        with (
            SqliteProjectRepository.open(project) as repo,
            pytest.raises(InvalidParameterError, match="no image with id 99"),
        ):
            run_analysis(repo, 99, PipelineConfig(detector="log", mode="detect"))

    def test_forgetting_the_image_forgets_its_results(self, project: Path) -> None:
        """`ON DELETE CASCADE`, and the first time M4-T02's `PRAGMA foreign_keys`
        is load-bearing rather than a precaution. A detection of a particle in a
        scan the project no longer knows about is litter — and unlike the image
        row, it is the *derived* half, so ADR-0040's argument does not apply."""
        with SqliteProjectRepository.open(project) as repo:
            image = repo.list_images()[0]
            run = run_analysis(repo, image.id, PipelineConfig(detector="log", mode="detect"))

            repo.remove_image(image.id)

            assert repo.runs_for(image.id) == []
            with pytest.raises(InvalidParameterError):
                repo.get_run(run.id)

    def test_the_measurement_file_outlives_the_cascade(self, project: Path) -> None:
        """Honest, and stated: the rows go, the file under `results/` stays.
        Deleting the operator's files is a decision this layer does not make
        (ADR-0040), and a file with no row is one `check_integrity` will show —
        `results/` is not covered by it yet, which ADR-0042 §3 records."""
        with SqliteProjectRepository.open(project) as repo:
            image = repo.list_images()[0]
            run = run_analysis(repo, image.id, PipelineConfig(detector="log", mode="baseline"))
            assert run.measurements_path is not None

            repo.remove_image(image.id)

            assert (repo.root / run.measurements_path).is_file()


class TestTheMigrationThatBroughtTheseTables:
    def test_a_project_written_at_v1_gains_the_tables(self, tmp_path: Path) -> None:
        """ADR-0039's mechanism, exercised for the first time on a database that
        already has rows in it — which is the case it exists for, and the one
        that had never been run."""
        root = tmp_path / "P"
        with SqliteProjectRepository.create(root, "P") as repo:
            (root / "images" / "a.spm").write_bytes(b"AFM")
            recorded = repo.add_image("images/a.spm", modality=Modality.AFM)
            # Back to the world as M4-T02 left it. Every table above v1 goes:
            # a database that claims v1 while carrying a v4 table is not a v1
            # database, and the migration is right to refuse it.
            revert_to(repo._conn, 1)

        with SqliteProjectRepository.open(root) as repo:
            assert schema_version(repo._conn) == SCHEMA_VERSION
            assert repo.list_images() == [recorded]
            assert repo.runs_for(recorded.id) == []
