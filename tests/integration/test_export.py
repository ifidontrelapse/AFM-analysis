"""A CSV somebody opens three months later (M4-T11, ADR-0048).

The stored measurement table already exists — ADR-0042 put it in
`results/run_<id>/measurements.csv` — so the tests that matter here are about
what an *export* has that storage does not: the provenance columns, more than
one run in one file, a name that does not overwrite yesterday's, and a refusal
to write a file that would misrepresent what happened.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.application.use_cases import export_measurements, run_analysis
from nanoscope.core.entities import PipelineConfig
from nanoscope.core.errors import AnalysisFailedError, MissingFileError
from nanoscope.core.values import Modality
from nanoscope.infrastructure.storage import SqliteProjectRepository

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from characterization.phantoms import afm_flat_monodisperse

BASELINE = PipelineConfig(detector="log", mode="baseline")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project with two analysed AFM images in it."""
    phantom = afm_flat_monodisperse()
    root = tmp_path / "P"
    with SqliteProjectRepository.create(root, "P") as repo:
        for name in ("monday.npy", "tuesday.npy"):
            source = tmp_path / name
            np.save(source, phantom.image)
            image = repo.import_image(
                source, modality=Modality.AFM, pixel_size_nm=phantom.pixel_size_nm
            )
            run_analysis(repo, image.id, BASELINE)
    return root


def read_export(root: Path, relative: str) -> pd.DataFrame:
    return pd.read_csv(root / relative)


class TestWhatAnExportHasThatStorageDoesNot:
    def test_every_row_says_which_scan_it_came_from(self, project: Path) -> None:
        """The stored table is filed *under* its run, so it does not repeat the
        fact. A CSV on a desktop has nothing around it, and a column of heights
        with no scan name is a column of numbers."""
        with SqliteProjectRepository.open(project) as repo:
            written = export_measurements(repo)
            table = read_export(repo.root, written)

        assert set(table["image"]) == {"monday.npy", "tuesday.npy"}
        assert list(table.columns[:6]) == [
            "image",
            "image_id",
            "run_id",
            "detector",
            "mode",
            "pixel_size_nm",
        ]

    def test_more_than_one_run_lands_in_one_file(self, project: Path) -> None:
        """Statistics across a dataset is the reason the measurements exist, and
        assembling it by hand from twelve files is not a workflow."""
        with SqliteProjectRepository.open(project) as repo:
            per_run = [
                len(repo.measurements_for(run))
                for image in repo.list_images()
                for run in repo.runs_for(image.id)
            ]
            table = read_export(repo.root, export_measurements(repo))

        assert len(table) == sum(per_run)
        assert table["run_id"].nunique() == 2

    def test_the_measured_columns_are_still_there(self, project: Path) -> None:
        """Provenance goes in front; nothing is dropped to make room."""
        with SqliteProjectRepository.open(project) as repo:
            table = read_export(repo.root, export_measurements(repo))

        assert {"height_nm", "area_px", "method"} <= set(table.columns)

    def test_it_lands_in_exports_with_a_relative_path(self, project: Path) -> None:
        with SqliteProjectRepository.open(project) as repo:
            written = export_measurements(repo)

            assert written.startswith("exports/")
            assert (repo.root / written).is_file()

    def test_one_run_can_be_exported_alone(self, project: Path) -> None:
        with SqliteProjectRepository.open(project) as repo:
            image = repo.list_images()[0]
            run = repo.runs_for(image.id)[0]

            table = read_export(repo.root, export_measurements(repo, [run]))

            assert set(table["run_id"]) == {run.id}


class TestNaming:
    def test_a_caller_can_name_it(self, project: Path) -> None:
        with SqliteProjectRepository.open(project) as repo:
            written = export_measurements(repo, file_name="for the paper")

            assert written == "exports/for the paper.csv"

    def test_a_name_cannot_escape_the_exports_directory(self, project: Path) -> None:
        """The name arrives from an operator's text field, and a `/` in it would
        write outside the project."""
        with SqliteProjectRepository.open(project) as repo:
            written = export_measurements(repo, file_name="../../etc/passwd")

            assert written.startswith("exports/")
            assert (repo.root / written).is_file()

    def test_two_exports_do_not_collide_by_default(self, project: Path) -> None:
        """An export is a snapshot, and two in one day are the normal case;
        replacing the first silently would lose work the operator believes they
        have."""
        with SqliteProjectRepository.open(project) as repo:
            first = export_measurements(repo, file_name="a")
            second = export_measurements(repo, file_name="b")

            assert first != second
            assert len(list((repo.root / "exports").iterdir())) == 2


class TestWhenThereIsNothingHonestToWrite:
    def test_a_detect_only_run_is_refused_rather_than_exported_empty(self, tmp_path: Path) -> None:
        """A file with headers and no rows is indistinguishable from "we measured
        and found nothing", which is a different statement."""
        phantom = afm_flat_monodisperse()
        source = tmp_path / "scan.npy"
        np.save(source, phantom.image)

        with SqliteProjectRepository.create(tmp_path / "Q", "Q") as repo:
            image = repo.import_image(
                source, modality=Modality.AFM, pixel_size_nm=phantom.pixel_size_nm
            )
            run_analysis(repo, image.id, PipelineConfig(detector="log", mode="detect"))

            with pytest.raises(AnalysisFailedError, match="nothing to export"):
                export_measurements(repo)

    def test_a_missing_stored_table_is_loud(self, project: Path) -> None:
        """An export silently missing one scan of twelve is a wrong dataset that
        looks right."""
        with SqliteProjectRepository.open(project) as repo:
            run = repo.runs_for(repo.list_images()[0].id)[0]
            assert run.measurements_path is not None
            (repo.root / run.measurements_path).unlink()

            with pytest.raises(MissingFileError):
                export_measurements(repo)
