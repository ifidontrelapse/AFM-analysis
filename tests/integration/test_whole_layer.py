"""M4, end to end, in the order an operator uses it (M4-T15).

The milestone's sixth exit criterion: *"integration tests cover the whole layer;
no Qt imported anywhere."* Every part of this layer already has its own file —
this one asks the different question of whether the parts work **as one thing**,
which is where a seam nobody owns shows up: a use case needing something no
adapter exposes, or a value that survives each hop and not the chain.

It is deliberately **one long test**, which is normally a smell. Here the
sequence *is* the subject: splitting it into fifteen independent tests would
produce fifteen more copies of what the per-task files already assert, and
would stop testing the thing this file exists for.

The Qt half of the criterion is in `tests/unit/test_import_graph.py`, where the
other dependency-direction guards live.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from nanoscope.app.logging import attach_project_log, configure_logging, detach_project_log
from nanoscope.application.commands import AddAnnotation, CommandStack, UpdateAnnotation
from nanoscope.application.jobs import JobRunner, JobState
from nanoscope.application.settings import Scope, Settings
from nanoscope.application.use_cases import (
    export_measurements,
    import_images,
    open_project,
    run_analysis,
)
from nanoscope.core.entities import ModelDescriptor, ModelFramework, ModelTask, PipelineConfig
from nanoscope.core.values import DeviceKind, Modality
from nanoscope.infrastructure.device import DeviceManager
from nanoscope.infrastructure.models import registry
from nanoscope.infrastructure.storage import JsonSettings, SqliteProjectRepository

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from characterization.phantoms import afm_flat_monodisperse


@pytest.fixture
def scans(tmp_path: Path) -> Path:
    """Two AFM phantoms on disk, standing in for a folder off the instrument."""
    folder = tmp_path / "from_the_instrument"
    folder.mkdir()
    phantom = afm_flat_monodisperse()
    for name in ("monday.npy", "tuesday.npy"):
        np.save(folder / name, phantom.image)
    return folder


def test_a_days_work(tmp_path: Path, scans: Path) -> None:
    """Configure, create, import, analyse, annotate, undo, export, close.

    Every task M4 shipped, in the order they happen, with nothing but Python.
    """
    phantom = afm_flat_monodisperse()
    project_dir = tmp_path / "Gold on mica"

    # M4-T14 — logging is configured before anything can fail.
    application_log = configure_logging(path=tmp_path / "state" / "nanoscope.log")
    assert application_log.parent.is_dir()

    # M4-T12 — one component decides where inference would run.
    device = DeviceManager().select(DeviceKind.CPU)
    assert device.device.kind is DeviceKind.CPU

    with SqliteProjectRepository.create(project_dir, "Gold on mica") as repo:
        attach_project_log(repo.root)  # M4-T14 — and this project keeps its own log

        # M4-T10 — a preference about the work goes in the work.
        settings = Settings(JsonSettings(tmp_path / "config.json"), repo)
        settings.set("detector", "log", Scope.PROJECT)
        settings.set("colormap", "afmhot")

        # M4-T13 — the model is a record, resolved without loading weights.
        model = repo.register_model(
            ModelDescriptor(
                model_id="particles-v12",
                task=ModelTask.DETECT,
                framework=ModelFramework.ULTRALYTICS,
                path="models/best12x.pt",
                provenance="the operator's own, registered by this test",
            )
        )
        assert callable(registry.resolve(model))

        # M4-T06 + M4-T04 — the import runs as a job and reports progress.
        seen: list[tuple[int, int]] = []
        with JobRunner(max_workers=1) as runner:
            job = runner.submit(
                "importing",
                lambda ctx: import_images(
                    repo,
                    sorted(scans.iterdir()),
                    modality=Modality.AFM,
                    pixel_size_nm=phantom.pixel_size_nm,
                    progress=ctx,
                ),
                listener=lambda j: seen.append((j.progress.done, j.progress.total)),
            )
            assert job.wait(60.0)

        assert job.state is JobState.SUCCEEDED, job.error
        assert job.result.is_complete
        assert (2, 2) in seen

        # M4-T05 — analysis, stored where ADR-0042 put it.
        images = repo.list_images()
        assert len(images) == 2
        runs = [
            # `mode` is spelled out because `PipelineConfig`'s default is
            # `"segment"`, which needs a SAM2 predictor that is not in this
            # repository — filed as B-068, not fixed here: changing a default
            # changes what happens for every caller who omits it.
            run_analysis(
                repo,
                image.id,
                PipelineConfig(detector=str(settings.get("detector")), mode="baseline"),
            )
            for image in images
        ]
        assert all(run.detections for run in runs)
        assert all(run.pixel_size_nm == phantom.pixel_size_nm for run in runs)

        # M4-T07 + M4-T08 — hand work, and taking it back.
        stack = CommandStack()
        added = stack.run(
            AddAnnotation(repo, images[0].id, (10.0, 10.0, 30.0, 30.0), label_text="contaminant")
        )
        assert added.annotation is not None
        stack.run(UpdateAnnotation(repo, added.annotation.id, label_text="dust"))
        stack.undo()
        assert repo.get_annotation(added.annotation.id).label == "contaminant"

        # M4-T11 — the file somebody opens three months later.
        export = export_measurements(repo, file_name="for the paper")
        exported = (repo.root / export).read_text(encoding="utf-8")
        assert exported.startswith("image,image_id,run_id,detector,mode,pixel_size_nm,")
        assert "monday.npy" in exported and "tuesday.npy" in exported

        detach_project_log()

    # M4-T14 — both logs have something in them, and they are JSON.
    assert json.loads(application_log.read_text(encoding="utf-8").splitlines()[0])["level"]
    project_log = project_dir / "logs" / "nanoscope.log"
    assert project_log.is_file()

    # M4-T01…T03 — and everything is still there in the next session.
    with SqliteProjectRepository.open(project_dir) as repo:
        opened = open_project(repo)

        assert opened.name == "Gold on mica"
        assert [image.display_name for image in opened.images] == ["monday.npy", "tuesday.npy"]
        assert opened.integrity.is_clean

        assert repo.get_setting("detector") == "log"
        assert [m.model_id for m in repo.list_models()] == ["particles-v12"]
        assert [a.label for a in repo.annotations_for(opened.images[0].id)] == ["contaminant"]

        stored_runs = [run for image in opened.images for run in repo.runs_for(image.id)]
        assert len(stored_runs) == 2
        assert all(len(repo.measurements_for(run)) == len(run.detections) for run in stored_runs)

        assert (repo.root / "exports" / "for the paper.csv").is_file()


def test_the_project_is_a_directory_an_operator_owns(tmp_path: Path, scans: Path) -> None:
    """The other half of what M4 promised, stated as files rather than as API.

    ADR-0003's whole argument is that the operator can open, copy and archive
    their work without us. After a day's work, this is what is on the disk.
    """
    project_dir = tmp_path / "P"
    phantom = afm_flat_monodisperse()

    with SqliteProjectRepository.create(project_dir, "P") as repo:
        import_images(
            repo,
            [sorted(scans.iterdir())[0]],
            modality=Modality.AFM,
            pixel_size_nm=phantom.pixel_size_nm,
        )
        image = repo.list_images()[0]
        run_analysis(repo, image.id, PipelineConfig(detector="log", mode="baseline"))
        export_measurements(repo, file_name="results")

    manifest = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "P"
    assert manifest["format_version"] == 1

    assert (project_dir / "database.sqlite").is_file()
    assert (project_dir / "images" / "monday.npy").is_file()
    assert list((project_dir / "results").glob("run_*/measurements.csv"))
    assert (project_dir / "exports" / "results.csv").is_file()

    # No WAL beside the database (ADR-0039 §4): a copy of the directory is a
    # complete copy of the work.
    assert not list(project_dir.glob("database.sqlite-*"))


# The "no Qt" half of the criterion is asserted in
# `tests/unit/test_import_graph.py`, statically over every module and in a
# subprocess for transitive imports. This file used to repeat it as
# `"PySide6" not in sys.modules`, which was the same in-process fragility
# `test_ports.py` was repaired for in M4-T15 — and it duly broke in M5-T02, when
# another test in this directory started monkeypatching the launcher. One
# subprocess check is worth more than two in-process ones (M5-T02).
