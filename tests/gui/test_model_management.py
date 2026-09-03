"""The model an operator registers is the model the detector loads (M8-T06, ADR-0086).

**M8's third exit criterion is the first class here:** *a trained model is
selectable for detection in M6 with no code change.* It was measurably false
until this task, and the reason was **W10** — a defect M4-T13 named, made
closable and handed to M5, which did not pay it. Measured before the fix, with
`particles-v1` registered in the project:

    detector options:                 [('log', True), ('yolo', True)]
    panel config yolo_model_path:     ./checkpoints/best12x.pt
    does the panel offer to pick one? False

and the path itself resolved against the working directory — `True` from the
repository root, where an untracked checkpoint sits, `False` from anywhere else.
When it was absent the failure was a raw `FileNotFoundError` out of the
framework, after the scan had already been preprocessed.

So the tests are about four things: the weights a run loads come from the
project, the choice is stored where a *project* stores things, a run records
which model produced it, and a missing file is said out loud rather than met
halfway through.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtCore import Qt

from nanoscope.app.container import Nanoscope
from nanoscope.application.settings import ACTIVE_MODEL_SETTING, Scope
from nanoscope.core.entities import PipelineConfig
from nanoscope.core.entities.model import ModelDescriptor, ModelFramework, ModelTask
from nanoscope.core.errors import MissingFileError, UnsupportedRequestError
from nanoscope.core.values import Modality
from nanoscope.gui.dialogs.models import (
    ACTIVE,
    COLUMNS,
    MISSING,
    SCORE_COLUMNS,
    ModelsDialog,
)
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import DetectionPanel
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")


def phantom(size: int = 48) -> np.ndarray:
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    z = rng.normal(0.0, 0.05, (size, size)).astype(np.float32)
    for cy, cx in ((12, 12), (30, 32)):
        z += 4.0 * np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / 12.0))
    return z.astype(np.float32)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "P"
    with SqliteProjectRepository.create(root, "P") as repo:
        source = tmp_path / "afm.npy"
        np.save(source, phantom())
        repo.import_image(source, modality=Modality.AFM, pixel_size_nm=2.0)
    return root


@pytest.fixture
def app(tmp_path: Path, project: Path) -> Iterator[Nanoscope]:
    with Nanoscope(settings_path=tmp_path / "settings.json") as container:
        container.open(project)
        yield container


@pytest.fixture
def session(app: Nanoscope) -> SessionViewModel:
    view = SessionViewModel(app)
    view.refresh()
    view.select_image(view.project.images[0].id)  # type: ignore[union-attr]
    return view


def weights_in(app: Nanoscope, relative: str = "models/run/best.pt") -> str:
    """A file where a trained run would have left one. Bytes, not a network."""
    repository: Any = app.repository
    path = Path(repository.root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not really a checkpoint")
    return relative


def register(app: Nanoscope, model_id: str = "particles-v1", **overrides: Any) -> ModelDescriptor:
    repository: Any = app.repository
    fields: dict[str, Any] = {
        "task": ModelTask.DETECT,
        "framework": ModelFramework.ULTRALYTICS,
        "path": weights_in(app, f"models/{model_id}/best.pt"),
        "provenance": f"trained here as {model_id}",
    }
    fields.update(overrides)
    return repository.register_model(ModelDescriptor(model_id=model_id, **fields))


class TestTheModelAProjectChoosesIsTheOneThatRuns:
    """M8's third exit criterion, and the close of W10."""

    def test_the_weights_a_run_loads_come_from_the_project(
        self, app: Nanoscope, session: SessionViewModel
    ) -> None:
        register(app)
        assert session.activate_model("particles-v1")

        seen: list[str] = []
        import nanoscope.application.use_cases.analysis as analysis

        original = analysis.run_pipeline

        def spy(data: object, cfg: PipelineConfig, predictor: object = None) -> object:
            seen.append(cfg.yolo_model_path)
            raise UnsupportedRequestError("stopped before inference, on purpose")

        analysis.run_pipeline = spy  # type: ignore[assignment]
        try:
            job = session.detect(PipelineConfig(detector="yolo", mode="detect"))
            assert job is not None and job.wait(10.0)
        finally:
            analysis.run_pipeline = original  # type: ignore[assignment]

        repository: Any = app.repository
        assert seen == [str(repository.path_of_model(repository.get_model("particles-v1")))]

    def test_no_path_resolves_against_the_working_directory(self) -> None:
        """W10 in one assertion. The default used to be
        `"./checkpoints/best12x.pt"`, which is `True` from the repository root
        and `False` from anywhere else — the same project, the same button, a
        different answer."""
        assert PipelineConfig().yolo_model_path == ""

    def test_a_run_with_no_weights_refuses_before_it_reads_anything(
        self, app: Nanoscope, session: SessionViewModel
    ) -> None:
        """Where M2-T10 put this class of refusal, for D-14's reason. Until now
        it could not happen, because the field defaulted to a path."""
        repository: Any = app.repository
        with pytest.raises(UnsupportedRequestError, match="names no weights"):
            from nanoscope.application.use_cases import run_analysis

            run_analysis(
                repository,
                session.image_id,
                PipelineConfig(detector="yolo", mode="detect"),
            )

    def test_a_detector_that_used_no_model_records_none(
        self, app: Nanoscope, session: SessionViewModel
    ) -> None:
        """`NULL` for a `log` run is honest, not lossy: it used no model."""
        job = session.detect(PipelineConfig(detector="log", mode="detect"))
        assert job is not None and job.wait(30.0)

        repository: Any = app.repository
        assert repository.runs_for(session.image_id)[0].model_id is None

    def test_a_run_records_which_model_produced_it(self, app: Nanoscope) -> None:
        """Schema v10. Without it, *which model found these particles?* is
        unanswerable the moment a project has two (ADR-0086)."""
        repository: Any = app.repository
        register(app)
        image_id = repository.list_images()[0].id
        from nanoscope.application.use_cases import run_analysis

        run = run_analysis(
            repository,
            image_id,
            PipelineConfig(detector="log", mode="detect"),
            model_id="particles-v1",
        )

        assert run.model_id == "particles-v1"
        assert repository.get_run(run.id).model_id == "particles-v1"


class TestWhereTheChoiceIsStored:
    def test_it_is_the_projects_answer_and_not_the_operators(
        self, app: Nanoscope, session: SessionViewModel
    ) -> None:
        """The first writer of a scope `Settings` has offered since M4-T10, and
        it is the right one by ADR-0047's test: a chosen model belongs to the
        project. In the application scope it would leak into every other."""
        register(app)
        assert session.activate_model("particles-v1")

        assert app.settings.scope_of(ACTIVE_MODEL_SETTING) is Scope.PROJECT
        assert app.application_settings.get_setting(ACTIVE_MODEL_SETTING) is None

    def test_it_survives_closing_and_reopening_the_project(
        self, app: Nanoscope, project: Path, session: SessionViewModel
    ) -> None:
        register(app)
        session.activate_model("particles-v1")

        session.close_project()
        assert session.active_model is None
        app.open(project)

        assert session.active_model == "particles-v1"

    def test_a_model_this_project_does_not_have_is_refused(self, session: SessionViewModel) -> None:
        """A stored id nothing resolves is a detection that fails later, for a
        reason nobody can see."""
        refusals: list[str] = []
        session.failed.connect(refusals.append)

        assert not session.activate_model("never-registered")
        assert session.active_model is None
        assert refusals

    def test_detecting_with_nothing_is_a_choice_that_can_be_made(
        self, app: Nanoscope, session: SessionViewModel
    ) -> None:
        register(app)
        session.activate_model("particles-v1")

        assert session.activate_model(None)
        assert session.active_model is None


class TestRegisteringWeightsFromThisMachine:
    def test_the_file_is_registered_where_it_is_and_not_copied(
        self, app: Nanoscope, session: SessionViewModel, tmp_path: Path
    ) -> None:
        """ADR-0050 decided this and stated the consequence: the project opens
        on another machine with that model unavailable. Copying gigabytes on an
        operator's behalf is a storage decision this layer does not make."""
        elsewhere = tmp_path / "shared" / "best.pt"
        elsewhere.parent.mkdir()
        elsewhere.write_bytes(b"weights")

        stored = session.register_model(
            elsewhere,
            model_id="shared",
            task=ModelTask.DETECT,
            framework=ModelFramework.ULTRALYTICS,
        )

        assert stored is not None
        assert stored.path == str(elsewhere)
        assert stored.is_external
        repository: Any = app.repository
        assert not (Path(repository.root) / "models" / "best.pt").exists()

    def test_weights_that_are_not_there_are_refused(
        self, session: SessionViewModel, tmp_path: Path
    ) -> None:
        refusals: list[str] = []
        session.failed.connect(refusals.append)

        assert (
            session.register_model(
                tmp_path / "nothing.pt",
                model_id="ghost",
                task=ModelTask.DETECT,
                framework=ModelFramework.ULTRALYTICS,
            )
            is None
        )
        assert refusals

    def test_a_checksum_is_computed_for_weights_that_are_there(
        self, session: SessionViewModel, tmp_path: Path
    ) -> None:
        """M8-T04's rule, met from the import side: a checksum describes the
        file the row points at, because it is taken from that file."""
        weights = tmp_path / "w.pt"
        weights.write_bytes(b"weights")

        stored = session.register_model(
            weights,
            model_id="w",
            task=ModelTask.DETECT,
            framework=ModelFramework.ULTRALYTICS,
        )

        assert stored is not None and stored.sha256


class TestAModelWhoseWeightsAreGone:
    def test_it_is_listed_as_missing_rather_than_hidden(
        self, app: Nanoscope, session: SessionViewModel
    ) -> None:
        """ADR-0040's dangling row, from the model side. Hiding it turns *that
        model is on the other machine* into *that model never existed*."""
        register(app, "gone", path="models/gone/best.pt")
        repository: Any = app.repository
        (Path(repository.root) / "models" / "gone" / "best.pt").unlink()

        dialog = ModelsDialog(session)

        assert dialog.table.rowCount() == 1
        assert _cell(dialog, 0, "Weights") == MISSING

    def test_a_run_naming_it_refuses_with_a_sentence(self, app: Nanoscope) -> None:
        """Not a framework's `FileNotFoundError` halfway through a run, which is
        what this application did until now (PROJECT_RULES §3)."""
        register(app, "gone", path="models/gone/best.pt")
        repository: Any = app.repository
        (Path(repository.root) / "models" / "gone" / "best.pt").unlink()
        from nanoscope.application.use_cases import run_analysis

        with pytest.raises(MissingFileError, match="gone"):
            run_analysis(
                repository,
                repository.list_images()[0].id,
                PipelineConfig(detector="log", mode="detect"),
                model_id="gone",
            )


class TestWhatTheDialogShows:
    def test_the_row_is_the_comparison(self, app: Nanoscope, session: SessionViewModel) -> None:
        """*Compare* is the records. What a model **does** to a scan is M8-T08's
        report through the M3-T15 harness, and answering it here from a record
        would be answering it wrong."""
        register(app, "a", input_size_px=640, class_map={0: "particle"})
        register(app, "b", input_size_px=320)
        dialog = ModelsDialog(session)

        assert dialog.table.rowCount() == 2
        sizes = {_cell(dialog, row, "Input (px)") for row in range(2)}
        assert sizes == {"640", "320"}
        assert {_cell(dialog, row, "Provenance") for row in range(2)} == {
            "trained here as a",
            "trained here as b",
        }

    def test_the_active_one_is_marked(self, app: Nanoscope, session: SessionViewModel) -> None:
        register(app, "a")
        register(app, "b")
        session.activate_model("b")
        dialog = ModelsDialog(session)

        marked = {
            _cell(dialog, row, "Model")
            for row in range(dialog.table.rowCount())
            if _cell(dialog, row, "Active") == ACTIVE
        }
        assert marked == {"b"}

    def test_activating_from_the_dialog_changes_what_runs(
        self, app: Nanoscope, session: SessionViewModel
    ) -> None:
        register(app, "a")
        dialog = ModelsDialog(session)
        dialog.table.selectRow(0)
        dialog.activate.click()

        assert session.active_model == "a"

    def test_an_empty_project_says_so(self, session: SessionViewModel) -> None:
        dialog = ModelsDialog(session)

        assert "No models yet" in dialog.note.text()

    def test_registered_but_unchosen_says_what_that_costs(
        self, app: Nanoscope, session: SessionViewModel
    ) -> None:
        register(app, "a")
        dialog = ModelsDialog(session)

        assert "No model is in use" in dialog.note.text()


class TestHowEachModelScores:
    """M8-T06 wrote down what it was not doing: *"Compare is the records, not a
    run of them — what a model does to a scan is M8-T08's report."* This is that
    half, in the same window, and it still runs no model: the annotations are the
    truth and the project's stored runs are the answer (ADR-0088).
    """

    def test_a_model_nobody_has_run_is_not_a_model_that_scored_badly(
        self, app: Nanoscope, session: SessionViewModel
    ) -> None:
        register(app)
        dialog = ModelsDialog(session)

        assert dialog.scores.rowCount() == 0
        assert "No model has been run" in dialog.score_note.text()

    def test_a_score_appears_from_a_stored_run(
        self, app: Nanoscope, session: SessionViewModel
    ) -> None:
        register(app)
        _annotate_and_detect(app, model_id="particles-v1", found=True)
        dialog = ModelsDialog(session)

        assert dialog.scores.rowCount() == 1
        assert _score(dialog, 0, "Model") == "particles-v1"
        assert _score(dialog, 0, "Recall") == "1.000"
        assert _score(dialog, 0, "Particles") == "1"

    def test_a_miss_is_visible_as_one(self, app: Nanoscope, session: SessionViewModel) -> None:
        register(app)
        _annotate_and_detect(app, model_id="particles-v1", found=False)
        dialog = ModelsDialog(session)

        assert _score(dialog, 0, "Recall") == "0.000"
        #: Nothing was reported, so there is no precision — blank, not 0.000.
        assert _score(dialog, 0, "Precision") == ""

    def test_it_says_the_split_is_no_longer_known(
        self, app: Nanoscope, session: SessionViewModel
    ) -> None:
        """A model this project never trained, or one whose dataset has been
        deleted from `cache/`: the score is over every scan, and the window says
        so rather than letting it read as generalisation."""
        register(app)
        _annotate_and_detect(app, model_id="particles-v1", found=True)
        dialog = ModelsDialog(session)

        assert "no longer say which scans" in dialog.score_note.text()
        assert "every scan" in _score(dialog, 0, "Scored on")


class TestTheDetectionPanelSaysWhatIsMissing:
    def test_registered_is_not_chosen(self, app: Nanoscope, session: SessionViewModel) -> None:
        """The matrix refuses a detector whose framework has **no registered
        model**; a project can have one registered and none in use, and without
        this the run preprocesses a scan and then refuses."""
        register(app)
        panel = DetectionPanel(session)
        _select_framework_detector(panel)

        assert not panel.run.isEnabled()
        assert "No model is in use" in panel.reason.text()

    def test_choosing_one_enables_it_without_a_restart(
        self, app: Nanoscope, session: SessionViewModel
    ) -> None:
        register(app)
        panel = DetectionPanel(session)
        _select_framework_detector(panel)

        session.activate_model("particles-v1")

        assert panel.run.isEnabled()
        assert panel.reason.text() == ""

    def test_the_menu_item_needs_a_project(self, app: Nanoscope) -> None:
        window = MainWindow(app)
        assert not window.models_action.isEnabled()

        window.session.refresh()
        assert window.models_action.isEnabled()


def _cell(dialog: ModelsDialog, row: int, column: str) -> str:
    item = dialog.table.item(row, COLUMNS.index(column))
    return "" if item is None else item.text()


def _select_framework_detector(panel: DetectionPanel) -> None:
    """Pick the detector that needs registered weights, without naming it.

    PROJECT_RULES §2.5 applies to the tests that police it too: the entry is
    found by asking the session which detector needs a model, which is the same
    question the panel asks.
    """
    for index in range(panel.detector.count()):
        option = panel.detector.itemData(index, Qt.ItemDataRole.UserRole)
        if option is not None and panel._session.needs_active_model(option.detector):
            panel.detector.setCurrentIndex(index)
            return
    raise AssertionError("no detector in this panel needs a registered model")


def _annotate_and_detect(app: Nanoscope, *, model_id: str, found: bool) -> None:
    """One drawn box, and one stored run by `model_id` that finds it or does not.

    `save_analysis(..., model_id=...)` is schema v10's column from M8-T06, which
    is what makes an evaluation possible without re-running anything.
    """
    import pandas as pd

    from nanoscope.core.entities import Detection, PipelineResult
    from nanoscope.core.entities.project import AnnotationSource

    repository: Any = app.repository
    image_id = repository.list_images()[0].id
    repository.add_annotation(
        image_id, label="particle", box=(10.0, 10.0, 20.0, 20.0), source=AnnotationSource.MANUAL
    )
    detections = (
        [Detection(x_px=15.0, y_px=15.0, radius_px=5.0, radius_nm=None, confidence=0.9)]
        if found
        else []
    )
    repository.save_analysis(
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


def _score(dialog: ModelsDialog, row: int, column: str) -> str:
    item = dialog.scores.item(row, SCORE_COLUMNS.index(column))
    return "" if item is None else item.text()
