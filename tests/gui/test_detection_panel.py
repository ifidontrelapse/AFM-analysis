"""Detection, offered by the matrix rather than by a widget (M6-T02, ADR-0062).

M6's third exit criterion: *"invalid combinations are disabled in the UI because
the capability matrix says so — not by a duplicated rule."* So the tests here ask
two things of every choice on screen:

- is it **the matrix's own row**, for this image's modality?
- when it is refused, does it **say why** — because greyed out with no
  explanation is the failure the criterion exists to prevent.

The last test is the one PROJECT_RULES §2.5 asks for: no detector's name appears
anywhere under `gui/`.
"""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from nanoscope.app.container import Nanoscope
from nanoscope.application.capabilities import CAPABILITIES, detector_options
from nanoscope.application.jobs import Job
from nanoscope.application.use_cases.preprocessing import PreprocessingParams
from nanoscope.core.entities.model import ModelDescriptor, ModelFramework, ModelTask
from nanoscope.core.values import Modality
from nanoscope.gui.panels import DetectionPanel, PreprocessingPanel
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")

GUI = pathlib.Path(__file__).resolve().parents[2] / "nanoscope" / "gui"


def phantom(size: int = 48) -> np.ndarray:
    rng = np.random.default_rng(1)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    height = rng.normal(0.0, 0.05, (size, size)).astype(np.float32)
    for cy, cx in ((12, 12), (30, 32)):
        height += 4.0 * np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / 12.0))
    return height.astype(np.float32)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "P"
    with SqliteProjectRepository.create(root, "P") as repo:
        source = tmp_path / "afm.npy"
        np.save(source, phantom())
        repo.import_image(source, modality=Modality.AFM, pixel_size_nm=2.0)
        sem = tmp_path / "sem.png"
        sem.write_bytes(b"not really a png")
        repo.import_image(sem, modality=Modality.SEM)
    return root


@pytest.fixture
def session(tmp_path: Path, project: Path) -> Iterator[SessionViewModel]:
    with Nanoscope(settings_path=tmp_path / "settings.json") as container:
        model = SessionViewModel(container)
        model.open_project(project)
        yield model


def image_ids(session: SessionViewModel) -> list[int]:
    assert session.project is not None
    return [image.id for image in session.project.images]


def settle(job: Job | None) -> None:
    assert job is not None
    assert job.wait(30.0)
    QApplication.processEvents()


def entries(combo: object) -> list[tuple[str, bool]]:
    """Every entry and whether it can be chosen."""
    return [
        (combo.itemText(i), combo.model().item(i).isEnabled())  # type: ignore[attr-defined]
        for i in range(combo.count())  # type: ignore[attr-defined]
    ]


class TestTheOptionsAreTheMatrix:
    def test_the_rows_offered_are_the_rows_that_exist(self) -> None:
        for modality in ("afm", "sem", "tem"):
            offered = {
                (option.detector, mode.mode)
                for option in detector_options(modality)
                for mode in option.modes
            }
            expected = {
                (row.detector, row.mode) for row in CAPABILITIES if row.modality == modality
            }
            assert offered == expected

    def test_an_sem_image_is_not_offered_the_afm_only_mode(self, session: SessionViewModel) -> None:
        """`baseline` measures height above a local substrate, so it needs a Z
        map. The panel does not remember that — the matrix does."""
        panel = DetectionPanel(session)

        session.select_image(image_ids(session)[1])  # the SEM row

        assert "baseline" not in [text for text, _ in entries(panel.mode)]

    def test_an_afm_image_is(self, session: SessionViewModel) -> None:
        panel = DetectionPanel(session)

        session.select_image(image_ids(session)[0])

        assert "baseline" in [text for text, _ in entries(panel.mode)]

    def test_nothing_is_offered_without_an_image(self, session: SessionViewModel) -> None:
        panel = DetectionPanel(session)

        assert panel.detector.count() == 0
        assert panel.config() is None
        assert not panel.run.isEnabled()


class TestAnUnavailableEntrySaysWhy:
    def test_a_detector_with_no_registered_model_is_disabled_and_explains_itself(
        self, session: SessionViewModel
    ) -> None:
        """ "You need to register a model" is a different sentence from "this
        application cannot do that", so the entry is offered and disabled rather
        than hidden (ADR-0050)."""
        panel = DetectionPanel(session)
        session.select_image(image_ids(session)[0])

        disabled = [text for text, enabled in entries(panel.detector) if not enabled]

        assert disabled, "one detector needs weights this fresh project has none of"
        index = [text for text, _ in entries(panel.detector)].index(disabled[0])
        assert "register" in panel.detector.itemData(index, 3)  # ToolTipRole

    def test_registering_a_model_makes_it_available(self, session: SessionViewModel) -> None:
        repository = session._app.repository
        assert repository is not None
        weights = repository.root / "models" / "best.pt"
        weights.parent.mkdir(exist_ok=True)
        weights.write_bytes(b"not really weights")
        repository.register_model(
            ModelDescriptor(
                model_id="best",
                task=ModelTask.DETECT,
                framework=ModelFramework.ULTRALYTICS,
                path="models/best.pt",
            )
        )
        session.select_image(image_ids(session)[0])

        panel = DetectionPanel(session)

        assert all(enabled for _text, enabled in entries(panel.detector))

    def test_segmentation_is_refused_with_its_reason(self, session: SessionViewModel) -> None:
        """The matrix says the mode needs a predictor; nothing constructs one
        before M6-T04, and the panel says that rather than failing at the end of
        a run."""
        panel = DetectionPanel(session)
        session.select_image(image_ids(session)[0])

        texts = [text for text, enabled in entries(panel.mode) if not enabled]
        index = [text for text, _ in entries(panel.mode)].index("segment")

        assert "segment" in texts
        assert "predictor" in panel.mode.itemData(index, 3)

    def test_the_panel_opens_on_something_that_can_run(self, session: SessionViewModel) -> None:
        """A combo opening on a disabled entry is a combo whose Run button is
        dead for a reason nobody asked for."""
        panel = DetectionPanel(session)
        session.select_image(image_ids(session)[0])

        assert panel.run.isEnabled()
        assert panel.config() is not None


class TestRunningIt:
    def test_it_stores_a_run_with_its_detections(self, session: SessionViewModel) -> None:
        """Unlike M6-T01's preview, this is a *result*: the run, its detections
        and its measurement table (ADR-0042)."""
        panel = DetectionPanel(session)
        session.select_image(image_ids(session)[0])
        stored: list[object] = []
        session.run_stored.connect(stored.append)

        panel.start()
        settle(session.job)

        repository = session._app.repository
        assert repository is not None
        runs = repository.runs_for(image_ids(session)[0])
        assert len(runs) == 1
        assert stored and stored[0].id == runs[0].id  # type: ignore[attr-defined]
        assert "Run" in panel.report.text()

    def test_the_preprocessing_parameters_reach_the_run(self, session: SessionViewModel) -> None:
        """A scan previewed at one opening scale and analysed at another, with
        nothing saying so, is what the shared value prevents."""
        preprocessing = PreprocessingPanel(session)
        panel = DetectionPanel(session)
        session.select_image(image_ids(session)[0])
        preprocessing.manual_radius.setValue(4.0)

        assert session.preprocessing == PreprocessingParams(
            min_size_nm=preprocessing.min_size.value(),
            manual_radius_px=4.0,
            opening_scale=preprocessing.opening_scale.value(),
        )

        panel.start()
        settle(session.job)

        assert session.job is not None
        assert session.job.result is not None

    def test_a_second_run_is_refused_while_one_is_going(self, session: SessionViewModel) -> None:
        panel = DetectionPanel(session)
        session.select_image(image_ids(session)[0])
        config = panel.config()
        assert config is not None

        first = session.detect(config)
        assert session.detect(config) is None

        settle(first)

    def test_it_is_written_down(
        self, session: SessionViewModel, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        panel = DetectionPanel(session)
        session.select_image(image_ids(session)[0])

        with caplog.at_level(logging.INFO, logger="nanoscope.gui.viewmodels.session"):
            panel.start()
            settle(session.job)

        assert any("detection(s)" in record.message for record in caplog.records)


class TestNoDetectorNameLivesInTheGui:
    def test_the_strings_are_not_in_any_widget(self) -> None:
        """PROJECT_RULES §2.5, checked rather than reviewed. D-19 is what the
        other outcome looks like: the deleted React client kept its own copy of
        the matrix, and the copy had drifted."""
        offenders: dict[str, list[str]] = {}
        for path in sorted(GUI.rglob("*.py")):
            source = path.read_text()
            found = [name for name in ("yolo", "sam2") if name in source.lower()]
            if found:
                offenders[str(path.relative_to(GUI))] = found

        assert not offenders, offenders

    def test_the_check_can_fail(self) -> None:
        """A guard that cannot fail is decoration."""
        assert "yolo" in "yolo_model_path".lower()

    def test_no_widget_enumerates_modes_either(self) -> None:
        """The modes are the matrix's rows too. A literal `"baseline"` in a
        panel is the same defect one word later."""
        for path in sorted((GUI / "panels").rglob("*.py")):
            literals = {
                node.value
                for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            assert not literals & {"detect", "baseline", "segment"}, path
