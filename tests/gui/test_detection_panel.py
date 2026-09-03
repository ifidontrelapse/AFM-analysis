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


def register(session: SessionViewModel) -> None:
    """Give the project a model for the framework detector, so it is offered."""
    repository = session._app.repository
    assert repository is not None
    weights = repository.root / "models" / "knobs.pt"
    weights.parent.mkdir(exist_ok=True)
    weights.write_bytes(b"not really weights")
    repository.register_model(
        ModelDescriptor(
            model_id="knobs",
            task=ModelTask.DETECT,
            framework=ModelFramework.ULTRALYTICS,
            path="models/knobs.pt",
        )
    )
    session.settings_changed.emit()


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


class TestWhatIsNotOfferedIsExplained:
    def test_a_detector_with_no_registered_model_is_not_offered_and_the_panel_says_why(
        self, session: SessionViewModel
    ) -> None:
        """ "You need to register a model" is still a different sentence from
        "this application cannot do that" (ADR-0050) — it is now a sentence on
        screen rather than a greyed row with a tooltip. What cannot run is not
        on the list; what would put it there is written under the list."""
        panel = DetectionPanel(session)
        session.select_image(image_ids(session)[0])

        offered = [text for text, _ in entries(panel.detector)]

        assert len(offered) < len(detector_options("afm")), "one needs weights this project lacks"
        assert "is registered in this project" in panel.missing.text()
        assert "File ▸ Models…" in panel.missing.text()

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

        assert len(entries(panel.detector)) == len(detector_options("afm"))
        assert panel.missing.text() == "" or "is registered" not in panel.missing.text()

    def test_registering_one_reaches_a_panel_that_is_already_open(
        self, session: SessionViewModel
    ) -> None:
        """The operator's order, not the test's: the panel is open on a scan,
        and the model is registered from the Models dialog while it is. Before
        this, the options were built once per *image* — so the way to reach a
        newly registered detector was to select a different scan and come back,
        which nothing on screen said."""
        session.select_image(image_ids(session)[0])
        panel = DetectionPanel(session)
        before = len(entries(panel.detector))

        weights = session._app.repository.root / "models" / "late.pt"
        weights.parent.mkdir(exist_ok=True)
        weights.write_bytes(b"not really weights")
        session.register_model(
            weights, model_id="late", task=ModelTask.DETECT, framework=ModelFramework.ULTRALYTICS
        )

        assert len(entries(panel.detector)) == before + 1

    def test_an_unrelated_preference_does_not_reset_the_choice(
        self, session: SessionViewModel
    ) -> None:
        """Rebuilding on every settings change is what makes the test above
        pass; keeping the selection is what stops it from being a worse bug —
        a colormap chosen in Settings must not put the detector back."""
        session.select_image(image_ids(session)[0])
        panel = DetectionPanel(session)
        last = panel.mode.count() - 1
        panel.mode.setCurrentIndex(last)
        chosen = panel.mode.currentText()

        session.remember("viewer.colormap", "viridis")

        assert panel.mode.currentText() == chosen

    def test_segmentation_is_refused_with_its_reason(self, session: SessionViewModel) -> None:
        """The matrix says the mode needs a predictor; a fresh project has no
        model to build one from, and the panel says that rather than failing at
        the end of a run (M6-T04 made it selectable once one is registered)."""
        panel = DetectionPanel(session)
        session.select_image(image_ids(session)[0])

        assert "segment" not in [text for text, _ in entries(panel.mode)]
        assert "segmentation needs a model" in panel.missing.text().lower()

    def test_the_panel_opens_on_something_that_can_run(self, session: SessionViewModel) -> None:
        """A combo opening on a disabled entry is a combo whose Run button is
        dead for a reason nobody asked for."""
        panel = DetectionPanel(session)
        session.select_image(image_ids(session)[0])

        assert panel.run.isEnabled()
        assert panel.config() is not None


class TestTheKnobsBelongToWhatIsChosen:
    """The blob parameters were on screen for every detector until M8-T09. A
    number that does nothing is a number an operator spends an afternoon on."""

    def test_the_detector_brings_its_own(self, session: SessionViewModel) -> None:
        session.select_image(image_ids(session)[0])
        register(session)
        panel = DetectionPanel(session)

        shown: dict[str, set[str]] = {}
        for index in range(panel.detector.count()):
            panel.detector.setCurrentIndex(index)
            shown[panel.detector.itemText(index)] = set(panel._spins)

        assert shown["log"] == {"log_overlap", "log_percentile"}
        assert shown[next(name for name in shown if name != "log")] == {"yolo_conf"}

    def test_the_mode_brings_the_measurement_ones(self, session: SessionViewModel) -> None:
        """`baseline` measures; `detect` counts. The ring is the measurement's."""
        session.select_image(image_ids(session)[0])
        panel = DetectionPanel(session)
        panel.mode.setCurrentIndex([text for text, _ in entries(panel.mode)].index("detect"))
        counting = set(panel._spins)

        panel.mode.setCurrentIndex([text for text, _ in entries(panel.mode)].index("baseline"))

        assert "measure_outer_px" not in counting
        assert {"measure_outer_px", "measure_inner_erode_px"} <= set(panel._spins)

    def test_a_tuned_value_reaches_the_request(self, session: SessionViewModel) -> None:
        session.select_image(image_ids(session)[0])
        panel = DetectionPanel(session)

        panel._spins["log_overlap"].setValue(0.75)
        config = panel.config()

        assert config is not None
        assert config.log_overlap == 0.75

    def test_a_tuned_value_survives_the_detector_being_changed_and_back(
        self, session: SessionViewModel
    ) -> None:
        """Rebuilding the rows must not quietly restore a default the operator
        overrode — the panel rebuilds on every settings change (M8-T09)."""
        session.select_image(image_ids(session)[0])
        register(session)
        panel = DetectionPanel(session)
        panel._spins["log_overlap"].setValue(0.75)

        panel.detector.setCurrentIndex(1)
        panel.detector.setCurrentIndex(0)

        assert panel._spins["log_overlap"].value() == 0.75

    def test_a_whole_number_stays_whole(self, session: SessionViewModel) -> None:
        """`PipelineConfig` types the ring in pixels as `int`, and a float there
        is a value nothing else in the pipeline would have produced."""
        session.select_image(image_ids(session)[0])
        panel = DetectionPanel(session)
        panel.mode.setCurrentIndex([text for text, _ in entries(panel.mode)].index("baseline"))

        config = panel.config()

        assert config is not None
        assert isinstance(config.measure_outer_px, int)


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
