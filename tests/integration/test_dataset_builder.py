"""Annotations become a dataset a trainer can read (M8-T02).

Integration rather than unit: the subject is a real project directory — scans
that go through preprocessing, PNGs and label files landing under `cache/`, and
a `DatasetSpec` a `TrainingProvider` will accept.

Two assertions carry the task and neither is about the file layout:

- **the picture is the one inference makes**, so a model is not trained on a
  distribution it will never be shown;
- **the split is by image**, so no scan has boxes in both halves.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from nanoscope.app.container import Nanoscope
from nanoscope.application.use_cases import build_dataset
from nanoscope.application.use_cases.dataset import DATASET_FILE, IMAGES_DIRECTORY, TRAIN, VAL
from nanoscope.core.entities.project import AnnotationSource
from nanoscope.core.errors import AnalysisFailedError, InvalidParameterError
from nanoscope.core.values import Modality, Polarity, default_polarity
from nanoscope.infrastructure.imaging.network_input import as_network_input
from nanoscope.infrastructure.storage import SqliteProjectRepository

SCANS = 10
BOXES_PER_SCAN = 3


def phantom(seed: int, size: int = 48) -> np.ndarray:
    """A tilted height map with a few bumps — enough for the substrate step."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    z = 0.05 * x + 0.02 * y + rng.normal(0.0, 0.2, (size, size)).astype(np.float32)
    for _ in range(4):
        cy, cx = rng.integers(6, size - 6, 2)
        z[cy - 3 : cy + 3, cx - 3 : cx + 3] += 12.0
    return z.astype(np.float32)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "P"
    with SqliteProjectRepository.create(root, "P") as repo:
        for index in range(SCANS):
            source = tmp_path / f"scan{index}.npy"
            np.save(source, phantom(index))
            record = repo.import_image(source, modality=Modality.AFM, pixel_size_nm=2.0)
            for box in range(BOXES_PER_SCAN):
                repo.add_annotation(
                    record.id,
                    label="particle",
                    box=(2.0 + box * 8, 2.0, 8.0 + box * 8, 8.0),
                    source=AnnotationSource.MANUAL,
                )
    return root


@pytest.fixture
def app(tmp_path: Path, project: Path) -> Iterator[Nanoscope]:
    with Nanoscope(settings_path=tmp_path / "settings.json") as container:
        container.open(project)
        yield container


def repo_of(app: Nanoscope) -> SqliteProjectRepository:
    assert app.repository is not None
    return app.repository  # type: ignore[return-value]


class TestWhatItBuilds:
    def test_it_returns_a_spec_that_adds_up(self, app: Nanoscope) -> None:
        report = build_dataset(repo_of(app), val_fraction=0.2, seed=1)

        assert report.spec.train_images + report.spec.val_images == SCANS
        assert report.spec.val_images == 2
        assert report.spec.classes == ("particle",)
        assert report.boxes == SCANS * BOXES_PER_SCAN
        assert not report.skipped

    def test_a_picture_and_a_label_land_for_every_scan(self, app: Nanoscope, project: Path) -> None:
        report = build_dataset(repo_of(app), seed=1)
        root = project / report.spec.root

        pictures = sorted(root.glob(f"{IMAGES_DIRECTORY}/*/*.png"))
        labels = sorted(root.glob("labels/*/*.txt"))
        assert len(pictures) == SCANS
        assert len(labels) == SCANS
        assert {p.stem for p in pictures} == {label.stem for label in labels}

    def test_it_lands_under_cache_because_it_is_re_creatable(
        self, app: Nanoscope, project: Path
    ) -> None:
        """PROJECT_RULES §5: anything under `cache/` is safely deletable. A
        dataset is derived from annotations that are still in the database."""
        report = build_dataset(repo_of(app))

        assert report.spec.root.startswith("cache/")
        assert (project / report.spec.root).is_dir()

    def test_the_manifest_names_the_halves_and_the_classes(
        self, app: Nanoscope, project: Path
    ) -> None:
        report = build_dataset(repo_of(app), seed=7)

        manifest = (project / report.spec.root / DATASET_FILE).read_text()
        assert f"train: {IMAGES_DIRECTORY}/{TRAIN}" in manifest
        assert f"val: {IMAGES_DIRECTORY}/{VAL}" in manifest
        assert "nc: 1" in manifest
        assert "0: 'particle'" in manifest
        # The seed is in the file because a rebuild that splits differently
        # makes two runs incomparable, and a person has to be able to check.
        assert "seed: 7" in manifest

    def test_with_nothing_held_out_val_still_resolves(self, app: Nanoscope, project: Path) -> None:
        """Found by M8-T03's contract suite, not by this file.

        A dataset built with `val_fraction=0.0` never creates `images/val`, and
        ultralytics **refuses the manifest** before the first epoch — measured:
        *"Dataset error"*. It also validates the final epoch whether or not it
        was asked to, so the directory has to exist and be readable.

        `val` therefore points at the training split, and the file says so. The
        numbers that come out of it are the model scored on what it trained on,
        and `LocalTrainingProvider` is what refuses to report them as validation
        (ADR-0081) — because ADR-0080's block means *a held-out set existed*.
        """
        report = build_dataset(repo_of(app), val_fraction=0.0, directory_name="none-held-out")

        manifest = (project / report.spec.root / DATASET_FILE).read_text()
        assert f"val: {IMAGES_DIRECTORY}/{TRAIN}" in manifest
        assert "not validation" in manifest
        assert (project / report.spec.root / IMAGES_DIRECTORY / TRAIN).is_dir()

    def test_two_builds_do_not_overwrite_each_other(self, app: Nanoscope) -> None:
        first = build_dataset(repo_of(app), directory_name="one")
        second = build_dataset(repo_of(app), directory_name="two")

        assert first.spec.root != second.spec.root


class TestThePictureIsTheOneInferenceMakes:
    """The decision the task turns on.

    A model trained on pictures made one way and used on pictures made another
    is measured on a question nobody asked, and the failure is silent — no
    exception, no wrong shape, just a detector that is worse than it should be.
    """

    def test_the_written_png_is_what_as_network_input_produces(
        self, app: Nanoscope, project: Path
    ) -> None:
        import cv2

        from nanoscope.application.use_cases import preprocess_image

        repository = repo_of(app)
        record = repository.list_images()[0]
        report = build_dataset(repository, directory_name="d", val_fraction=0.0)

        expected = as_network_input(
            preprocess_image(repository, record.id).z_result,
            polarity=default_polarity(record.modality),
        )
        stem = Path(record.relative_path).stem
        written = cv2.imread(
            str(project / report.spec.root / IMAGES_DIRECTORY / TRAIN / f"{stem}.png"),
            cv2.IMREAD_GRAYSCALE,
        )

        assert np.array_equal(written, expected)

    def test_the_detector_and_the_builder_call_the_same_function(self) -> None:
        """Not a copy of it. `display.py` kept a second copy of a four-entry
        extension map and a folder of scans would not open (2026-08-30); this
        one would produce a worse model instead of an error message.
        """
        from nanoscope.infrastructure.models import yolo

        assert yolo.as_network_input is as_network_input

    def test_the_picture_is_built_from_z_above_not_the_raw_file(
        self, app: Nanoscope, project: Path
    ) -> None:
        """`detect` is handed `z_flat - substrate`. A dataset made from raw
        height maps would teach a model the tilt and the substrate that
        inference has already removed — and these phantoms are tilted."""
        import cv2

        from nanoscope.application.use_cases.display import load_for_display

        repository = repo_of(app)
        record = repository.list_images()[0]
        report = build_dataset(repository, directory_name="d", val_fraction=0.0)

        raw = as_network_input(
            load_for_display(repository, record.id).data, polarity=Polarity.BRIGHT_ON_DARK
        )
        written = cv2.imread(
            str(
                project
                / report.spec.root
                / IMAGES_DIRECTORY
                / TRAIN
                / f"{Path(record.relative_path).stem}.png"
            ),
            cv2.IMREAD_GRAYSCALE,
        )

        assert not np.array_equal(written, raw)


class TestTheSplit:
    def test_no_scan_has_boxes_in_both_halves(self, app: Nanoscope, project: Path) -> None:
        """Leakage, and the one thing here a reviewer cannot see from the output.

        Two boxes off one scan, one in each half, makes the validation score a
        measurement of how well the model memorised that scan's substrate — and
        every number M8-T08 reports is inflated by it.
        """
        report = build_dataset(repo_of(app), val_fraction=0.3, seed=3)
        root = project / report.spec.root

        trained = {p.stem for p in root.glob(f"{IMAGES_DIRECTORY}/{TRAIN}/*.png")}
        held = {p.stem for p in root.glob(f"{IMAGES_DIRECTORY}/{VAL}/*.png")}

        assert trained and held
        assert not trained & held

    def test_the_same_seed_splits_the_same_way(self, app: Nanoscope, project: Path) -> None:
        first = build_dataset(repo_of(app), val_fraction=0.3, seed=5, directory_name="a")
        second = build_dataset(repo_of(app), val_fraction=0.3, seed=5, directory_name="b")

        assert _held_out(project, first.spec.root) == _held_out(project, second.spec.root)

    def test_a_different_seed_can_split_differently(self, app: Nanoscope, project: Path) -> None:
        """A guard on the guard: a "deterministic" split that ignores the seed
        would pass the test above and mean nothing."""
        splits = {
            frozenset(
                _held_out(
                    project,
                    build_dataset(
                        repo_of(app), val_fraction=0.3, seed=seed, directory_name=f"s{seed}"
                    ).spec.root,
                )
            )
            for seed in range(6)
        }

        assert len(splits) > 1

    def test_holding_nothing_out_is_legal(self, app: Nanoscope) -> None:
        """`val_images == 0` means the `validation` metric block is absent for
        every epoch, which is a state ADR-0080 already defined."""
        report = build_dataset(repo_of(app), val_fraction=0.0)

        assert report.spec.val_images == 0
        assert report.spec.train_images == SCANS

    def test_a_fifth_of_four_scans_is_none_of_them(self, tmp_path: Path) -> None:
        """Rounded down and then not up: holding out one of four is a 25%
        validation set reported as 20%, and zero is the honest answer."""
        root = tmp_path / "Q"
        with SqliteProjectRepository.create(root, "Q") as repo:
            for index in range(4):
                source = tmp_path / f"q{index}.npy"
                np.save(source, phantom(index))
                record = repo.import_image(source, modality=Modality.AFM, pixel_size_nm=2.0)
                repo.add_annotation(
                    record.id,
                    label="particle",
                    box=(2.0, 2.0, 8.0, 8.0),
                    source=AnnotationSource.MANUAL,
                )
            report = build_dataset(repo, val_fraction=0.2)

        assert report.spec.val_images == 0

    @pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.5])
    def test_a_fraction_that_is_not_one_is_refused(self, app: Nanoscope, fraction: float) -> None:
        with pytest.raises(InvalidParameterError):
            build_dataset(repo_of(app), val_fraction=fraction)


class TestWhatItRefuses:
    def test_a_project_with_no_annotations_is_refused(self, tmp_path: Path) -> None:
        """An empty dataset is indistinguishable from "nothing was drawn", which
        is a different statement (ADR-0048's rule, third site)."""
        root = tmp_path / "Empty"
        with SqliteProjectRepository.create(root, "Empty") as repo:
            source = tmp_path / "lonely.npy"
            np.save(source, phantom(0))
            repo.import_image(source, modality=Modality.AFM)

            with pytest.raises(AnalysisFailedError, match="no annotation"):
                build_dataset(repo)

    def test_the_caller_chooses_which_sources_go_in(self, app: Nanoscope) -> None:
        """ADR-0044: a model trained on its own output is confirming itself."""
        repository = repo_of(app)
        record = repository.list_images()[0]
        repository.add_annotation(
            record.id,
            label="adopted",
            box=(1.0, 1.0, 5.0, 5.0),
            source=AnnotationSource.FROM_DETECTION,
        )

        hand_drawn = build_dataset(
            repository, sources=(AnnotationSource.MANUAL,), directory_name="hand"
        )
        everything = build_dataset(repository, directory_name="all")

        assert hand_drawn.spec.classes == ("particle",)
        assert everything.spec.classes == ("adopted", "particle")


def _held_out(project: Path, root: str) -> set[str]:
    return {p.stem for p in (project / root / IMAGES_DIRECTORY / VAL).glob("*.png")}


def test_a_provider_accepts_what_the_builder_produced(app: Nanoscope, project: Path) -> None:
    """The seam closing: M8-T01 declared `DatasetSpec` and M8-T02 is the first
    thing that makes one, so this is the first time the two halves meet.

    Against the fake, because `LocalTrainingProvider` is M8-T03 and brings
    ultralytics with it — what is asserted is that the spec is *acceptable*, not
    that anything trains.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "contract"))
    from fake_provider import FakeTrainingProvider

    from nanoscope.core.entities.training import TrainingConfig

    spec = build_dataset(repo_of(app), val_fraction=0.2, seed=1).spec
    provider = FakeTrainingProvider(project)

    run = provider.start(
        spec,
        TrainingConfig(
            base_model="n.pt", epochs=2, image_size_px=640, output_directory="models/first"
        ),
    )

    assert run.dataset.root == spec.root
    assert not run.is_finished
    provider.cancel(run.run_id)
