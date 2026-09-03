"""Models as records, and the registry that resolves them (M4-T13, ADR-0050).

M4's fourth exit criterion: *"model registry resolves `yolo` and `sam2` to
providers via `ModelDescriptor`"*. It is met here without a single weight file,
because the registry hands back **factories** — nothing is loaded until somebody
calls one, which is also what keeps "what models does this project have?" a
cheap question.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from nanoscope.core.entities import ModelDescriptor, ModelFramework, ModelTask
from nanoscope.core.entities.device import Device
from nanoscope.core.errors import InvalidParameterError, UnsupportedRequestError
from nanoscope.core.values import DeviceKind
from nanoscope.infrastructure.models import registry
from nanoscope.infrastructure.storage import SqliteProjectRepository
from nanoscope.infrastructure.storage.project_repository import sha256_of

YOLO = ModelDescriptor(
    model_id="particles-v12",
    task=ModelTask.DETECT,
    framework=ModelFramework.ULTRALYTICS,
    path="models/best12x.pt",
    input_size_px=640,
    class_map={0: "particle"},
    provenance="trained 2026-08-01 on 412 annotations",
    sha256="a" * 64,
)


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[SqliteProjectRepository]:
    with SqliteProjectRepository.create(tmp_path / "P", "P") as repository:
        yield repository


class TestAModelIsARecord:
    def test_it_round_trips(self, repo: SqliteProjectRepository) -> None:
        stored = repo.register_model(YOLO)

        assert repo.get_model("particles-v12") == stored
        assert stored.class_map == {0: "particle"}
        assert stored.provenance.startswith("trained")
        assert stored.sha256 == "a" * 64

    def test_registering_it_records_when(self, repo: SqliteProjectRepository) -> None:
        assert repo.register_model(YOLO).registered_utc

    def test_the_same_id_replaces(self, repo: SqliteProjectRepository) -> None:
        """Retraining produces a new file under the name the configuration
        already uses; two rows for one id would make "which one" a question."""
        repo.register_model(YOLO)

        from dataclasses import replace

        repo.register_model(replace(YOLO, provenance="retrained 2026-08-12"))

        assert len(repo.list_models()) == 1
        assert repo.get_model("particles-v12").provenance == "retrained 2026-08-12"

    def test_an_unknown_id_is_refused_by_name(self, repo: SqliteProjectRepository) -> None:
        """The id came from a configuration, so a typo there is the likely cause
        and the message has to carry it back."""
        with pytest.raises(InvalidParameterError, match="particles-v99"):
            repo.get_model("particles-v99")

    def test_weights_inside_the_project_are_stored_relative(
        self, repo: SqliteProjectRepository
    ) -> None:
        from dataclasses import replace

        absolute = repo.root / "models" / "best.pt"
        stored = repo.register_model(replace(YOLO, path=str(absolute)))

        assert stored.path == "models/best.pt"
        assert not stored.is_external
        assert repo.path_of_model(stored) == absolute

    def test_a_shared_checkpoint_keeps_its_absolute_path(
        self, repo: SqliteProjectRepository, tmp_path: Path
    ) -> None:
        """Nobody copies a 137 MB checkpoint into every project. The consequence
        is stated rather than prevented: this project opens on another machine
        and that model is simply unavailable there (ADR-0050)."""
        from dataclasses import replace

        shared = tmp_path / "checkpoints" / "best12x.pt"
        stored = repo.register_model(replace(YOLO, path=str(shared)))

        assert stored.path == str(shared)
        assert stored.is_external
        assert repo.path_of_model(stored) == shared

    def test_a_checksum_is_computed_from_the_file_when_nobody_gave_one(
        self, repo: SqliteProjectRepository
    ) -> None:
        """M8-T04, and this module's oldest rule (ADR-0040): a checksum
        describes the file the row points at, so it is taken here rather than
        accepted as an argument. ADR-0050 left it `None` *if nobody computed
        it*; a run that just wrote the weights is somebody."""
        from dataclasses import replace

        weights = repo.root / "models" / "run-1" / "best.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"not a model")
        stored = repo.register_model(replace(YOLO, path="models/run-1/best.pt", sha256=None))

        assert stored.sha256 == sha256_of(weights)
        assert repo.get_model("particles-v12").sha256 == stored.sha256

    def test_a_checksum_the_caller_gave_is_kept(self, repo: SqliteProjectRepository) -> None:
        """Nothing re-reads 137 MB to second-guess a caller who already knows."""
        from dataclasses import replace

        weights = repo.root / "models" / "shared.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b"not a model")

        assert repo.register_model(replace(YOLO, path="models/shared.pt")).sha256 == "a" * 64

    def test_a_model_whose_weights_are_not_here_has_no_checksum(
        self, repo: SqliteProjectRepository
    ) -> None:
        """An absent file is a state, not a hash of nothing (ADR-0025's rule)."""
        from dataclasses import replace

        assert repo.register_model(replace(YOLO, sha256=None)).sha256 is None

    def test_they_survive_the_session(self, tmp_path: Path) -> None:
        with SqliteProjectRepository.create(tmp_path / "Q", "Q") as repo:
            stored = repo.register_model(YOLO)

        with SqliteProjectRepository.open(tmp_path / "Q") as repo:
            assert repo.list_models() == [stored]


class TestTheRegistryResolves:
    def test_yolo_resolves_to_a_provider(self) -> None:
        """The exit criterion, without a weight file in sight."""
        factory = registry.resolve(YOLO)

        assert callable(factory)

    def test_sam2_resolves_to_a_provider(self) -> None:
        from dataclasses import replace

        segmenter = replace(YOLO, framework=ModelFramework.SAM2, task=ModelTask.SEGMENT)

        assert callable(registry.resolve(segmenter))

    def test_resolving_loads_nothing(self, tmp_path: Path) -> None:
        """A registry that constructed on lookup would make listing a project's
        models expensive, and impossible in CI where no weights exist. This
        resolves a model whose file does not exist at all."""
        from dataclasses import replace

        missing = replace(YOLO, path=str(tmp_path / "does-not-exist.pt"))

        assert callable(registry.resolve(missing))

    def test_an_unknown_framework_is_refused_and_says_what_it_knows(self) -> None:
        class Pretend:
            framework = "tensorflow"
            model_id = "x"

        with pytest.raises(UnsupportedRequestError) as excinfo:
            registry.resolve(Pretend())  # type: ignore[arg-type]

        assert "tensorflow" in str(excinfo.value)
        assert "ultralytics" in str(excinfo.value)

    def test_every_registered_framework_is_listed(self) -> None:
        assert set(registry.frameworks()) == {ModelFramework.ULTRALYTICS, ModelFramework.SAM2}

    def test_the_device_reaches_the_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The gap ADR-0049 named: M4-T12 resolves a device and nothing consumed
        it. The factory is where it arrives, so no provider ever asks torch
        where it should run (ADR-0004)."""
        seen: dict[str, object] = {}

        class FakeYolo:
            def __init__(self, model_path: str, device: str | None = None) -> None:
                seen["path"] = model_path
                seen["device"] = device

        monkeypatch.setattr("nanoscope.infrastructure.models.yolo.YoloDetector", FakeYolo)
        device = Device(kind=DeviceKind.CUDA, name="A card", torch_name="cuda:0")

        registry.resolve(YOLO)(Path("/weights/best.pt"), device)

        assert seen == {"path": "/weights/best.pt", "device": "cuda:0"}
