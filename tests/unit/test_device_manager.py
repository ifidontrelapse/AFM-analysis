"""Every backend, on a machine that has none of them (M4-T12, ADR-0049).

There is no GPU in CI and no Mac anywhere near this project, so the probe is
tested against a **fake torch**: a module object with exactly the attributes the
manager reads. That is not a compromise — it is the only way to test the ROCm
branch at all, and the ROCm branch is the one a naive implementation gets wrong.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from nanoscope.core.values import DeviceKind
from nanoscope.infrastructure.device import DeviceManager


def fake_torch(
    *,
    cuda: int = 0,
    hip: str | None = None,
    mps: bool = False,
    names: tuple[str, ...] = (),
) -> types.ModuleType:
    """A stand-in torch that answers exactly what the probe asks it.

    Args:
        cuda: how many devices `torch.cuda` reports.
        hip: the ROCm version string, which is what makes those devices AMD.
        mps: whether Apple's backend says it is available.
        names: device names, defaulting to something plausible.
    """
    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: cuda > 0,
        device_count=lambda: cuda,
        get_device_name=lambda index: names[index] if index < len(names) else f"Device {index}",
    )
    module.version = types.SimpleNamespace(hip=hip)  # type: ignore[attr-defined]
    module.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        mps=types.SimpleNamespace(is_available=lambda: mps)
    )
    return module


@pytest.fixture
def install_torch(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Put a fake torch where `import torch` will find it."""

    def install(module: types.ModuleType | None) -> DeviceManager:
        if module is None:
            monkeypatch.setitem(sys.modules, "torch", None)
        else:
            monkeypatch.setitem(sys.modules, "torch", module)
        return DeviceManager()

    return install


class TestProbing:
    def test_without_torch_there_is_still_a_cpu(self, install_torch: Any) -> None:
        """CI installs no torch on purpose. A device manager that raises on the
        import is one no test can run — and the machine still has a processor."""
        manager = install_torch(None)

        devices = manager.available()

        assert [device.kind for device in devices] == [DeviceKind.CPU]
        assert devices[0].torch_name == "cpu"

    def test_a_cpu_only_torch_reports_the_cpu(self, install_torch: Any) -> None:
        manager = install_torch(fake_torch())

        assert [device.kind for device in manager.available()] == [DeviceKind.CPU]

    def test_a_cuda_card_is_found_and_named(self, install_torch: Any) -> None:
        manager = install_torch(fake_torch(cuda=1, names=("NVIDIA GeForce RTX 4090",)))

        first = manager.available()[0]

        assert first.kind is DeviceKind.CUDA
        assert first.name == "NVIDIA GeForce RTX 4090"
        assert first.torch_name == "cuda:0"

    def test_an_amd_card_is_rocm_and_not_cuda(self, install_torch: Any) -> None:
        """The branch a naive probe gets wrong: a ROCm build answers
        `cuda.is_available()` with True and serves AMD cards through the same
        API, so without `torch.version.hip` a Radeon is reported as CUDA."""
        manager = install_torch(fake_torch(cuda=1, hip="6.0.0", names=("AMD Radeon RX 7900 XTX",)))

        first = manager.available()[0]

        assert first.kind is DeviceKind.ROCM
        assert first.name == "AMD Radeon RX 7900 XTX"

    def test_apples_backend_is_found(self, install_torch: Any) -> None:
        manager = install_torch(fake_torch(mps=True))

        assert [device.kind for device in manager.available()] == [DeviceKind.MPS, DeviceKind.CPU]

    def test_every_card_is_listed(self, install_torch: Any) -> None:
        manager = install_torch(fake_torch(cuda=2, names=("A", "B")))

        assert [device.torch_name for device in manager.available()] == ["cuda:0", "cuda:1", "cpu"]

    def test_the_best_comes_first_and_the_cpu_last(self, install_torch: Any) -> None:
        manager = install_torch(fake_torch(cuda=1, mps=True))

        assert [device.kind for device in manager.available()] == [
            DeviceKind.CUDA,
            DeviceKind.MPS,
            DeviceKind.CPU,
        ]

    def test_probing_happens_once(self, install_torch: Any) -> None:
        """It imports torch and queries a driver, which a settings dialog must
        not do on every repaint."""
        calls = []
        module = fake_torch(cuda=1)
        original = module.cuda.is_available
        module.cuda.is_available = lambda: (calls.append(1), original())[1]
        manager = install_torch(module)

        manager.available()
        manager.available()

        assert len(calls) == 1

    def test_refresh_asks_again(self, install_torch: Any) -> None:
        """For the operator who fixed a driver without restarting."""
        manager = install_torch(fake_torch())
        assert [d.kind for d in manager.available()] == [DeviceKind.CPU]

        monkeypatched = fake_torch(cuda=1)
        sys.modules["torch"] = monkeypatched

        assert [d.kind for d in manager.refresh()] == [DeviceKind.CUDA, DeviceKind.CPU]


class TestSelecting:
    def test_with_no_preference_it_takes_the_best(self, install_torch: Any) -> None:
        manager = install_torch(fake_torch(cuda=1))

        selection = manager.select()

        assert selection.device.kind is DeviceKind.CUDA
        assert not selection.is_fallback

    def test_a_preference_that_exists_is_honoured(self, install_torch: Any) -> None:
        manager = install_torch(fake_torch(cuda=1))

        selection = manager.select(DeviceKind.CPU)

        assert selection.device.kind is DeviceKind.CPU
        assert not selection.is_fallback
        assert selection.reason == ""

    def test_an_unavailable_preference_falls_back_and_says_why(self, install_torch: Any) -> None:
        """ADR-0004 asked for the reason in those words. A fallback nobody is
        told about is a silent forty-fold slowdown that reads as the application
        being slow."""
        manager = install_torch(fake_torch())

        selection = manager.select(DeviceKind.CUDA)

        assert selection.device.kind is DeviceKind.CPU
        assert selection.requested is DeviceKind.CUDA
        assert selection.is_fallback
        assert "CUDA was requested" in selection.reason
        assert "CPU" in selection.reason

    def test_it_never_raises_for_hardware_that_is_not_there(self, install_torch: Any) -> None:
        manager = install_torch(None)

        for kind in DeviceKind:
            assert manager.select(kind).device.kind is DeviceKind.CPU

    def test_the_fallback_is_logged(
        self, install_torch: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        manager = install_torch(fake_torch())

        with caplog.at_level("WARNING"):
            manager.select(DeviceKind.MPS)

        assert "device fallback" in caplog.text
