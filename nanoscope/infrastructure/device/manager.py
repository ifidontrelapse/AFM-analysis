"""The one component that decides where inference runs (M4-T12, ADR-0004).

W8, closed: before this, nothing chose. `grep -r cuda nanoscope/infrastructure/
models` returned nothing at all, so every inference this project ever ran went
wherever torch felt like putting it — which on a machine with a GPU and a
mis-built torch is the CPU, silently, at a fortieth of the speed.

Three things here are decisions rather than plumbing (ADR-0049):

- **No torch is CPU, not an error.** CI installs no torch at all, and a machine
  without it still has a processor.
- **ROCm is told apart by `torch.version.hip`**, because a ROCm build answers
  `torch.cuda.is_available()` with `True` and serves AMD cards through the
  `torch.cuda` API. Without the check, a Radeon is reported as CUDA.
- **A fallback says why, in a sentence.** ADR-0004 asked for it in those words.
"""

from __future__ import annotations

import logging
from typing import Any

from nanoscope.core.entities.device import Device, DeviceSelection
from nanoscope.core.values import DeviceKind

logger = logging.getLogger(__name__)

#: Best first. Not measured — there is no AMD card or Mac here to measure with —
#: so it is a stated convention: a discrete NVIDIA card is what this project's
#: operator actually has, ROCm is the same shape of hardware through a younger
#: stack, and MPS is unified memory on a laptop. Reorder here, nowhere else.
PREFERENCE_ORDER = (DeviceKind.CUDA, DeviceKind.ROCM, DeviceKind.MPS, DeviceKind.CPU)

CPU = Device(kind=DeviceKind.CPU, name="CPU", torch_name="cpu")


class DeviceManager:
    """Probes what is usable, and resolves a preference against it.

    Satisfies `core.ports.DeviceProvider`. Probing is done once and cached: it
    imports torch and queries a driver, which is not something a settings dialog
    should do on every repaint. `refresh()` exists for the case that actually
    changes the answer — an operator plugging in an eGPU, or fixing a driver
    without restarting.
    """

    def __init__(self) -> None:
        self._devices: list[Device] | None = None

    def available(self) -> list[Device]:
        """Every usable device, best first. Never empty: the CPU is always there."""
        if self._devices is None:
            self._devices = self._probe()
        return list(self._devices)

    def refresh(self) -> list[Device]:
        """Probe again, for a machine whose hardware changed under us."""
        self._devices = None
        return self.available()

    def select(self, preferred: DeviceKind | None = None) -> DeviceSelection:
        """The caller's preference, else the best available, else the CPU.

        Args:
            preferred: what the operator chose, or `None` to let the policy
                decide — which is the setting's default and the common case.

        Returns:
            The device to use, and a readable reason when it is not the one that
            was asked for. Never raises: a caller who asked for CUDA on a laptop
            wants their analysis to run, and wants to be told it ran slowly.
        """
        devices = self.available()
        best = devices[0]

        if preferred is None:
            return DeviceSelection(device=best)

        for device in devices:
            if device.kind is preferred:
                return DeviceSelection(device=device, requested=preferred)

        reason = (
            f"{preferred.upper()} was requested but no {preferred.upper()} device is available "
            f"— using {best.name}"
        )
        logger.warning("device fallback: %s", reason)
        return DeviceSelection(device=best, requested=preferred, reason=reason)

    def _probe(self) -> list[Device]:
        """Ask torch what exists, and put the CPU at the end of whatever it says."""
        torch = _import_torch()
        if torch is None:
            logger.info("torch is not installed: inference will run on the CPU")
            return [CPU]

        found = _accelerators(torch)
        found.sort(key=lambda device: PREFERENCE_ORDER.index(device.kind))
        return [*found, CPU]


def _import_torch() -> Any | None:
    """torch, or `None` if it is not installed.

    Imported here rather than at module scope for two reasons: it costs about a
    second and pulls in CUDA libraries, and this module has to be importable in
    CI, which installs no torch on purpose (`pyproject.toml`, the `ci` group).
    """
    try:
        import torch
    except ImportError:
        return None
    return torch


def _accelerators(torch: Any) -> list[Device]:
    """Every non-CPU device this torch can see.

    A ROCm build reports its cards through `torch.cuda` — same API, different
    hardware — so the kind is decided by `torch.version.hip`, not by which
    function answered.
    """
    devices: list[Device] = []

    if torch.cuda.is_available():
        kind = DeviceKind.ROCM if getattr(torch.version, "hip", None) else DeviceKind.CUDA
        for index in range(torch.cuda.device_count()):
            devices.append(
                Device(
                    kind=kind,
                    name=torch.cuda.get_device_name(index),
                    #: `cuda:N` even for ROCm: it is what a ROCm torch expects,
                    #: and the *kind* is what says which hardware it is.
                    torch_name=f"cuda:{index}",
                )
            )

    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        devices.append(Device(kind=DeviceKind.MPS, name="Apple GPU (MPS)", torch_name="mps"))

    return devices
