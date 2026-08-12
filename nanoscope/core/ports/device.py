"""Where inference runs, asked without knowing what torch is (M4-T12).

The third port to pay out `core/ports/__init__.py`'s table, and the one
ADR-0004 named when it wrote *"it is defined as a port in `core/ports/device.py`
and implemented in `infrastructure/device/`"*.

The point of the boundary: `application` and `gui` may ask which devices exist
and choose one, and neither imports torch to do it — the import that costs a
second and pulls in CUDA libraries stays behind the adapter.
"""

from __future__ import annotations

from typing import Protocol

from nanoscope.core.entities.device import Device, DeviceSelection
from nanoscope.core.values import DeviceKind


class DeviceProvider(Protocol):
    """The one authority on what hardware is usable (ADR-0004)."""

    def available(self) -> list[Device]:
        """Every device inference could run on, best first.

        Always at least one: the CPU is always there, including on a machine
        with no torch installed at all.
        """
        ...

    def select(self, preferred: DeviceKind | None = None) -> DeviceSelection:
        """Choose one: the caller's preference, else the best available, else CPU.

        Never raises for an unavailable preference. It falls back and says why,
        because a caller asking for CUDA on a laptop wants their analysis to run
        — and wants to be told it ran slowly.
        """
        ...
