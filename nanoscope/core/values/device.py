"""Which compute device inference runs on.

Named in `core` rather than `infrastructure` so a port can mention it without the
domain importing torch. The `DeviceManager` that resolves one arrives in M4-T12.
"""

from __future__ import annotations

from enum import StrEnum


class DeviceKind(StrEnum):
    """The backends the project targets. `CPU` is the only one always available."""

    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"
