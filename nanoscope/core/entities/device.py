"""What the application knows about where inference can run (M4-T12).

In `core` so a use case can say "run this on the selected device" without
importing torch — which is why `DeviceKind` was put here in M2-T02 and left
unadopted until there was a manager to resolve one.
"""

from __future__ import annotations

from dataclasses import dataclass

from nanoscope.core.values import DeviceKind


@dataclass(frozen=True)
class Device:
    """One place inference could run.

    `torch_name` is what a framework is handed — `"cuda:0"`, `"mps"`, `"cpu"`.
    It is the only string in this project allowed to look like a torch device,
    and it is produced by the manager rather than written anywhere else
    (PROJECT_RULES §2.6).
    """

    kind: DeviceKind
    #: What to show a person: "NVIDIA GeForce RTX 4090", "Apple M2", "CPU".
    name: str
    torch_name: str

    def __str__(self) -> str:
        return f"{self.name} ({self.kind})"


@dataclass(frozen=True)
class DeviceSelection:
    """The device that was chosen, and — when it was not the one asked for — why.

    The `reason` is the half of ADR-0004 that is easy to skip: *"it reports
    capability and the reason for a fallback in language a user can read"*. A
    fallback nobody is told about is a silent forty-fold slowdown that looks
    like the application being slow.
    """

    device: Device
    #: What the caller asked for, or `None` if they let the policy decide.
    requested: DeviceKind | None = None
    #: Empty when the request was honoured. A sentence, not a code, when it was
    #: not: "CUDA was requested but no CUDA device is available — using CPU".
    reason: str = ""

    @property
    def is_fallback(self) -> bool:
        return bool(self.reason)
