"""Value objects — no identity, compared by value, safe to pass anywhere.

Added in M2-T02 and **deliberately not yet adopted**: `PipelineResult.modality`
is still `str`, `pixel_size_nm` is still a bare `float`. Adoption changes what
`dataclasses.asdict` produces and what the characterization golden records, so each
one moves with the task that has a consumer for it —

- `Modality` → M2-T10, the owned capability matrix
- `Polarity` → M3-T10, TEM detection (audit D-12, decision B3)
- `PixelScale` → M2-T03…T07, as the science modules move
- `DeviceKind` → M4-T12, the `DeviceManager`

They are unused on purpose. **M2-T13 (retire dead code) must not delete them.**
"""

from nanoscope.core.values.device import DeviceKind
from nanoscope.core.values.modality import Modality, Polarity
from nanoscope.core.values.scale import PixelScale

__all__ = [
    "DeviceKind",
    "Modality",
    "PixelScale",
    "Polarity",
]
