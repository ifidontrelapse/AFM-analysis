"""Where inference runs — probing it, and choosing (M4-T12, ADR-0004/ADR-0049).

The only place in this project allowed to ask torch about hardware. Everything
above it speaks `DeviceKind` and `Device`, which live in `core` and know nothing
about frameworks.
"""

from nanoscope.infrastructure.device.manager import DeviceManager

__all__ = ["DeviceManager"]
