"""
Shared dataclasses for the AFM nanoparticle analysis pipeline.

**Shim.** The definitions moved to `nanoscope.core.entities` in M2-T02; this module
re-exports them and defines nothing. That is deliberate: `src/pipeline.py`,
`src/afm_io.py`, `src/detection/base.py`, `src/visualization.py` and the
characterization harness still import from here, and a second copy of `Detection`
would make `isinstance` fail across the boundary while both packages exist.

Deleted in M2-T15, once nothing imports `src`.
"""

from __future__ import annotations

from nanoscope.core.entities import (
    AFMRawData,
    Detection,
    MicroscopyData,
    PipelineConfig,
    PipelineResult,
    PreprocessingResult,
)

__all__ = [
    "AFMRawData",
    "Detection",
    "MicroscopyData",
    "PipelineConfig",
    "PipelineResult",
    "PreprocessingResult",
]
