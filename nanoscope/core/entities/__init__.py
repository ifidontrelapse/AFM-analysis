"""Domain records — the data the pipeline reads, produces and hands on.

The public surface of the layer: import from `nanoscope.core.entities`, not from
the modules underneath, so a later split does not break callers.

Everything here arrived in M2-T02 as a verbatim move of `src/types.py`, which was
the dependency root of the old package. `src/types.py` now re-exports these very
objects, so there is exactly one `Detection` class in the process — two would make
`isinstance` lie across the boundary while both packages exist.
"""

from nanoscope.core.entities.detection import Detection
from nanoscope.core.entities.device import Device, DeviceSelection
from nanoscope.core.entities.image import AFMRawData, MicroscopyData, PreprocessingResult
from nanoscope.core.entities.model import ModelDescriptor, ModelFramework, ModelTask
from nanoscope.core.entities.pipeline import PipelineConfig, PipelineResult
from nanoscope.core.entities.project import (
    AnalysisRun,
    Annotation,
    AnnotationSource,
    ImageRecord,
    ImportFailure,
    ImportReport,
    IntegrityReport,
    OpenedProject,
)

__all__ = [
    "AFMRawData",
    "AnalysisRun",
    "Annotation",
    "AnnotationSource",
    "Detection",
    "Device",
    "DeviceSelection",
    "ImageRecord",
    "ImportFailure",
    "ImportReport",
    "IntegrityReport",
    "MicroscopyData",
    "ModelDescriptor",
    "ModelFramework",
    "ModelTask",
    "OpenedProject",
    "PipelineConfig",
    "PipelineResult",
    "PreprocessingResult",
]
