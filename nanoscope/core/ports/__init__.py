"""Ports — the interfaces `core` owns and adapters implement.

The dependency rule in one sentence: `core` declares what it needs, and
`infrastructure` provides it. Arrows point inward, so nothing here may import
from `application`, `infrastructure` or `gui`.

## What is here, and what deliberately is not

`Detector` is the only port in this package today, and that is a decision rather
than an unfinished job. M2-T08 was written to define seven at once — `Detector`,
`Segmenter`, `ImageLoader`, `ProjectRepository`, `TrainingProvider`,
`DeviceProvider`, `LogSink`. Six of them have no implementation, no caller, and no
second candidate implementation anywhere in the repository.

An interface written before its first adapter is a guess about a shape, and it
gets rewritten the moment real code has to fit through it — except that by then it
is quoted in a document and looks decided. `Detector` is different: `LogDetector`
and `YoloDetector` both satisfy it *today*, from opposite layers, which is exactly
the situation an abstraction is for.

The rest ship with their first adapter, each with the task that brings one:

| Port | Arrives with | Task |
|---|---|---|
| `LogSink` | the structured logger replacing 13 `print` calls | M2-T11 |
| `ImageLoader` | a loader class, once `application` has a use case that needs one | M2-T10 / M6 |
| `Segmenter` | the first SAM2 wrapper that is a class rather than a function | M4 |
| `DeviceProvider` | `DeviceManager` — CPU / CUDA / ROCm / MPS | M4-T12 |
| `ProjectRepository` | the SQLite schema and its repository | M6 |
| `TrainingProvider` | local and remote training | M7 |

This table is the commitment; an empty `Protocol` would only have been the
appearance of one.
"""

from nanoscope.core.ports.detector import Detector

__all__ = ["Detector"]
