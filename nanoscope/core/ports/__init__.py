"""Ports — the interfaces `core` owns and adapters implement.

The dependency rule in one sentence: `core` declares what it needs, and
`infrastructure` provides it. Arrows point inward, so nothing here may import
from `application`, `infrastructure` or `gui`.

## What is here, and what deliberately is not

Two ports live here, and the ones that do not are a decision rather than an
unfinished job. M2-T08 was written to define seven at once — `Detector`,
`Segmenter`, `ImageLoader`, `ProjectRepository`, `TrainingProvider`,
`DeviceProvider`, `LogSink` — and defined one, because the other six had no
implementation, no caller, and no second candidate anywhere in the repository.

An interface written before its first adapter is a guess about a shape, and it
gets rewritten the moment real code has to fit through it — except that by then it
is quoted in a document and looks decided. `Detector` was different: `LogDetector`
and `YoloDetector` both satisfy it *today*, from opposite layers, which is exactly
the situation an abstraction is for.

The rest ship with their first adapter, each with the task that brings one:

| Port | Arrives with | Task |
|---|---|---|
| `ProjectRepository` | ✅ **arrived** with `SqliteProjectRepository` | M4-T03 |
| `ImageLoader` | a loader class, once `application` has a use case that needs one | M2-T10 / M6 |
| `Segmenter` | the first SAM2 wrapper that is a class rather than a function | M4 |
| `DeviceProvider` | ✅ **arrived** with `DeviceManager` | M4-T12 |
| `SettingsStore` | ✅ **arrived** with `JsonSettings` and the repository | M4-T10 |
| `TrainingProvider` | local and remote training | M7 |

This table is the commitment; an empty `Protocol` would only have been the
appearance of one. `ProjectRepository` is the first row it has paid out: the port
landed in the same commit as the adapter, and one layer up `application` needs it
to talk about a project without importing `sqlite3` (M4-T04).
"""

from nanoscope.core.ports.detector import Detector
from nanoscope.core.ports.device import DeviceProvider
from nanoscope.core.ports.project_repository import ProjectRepository
from nanoscope.core.ports.settings import SettingsStore

__all__ = ["Detector", "DeviceProvider", "ProjectRepository", "SettingsStore"]
