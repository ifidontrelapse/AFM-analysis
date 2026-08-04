"""AFM nanoparticle analysis — legacy import package, being dissolved by M2.

**This file deliberately imports nothing.** It used to re-export `run_pipeline`,
`PipelineConfig`, `PipelineResult`, `Detection`, `LogDetector` and `YoloDetector`,
which made it the cause of all five import cycles the audit found (D-18): Python
runs a package's `__init__` before any submodule, so `import src.types` — the
documented "dependency root" — pulled in the pipeline, the detectors, SAM2 and
matplotlib before it could give you a dataclass.

Nothing in the repository or the notebooks used `from src import X`; every caller
already imports the submodule. Emptying this file broke all five cycles at once
and cost no caller anything (M2-T09).

Import from `nanoscope` for new code. The modules here are re-export shims and go
away in M2-T15.
"""
