"""Use cases — the orchestrations the application offers.

Each one sequences `core.science` and the adapters in `infrastructure`, and owns
neither. Both arrived in M2-T15, when `src/` was deleted.

`projects` joined them in M4-T04 — the first module here written for the
application layer rather than moved into it, and the first that depends on a
port instead of on a concrete adapter.

The modules are `pipeline` and `preprocessing`, not `run_pipeline` and
`run_preprocessing`: a module and a function with the same name shadow each other
here, so `import ...use_cases.run_pipeline` would hand back the function and
`monkeypatch.setattr` on it would fail with a confusing AttributeError. Found by a
test, in M2-T15.
"""

from nanoscope.application.use_cases.pipeline import run_pipeline
from nanoscope.application.use_cases.preprocessing import run_preprocessing
from nanoscope.application.use_cases.projects import import_images, open_project

__all__ = ["import_images", "open_project", "run_pipeline", "run_preprocessing"]
