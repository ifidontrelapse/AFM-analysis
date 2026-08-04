"""Use cases — the orchestrations the application offers.

Each one sequences `core.science` and the adapters in `infrastructure`, and owns
neither. Both arrived in M2-T15, when `src/` was deleted.

The modules are `pipeline` and `preprocessing`, not `run_pipeline` and
`run_preprocessing`: a module and a function with the same name shadow each other
here, so `import ...use_cases.run_pipeline` would hand back the function and
`monkeypatch.setattr` on it would fail with a confusing AttributeError. Found by a
test, in M2-T15.
"""

from nanoscope.application.use_cases.pipeline import run_pipeline
from nanoscope.application.use_cases.preprocessing import run_preprocessing

__all__ = ["run_pipeline", "run_preprocessing"]
