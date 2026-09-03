"""Producing models, which this project could not do until M8.

ADR-0006 put training behind a port and on the other side of a wall from
inference: *"No module under `infrastructure/models/` imports
`infrastructure/training/`, and vice versa."* A test says so
(`tests/unit/test_import_graph.py`), and until this package existed it could
only check one direction.

Heavy by nature — ultralytics and torch — and every import of them is inside the
function that needs one, so constructing a provider costs nothing and CI, which
installs neither, can still import this package.
"""

from nanoscope.infrastructure.training.local import LocalTrainingProvider
from nanoscope.infrastructure.training.remote import RemoteTrainingProvider

__all__ = ["LocalTrainingProvider", "RemoteTrainingProvider"]
