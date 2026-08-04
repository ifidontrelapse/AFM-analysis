"""The `Detector` port is a description of working code, not an aspiration.

Two checks, and the important one is not the assertions — it is that mypy verifies
the *signatures* structurally when `make types` runs. `runtime_checkable` only
proves a method with the right name exists.

`YoloDetector` is instantiated but never run: constructing it touches no model
weights and imports no torch (they are function-local, M1-T08/M2-T07), which is
also what makes this test runnable in CI's torch-free environment.
"""

from __future__ import annotations

from nanoscope.core.ports import Detector
from nanoscope.core.science.detection import LogDetector
from nanoscope.infrastructure.models import YoloDetector


def _accepts_a_detector(detector: Detector) -> Detector:
    """mypy checks the structural match here — including the signature."""
    return detector


def test_both_detectors_satisfy_the_port() -> None:
    # From `core.science` and from `infrastructure.models` — opposite layers,
    # one contract, and neither class imports `nanoscope.core.ports`.
    assert isinstance(_accepts_a_detector(LogDetector()), Detector)
    assert isinstance(_accepts_a_detector(YoloDetector()), Detector)


def test_the_port_rejects_something_without_detect() -> None:
    # A port that accepts anything is decoration. This is the negative case.
    class NotADetector:
        def find(self) -> None: ...

    assert not isinstance(NotADetector(), Detector)


def test_importing_the_domain_pulls_in_no_torch() -> None:
    """The dependency rule, as a fact about `sys.modules` rather than a diagram.

    M2-T09 generalises this into a full import-graph test; until then this is the
    one edge that would hurt most, and it is the reason CI can run without a
    750 MB CUDA wheel.
    """
    import sys

    import nanoscope.core.science  # noqa: F401  — imported for its side effects

    assert "torch" not in sys.modules
    assert "ultralytics" not in sys.modules
