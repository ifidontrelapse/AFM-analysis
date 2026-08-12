"""The `Detector` port is a description of working code, not an aspiration.

Two checks, and the important one is not the assertions — it is that mypy verifies
the *signatures* structurally when `make types` runs. `runtime_checkable` only
proves a method with the right name exists.

`YoloDetector` is instantiated but never run: constructing it touches no model
weights and imports no torch (they are function-local, M1-T08/M2-T07), which is
also what makes this test runnable in CI's torch-free environment.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

from nanoscope.core.ports import Detector
from nanoscope.core.science.detection import LogDetector
from nanoscope.infrastructure.models import YoloDetector


def _accepts_a_detector(detector: Detector) -> Detector:
    """mypy checks the structural match here — including the signature."""
    return detector


ROOT = pathlib.Path(__file__).resolve().parents[2]


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

    **In a subprocess**, since M4-T15. It used to read this process's
    `sys.modules`, which made it a test any *earlier* test could break — and one
    finally did: the end-to-end walkthrough probes the real hardware, which
    imports torch on purpose, and this assertion then failed for a reason that
    had nothing to do with the domain. An in-process `sys.modules` assertion is
    a claim about the whole suite, not about the import it names.

    M2-T09's import-graph test already runs its weight check this way and says
    so in its own docstring; this is the same repair, one file over.
    """
    code = "import sys; import nanoscope.core.science; print(' '.join(sorted(sys.modules)))"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT
    )
    loaded = set(out.stdout.split())

    assert "torch" not in loaded
    assert "ultralytics" not in loaded
