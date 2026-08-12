"""Library code reports through `logging`, never through `print` (M2-T11, D-23).

Two kinds of check. The first is a grep — crude, and the only thing that stays true
for code nobody thought to test. The second exercises the real call paths and
asserts the records come out where an application can catch them.
"""

from __future__ import annotations

import ast
import logging
import pathlib

import numpy as np
import pytest

from nanoscope.core.science.detection.log import detect_particles
from nanoscope.core.science.preprocessing.substrate import estimate_rough_radius

ROOT = pathlib.Path(__file__).resolve().parents[2]
#: The console entry point is excluded, and it is the only exclusion (M5-T01,
#: ADR-0052 §5). PROJECT_RULES §3 forbids `print` in **library** code, which is
#: what this test enforces and what M2-T11 deleted thirteen calls to satisfy —
#: but a command-line program's stdout is its user interface, not a diagnostic
#: channel, and a CLI that logs instead of printing has no output. Scoped to the
#: one module with a terminal on the other end; `ruff`'s own `T20` carries the
#: same exception, in `pyproject.toml`, for the same reason.
CLI = ROOT / "nanoscope" / "app" / "main.py"

LIBRARY = sorted(path for path in (ROOT / "nanoscope").rglob("*.py") if path != CLI)


def test_the_library_glob_is_not_empty() -> None:
    assert len(LIBRARY) >= 20, LIBRARY


@pytest.mark.parametrize("path", LIBRARY, ids=lambda p: str(p.relative_to(ROOT / "nanoscope")))
def test_no_print_in_library_code(path: pathlib.Path) -> None:
    calls = [
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    assert not calls, (
        f"{path.relative_to(ROOT)} calls print() at line(s) {[c.lineno for c in calls]}. "
        "Library code cannot decide where output goes — use the module logger."
    )


def test_a_flat_image_warns_instead_of_printing(caplog: pytest.LogCaptureFixture) -> None:
    """The substrate fallback path: no objects found, so a default radius is used."""
    flat = np.zeros((32, 32), dtype=np.float32)
    with caplog.at_level(logging.WARNING, logger="nanoscope.core.science.preprocessing.substrate"):
        radius = estimate_rough_radius(flat, pixel_size_nm=1.0, min_size_nm=2)

    assert radius == pytest.approx(2)  # the minimum-size floor, unchanged by M2-T11
    assert "too" in caplog.text and "flat" in caplog.text
    assert caplog.records[0].levelno == logging.WARNING


def test_detection_reports_its_result_at_info(caplog: pytest.LogCaptureFixture) -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(0, 0.1, (64, 64)).astype(np.float32)
    ys, xs = np.mgrid[0:64, 0:64]
    for cy, cx in ((16, 16), (16, 48), (48, 16), (48, 48)):
        z += 5.0 * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / 18.0)

    sizes = {"radii_px": np.array([3.0, 4.0])}
    with caplog.at_level(logging.INFO, logger="nanoscope.core.science.detection.log"):
        blobs = detect_particles(z, 1.0, sizes, 0.3, 0.05, 20.0)

    assert len(blobs) > 0
    assert "found" in caplog.text and "particles" in caplog.text
    # Lazy %-formatting, not an f-string: the message template survives so log
    # aggregators can group records, and the arguments are not rendered when the
    # level is off.
    assert caplog.records[-1].args is not None


def test_nothing_is_emitted_when_the_caller_does_not_ask(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A library must not configure logging; silence is the default (D-23)."""
    flat = np.zeros((32, 32), dtype=np.float32)
    with caplog.at_level(logging.CRITICAL):
        estimate_rough_radius(flat, pixel_size_nm=1.0, min_size_nm=2)
    assert caplog.records == []
