"""The characterization golden, run as a test instead of by discipline.

This is the only check standing between M2's module moves and a silent numerical
change (ADR-0008, PROJECT_RULES §4.2). It wraps `capture.py`; it does not
duplicate or reimplement any part of the comparison.

    pytest                    # includes this — ~100 s, the merge gate
    pytest -m "not slow"      # skips it — the inner loop

A drift here means one of two things: the refactor has a bug, or the change was
intentional. Intentional needs an ADR, a test, the regenerated golden in the
same commit, and a quantified delta in `docs/Progress.md`. `--write` on its own
is a rule violation.
"""

from __future__ import annotations

import capture  # tests/characterization is on sys.path — same directory as this file
import pytest

MAX_REPORTED = 80


@pytest.mark.slow
def test_characterization_baseline_is_stable() -> None:
    """Every recorded number must survive the current working tree unchanged."""
    diffs = capture.diff_against_golden()

    report = "\n".join(f"  {d}" for d in diffs[:MAX_REPORTED])
    if len(diffs) > MAX_REPORTED:
        report += f"\n  ... and {len(diffs) - MAX_REPORTED} more"
    assert not diffs, f"CHARACTERIZATION DRIFT: {len(diffs)} difference(s)\n{report}"
