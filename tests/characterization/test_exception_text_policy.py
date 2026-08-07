"""What the golden promises about exception text (B-058, ADR-0022).

`capture.py` records the exception an input provokes, and until now it compared
the whole message. Most of those sentences are not ours: `too many values to
unpack (expected 2)` is CPython's, and CPython 3.14 reworded it, which the first
real CI run reported as characterization drift with no scientific change behind
it (M1-T08).

The policy: the exception **type** and the function it came out of are always
compared; the **message** is compared only when this project wrote it. These
tests pin both halves — the discriminator, and the fact that `compare` really
does skip an `_unchecked` key.
"""

from __future__ import annotations

import traceback

import capture
import pytest

from nanoscope.core.science.measurement import measure_height
from nanoscope.core.science.preprocessing import estimate_radius_otsu


def _last_frame(fn, *args) -> traceback.FrameSummary:
    with pytest.raises(Exception) as exc_info:
        fn(*args)
    return traceback.extract_tb(exc_info.value.__traceback__)[-1]


class TestWhoWroteTheMessage:
    def test_our_own_raise_is_ours(self) -> None:
        """`estimate_radius_otsu`'s message names the parameter and its value —
        PROJECT_RULES §3 requires that, so the golden must hold it exactly."""
        import numpy as np

        z = np.zeros((32, 32), dtype=np.float32)
        z[8:24, 8:24] = 5.0
        frame = _last_frame(estimate_radius_otsu, z, 1.0, 500)
        assert capture._we_wrote_this_message(frame)

    def test_an_interpreter_message_in_our_file_is_not_ours(self) -> None:
        """The M1-T08 case: a line in our own module raises a sentence NumPy or
        CPython composed, and a new version may reword it.

        The example moved in M3-T13 (ADR-0030). It used to be `flatten_plane`
        on a 1-D array — `h, w = z.shape`, CPython's "not enough values to
        unpack" — and that input is now refused by name at the entry.
        `measure_height` with masks of two different shapes reaches the same
        class of message, from `ring & substrate_mask`, and is not a rejection
        this project has written a rule for."""
        import numpy as np

        frame = _last_frame(
            measure_height,
            np.zeros((8, 8), dtype=np.float32),
            np.ones((4, 4), dtype=bool),
            np.ones((8, 8), dtype=bool),
            0.0,
        )
        assert "nanoscope" in frame.filename
        assert "raise " not in (frame.line or "")
        assert not capture._we_wrote_this_message(frame)

    def test_a_library_raise_is_not_ours(self) -> None:
        """A library's own explicit `raise`, so the filename check is not
        redundant with the `raise` check — both signals are needed.

        `scipy.linalg.lstsq` is the one that mattered historically: this is the
        exact error `flatten_plane` used to surface for a map containing NaN,
        before ADR-0030 made that rejection ours and put the parameter's name in
        it. The frame is still the right test subject; nothing in `nanoscope`
        reaches it any more."""
        import numpy as np
        from scipy.linalg import lstsq

        a = np.array([[1.0, 1.0], [1.0, np.nan]])
        frame = _last_frame(lstsq, a, np.array([1.0, 2.0]))
        assert "raise " in (frame.line or "")
        assert "nanoscope" not in frame.filename
        assert not capture._we_wrote_this_message(frame)


class TestTheComparatorHonoursIt:
    def test_a_reworded_foreign_message_is_not_drift(self) -> None:
        old = {"ok": False, "error_type": "ValueError", "error_message_unchecked": "old wording"}
        new = {"ok": False, "error_type": "ValueError", "error_message_unchecked": "NEW wording"}
        diffs: list[str] = []
        capture.compare(new, old, "x", diffs)
        assert diffs == []

    def test_a_reworded_message_of_ours_is_drift(self) -> None:
        old = {"ok": False, "error_type": "ValueError", "error_message": "min_size_nm=5 nm"}
        new = {"ok": False, "error_type": "ValueError", "error_message": "min_size_nm=6 nm"}
        diffs: list[str] = []
        capture.compare(new, old, "x", diffs)
        assert len(diffs) == 1

    def test_the_type_is_still_compared_either_way(self) -> None:
        """Loosening the wording must not loosen what actually failed."""
        old = {"ok": False, "error_type": "ValueError", "error_message_unchecked": "same"}
        new = {"ok": False, "error_type": "TypeError", "error_message_unchecked": "same"}
        diffs: list[str] = []
        capture.compare(new, old, "x", diffs)
        assert len(diffs) == 1
        assert "error_type" in diffs[0]
