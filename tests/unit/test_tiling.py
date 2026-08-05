"""The tiled YOLO backend has never tiled (M3-T21, ADR-0021, decision B7).

`_prepare_image` returns exactly one `yolo_size` square, and the crop shape is
the same square, so `get_crops_xy` computes one step per axis: the sliding window
covers the whole image in a single tile. The tiled backend therefore did the
direct backend's work, more slowly, and small particles were never seen at native
resolution — which is the only reason tiling exists.

B7 answered: keep the backend, stop defaulting to it. These tests pin the
arithmetic, so that a future change to `yolo_size` or the crop shape is measured
against a fact rather than against a comment.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from nanoscope.core.entities import PipelineConfig
from nanoscope.infrastructure.models import YoloDetector


def _detector(**kw: object) -> YoloDetector:
    """A detector without touching torch — `__init__` imports nothing heavy."""
    return YoloDetector(**kw)  # type: ignore[arg-type]


def test_the_default_is_the_direct_backend() -> None:
    assert _detector().use_tiling is False
    assert PipelineConfig().yolo_use_tiling is False


def test_the_prepared_image_is_exactly_one_crop() -> None:
    """The measurement behind the decision: input side == crop side == 640."""
    det = _detector()
    assert det._crop_steps(det.yolo_size) == 1


@pytest.mark.parametrize("overlap", [0, 25, 50, 75])
def test_one_crop_whatever_the_overlap(overlap: int) -> None:
    """`int((side - shape) / step) + 1` is 1 for any step when side == shape, so
    tuning the overlap cannot rescue this — the input size is the only lever."""
    det = _detector(overlap_x=overlap)
    assert det._crop_steps(det.yolo_size) == 1


def test_a_large_enough_input_would_actually_tile() -> None:
    """The threshold the ADR quotes: `shape * (2 - overlap/100)` = 1120 px at
    640/25. One pixel below it, the window still fits in a single crop."""
    det = _detector()
    assert det._crop_steps(1120) == 2
    assert det._crop_steps(1119) == 1


def test_asking_for_tiling_anyway_says_what_it_will_do(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Opting in is allowed; being told it changes nothing is the point. Asserted
    against the guard rather than through `_detect_tiled`, which would run real
    inference — outside the gate by PROJECT_RULES §6."""
    det = _detector(use_tiling=True)
    with caplog.at_level(logging.WARNING):
        degenerate = det._warn_if_single_crop(np.zeros((640, 640, 3), dtype=np.uint8))
    assert degenerate
    assert "one 640 px crop" in caplog.text


def test_a_genuinely_tiled_input_is_not_warned_about() -> None:
    """The guard must go quiet the day the input grows, or it becomes noise."""
    det = _detector(use_tiling=True)
    assert det._warn_if_single_crop(np.zeros((1120, 1120, 3), dtype=np.uint8)) is False
