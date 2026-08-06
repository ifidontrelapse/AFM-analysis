"""A detection carries the score its detector gave it, or none at all (D-09, ADR-0028).

`_boxes_to_detections` never assigned `confidence`, so every YOLO detection took
the dataclass default of `1.0` — including a box that had only just cleared
`cfg.yolo_conf`. The model scores every box and the threshold *filters* on those
scores; only the reporting threw them away.

`1.0` was a substitute value, and M3 has spent four ADRs deleting substitute
values. So the default is now `None`, which is also the honest answer for the LoG
detector: its blob response is not a probability.

Inference is outside the gate (PROJECT_RULES §6), so these tests exercise the two
conversion seams — both `staticmethod`s, neither needing weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.core.entities import Detection
from nanoscope.core.science.detection import BaseDetector

BOXES = np.array([[10.0, 10.0, 30.0, 34.0], [100.0, 100.0, 140.0, 138.0]])


def _yolo():
    from nanoscope.infrastructure.models import YoloDetector

    return YoloDetector


class TestTheScoreReachesTheEntity:
    def test_each_box_carries_its_own_score(self) -> None:
        """The defect: both of these used to read 1.0."""
        dets = _yolo()._boxes_to_detections(BOXES, 2.0, np.array([0.91, 0.53]))

        assert [d.confidence for d in dets] == pytest.approx([0.91, 0.53])

    def test_the_scores_stay_with_their_own_boxes(self) -> None:
        """Order matters more than presence: a score attached to the wrong box is
        worse than no score, because it reads as a measurement of that box."""
        dets = _yolo()._boxes_to_detections(BOXES, 2.0, np.array([0.51, 0.99]))

        smaller, larger = sorted(dets, key=lambda d: d.radius_px)
        assert smaller.confidence == pytest.approx(0.51)  # the 20x24 box
        assert larger.confidence == pytest.approx(0.99)  # the 40x38 box

    def test_a_length_mismatch_is_an_error_not_a_silent_truncation(self) -> None:
        """`zip` would drop the tail and return a shorter, plausible-looking
        list. PROJECT_RULES §3: the error names what it got."""
        with pytest.raises(ValueError, match="1 confidences for 2 boxes"):
            _yolo()._boxes_to_detections(BOXES, 2.0, np.array([0.9]))

    def test_no_scores_given_means_no_scores_reported(self) -> None:
        """The conversion does not invent one when a caller has none."""
        dets = _yolo()._boxes_to_detections(BOXES, 2.0)

        assert all(d.confidence is None for d in dets)


class TestTheDefaultIsAbsence:
    def test_a_bare_detection_has_no_confidence(self) -> None:
        """It used to claim 1.0 — certainty nothing had computed."""
        det = Detection(x_px=1.0, y_px=2.0, radius_px=3.0, radius_nm=6.0)

        assert det.confidence is None

    def test_log_detections_report_no_score(self) -> None:
        """The LoG path has no score to give. Its response is not a probability,
        and inventing one from it would be a scientific claim (ADR-0028)."""
        blobs = np.array([[16.0, 16.0, 3.0, 8.4], [48.0, 48.0, 4.0, 11.2]])

        dets = BaseDetector._blobs_to_detections(blobs)

        assert dets  # the conversion still works
        assert all(d.confidence is None for d in dets)
        assert all(d.radius_px > 0 for d in dets)

    def test_a_score_of_zero_survives_being_reported(self) -> None:
        """`0.0` is falsy, so any `confidence or default` phrasing would erase
        the least confident detection there is — the `or` family of bugs this
        milestone has already fixed twice (ADR-0025)."""
        dets = _yolo()._boxes_to_detections(BOXES[:1], None, np.array([0.0]))

        assert dets[0].confidence == 0.0
        assert dets[0].confidence is not None
