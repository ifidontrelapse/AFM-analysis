"""Precision, recall and localisation against ground truth (M3-T15, ADR-0032).

The measurement five tasks in this milestone had to do without. These tests are
about the *scoring*, not about any detector: a metric that is wrong is worse than
no metric, because the number looks like evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.core.entities import Detection
from nanoscope.core.errors import InvalidInputError, InvalidParameterError
from nanoscope.core.science.evaluation import evaluate_detections, match_detections

TRUTH_YX = np.array([[10.0, 10.0], [10.0, 40.0], [40.0, 10.0], [40.0, 40.0]])
TRUTH_R = np.array([5.0, 5.0, 5.0, 5.0])


def _det(y: float, x: float, r: float = 5.0) -> Detection:
    return Detection(x_px=x, y_px=y, radius_px=r, radius_nm=None)


def _perfect() -> list[Detection]:
    return [_det(y, x, r) for (y, x), r in zip(TRUTH_YX, TRUTH_R, strict=True)]


class TestThePerfectCase:
    def test_everything_found_exactly(self) -> None:
        m = evaluate_detections(_perfect(), TRUTH_YX, TRUTH_R)

        assert (m.true_positives, m.false_positives, m.false_negatives) == (4, 0, 0)
        assert (m.precision, m.recall, m.f1) == (1.0, 1.0, 1.0)
        assert m.mean_localisation_error_px == 0.0
        assert m.mean_radius_error_px == 0.0

    def test_the_nanometre_error_is_the_pixel_error_scaled(self) -> None:
        dets = [_det(y + 1.0, x) for y, x in TRUTH_YX]

        m = evaluate_detections(dets, TRUTH_YX, TRUTH_R, pixel_size_nm=2.5)

        assert m.mean_localisation_error_px == pytest.approx(1.0)
        assert m.mean_localisation_error_nm == pytest.approx(2.5)

    def test_without_a_scale_the_nanometre_error_is_absent(self) -> None:
        """ADR-0019 again: unknown is `None`, never the pixel value wearing
        nanometre units."""
        m = evaluate_detections(_perfect(), TRUTH_YX, TRUTH_R)

        assert m.mean_localisation_error_px == 0.0
        assert m.mean_localisation_error_nm is None


class TestWhatCountsAsAMatch:
    def test_a_centre_inside_the_particle_matches(self) -> None:
        m = evaluate_detections([_det(10.0, 14.0)], TRUTH_YX, TRUTH_R)

        assert m.true_positives == 1

    def test_a_centre_outside_it_does_not(self) -> None:
        m = evaluate_detections([_det(10.0, 16.0)], TRUTH_YX, TRUTH_R)

        assert (m.true_positives, m.false_positives) == (0, 1)

    def test_the_tolerance_is_the_particles_own_radius_not_a_pixel_count(self) -> None:
        """The reason a fixed pixel threshold is wrong: the same offset is a hit
        on a large particle and a miss on a small one, and the phantom set runs
        from 1.95 to 29.3 nm/px."""
        truth = np.array([[10.0, 10.0], [40.0, 40.0]])
        radii = np.array([10.0, 2.0])
        dets = [_det(10.0, 17.0), _det(40.0, 47.0)]

        result = match_detections(dets, truth, radii)

        assert list(result.pairs[:, 1]) == [0]
        assert list(result.unmatched_detections) == [1]

    def test_a_looser_factor_admits_more(self) -> None:
        assert evaluate_detections([_det(10.0, 16.0)], TRUTH_YX, TRUTH_R).true_positives == 0
        assert (
            evaluate_detections(
                [_det(10.0, 16.0)], TRUTH_YX, TRUTH_R, match_factor=2.0
            ).true_positives
            == 1
        )


class TestOneDetectionPerParticle:
    def test_ten_boxes_on_one_particle_are_one_hit_and_nine_false_positives(self) -> None:
        """The property that makes precision mean anything. A detector that
        fires repeatedly on the same particle must be charged for it."""
        dets = [_det(10.0 + i * 0.2, 10.0) for i in range(10)]

        m = evaluate_detections(dets, TRUTH_YX, TRUTH_R)

        assert (m.true_positives, m.false_positives, m.false_negatives) == (1, 9, 3)
        assert m.precision == pytest.approx(0.1)
        assert m.recall == pytest.approx(0.25)

    def test_a_particle_is_not_found_twice(self) -> None:
        result = match_detections([_det(10.0, 10.0), _det(10.0, 11.0)], TRUTH_YX, TRUTH_R)

        assert len(result.pairs) == 1
        assert len(result.unmatched_detections) == 1

    def test_the_assignment_is_optimal_and_not_greedy(self) -> None:
        """A case where greedy nearest-first picks the worse pairing.

        Particles at x = 0 and x = 3, detections at x = 1 and x = -2, all four
        pairs admissible:

            d(A, p0) = 1   d(A, p1) = 2
            d(B, p0) = 2   d(B, p1) = 5

        Greedy takes the globally smallest distance first — A to p0 at 1.0 — and
        is then left with B to p1 at 5.0, for a total of **6.0**. The optimal
        assignment is A to p1 and B to p0: **4.0**. Both score two true
        positives, so only the localisation error can tell them apart, which is
        why it is computed from the assignment and not from the nearest
        neighbour of each detection.
        """
        truth = np.array([[0.0, 0.0], [0.0, 3.0]])
        radii = np.array([6.0, 6.0])
        dets = [_det(0.0, 1.0), _det(0.0, -2.0)]  # A, B

        result = match_detections(dets, truth, radii)
        total = float(result.distances_px.sum())

        assert len(result.pairs) == 2
        assert total == pytest.approx(4.0)
        assert total < 6.0  # what nearest-first would have cost
        # A (index 0) is paired with particle 1, B (index 1) with particle 0.
        assert sorted(map(tuple, result.pairs)) == [(0, 1), (1, 0)]


class TestTheEmptyCases:
    def test_no_detections_on_a_populated_image(self) -> None:
        m = evaluate_detections([], TRUTH_YX, TRUTH_R)

        assert (m.true_positives, m.false_negatives) == (0, 4)
        assert m.recall == 0.0
        assert m.precision is None  # nothing was reported, so nothing was right or wrong
        assert m.f1 is None

    def test_detections_on_an_empty_image(self) -> None:
        m = evaluate_detections([_det(1.0, 1.0)], np.empty((0, 2)), np.empty(0))

        assert (m.true_positives, m.false_positives) == (0, 1)
        assert m.precision == 0.0
        assert m.recall is None

    def test_nothing_at_all(self) -> None:
        """A detector that reported nothing on an image with nothing in it has
        no precision and no recall. `1.0` would be a substitute value — the
        seventh this milestone would have had to delete."""
        m = evaluate_detections([], np.empty((0, 2)), np.empty(0))

        assert (m.true_positives, m.false_positives, m.false_negatives) == (0, 0, 0)
        assert m.precision is None
        assert m.recall is None
        assert m.f1 is None
        assert m.mean_localisation_error_px is None


class TestTheRadiusError:
    def test_a_systematic_underestimate_shows_in_the_sign(self) -> None:
        """The reason the signed error is reported next to the absolute one: a
        detector that reports every radius 1 px small is a calibration problem,
        and one that scatters by 1 px is a noise problem. The mean absolute
        error is the same number for both."""
        biased = [_det(y, x, r=4.0) for y, x in TRUTH_YX]
        scattered = [
            _det(y, x, r=5.0 + s) for (y, x), s in zip(TRUTH_YX, [1, -1, 1, -1], strict=True)
        ]

        m_biased = evaluate_detections(biased, TRUTH_YX, TRUTH_R)
        m_scattered = evaluate_detections(scattered, TRUTH_YX, TRUTH_R)

        assert m_biased.mean_radius_error_px == m_scattered.mean_radius_error_px == 1.0
        assert m_biased.mean_signed_radius_error_px == -1.0
        assert m_scattered.mean_signed_radius_error_px == 0.0

    def test_only_matched_pairs_contribute(self) -> None:
        """A false positive has no true radius to be wrong about."""
        dets = [*_perfect(), _det(80.0, 80.0, r=99.0)]

        m = evaluate_detections(dets, TRUTH_YX, TRUTH_R)

        assert m.false_positives == 1
        assert m.mean_radius_error_px == 0.0


class TestTheInputsAreChecked:
    def test_ground_truth_must_be_pairs_of_coordinates(self) -> None:
        with pytest.raises(InvalidInputError, match=r"\(N, 2\)"):
            evaluate_detections(_perfect(), np.zeros((4, 3)), TRUTH_R)

    def test_one_radius_per_centre(self) -> None:
        with pytest.raises(InvalidInputError, match="one radius per centre"):
            evaluate_detections(_perfect(), TRUTH_YX, np.array([5.0, 5.0]))

    def test_radii_must_be_positive(self) -> None:
        with pytest.raises(InvalidInputError, match="positive"):
            evaluate_detections(_perfect(), TRUTH_YX, np.array([5.0, 0.0, 5.0, 5.0]))

    def test_the_match_factor_must_be_positive(self) -> None:
        with pytest.raises(InvalidParameterError, match="match_factor"):
            evaluate_detections(_perfect(), TRUTH_YX, TRUTH_R, match_factor=0.0)

    def test_detections_must_be_entities(self) -> None:
        """Tuples would work by accident today and break the moment the entity
        gains a field. The taxonomy from ADR-0030 says so out loud."""
        with pytest.raises(InvalidInputError, match="Detection entities"):
            evaluate_detections([(10.0, 10.0, 5.0)], TRUTH_YX, TRUTH_R)


class TestAgainstARealDetector:
    def test_the_log_detector_finds_the_particles_in_a_synthetic_scene(self) -> None:
        """End to end on the smallest honest scene: the harness is only worth
        having if it can be pointed at a detector and produce a number."""
        from nanoscope.core.science.detection import LogDetector

        size = 96
        ys, xs = np.mgrid[0:size, 0:size]
        z = np.zeros((size, size), dtype=np.float32)
        centres = np.array([[24.0, 24.0], [24.0, 72.0], [72.0, 24.0], [72.0, 72.0]])
        radii = np.full(4, 6.0)
        for cy, cx in centres:
            z += 10.0 * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * 4.0**2))

        detections = LogDetector(threshold=0.05).detect(z, 1.0, sizes={"radii_px": radii})
        m = evaluate_detections(detections, centres, radii, pixel_size_nm=1.0)

        assert m.recall == 1.0
        assert m.mean_localisation_error_px < 1.0
