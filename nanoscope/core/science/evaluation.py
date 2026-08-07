"""Scoring a detector against ground truth (M3-T15, ADR-0032).

Until this module existed the project could measure *change* and not *quality*.
The characterization golden catches a number that moved; nothing said whether the
number was any good, so five tasks in this milestone had to write "not claimed"
where they wanted to write "better" — M3-T03, M3-T10, M3-T21, M3-T05 and M3-T14.

What is scored here is detection: did we find the particles that are there, did
we invent any that are not, and how far from the truth did we land. Segmentation
quality (mask IoU) is not, because the phantoms carry centres and radii rather
than masks.

Two rules make the numbers mean what they say:

- **A match is a centre inside the particle.** Scale-free by construction, unlike
  a fixed pixel distance: the phantom set runs from 1.95 to 29.3 nm/px, so one
  pixel threshold would be two different physical tolerances.
- **One detection per particle, assigned optimally.** Ten boxes on one particle
  are one hit and nine false positives. Greedy nearest-first would also give one
  hit, but it can pair the wrong two and inflate the localisation error, so the
  assignment minimises total distance over the admissible pairs
  (`scipy.optimize.linear_sum_assignment`).

**A phantom is not a sample.** Scoring well here licenses "this change improved
detection on the phantom set" and nothing about real scans, which is B6/M3-T16.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from nanoscope.core.entities import Detection
from nanoscope.core.errors import InvalidInputError
from nanoscope.core.validation import ensure_positive


@dataclass(frozen=True)
class MatchResult:
    """Which detection answers which particle.

    Attributes:
        pairs:        `(M, 2)` array of `[detection_index, truth_index]`, one row
                      per matched pair, ordered by truth index.
        unmatched_detections: indices into the detection list — false positives.
        unmatched_truth:      indices into the ground truth — misses.
        distances_px: `(M,)` centre-to-centre distance of each pair, in pixels,
                      in the same order as `pairs`.
    """

    pairs: np.ndarray
    unmatched_detections: np.ndarray
    unmatched_truth: np.ndarray
    distances_px: np.ndarray


@dataclass(frozen=True)
class DetectionMetrics:
    """What a detector scored, with every field named for what it is.

    The `_nm` fields are `None` when the image has no known scale — absent, never
    a pixel count wearing nanometre units (ADR-0019).

    `precision` and `recall` are `None`, not 0.0 or 1.0, when their denominator
    is zero: a detector that reported nothing on an empty image has no precision,
    and inventing one would be the seventh substitute value this milestone has
    deleted.
    """

    n_truth: int
    n_detected: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None
    mean_localisation_error_px: float | None
    median_localisation_error_px: float | None
    mean_localisation_error_nm: float | None
    mean_radius_error_px: float | None
    median_radius_error_px: float | None
    mean_signed_radius_error_px: float | None
    match_factor: float


def _centres_and_radii(detections: Sequence[Detection]) -> tuple[np.ndarray, np.ndarray]:
    """`(N, 2)` centres as `[y, x]` and `(N,)` radii, from the entities.

    `[y, x]`, because that is the project's array convention (PROJECT_RULES §3)
    and the ground truth is stored that way; `Detection` spells the same point
    `x_px`, `y_px`, so the swap happens once, here.
    """
    if not all(isinstance(d, Detection) for d in detections):
        raise InvalidInputError(
            "detections must be a sequence of Detection entities; "
            f"got {[type(d).__name__ for d in detections][:3]}."
        )
    if len(detections) == 0:
        return np.empty((0, 2)), np.empty(0)
    centres = np.array([[d.y_px, d.x_px] for d in detections], dtype=np.float64)
    radii = np.array([d.radius_px for d in detections], dtype=np.float64)
    return centres, radii


def _ensure_truth(centres_yx_px: np.ndarray, radii_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centres = np.asarray(centres_yx_px, dtype=np.float64)
    radii = np.asarray(radii_px, dtype=np.float64)
    if centres.ndim != 2 or (centres.size and centres.shape[1] != 2):
        raise InvalidInputError(
            f"truth_centres_yx_px must have shape (N, 2) — [y, x] — got {centres.shape}."
        )
    if radii.ndim != 1 or len(radii) != len(centres):
        raise InvalidInputError(
            f"truth_radii_px must be one radius per centre: {len(centres)} centres, "
            f"{radii.shape} radii."
        )
    if radii.size and not np.all(radii > 0):
        raise InvalidInputError("truth_radii_px must all be positive.")
    return centres, radii


def match_detections(
    detections: Sequence[Detection],
    truth_centres_yx_px: np.ndarray,
    truth_radii_px: np.ndarray,
    *,
    match_factor: float = 1.0,
) -> MatchResult:
    """Pair detections with the particles they found, one to one.

    A pair is *admissible* when the detection's centre lies within
    `match_factor * radius` of the particle's centre — scale-free, because the
    tolerance is the particle's own size. Among the admissible pairs the
    assignment minimising total distance is chosen, so a detection cannot be
    credited to two particles and a particle cannot be found twice.

    Args:
        detections:          what the detector returned
        truth_centres_yx_px: `(N, 2)` true centres, `[y, x]`
        truth_radii_px:      `(N,)` true radii, in pixels
        match_factor:        multiplier on the true radius; 1.0 means the centre
                             must land inside the particle

    Returns:
        A `MatchResult`. Every detection appears exactly once, in `pairs` or in
        `unmatched_detections`; so does every particle.

    Raises:
        InvalidInputError: if the ground truth is not `(N, 2)` centres with `N`
            positive radii, or `match_factor` is not positive.
    """
    ensure_positive(match_factor, "match_factor")
    centres, _ = _centres_and_radii(detections)
    truth_centres, truth_radii = _ensure_truth(truth_centres_yx_px, truth_radii_px)

    n_det, n_truth = len(centres), len(truth_centres)
    if n_det == 0 or n_truth == 0:
        return MatchResult(
            pairs=np.empty((0, 2), dtype=int),
            unmatched_detections=np.arange(n_det),
            unmatched_truth=np.arange(n_truth),
            distances_px=np.empty(0),
        )

    # (n_det, n_truth) centre-to-centre distances.
    deltas = centres[:, None, :] - truth_centres[None, :, :]
    distances = np.sqrt((deltas**2).sum(axis=2))
    admissible = distances <= match_factor * truth_radii[None, :]

    # An inadmissible pair must never be chosen, so it costs more than any
    # admissible assignment could: `linear_sum_assignment` minimises the total.
    forbidden = distances.max() * (n_det + n_truth) + 1.0
    cost = np.where(admissible, distances, forbidden)
    rows, cols = linear_sum_assignment(cost)

    keep = admissible[rows, cols]
    rows, cols = rows[keep], cols[keep]
    order = np.argsort(cols)
    rows, cols = rows[order], cols[order]

    return MatchResult(
        pairs=np.stack([rows, cols], axis=1),
        unmatched_detections=np.setdiff1d(np.arange(n_det), rows),
        unmatched_truth=np.setdiff1d(np.arange(n_truth), cols),
        distances_px=distances[rows, cols],
    )


def evaluate_detections(
    detections: Sequence[Detection],
    truth_centres_yx_px: np.ndarray,
    truth_radii_px: np.ndarray,
    *,
    match_factor: float = 1.0,
    pixel_size_nm: float | None = None,
) -> DetectionMetrics:
    """Score a detector against ground truth.

    Args:
        detections:          what the detector returned
        truth_centres_yx_px: `(N, 2)` true centres, `[y, x]`
        truth_radii_px:      `(N,)` true radii, in pixels
        match_factor:        see `match_detections`
        pixel_size_nm:       nm per pixel, or `None` when unknown — then the
                             `_nm` fields are `None` rather than a pixel count

    Returns:
        `DetectionMetrics`. Ratios whose denominator is zero are `None`, not a
        substituted 0.0 or 1.0.

    Raises:
        InvalidInputError, InvalidParameterError: as `match_detections`, plus a
            non-positive `pixel_size_nm`.
    """
    ensure_positive(pixel_size_nm, "pixel_size_nm", allow_none=True)
    _, radii = _centres_and_radii(detections)
    _, truth_radii = _ensure_truth(truth_centres_yx_px, truth_radii_px)
    result = match_detections(
        detections, truth_centres_yx_px, truth_radii_px, match_factor=match_factor
    )

    n_truth, n_detected = len(truth_radii), len(radii)
    tp = len(result.pairs)
    fp = n_detected - tp
    fn = n_truth - tp

    precision = tp / n_detected if n_detected else None
    recall = tp / n_truth if n_truth else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    if tp:
        radius_error = radii[result.pairs[:, 0]] - truth_radii[result.pairs[:, 1]]
        mean_px = float(result.distances_px.mean())
        metrics = {
            "mean_localisation_error_px": mean_px,
            "median_localisation_error_px": float(np.median(result.distances_px)),
            "mean_localisation_error_nm": (
                None if pixel_size_nm is None else mean_px * pixel_size_nm
            ),
            # Signed as well as absolute: a detector that reports every radius
            # 20 % small is a different problem from one that scatters, and the
            # mean absolute error alone cannot tell them apart.
            "mean_radius_error_px": float(np.abs(radius_error).mean()),
            "median_radius_error_px": float(np.median(np.abs(radius_error))),
            "mean_signed_radius_error_px": float(radius_error.mean()),
        }
    else:
        metrics = dict.fromkeys(
            (
                "mean_localisation_error_px",
                "median_localisation_error_px",
                "mean_localisation_error_nm",
                "mean_radius_error_px",
                "median_radius_error_px",
                "mean_signed_radius_error_px",
            )
        )

    return DetectionMetrics(
        n_truth=n_truth,
        n_detected=n_detected,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        match_factor=match_factor,
        **metrics,
    )
