"""A height that is not a number is not a measurement (B-059, ADR-0033).

`measure_all_baseline` drops a particle whose height is not positive — artefacts,
by the guard's own comment. It was written `if metrics["height_nm"] <= 0`, and
**`nan <= 0` is `False`**, so the one value most obviously an artefact was the
one value that survived.

ADR-0018 had already ruled on this comparison, in this milestone, for this
reason. These tests pin the third site it applies to, and the warning that keeps
the empty table from reading as "no particles here".
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from nanoscope.core.science.measurement import BASELINE_COLUMNS, measure_all_baseline

BLOBS = np.array([[32.0, 32.0, 4.0, 8.0], [20.0, 20.0, 4.0, 8.0]])


def _constant(size: int = 64, value: float = 3.0) -> np.ndarray:
    """A map with one value. Otsu returns that value, so `z_above < thresh` is
    empty — the only route to a `nan` baseline there is."""
    return np.full((size, size), value, dtype=np.float32)


def _two_particles(size: int = 64) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size]
    z = np.zeros((size, size), dtype=np.float32)
    for cy, cx in ((32, 32), (20, 20)):
        z += 10.0 * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * 4.0**2))
    return z


class TestTheReproduction:
    def test_a_nan_height_does_not_reach_the_table(self) -> None:
        """The defect: two rows, both `NaN`, in a table of measurements."""
        z = _constant()

        df = measure_all_baseline(z, z, BLOBS)

        assert not df["height_nm"].isna().any()
        assert df.empty

    def test_the_table_still_has_its_schema(self) -> None:
        """Dropping the rows must not undo ADR-0027: an empty result is still a
        table a consumer can read by name."""
        z = _constant()

        df = measure_all_baseline(z, z, BLOBS)

        assert list(df.columns) == list(BASELINE_COLUMNS)

    def test_the_comparison_is_the_one_that_catches_nan(self) -> None:
        """`nan <= 0` is False and `not nan > 0` is True. That is the whole
        defect, and it is worth one assertion of its own, because the next
        person to write this guard will reach for `<=` again."""
        assert (np.float64("nan") <= 0) is np.False_
        assert not np.float64("nan") > 0


class TestItSaysWhy:
    def test_an_empty_substrate_is_warned_about(self, caplog: pytest.LogCaptureFixture) -> None:
        """The other half. Without this the fix turns two `NaN` rows into zero
        rows, which reads exactly like "there was nothing here" — and that
        sentence is how the defect stayed invisible for a milestone."""
        z = _constant()

        with caplog.at_level(logging.WARNING):
            measure_all_baseline(z, z, BLOBS)

        assert "substrate mask is empty" in caplog.text
        assert "no global baseline" in caplog.text

    def test_an_ordinary_map_says_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        """A warning that fires on the working path is a warning nobody reads."""
        z = _two_particles()

        with caplog.at_level(logging.WARNING):
            measure_all_baseline(z, z, BLOBS)

        assert "substrate mask is empty" not in caplog.text


class TestNothingElseMoved:
    def test_real_particles_are_still_measured(self) -> None:
        z = _two_particles()

        df = measure_all_baseline(z, z, BLOBS)

        assert len(df) == 2
        assert (df["height_nm"] > 0).all()

    def test_a_negative_height_is_still_dropped(self) -> None:
        """The guard's original job, unchanged: `not h > 0` and `h <= 0` agree on
        every number, and differ only on `nan`."""
        z = _two_particles()
        inverted = -z

        df = measure_all_baseline(inverted, z, BLOBS)

        assert df.empty

    def test_a_zero_height_is_still_dropped(self) -> None:
        """The boundary the two spellings share."""
        z = _two_particles()

        df = measure_all_baseline(np.zeros_like(z), z, BLOBS)

        assert df.empty


class TestTheEmptySubstrateIsAllOrNothing:
    def test_no_particle_can_be_measured_when_the_substrate_is_empty(self) -> None:
        """Found while writing these tests, and worth pinning: the "partial
        success" case does not exist on this route.

        `get_clean_ring` intersects the ring with the substrate mask, so an empty
        substrate leaves *every* particle without a ring, every particle falls
        back to the baseline that is `nan`, and the whole table goes. There is
        no scan where some rows survive a `nan` global baseline — which is
        exactly why the warning has to name the substrate rather than the rows.
        """
        z = _constant()
        many = np.array([[float(y), float(x), 4.0, 8.0] for y in (16, 32, 48) for x in (16, 32)])

        df = measure_all_baseline(z, z, many)

        assert df.empty

    def test_and_when_the_substrate_exists_every_baseline_is_a_number(self) -> None:
        z = _two_particles()

        df = measure_all_baseline(z, z, BLOBS)

        assert df["baseline_nm"].notna().all()
        assert df["height_nm"].notna().all()
