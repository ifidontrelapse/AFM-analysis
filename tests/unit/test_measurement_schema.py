"""An empty measurement table still has its columns (D-08, ADR-0027).

`measure_all_baseline` drops a particle whose mask runs past the image edge and
one whose height comes out non-positive. Both are ordinary outcomes, and when
they take the last row `pd.DataFrame([])` produced a table with **zero columns**,
so every consumer reading by name got a `KeyError` rather than an empty column.

The pair of tests that matters here is the empty one and the populated one: a
declared schema is only worth anything while it still describes what the
populated path emits.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.core.science.measurement import (
    BASELINE_COLUMNS,
    empty_baseline_table,
    measure_all_baseline,
)


def _one_particle(size: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A single Gaussian bump, and the blob array a detector would hand over."""
    ys, xs = np.mgrid[0:size, 0:size]
    z = 10.0 * np.exp(-((ys - 32) ** 2 + (xs - 32) ** 2) / (2 * 5.0**2))
    blobs = np.array([[32.0, 32.0, 5.0, 10.0]])  # y, x, sigma, radius_nm
    return z.astype(np.float32), z.astype(np.float32), blobs


class TestTheEmptyTable:
    def test_no_blobs_still_yields_every_column(self) -> None:
        """The reproduction: `pd.DataFrame([])` has no columns at all."""
        z, z_above, _ = _one_particle()

        df = measure_all_baseline(z, z_above, np.empty((0, 4)))

        assert df.empty
        assert list(df.columns) == list(BASELINE_COLUMNS)

    def test_a_consumer_can_read_a_column_by_name(self) -> None:
        """D-08 as the audit reproduced it — `KeyError: 'height_nm'`, from a
        pipeline result that simply found nothing."""
        z, z_above, _ = _one_particle()

        df = measure_all_baseline(z, z_above, np.empty((0, 4)))

        assert len(df["height_nm"]) == 0  # not a KeyError
        assert df["height_nm"].sum() == 0.0

    def test_the_dtypes_are_part_of_the_promise(self) -> None:
        """An empty `object` column and an empty `float64` column answer
        `.mean()` differently, so the schema declares both."""
        df = empty_baseline_table()

        assert {name: str(df[name].dtype) for name in df.columns} == BASELINE_COLUMNS

    def test_every_particle_being_rejected_reads_the_same_as_none_detected(self) -> None:
        """The path the audit named: blobs *were* found and every one of them was
        dropped — here by the edge rule, with both centres off the image, so the
        mask has fewer than 4 pixels. The result must be the empty table, not a
        zero-column one; the caller cannot tell the two rejections apart and does
        not need to."""
        z, z_above, _ = _one_particle()
        blobs = np.array([[-100.0, -100.0, 2.0, 4.0], [200.0, 200.0, 2.0, 4.0]])

        df = measure_all_baseline(z, z_above, blobs)

        assert df.empty
        assert list(df.columns) == list(BASELINE_COLUMNS)


class TestThePopulatedTable:
    def test_it_emits_exactly_the_declared_schema(self) -> None:
        """The guard that keeps the declaration honest. If a future column is
        added to the row dict and not to `BASELINE_COLUMNS`, the empty table
        silently stops matching the populated one — and nothing else would
        notice, because the golden's empty case has no columns to compare."""
        z, z_above, blobs = _one_particle()

        df = measure_all_baseline(z, z_above, blobs)

        assert not df.empty
        assert list(df.columns) == list(BASELINE_COLUMNS)

    def test_the_dtypes_match_the_declaration_too(self) -> None:
        """pandas infers these from the values; the declaration has to agree
        with what it infers, or the empty and populated tables differ in a way
        `df.dtypes` shows and `df.columns` does not."""
        z, z_above, blobs = _one_particle()

        df = measure_all_baseline(z, z_above, blobs)

        assert {name: str(df[name].dtype) for name in df.columns} == BASELINE_COLUMNS

    def test_concatenating_an_empty_and_a_populated_table_changes_nothing(self) -> None:
        """The practical consequence for a caller batching scans: an empty
        result contributes no rows and no columns of its own."""
        pd = pytest.importorskip("pandas")
        z, z_above, blobs = _one_particle()
        populated = measure_all_baseline(z, z_above, blobs)

        combined = pd.concat([empty_baseline_table(), populated], ignore_index=True)

        assert list(combined.columns) == list(BASELINE_COLUMNS)
        assert len(combined) == len(populated)
