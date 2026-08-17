"""The measurement reference against the schema it describes (M7-T10, ADR-0079).

PROJECT_RULES §8: *"documentation that contradicts the code is worse than no
documentation"*. A document is the one artefact nothing executes, so it drifts
first and silently — M5-T03's refrain applies to prose too: **the rule and its
enforcement ship together, or only the rule does.**

What is checked is the vocabulary, not the sentences: every column the schema can
declare is named in `docs/Measurements.md`, and the document names no column the
schema does not have. A column added in M8 fails this test until somebody writes
down what it means.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from nanoscope.core.science.measurement.schema import (
    CORE_COLUMNS,
    GEOMETRY_COLUMNS,
    HEIGHT_COLUMNS,
    measurement_columns,
)

DOCUMENT = pathlib.Path(__file__).resolve().parents[2] / "docs" / "Measurements.md"

#: Every block, so a column that only exists for one producer still has to be
#: documented — the SEM/TEM geometry block is exactly the one nobody runs in CI.
EVERY_COLUMN = measurement_columns(detector=True, height=True, geometry=True, segmentation=True)

#: Words in backticks in the document. The names are quoted as code there, which
#: is what makes this checkable without parsing prose.
QUOTED = re.compile(r"`([a-z][a-z0-9_]*)`")


@pytest.fixture(scope="module")
def document() -> str:
    return DOCUMENT.read_text(encoding="utf-8")


@pytest.mark.parametrize("column", sorted(EVERY_COLUMN))
def test_every_column_is_documented(document: str, column: str) -> None:
    """A column nobody wrote a paragraph about is a number nobody can defend."""
    assert f"`{column}`" in document, (
        f"{column} is in the measurement schema and not in docs/Measurements.md. "
        "A column arrives with its meaning or it does not arrive."
    )


def test_the_document_invents_no_column(document: str) -> None:
    """The other direction, which is how a document goes stale: a column that was
    renamed stays described under its old name for a milestone."""
    #: Only the words that look like schema columns are considered — the document
    #: quotes function names, file names and parameters too, and this test is
    #: about the *table's* vocabulary.
    suspects = {
        word
        for word in QUOTED.findall(document)
        if word.endswith(("_px", "_nm", "_nm2")) or word in {"circularity", "aspect_ratio"}
    }
    #: Named in the document for what they are: not columns.
    not_columns = {
        "x1",
        "y1",
        "x2",
        "y2",
        "pixel_size_nm",
        "nm_per_pixel",
        "min_ring_px",
        "measure_outer_px",
        "measure_inner_erode_px",
        "sam2_outer_ring_px",
        "sam2_inner_erode_px",
        "inner_erode_px",
        "outer_px",
        "equivalent_diameter_area",
    }

    assert not suspects - set(EVERY_COLUMN) - not_columns


def test_the_check_can_fail() -> None:
    """A guard that cannot fail is decoration (M5-T03's rule, fourth site)."""
    assert "`height_nm`" in DOCUMENT.read_text(encoding="utf-8")
    assert "`no_such_column_nm`" not in DOCUMENT.read_text(encoding="utf-8")


def test_the_two_traps_are_stated(document: str) -> None:
    """The two things this document exists to say, and the reason it was written:
    a reader who misses either draws a wrong conclusion from a right number."""
    #: `area_px` is a *detector's* disk on one path and a real mask on the other.
    assert "biggest trap" in document
    for column in ("area_px", "particle_id"):
        assert f"`{column}`" in document
    #: And the block that is only reachable through a path CI never runs.
    assert set(GEOMETRY_COLUMNS) | set(HEIGHT_COLUMNS) | set(CORE_COLUMNS) <= set(EVERY_COLUMN)
