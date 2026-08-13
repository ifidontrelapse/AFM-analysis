"""What a run says about the sample, rather than about one particle (M6-T06).

Nobody measures thirty particles to read thirty numbers. They measure thirty to
say *"the particles on this sample are 15 nm across, give or take 4"* — and that
sentence is the reason the measurement table exists.

**Here rather than in a widget** (`docs/Roadmap.md`, M6: *the UI must not
introduce its own defaults*). A panel that computes a mean is a second place
where "what is the mean of this column" is decided, and the first `NaN` makes the
two answers differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

#: How the histogram picks its bins: numpy's own rule, the larger of Sturges and
#: Freedman-Diaconis. A fixed 20 would be an invented parameter, and the shape of
#: a histogram *is* the claim it makes.
BIN_RULE = "auto"

#: Columns that are identifiers rather than measurements. Averaging a
#: `particle_id` produces a number with no meaning attached to it.
_NOT_MEASUREMENTS = frozenset({"particle_id"})


@dataclass(frozen=True)
class Summary:
    """One column of one run, described.

    `count` is how many rows had a **finite** value, which is not the number of
    rows: a producer may leave a column absent for some particles, and a mean
    over `NaN` is `NaN` rather than an answer (ADR-0033's rule, one layer up).
    """

    column: str
    count: int
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float


def numeric_columns(table: pd.DataFrame) -> tuple[str, ...]:
    """The columns worth summarising, in the order an operator reads them.

    Derived from the table in hand, never assumed: which columns exist depends
    on the producer (ADR-0031) and on whether the scale was known (ADR-0025).
    A column that is entirely absent — every `_nm` column of an unscaled scan —
    is **not offered**, because `nan ± nan` is not a statistic.

    Physical quantities come first: an operator asks about nanometres before
    they ask about pixels.
    """
    usable = [
        str(name)
        for name in table.columns
        if str(name) not in _NOT_MEASUREMENTS
        #: `.kind`, not `np.issubdtype`: a pandas extension dtype — the string
        #: column every measurement table carries (`method`, ADR-0031) — is not
        #: a numpy dtype at all, and `issubdtype` raises on it rather than
        #: answering "no".
        and getattr(table[name].dtype, "kind", "O") in "fiu"
        and bool(np.isfinite(table[name].to_numpy(dtype=float)).any())
    ]
    return tuple(sorted(usable, key=lambda name: (not name.endswith("_nm"), name)))


def summarise(table: pd.DataFrame, column: str) -> Summary | None:
    """Count, centre and spread of one column, over its finite values.

    Returns:
        The summary, or `None` when the column is absent or has nothing finite
        in it — which is a state, and not a row of zeros pretending to be one.
    """
    if column not in table.columns:
        return None
    values = table[column].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None

    return Summary(
        column=column,
        count=int(finite.size),
        mean=float(np.mean(finite)),
        median=float(np.median(finite)),
        #: The sample standard deviation (`ddof=1`), because these are particles
        #: measured *from* a sample rather than the whole population of them.
        #: Undefined for one particle, and reported as such.
        std=float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan"),
        minimum=float(np.min(finite)),
        maximum=float(np.max(finite)),
    )


def histogram(table: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Counts and bin edges for one column, binned by `BIN_RULE`.

    Returns:
        `(counts, edges)` as `np.histogram` returns them — empty arrays when
        there is nothing finite to bin.
    """
    if column not in table.columns:
        return np.zeros(0, dtype=int), np.zeros(0)
    values = table[column].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0)
    return np.histogram(finite, bins=BIN_RULE)
