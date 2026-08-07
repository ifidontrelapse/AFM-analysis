"""The checks every numerical entry point runs before it computes (D-15, ADR-0030).

One implementation, called at each entry point, rather than a hand-written check
per function — because the defect D-15 describes is not "there are no checks",
it is that the eleven ways in were answering the same question differently.

The contract these functions state, once:

    A height map is a 2-D NumPy array, non-empty, of a numeric dtype, and
    finite.

Finiteness is the half that is a decision rather than a structural fact, and
ADR-0030 records it: `flatten_plane` — step one of the documented chain — has
always rejected NaN through `scipy.lstsq`, while `flatten_lines` propagated it
and `detect_particles` answered a NaN map with "no particles". Rejecting at the
entry makes the contract the whole library's instead of the first step's.

The cost is one `np.isfinite` pass, O(n) and measured at ~0.05 ms on 512x512 —
against a `detect_particles` call three orders of magnitude slower.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nanoscope.core.errors import InvalidImageError, InvalidParameterError


def ensure_height_map(z: Any, name: str = "z") -> np.ndarray:
    """Check that `z` is a map this library can compute on, and return it.

    Args:
        z:    the candidate array
        name: the parameter's name in the caller's signature, so the message
              names what the caller wrote rather than what we call it

    Returns:
        `z` unchanged. It is returned rather than only checked so a call site
        can read `z = ensure_height_map(z)` and have one statement to delete if
        this ever needs to become a coercion.

    Raises:
        InvalidImageError: if it is not a 2-D, non-empty, numeric, finite array.
            The message names the parameter, what was wrong, and the value that
            was wrong — a shape, a dtype, or how many values were not finite.
    """
    if not isinstance(z, np.ndarray):
        raise InvalidImageError(
            f"{name} must be a numpy array, got {type(z).__name__}. "
            "Load the image with nanoscope.infrastructure.storage first."
        )
    if z.ndim != 2:
        raise InvalidImageError(
            f"{name} must be a 2-D array indexed [y, x], got {z.ndim} dimensions "
            f"with shape {z.shape}."
        )
    if z.size == 0:
        raise InvalidImageError(f"{name} is empty: shape {z.shape}.")
    if not (np.issubdtype(z.dtype, np.integer) or np.issubdtype(z.dtype, np.floating)):
        # Booleans are excluded on purpose: a mask is not a height map, and the
        # one place a bool array meant "topography" was D-13's truncation bug.
        raise InvalidImageError(f"{name} must hold integers or real numbers, got dtype {z.dtype}.")
    if np.issubdtype(z.dtype, np.floating):
        finite = np.isfinite(z)
        if not finite.all():
            n_bad = int(finite.size - finite.sum())
            n_nan = int(np.isnan(z).sum())
            raise InvalidImageError(
                f"{name} contains {n_bad} value(s) that are not finite "
                f"({n_nan} nan, {n_bad - n_nan} inf); a height map must be finite. "
                "Repair or crop the scan before analysing it."
            )
    return z


def ensure_mask(mask: Any, name: str = "mask") -> np.ndarray:
    """Check that `mask` is a 2-D boolean array, and return it.

    The mirror of `ensure_height_map`: where that one refuses a boolean array
    because a mask is not topography, this one refuses a float array because a
    membership question has no intermediate answers. `mask.astype(bool)` on a
    float array is a silent threshold at zero, which is the kind of substitution
    this milestone has spent five ADRs deleting.
    """
    if not isinstance(mask, np.ndarray):
        raise InvalidImageError(f"{name} must be a numpy array, got {type(mask).__name__}.")
    if mask.ndim != 2:
        raise InvalidImageError(
            f"{name} must be a 2-D array indexed [y, x], got {mask.ndim} dimensions "
            f"with shape {mask.shape}."
        )
    if mask.size == 0:
        raise InvalidImageError(f"{name} is empty: shape {mask.shape}.")
    if mask.dtype != np.bool_:
        raise InvalidImageError(
            f"{name} must be a boolean array, got dtype {mask.dtype}. Threshold it "
            "explicitly rather than relying on a cast."
        )
    return mask


def ensure_positive(value: Any, name: str, *, allow_none: bool = False) -> Any:
    """Check that a scalar parameter is strictly positive, and return it.

    `nan` fails, which is the intent: `not nan > 0` is `True`, and the same
    comparison ADR-0018 chose for exactly that reason.

    Args:
        value:      the caller's number, or None
        name:       parameter name, for the message
        allow_none: whether "unknown" is a legal value here. `None` is a state
                    with a meaning (ADR-0019/0025), not a missing argument

    Raises:
        InvalidParameterError: if the value is absent where it is required, or
            present and not strictly positive.
    """
    if value is None:
        if allow_none:
            return None
        raise InvalidParameterError(f"{name} is required and must be positive, got None.")
    if not value > 0:
        raise InvalidParameterError(f"{name} must be positive, got {value!r}.")
    return value


def ensure_non_negative(value: Any, name: str) -> Any:
    """Check that a scalar parameter is zero or greater, and return it.

    Zero is legal where the parameter is a *floor* — `min_size_nm=0` means "keep
    everything", which ADR-0025 already relies on for an unscaled image.
    """
    if value is None:
        raise InvalidParameterError(f"{name} is required, got None.")
    if not value >= 0:
        raise InvalidParameterError(f"{name} must be zero or greater, got {value!r}.")
    return value
