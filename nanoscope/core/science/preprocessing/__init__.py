"""AFM preprocessing: flatten the map, then estimate what is underneath it.

The public surface of the subpackage. `src/preprocess.py` re-exports exactly these
objects until M2-T15 deletes it.

Split into two modules on the way in — levelling and substrate estimation share no
state and are called at different stages — but nothing else changed: same
algorithms, same constants, zero golden drift.
"""

from nanoscope.core.science.preprocessing.flatten import flatten_lines, flatten_plane
from nanoscope.core.science.preprocessing.substrate import (
    build_substrate_map,
    estimate_radius_otsu,
    estimate_rough_radius,
    get_substrate_map,
)

__all__ = [
    "build_substrate_map",
    "estimate_radius_otsu",
    "estimate_rough_radius",
    "flatten_lines",
    "flatten_plane",
    "get_substrate_map",
]
