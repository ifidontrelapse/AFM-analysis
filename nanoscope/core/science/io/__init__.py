"""Parsing instrument files into arrays. No disk access beyond the read itself.

Nanoscope SPM today; the loaders that choose a file and hand it here live in
`nanoscope.infrastructure.storage` (M2-T04).
"""

from nanoscope.core.science.io.nanoscope_spm import _read_nanoscope_z

__all__ = ["_read_nanoscope_z"]
