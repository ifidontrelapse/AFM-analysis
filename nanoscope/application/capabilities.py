"""Which (modality, detector, mode) combinations the pipeline supports — once.

Before M2-T10 these rules existed in three places: as `if` statements scattered
through `src/pipeline.py`, as a table in `PROJECT_CONTEXT.md` §"Execution matrix",
and — until ADR-0012 deleted it — hardcoded in the React client, where the audit
found it had already drifted (D-19). Three copies of a rule is three chances to
disagree, and prose cannot be executed.

This module is the copy that runs. The table in `PROJECT_CONTEXT.md` now documents
it rather than restating it.

**The other half of the task is *when*.** `src/pipeline.py` used to validate after
detection: asking for AFM + YOLO + baseline ran a full inference pass and then
raised `ValueError` — minutes of GPU work for a request that was invalid before
any compute started (audit D-14). `validate_request` is called first now, and the
messages are unchanged so anything matching on the text still works.
"""

from __future__ import annotations

from dataclasses import dataclass

# The modes, spelled as they appear in `PipelineConfig.mode`.
_DETECT = "detect"
_BASELINE = "baseline"
_SEGMENT = "segment"


@dataclass(frozen=True)
class Capability:
    """One row of the execution matrix.

    `requires_predictor` is a property of the mode rather than of the row, but it
    is stated per row so a reader never has to hold a second rule in their head.
    """

    modality: str
    detector: str
    mode: str
    requires_predictor: bool


def _rows() -> tuple[Capability, ...]:
    supported: list[Capability] = []
    for modality in ("afm", "sem", "tem"):
        for detector in ("log", "yolo"):
            supported.append(Capability(modality, detector, _DETECT, requires_predictor=False))
            supported.append(Capability(modality, detector, _SEGMENT, requires_predictor=True))
    # `baseline` measures height above a local substrate, so it needs a Z map —
    # AFM only. And it consumes the LoG blob array (sigma per particle) to build
    # circular masks, which YOLO does not produce: it returns boxes.
    supported.append(Capability("afm", "log", _BASELINE, requires_predictor=False))
    return tuple(supported)


CAPABILITIES: tuple[Capability, ...] = _rows()


def find(modality: str, detector: str, mode: str) -> Capability | None:
    """The matching row, or `None` if the combination is unsupported."""
    for row in CAPABILITIES:
        if (row.modality, row.detector, row.mode) == (modality, detector, mode):
            return row
    return None


def validate_request(modality: str, detector: str, mode: str, has_predictor: bool) -> Capability:
    """Check a request **before any inference runs**. Returns the matching row.

    Raises:
        ValueError: with the same wording `src/pipeline.py` used before M2-T10,
            so callers matching on the message are unaffected.
    """
    if detector not in ("log", "yolo"):
        raise ValueError(f"Unknown detector: {detector!r}")

    row = find(modality, detector, mode)
    if row is None:
        # Order matters: report the most specific reason, the way the old
        # sequential `if`s did, rather than a generic "unsupported combination".
        if mode == _BASELINE and modality != "afm":
            raise ValueError("mode='baseline' is only supported for AFM data")
        if mode == _BASELINE:
            raise ValueError("mode='baseline' requires detector='log'")
        raise ValueError(
            f"Unsupported combination: modality={modality!r}, detector={detector!r}, mode={mode!r}"
        )

    if row.requires_predictor and not has_predictor:
        raise ValueError("predictor must be provided when mode='segment'")

    return row
