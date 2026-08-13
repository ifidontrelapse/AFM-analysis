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

from collections.abc import Collection
from dataclasses import dataclass

from nanoscope.core.entities.model import ModelFramework
from nanoscope.core.errors import UnsupportedRequestError

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
        UnsupportedRequestError: with the same wording `src/pipeline.py` used
            before M2-T10, so callers matching on the message are unaffected —
            and, since ADR-0030, a `ValueError` as well, so callers matching on
            the *type* are unaffected too. Nothing here is malformed: the
            request is well-formed and this version has no path for it.
    """
    if detector not in ("log", "yolo"):
        raise UnsupportedRequestError(f"Unknown detector: {detector!r}")

    row = find(modality, detector, mode)
    if row is None:
        # Order matters: report the most specific reason, the way the old
        # sequential `if`s did, rather than a generic "unsupported combination".
        if mode == _BASELINE and modality != "afm":
            raise UnsupportedRequestError("mode='baseline' is only supported for AFM data")
        if mode == _BASELINE:
            raise UnsupportedRequestError("mode='baseline' requires detector='log'")
        raise UnsupportedRequestError(
            f"Unsupported combination: modality={modality!r}, detector={detector!r}, mode={mode!r}"
        )

    if row.requires_predictor and not has_predictor:
        raise UnsupportedRequestError("predictor must be provided when mode='segment'")

    return row


# ── What a UI may offer (M6-T02, ADR-0062) ───────────────────────────────────
#
# M6's third exit criterion: *"invalid combinations are disabled in the UI
# **because the capability matrix says so** — not by a duplicated rule."* The
# matrix has had one caller since M2-T10, and it validates a request that has
# already been assembled. This is the other direction: **what may be asked for**,
# so a panel cannot express an invalid request in the first place.
#
# It lives here rather than in `gui/` for the reason PROJECT_RULES §2.5 gives:
# the strings `"log"` and `"yolo"` may not appear in a widget, and a widget that
# knows one detector's name is a widget that will grow an `if` about it.


@dataclass(frozen=True)
class ModeOption:
    """One mode of one detector, and whether it can run right now."""

    mode: str
    available: bool
    #: Why not — a sentence for an operator, never an error code. An entry
    #: greyed out with no explanation is the failure this criterion is against,
    #: not a lesser version of it.
    reason: str | None = None


@dataclass(frozen=True)
class DetectorOption:
    """One detector, its modes for a given modality, and what it needs."""

    detector: str
    modes: tuple[ModeOption, ...]
    available: bool
    reason: str | None = None


#: What each detector needs loaded before it can run, as the framework a
#: `ModelDescriptor` carries. `None` means "nothing but NumPy" — which is why
#: the LoG detector is the one that works in CI, on a fresh install, and on a
#: machine with no GPU (ADR-0050's registry is keyed by exactly this).
DETECTOR_FRAMEWORKS: dict[str, ModelFramework | None] = {
    "log": None,
    "yolo": ModelFramework.ULTRALYTICS,
}


def detector_options(
    modality: str,
    *,
    frameworks: Collection[ModelFramework] = (),
    has_predictor: bool = False,
) -> tuple[DetectorOption, ...]:
    """What a panel may offer for this modality, and why the rest is refused.

    Args:
        modality: the image's own, which is what makes `baseline` an AFM-only
            row rather than a rule a widget remembers.
        frameworks: the frameworks this project has a **detection** model
            registered for (ADR-0050). A detector whose framework is missing is
            offered and disabled, not hidden: "you need to register a model" is
            a different sentence from "this application cannot do that".
        has_predictor: whether a segmentation predictor exists. Nothing
            constructs one before M6-T04, so today this is `False` and every
            `segment` row says so.

    Returns:
        One entry per detector the matrix knows, in its own order, each with its
        modes for this modality.
    """
    options: list[DetectorOption] = []
    for detector, framework in DETECTOR_FRAMEWORKS.items():
        needs_model = framework is not None and framework not in frameworks
        reason = (
            f"no {framework} model is registered in this project; register one to use it"
            if needs_model
            else None
        )
        modes = tuple(
            ModeOption(
                mode=row.mode,
                available=not needs_model and (has_predictor or not row.requires_predictor),
                reason=reason
                or (
                    "segmentation needs a model registered for it in this project"
                    if row.requires_predictor and not has_predictor
                    else None
                ),
            )
            for row in CAPABILITIES
            if row.modality == modality and row.detector == detector
        )
        options.append(
            DetectorOption(
                detector=detector,
                modes=modes,
                available=not needs_model and any(mode.available for mode in modes),
                reason=reason,
            )
        )
    return tuple(options)
