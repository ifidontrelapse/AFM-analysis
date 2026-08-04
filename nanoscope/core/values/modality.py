"""What kind of image this is, and which way the particles read.

Both replace string literals that already exist in the code — `"afm" | "sem" |
"tem"` in `PipelineResult.modality`, and the bright-on-dark assumption the LoG
detector makes without saying so.

Neither is adopted yet: `PipelineResult.modality` stays `str` until M2-T10, because
the characterization golden serializes that field and an enum would change what
`dataclasses.asdict` produces.
"""

from __future__ import annotations

from enum import StrEnum


class Modality(StrEnum):
    """The instrument an image came from.

    `StrEnum`, so `Modality.AFM == "afm"` and `f"{Modality.AFM}" == "afm"` both
    hold: the existing comparisons against the literals keep working while M2-T10
    adopts this one call site at a time.
    """

    AFM = "afm"
    SEM = "sem"
    TEM = "tem"


class Polarity(StrEnum):
    """Whether particles are brighter or darker than their background.

    AFM height maps and SEM images put particles above the background; TEM puts
    them below it. The detector currently keeps the bright side unconditionally
    and therefore finds 0 of 22 particles on TEM — audit defect **D-12**, and the
    open decision **B3** is whether polarity is configured per modality or
    detected. This type is the vocabulary that decision needs; M3-T10 wires it.
    """

    BRIGHT_ON_DARK = "bright_on_dark"
    DARK_ON_BRIGHT = "dark_on_bright"
