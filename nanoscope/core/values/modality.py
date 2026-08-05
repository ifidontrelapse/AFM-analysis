"""What kind of image this is, and which way the particles read.

Both replace string literals that already exist in the code — `"afm" | "sem" |
"tem"` in `PipelineResult.modality`, and the bright-on-dark assumption the LoG
detector makes without saying so.

`PipelineResult.modality` stays `str`, because the characterization golden
serializes that field and an enum would change what `dataclasses.asdict`
produces. `Polarity` **is** adopted, by M3-T10 (ADR-0023): both detectors take
one, and `default_polarity` below is where each instrument's convention lives.
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
    them below it. Both detectors used to keep the bright side unconditionally,
    and so found 0 of 22 particles on TEM — audit defect **D-12**. Decision **B3**
    answered it: polarity is *configured*, with a per-modality default, not
    detected from the image (ADR-0023).
    """

    BRIGHT_ON_DARK = "bright_on_dark"
    DARK_ON_BRIGHT = "dark_on_bright"


#: What each instrument produces when nobody says otherwise (ADR-0023 / B3).
#: AFM height maps put particles above the substrate and SEM electrons scatter
#: off them brightly; TEM images them by absorption, so they come out dark.
_DEFAULT_POLARITY = {
    Modality.AFM: Polarity.BRIGHT_ON_DARK,
    Modality.SEM: Polarity.BRIGHT_ON_DARK,
    Modality.TEM: Polarity.DARK_ON_BRIGHT,
}


def default_polarity(modality: str) -> Polarity:
    """The polarity of `modality` when the caller states none.

    A default, not a detection: it is what the instrument conventionally
    produces, and an operator whose sample breaks the convention overrides it
    (ADR-0023).

    Raises:
        ValueError: for a modality that does not exist. Every modality that does
            has an entry, so there is no silent fallback to guess wrong with.
    """
    return _DEFAULT_POLARITY[Modality(modality)]
