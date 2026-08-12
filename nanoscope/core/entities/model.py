"""What a model is, once it stops being a path in a default argument (M4-T13).

W10: `yolo_model_path: str = "./checkpoints/best12x.pt"` — a relative path to a
file nobody promises exists, with no version, no checksum and no record of what
it was trained on, repeated in two places.

ADR-0005 replaced it with a record. This is that record, and it lives in `core`
because `application` decides *which* model to use and must not import
ultralytics to hold the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ModelTask(StrEnum):
    """What a model does. What it *is* — YOLO, SAM2 — is the framework."""

    DETECT = "detect"
    SEGMENT = "segment"


class ModelFramework(StrEnum):
    """Which provider can load it.

    The string the registry is keyed by, and the only place these names are
    allowed outside `infrastructure/models` (PROJECT_RULES §2.5 keeps `"yolo"`
    and `"sam2"` out of `gui/` and `core/science`; a registry key in `core` is
    the seam that rule expects).
    """

    ULTRALYTICS = "ultralytics"
    SAM2 = "sam2"


@dataclass(frozen=True)
class ModelDescriptor:
    """One model a project can use, and everything needed to say which it is.

    ADR-0005's list, in a record: *id, task, framework, path, input size, class
    map, provenance, checksum*.
    """

    #: What a configuration names, chosen by whoever registered it, unique in
    #: the project. Not a hash: an operator names their model, and the checksum
    #: answers the different question of whether the file still matches.
    model_id: str
    task: ModelTask
    framework: ModelFramework
    #: Relative to the project root when the weights live in `models/`, absolute
    #: when they are a shared file the operator keeps elsewhere. The second is
    #: allowed and has a consequence: the project opens on another machine and
    #: that model is unavailable there (ADR-0050).
    path: str
    #: The side length the provider feeds the network, or `None` when the
    #: framework decides. Never a guess.
    input_size_px: int | None = None
    #: Class index → name, as the weights were trained. Empty when the model has
    #: one unnamed class, which is every detector in this project today.
    class_map: dict[int, str] = field(default_factory=dict)
    #: Where it came from, in whatever words the person who registered it used:
    #: "trained 2026-08-01 on 412 annotations", "downloaded from …". Free text,
    #: because provenance that must fit a schema stops being recorded.
    provenance: str = ""
    #: SHA-256 of the weights when they were registered, or `None` if nobody
    #: computed it. Stored so a later reader *can* ask whether the file changed
    #: — checking it is a read of a very large file, and whose job that is
    #: belongs to whoever asks.
    sha256: str | None = None
    registered_utc: str = ""

    @property
    def is_external(self) -> bool:
        """True when the weights live outside the project directory."""
        return self.path.startswith("/")
