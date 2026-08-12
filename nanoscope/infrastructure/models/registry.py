"""Which provider loads which model (M4-T13, ADR-0005).

ADR-0005 wrote the shape: *"adding a model means adding a provider and one
registry line. No other file changes."* This is the registry, and it is keyed by
`ModelFramework` — the string a `ModelDescriptor` carries — so nothing above
`infrastructure` has to know that ultralytics exists (PROJECT_RULES §2.5).

**It hands back factories, never instances.** Building a detector loads weights
off a disk; a registry that constructs on lookup makes "what models does this
project have?" an expensive question, and an impossible one in CI, where the
weights do not exist at all. `resolve()` returns something callable and the
caller decides when to pay for it.

The device from M4-T12 arrives here, which is the gap ADR-0049 named: a factory
takes the resolved `Device` and hands it to the provider, so no provider ever
asks torch where it should run (ADR-0004, PROJECT_RULES §2.6).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from nanoscope.core.entities.device import Device
from nanoscope.core.entities.model import ModelDescriptor, ModelFramework
from nanoscope.core.errors import UnsupportedRequestError

#: What a factory is: given the resolved weights path and the device chosen for
#: it, produce the thing that runs. `Any` is honest here — a detector and a
#: segmenter have different shapes, and `ModelDescriptor.task` is what says
#: which one a caller asked for.
ModelFactory = Callable[[Path, Device], Any]

_REGISTRY: dict[ModelFramework, ModelFactory] = {}


def register(framework: ModelFramework, factory: ModelFactory) -> None:
    """Say which factory loads this framework. The one line ADR-0005 promised."""
    _REGISTRY[framework] = factory


def frameworks() -> tuple[ModelFramework, ...]:
    """Every framework that can be loaded here, for a dialog to offer."""
    return tuple(_REGISTRY)


def resolve(descriptor: ModelDescriptor) -> ModelFactory:
    """The factory for this model — **not** the model.

    Nothing is read from disk. That is what makes this callable in CI, where no
    weights exist, and what keeps listing a project's models cheap.

    Raises:
        UnsupportedRequestError: the descriptor names a framework this version
            cannot load. Nothing is malformed — the request is well-formed and
            has no implementation here, which is exactly what that error means
            (ADR-0030).
    """
    factory = _REGISTRY.get(descriptor.framework)
    if factory is None:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise UnsupportedRequestError(
            f"no provider for framework {descriptor.framework!r} "
            f"(model {descriptor.model_id!r}); this version can load: {known}"
        )
    return factory


def _ultralytics(path: Path, device: Device) -> Any:
    """A YOLO detector on the chosen device.

    Imported inside the function: ultralytics pulls in torch, and importing this
    module must stay free (PROJECT_RULES import hygiene).
    """
    from nanoscope.infrastructure.models.yolo import YoloDetector

    return YoloDetector(model_path=str(path), device=device.torch_name)


def _sam2(path: Path, device: Device) -> Any:
    """A SAM2 predictor on the chosen device.

    Returns the predictor rather than a wrapper, because the SAM2 code in this
    project is functions taking a predictor (`run_sam2_from_blobs`) rather than
    a class — and inventing a class here to make the registry symmetrical would
    be an abstraction with one caller and no second implementation.
    """
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    return SAM2ImagePredictor.from_pretrained(str(path), device=device.torch_name)


register(ModelFramework.ULTRALYTICS, _ultralytics)
register(ModelFramework.SAM2, _sam2)
