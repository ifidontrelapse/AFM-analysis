"""Annotations become something a trainer can read (M8-T02).

M8-T01 said what a training run consumes — a `DatasetSpec`: a directory, class
names, and how many images are in each half. Nothing produced one. This does.

M7-T09 wrote the labels and **named the half it was not writing**: ADR-0078
stopped before `data.yaml` and the split because *"a split is a dataset decision
— how much to hold out, stratified by what — and it belongs to M8-T02 rather
than to the task that happened to write the labels first."*

**The decision this module turns on is that a height map is not an image.** The
scans here are `float32` arrays in nanometres; a trainer reads PNG. Something has
to make a picture, and what makes it decides what the model learns — so it is
made by `as_network_input`, the same function `YoloDetector._prepare_image`
calls, from `z_above` (`z_flat - substrate`), the same array `detect` is handed.
A dataset built any other way trains a model on a distribution inference never
sees, and nothing about that failure is loud (ADR-0081).
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from nanoscope.application.use_cases.annotations import LABELS_DIRECTORY, _to_label
from nanoscope.application.use_cases.display import load_for_display
from nanoscope.application.use_cases.preprocessing import preprocess_image
from nanoscope.core.entities.project import Annotation, AnnotationSource, ImageRecord
from nanoscope.core.entities.training import DatasetSpec
from nanoscope.core.errors import AnalysisFailedError, InvalidParameterError
from nanoscope.core.ports import ProjectRepository
from nanoscope.core.values import Modality, default_polarity
from nanoscope.infrastructure.imaging.network_input import as_network_input

logger = logging.getLogger(__name__)

#: Where the pictures go, beside the labels M7-T09 already names `labels/`.
IMAGES_DIRECTORY = "images"

#: The two halves, and the names every trainer in this ecosystem expects.
TRAIN, VAL = "train", "val"

#: The manifest ultralytics is pointed at. Not written by M7-T09's export, and
#: this is the module ADR-0078 said would write it.
DATASET_FILE = "data.yaml"

#: How much is held out when the caller does not say. A fifth is the ordinary
#: default in this ecosystem; it is a parameter because the right number depends
#: on how many scans there are, and forty is a different problem from four.
DEFAULT_VAL_FRACTION = 0.2


@dataclass(frozen=True)
class DatasetReport:
    """What was built, and what it left out.

    The spec is what a provider needs; the rest is what a person needs in order
    to believe it — the same division `ImportReport` made in M4-T06, and for the
    same reason: a batch that quietly does part of the job is one an operator
    finds out about later.
    """

    spec: DatasetSpec
    boxes: int
    #: Scans that carried annotations and could not be prepared, with the reason.
    #: Reported rather than raised: eleven usable scans and one unreadable one is
    #: a dataset, and a builder that refuses the lot teaches an operator to stop
    #: asking (ADR-0040's rule, from the building side).
    skipped: tuple[tuple[str, str], ...] = ()


def build_dataset(
    repository: ProjectRepository,
    *,
    sources: Iterable[AnnotationSource] | None = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = 0,
    directory_name: str | None = None,
) -> DatasetReport:
    """Turn the project's annotations into a dataset directory under `cache/`.

    Args:
        repository: an open project.
        sources: which kinds of annotation to train on. `None` means all of
            them; `(AnnotationSource.MANUAL,)` is the set that cannot confirm
            itself. The caller names the scope rather than this picking a
            default that hides the question — ADR-0044's rule, and M7-T09's
            reading of it: *a model trained on its own output is confirming
            itself.*
        val_fraction: how much to hold out, by **image**. `0.0` builds a
            train-only dataset, which is legal and means `DatasetSpec.val_images`
            is 0 and every epoch's `validation` metric block is absent
            (ADR-0080).
        seed: which shuffle. Recorded in `data.yaml`, because a rebuild that
            splits differently makes two runs incomparable.
        directory_name: what to call it under `cache/`. Defaults to a timestamped
            name, so building twice does not overwrite the dataset a run is
            still training from.

    Returns:
        The spec a `TrainingProvider` takes, and what was written.

    Raises:
        AnalysisFailedError: no annotation of those kinds exists, or none of the
            scans carrying them could be prepared. An empty dataset is
            indistinguishable from *"nothing was drawn"*, which is a different
            statement (ADR-0048's rule, third site).
        InvalidParameterError: `val_fraction` is not a fraction.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise InvalidParameterError(
            f"val_fraction={val_fraction}: hold out somewhere in [0, 1). Holding out "
            "everything leaves nothing to train on"
        )

    per_image = [
        (record, kept)
        for record in repository.list_images()
        if (kept := _selected(repository.annotations_for(record.id), sources))
    ]
    if not per_image:
        raise AnalysisFailedError(
            "nothing to build: no annotation of that kind exists in this project"
        )

    classes = tuple(sorted({one.label for _record, kept in per_image for one in kept}))
    index_of = {name: index for index, name in enumerate(classes)}
    root = directory_name or f"dataset-{datetime.now(UTC):%Y%m%d-%H%M%S}"

    held_out = _validation_set(per_image, val_fraction, seed)
    counts = {TRAIN: 0, VAL: 0}
    boxes = 0
    skipped: list[tuple[str, str]] = []

    for record, kept in per_image:
        try:
            picture = _picture_of(repository, record)
        except (AnalysisFailedError, InvalidParameterError, OSError) as refusal:
            #: Named, counted and carried on. The alternative is one unreadable
            #: scan costing an operator the other eleven.
            logger.warning("skipping %s: %s", record.display_name, refusal)
            skipped.append((record.display_name, str(refusal)))
            continue

        half = VAL if record.id in held_out else TRAIN
        stem = Path(record.relative_path).stem
        height, width = picture.shape[:2]
        repository.write_cache_image(f"{root}/{IMAGES_DIRECTORY}/{half}/{stem}.png", picture)
        repository.write_cache_text(
            f"{root}/{LABELS_DIRECTORY}/{half}/{stem}.txt",
            "\n".join(
                _to_label(one, index_of[one.label], width=width, height=height) for one in kept
            )
            + "\n",
        )
        counts[half] += 1
        boxes += len(kept)

    if not counts[TRAIN]:
        raise AnalysisFailedError(
            "nothing to build: none of the annotated scans could be prepared — "
            + "; ".join(f"{name} ({why})" for name, why in skipped)
        )

    written = repository.write_cache_text(
        f"{root}/{DATASET_FILE}", _manifest(classes, seed=seed, val_fraction=val_fraction)
    )
    spec = DatasetSpec(
        #: Read back off the path the adapter chose rather than assembled here:
        #: where a dataset lands is the adapter's decision (M4-T11's division).
        root=str(Path(written).parent.as_posix()),
        classes=classes,
        train_images=counts[TRAIN],
        val_images=counts[VAL],
    )
    logger.info(
        "built %s: %d train, %d val, %d box(es) over %d class(es)",
        spec.root,
        spec.train_images,
        spec.val_images,
        boxes,
        len(classes),
    )
    return DatasetReport(spec=spec, boxes=boxes, skipped=tuple(skipped))


def _validation_set(
    per_image: Sequence[tuple[ImageRecord, Sequence[Annotation]]],
    val_fraction: float,
    seed: int,
) -> set[int]:
    """Which image **ids** are held out.

    **By image, never by box**, and this is the line in the task that a reviewer
    cannot see from the output. Two boxes off one scan, one in each half, is
    leakage: the validation score then measures how well the model memorised
    that scan's substrate, its instrument noise and its particle population, and
    every number M8-T08 reports is quietly inflated by it.

    Seeded and sorted first, so the same project and the same seed give the same
    split — two runs that split differently cannot be compared, and the seed
    goes into `data.yaml` where a person can read it.
    """
    ids = sorted(record.id for record, _kept in per_image)
    holding = int(len(ids) * val_fraction)
    if not holding:
        #: Rounding down, and then not up: asking for a fifth of four scans is
        #: asking for 0.8, and holding out one of four is a validation set of
        #: 25% reported as 20%. Zero is the honest answer, and `val_images == 0`
        #: already means something specific (ADR-0080).
        return set()

    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    return set(shuffled[:holding])


def _picture_of(repository: ProjectRepository, record: ImageRecord) -> np.ndarray:
    """The `uint8` picture this scan trains as.

    AFM goes through preprocessing, because `z_above` — `z_flat - substrate` — is
    what `detect` is handed and what every detection in this project was ever
    made from. Training on raw height maps would teach a model the tilt and the
    substrate that inference has already removed.

    SEM and TEM have no substrate to build (ADR-0031) and are analysed as they
    are, so the loaded image is the picture. Both then go through the one
    function inference also calls.
    """
    if record.modality is Modality.AFM:
        z = preprocess_image(repository, record.id).z_result
    else:
        z = load_for_display(repository, record.id).data
    return as_network_input(z, polarity=default_polarity(record.modality))


def _manifest(classes: tuple[str, ...], *, seed: int, val_fraction: float) -> str:
    """`data.yaml`, written by hand rather than with a YAML library.

    Four keys and a list of short names: a dependency to emit that is a
    dependency to install, and `application` has none of its own today. The
    quoting is the one thing that could go wrong, so every name is quoted and a
    quote inside one is doubled — which is YAML's own escape.

    `path: .` and relative halves, so the directory can be moved or copied and
    still resolves — the same property that makes a project a plain directory
    (ADR-0003).
    """
    names = "\n".join(f"  {index}: {_quoted(name)}" for index, name in enumerate(classes))
    return (
        "# Written by nanoscope (M8-T02). Rebuilt from the project's annotations;\n"
        "# safely deletable, like everything under cache/ (PROJECT_RULES §5).\n"
        f"# seed: {seed}  val_fraction: {val_fraction}\n"
        "path: .\n"
        f"train: {IMAGES_DIRECTORY}/{TRAIN}\n"
        f"val: {IMAGES_DIRECTORY}/{VAL}\n"
        f"nc: {len(classes)}\n"
        f"names:\n{names}\n"
    )


def _quoted(name: str) -> str:
    doubled = name.replace("'", "''")
    return f"'{doubled}'"


def _selected(
    annotations: Sequence[Annotation], sources: Iterable[AnnotationSource] | None
) -> list[Annotation]:
    wanted = None if sources is None else set(sources)
    return [one for one in annotations if wanted is None or one.source in wanted]
