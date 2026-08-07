"""One measurement schema across the four producers (D-16, D-17, ADR-0031).

The audit counted the columns each producer emits and found four different sets.
Reading them for this task turned up three faults rather than one:

- one quantity under two names — `score`/`sam_score`, `area_px`/`mask_area_px`;
- **two quantities under one name** — `radius_nm` was the detector's blob radius
  in one table and the measured mask's radius in another;
- columns that varied *per row*, because both SAM2 producers built records with
  `if k in res`.

There are no SAM2 weights on this machine and none in CI (PROJECT_RULES §6), so
the two segmentation producers are driven here by a stub predictor. That is the
only way this half of D-17 is testable at all, and it is enough: the defect was
never in the network, it was in the eight lines that assemble a row from its
output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.entities import Detection
from nanoscope.core.science.measurement import BASELINE_COLUMNS, measure_all_baseline
from nanoscope.core.science.measurement.schema import (
    CORE_COLUMNS,
    blocks_for,
    empty_measurement_table,
    measurement_columns,
)
from nanoscope.infrastructure.models.sam2 import run_sam2_from_blobs, run_sam2_from_boxes


class StubPredictor:
    """What SAM2 is, from the record assembly's point of view: something that
    turns a prompt into a mask and a score.

    `predict` returns three candidate masks and their scores, exactly as
    `SAM2ImagePredictor` does, so `masks_pred[np.argmax(scores)]` is exercised
    rather than bypassed.
    """

    def __init__(self, size: int = 64, radius: int = 6) -> None:
        self.size, self.radius = size, radius
        self.images: list[np.ndarray] = []

    def set_image(self, image: np.ndarray) -> None:
        self.images.append(image)

    def predict(self, point_coords, point_labels, box, multimask_output):
        cx, cy = float(point_coords[0, 0]), float(point_coords[0, 1])
        ys, xs = np.mgrid[0 : self.size, 0 : self.size]
        disk = ((xs - cx) ** 2 + (ys - cy) ** 2) <= self.radius**2
        smaller = ((xs - cx) ** 2 + (ys - cy) ** 2) <= (self.radius // 2) ** 2
        masks = np.stack([smaller, disk, np.zeros_like(disk)])
        return masks, np.array([0.30, 0.87, 0.10]), None


def _scene(size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """A height map with two bumps, and the same map as the detector's input."""
    ys, xs = np.mgrid[0:size, 0:size]
    z = np.zeros((size, size), dtype=np.float32)
    for cy, cx in ((20, 20), (44, 44)):
        z += 12.0 * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * 4.0**2))
    return z, z


BLOBS = np.array([[20.0, 20.0, 4.0, 8.0], [44.0, 44.0, 4.0, 8.0]])
BOXES = np.array([[13.0, 13.0, 27.0, 27.0], [37.0, 37.0, 51.0, 51.0]])


def _producers() -> dict[str, tuple[pd.DataFrame, dict]]:
    """Every producer's table, with the blocks it claims to emit."""
    z, image = _scene()
    afm_blobs, _ = run_sam2_from_blobs(StubPredictor(), z, image, BLOBS, nm_per_pixel=2.0)
    afm_boxes, _ = run_sam2_from_boxes(StubPredictor(), z, image, BOXES, nm_per_pixel=2.0)
    sem_blobs, _ = run_sam2_from_blobs(StubPredictor(), None, image, BLOBS, nm_per_pixel=2.0)
    sem_boxes, _ = run_sam2_from_boxes(StubPredictor(), None, image, BOXES, nm_per_pixel=2.0)
    baseline = measure_all_baseline(z, image, BLOBS)
    return {
        "measure_all_baseline": (baseline, {"detector": True, "height": True}),
        "sam2_blobs_afm": (afm_blobs, {"detector": True, "segmentation": True, "height": True}),
        "sam2_boxes_afm": (afm_boxes, {"segmentation": True, "height": True}),
        "sam2_blobs_image": (
            sem_blobs,
            {"detector": True, "segmentation": True, "geometry": True},
        ),
        "sam2_boxes_image": (sem_boxes, {"segmentation": True, "geometry": True}),
    }


class TestEveryProducerEmitsTheDeclaredSchema:
    @pytest.mark.parametrize("name", sorted(_producers()))
    def test_the_columns_are_exactly_the_declaration(self, name: str) -> None:
        """The drift guard ADR-0027 established, now applied to all five tables:
        a declared schema is worth exactly its agreement with what the code
        emits, and only a populated run can prove that."""
        df, blocks = _producers()[name]

        assert not df.empty
        assert list(df.columns) == list(measurement_columns(**blocks))

    @pytest.mark.parametrize("name", sorted(_producers()))
    def test_the_core_is_in_every_one_of_them(self, name: str) -> None:
        """`particle_id x_px y_px area_px method` — the columns a consumer can
        read without first asking which producer it is holding."""
        df, _ = _producers()[name]

        assert set(CORE_COLUMNS) <= set(df.columns)

    @pytest.mark.parametrize("name", sorted(_producers()))
    def test_no_row_has_a_missing_value_in_a_block_it_claims(self, name: str) -> None:
        """The `if k in res` fault: two particles in one call could have
        different columns, and the DataFrame was the union with NaN where a key
        happened to be absent."""
        df, _ = _producers()[name]

        assert not df.isna().to_numpy().any()


class TestOneNamePerQuantity:
    def test_the_segmenter_score_has_one_name(self) -> None:
        """`score` and `sam_score` were the same SAM2 number, emitted by two
        functions that were copy-pasted and drifted (the audit's own note)."""
        producers = _producers()
        for name in ("sam2_blobs_afm", "sam2_boxes_afm"):
            df = producers[name][0]
            assert "mask_score" in df.columns
            assert "score" not in df.columns
            assert "sam_score" not in df.columns

    def test_the_area_of_a_mask_has_one_name(self) -> None:
        producers = _producers()
        for name, (df, _) in producers.items():
            assert "area_px" in df.columns, name
            assert "mask_area_px" not in df.columns, name

    def test_the_detector_radius_and_the_measured_radius_have_different_names(self) -> None:
        """The fault the audit did *not* name, and the worse of the two: a
        reader concatenating the baseline table with the SEM/TEM one used to get
        a single `radius_nm` column holding two different measurements."""
        producers = _producers()
        baseline = producers["measure_all_baseline"][0]
        measured = producers["sam2_blobs_image"][0]

        assert "detector_radius_nm" in baseline.columns
        assert "radius_nm" not in baseline.columns
        assert "radius_nm" in measured.columns
        assert "detector_radius_nm" in measured.columns

        joined = pd.concat([baseline, measured], ignore_index=True)
        assert joined["detector_radius_nm"].notna().all()

    def test_the_two_radii_are_genuinely_different_numbers(self) -> None:
        """Which is why they cannot share a column: the stub segments a disk of
        radius 6, and the prompt says the blob's radius is 8 nm at 2 nm/px."""
        df = _producers()["sam2_blobs_image"][0]

        assert not np.allclose(df["radius_nm"], df["detector_radius_nm"])


class TestTheEmptyTableOfEachKind:
    @pytest.mark.parametrize(
        "blocks",
        [{}, {"height": True}, {"geometry": True}, {"detector": True, "segmentation": True}],
        ids=["core", "afm", "image", "prompted-and-scored"],
    )
    def test_an_empty_table_has_the_columns_and_dtypes_of_its_kind(self, blocks: dict) -> None:
        df = empty_measurement_table(**blocks)

        assert df.empty
        assert list(df.columns) == list(measurement_columns(**blocks))
        assert [str(d) for d in df.dtypes] == list(measurement_columns(**blocks).values())

    def test_a_modality_decides_which_blocks_exist(self) -> None:
        """AFM measures heights, SEM and TEM measure shapes — the rule
        `run_pipeline`'s detect mode needed, and the one ADR-0027 left open."""
        assert blocks_for("afm") == {"height": True, "geometry": False}
        assert blocks_for("sem") == {"height": False, "geometry": True}
        assert blocks_for("tem") == {"height": False, "geometry": True}

    def test_detect_mode_returns_a_table_with_columns(self) -> None:
        """`pd.DataFrame()` — zero columns — was what detect mode returned, the
        one place M3-T12 left D-08 alive on purpose."""
        afm = empty_measurement_table(**blocks_for("afm"))
        image = empty_measurement_table(**blocks_for("tem"))

        assert "height_nm" in afm.columns
        assert "circularity" in image.columns
        assert afm.empty and image.empty


class TestConcatenatingProducers:
    def test_two_tables_of_the_same_kind_concatenate_exactly(self) -> None:
        df = _producers()["sam2_blobs_afm"][0]

        joined = pd.concat([df, df], ignore_index=True)

        assert list(joined.columns) == list(df.columns)
        assert len(joined) == 2 * len(df)

    def test_a_populated_table_and_its_empty_form_are_the_populated_table(self) -> None:
        df, blocks = _producers()["sam2_boxes_afm"]

        joined = pd.concat([empty_measurement_table(**blocks), df], ignore_index=True)

        assert list(joined.columns) == list(df.columns)
        pd.testing.assert_frame_equal(joined, df, check_dtype=False)


class TestTheBoundingBox:
    def test_a_detection_with_no_box_says_so(self) -> None:
        """D-16: `field(default_factory=tuple)` produced `()` — a
        `tuple[int, int, int, int]` with no elements in it, which the annotation
        promised had four. A LoG detection has no box at all."""
        det = Detection(x_px=1.0, y_px=2.0, radius_px=3.0, radius_nm=4.0)

        assert det.bbox is None

    def test_a_detection_with_a_box_carries_four_numbers(self) -> None:
        det = Detection(x_px=1.0, y_px=2.0, radius_px=3.0, radius_nm=4.0, bbox=(1, 2, 3, 4))

        assert det.bbox is not None
        assert len(det.bbox) == 4

    def test_the_empty_tuple_is_not_how_absence_is_spelled(self) -> None:
        """`()` is falsy and so is `None`, so a caller writing `if det.bbox:`
        saw no difference — which is how the broken promise survived. The
        difference is visible to anyone who asks the question properly."""
        absent = Detection(x_px=0.0, y_px=0.0, radius_px=1.0, radius_nm=None)

        assert absent.bbox is None
        assert absent.bbox != ()


class TestNoNumberMoved:
    def test_the_baseline_producer_still_measures_what_it_measured(self) -> None:
        """The rename is a rename. Heights, baselines and areas are unchanged;
        only the names, the dtypes of the two centre columns and the added
        `peak_nm` differ — and `peak_nm` is `height_nm + baseline_nm`, which is
        the definition it was always computed from."""
        z, image = _scene()

        df = measure_all_baseline(z, image, BLOBS)

        assert list(df.columns) == list(BASELINE_COLUMNS)
        np.testing.assert_allclose(df["peak_nm"], df["height_nm"] + df["baseline_nm"])
        assert df["x_px"].dtype == np.float64
        # The centre is the rounded one the mask was built at, not the blob's
        # subpixel centre: reporting the latter would move the measurement.
        np.testing.assert_array_equal(df["x_px"].to_numpy(), np.round(BLOBS[:, 1]))
