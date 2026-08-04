"""The execution matrix, and the promise that it is checked before any inference.

The second half matters more than the first. Validation used to run *after*
detection (audit D-14), so an invalid request cost a full YOLO or SAM2 pass before
raising. `test_invalid_requests_are_rejected_before_any_detector_is_built` is the
test that would notice if that ever regressed — it fails if `run_pipeline` touches
a detector at all.

None of this is covered by the characterization golden: it never calls
`run_pipeline`. These tests are the only thing standing behind the change.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.application.capabilities import CAPABILITIES, find, validate_request
from nanoscope.core.entities import MicroscopyData, PipelineConfig, PreprocessingResult


class TestTheMatrix:
    def test_baseline_exists_only_for_afm_plus_log(self) -> None:
        # Height above a local substrate needs a Z map (AFM only) and the LoG
        # blob array to build circular masks — YOLO returns boxes, not sigmas.
        baseline = {(c.modality, c.detector) for c in CAPABILITIES if c.mode == "baseline"}
        assert baseline == {("afm", "log")}

    def test_detect_and_segment_exist_for_every_modality_and_detector(self) -> None:
        for modality in ("afm", "sem", "tem"):
            for detector in ("log", "yolo"):
                assert find(modality, detector, "detect") is not None
                assert find(modality, detector, "segment") is not None

    def test_only_segment_requires_a_predictor(self) -> None:
        assert {c.mode for c in CAPABILITIES if c.requires_predictor} == {"segment"}

    def test_the_matrix_matches_the_table_in_project_context(self) -> None:
        # PROJECT_CONTEXT §"Execution matrix" documents 13 supported rows.
        # If this number changes, that table needs the same edit — which is the
        # remaining duplication, and it is prose, so it cannot be executed.
        assert len(CAPABILITIES) == 13


class TestValidation:
    @pytest.mark.parametrize(
        ("modality", "detector", "mode", "message"),
        [
            ("sem", "log", "baseline", "mode='baseline' is only supported for AFM data"),
            ("tem", "yolo", "baseline", "mode='baseline' is only supported for AFM data"),
            ("afm", "yolo", "baseline", "mode='baseline' requires detector='log'"),
            ("afm", "sift", "detect", "Unknown detector: 'sift'"),
        ],
    )
    def test_rejections_keep_their_original_wording(
        self, modality: str, detector: str, mode: str, message: str
    ) -> None:
        # The messages are the ones src/pipeline.py raised before M2-T10 moved
        # the rules. Anything matching on the text is unaffected by the move.
        with pytest.raises(ValueError, match=message.replace("(", r"\(")):
            validate_request(modality, detector, mode, has_predictor=False)

    def test_segment_without_a_predictor_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="predictor must be provided"):
            validate_request("afm", "log", "segment", has_predictor=False)
        assert validate_request("afm", "log", "segment", has_predictor=True).mode == "segment"

    def test_a_valid_request_returns_its_row(self) -> None:
        row = validate_request("afm", "log", "baseline", has_predictor=False)
        assert (row.modality, row.detector, row.mode) == ("afm", "log", "baseline")


class TestValidationHappensFirst:
    """D-14: the whole point is *when* the check runs, not that it runs."""

    @staticmethod
    def _afm_data() -> PreprocessingResult:
        z = np.zeros((8, 8), dtype=np.float32)
        return PreprocessingResult(
            z_raw=z,
            z_flat=z,
            z_result=z,
            substrate=z,
            pixel_size_nm=1.0,
            scan_size_nm=8.0,
            sizes={},
            opening_radius=1,
        )

    def test_invalid_requests_are_rejected_before_any_detector_is_built(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nanoscope.application.use_cases import pipeline

        def _explode(*args: object, **kwargs: object) -> None:
            raise AssertionError("a detector was constructed before validation")

        monkeypatch.setattr(pipeline, "LogDetector", _explode)
        monkeypatch.setattr(pipeline, "YoloDetector", _explode)

        # AFM + YOLO + baseline: the exact combination from D-14 that used to burn
        # a full inference pass before raising.
        with pytest.raises(ValueError, match="requires detector='log'"):
            pipeline.run_pipeline(
                self._afm_data(), PipelineConfig(detector="yolo", mode="baseline")
            )

        # And segment-without-predictor, which used to be checked even later —
        # after detection had already finished.
        with pytest.raises(ValueError, match="predictor must be provided"):
            pipeline.run_pipeline(self._afm_data(), PipelineConfig(detector="yolo", mode="segment"))

    def test_sem_data_reaches_validation_with_its_own_modality(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nanoscope.application.use_cases import pipeline

        def _explode(*args: object, **kwargs: object) -> None:
            raise AssertionError("a detector was constructed before validation")

        monkeypatch.setattr(pipeline, "LogDetector", _explode)
        data = MicroscopyData(image=np.zeros((8, 8), np.uint8), nm_per_pixel=1.0, modality="sem")
        with pytest.raises(ValueError, match="only supported for AFM data"):
            pipeline.run_pipeline(data, PipelineConfig(detector="log", mode="baseline"))
