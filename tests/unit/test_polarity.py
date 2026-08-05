"""Which way the particles read is configured, not guessed (D-12, ADR-0023, B3).

Both detectors kept the bright side of the image unconditionally. TEM particles
are conventionally dark on bright, so on TEM the detector characterised the
background and returned **0 of 22** on the audit's phantom.

B3 answered: polarity is a setting with a per-modality default. These tests pin
the default, the inversion, and the override — and, most importantly, that the
bright-on-dark path is untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.core.entities import MicroscopyData, PipelineConfig
from nanoscope.core.science.detection import LogDetector
from nanoscope.core.values import Modality, Polarity, default_polarity

SIZES = {"radii_px": np.array([3.0, 5.0])}


def _bright_on_dark(size: int = 64) -> np.ndarray:
    """Four bright caps on a dark background — the AFM/SEM convention."""
    ys, xs = np.mgrid[0:size, 0:size]
    z = np.zeros((size, size), dtype=np.float32)
    for cy, cx in ((16, 16), (16, 48), (48, 16), (48, 48)):
        z += 6.0 * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * 4.0**2))
    return z


def _dark_on_bright(size: int = 64) -> np.ndarray:
    """The same four particles, imaged the TEM way."""
    return _bright_on_dark(size).max() - _bright_on_dark(size)


class TestTheDefault:
    @pytest.mark.parametrize(
        ("modality", "expected"),
        [
            ("afm", Polarity.BRIGHT_ON_DARK),
            ("sem", Polarity.BRIGHT_ON_DARK),
            ("tem", Polarity.DARK_ON_BRIGHT),
        ],
    )
    def test_each_instrument_gets_its_convention(self, modality: str, expected: Polarity) -> None:
        assert default_polarity(modality) is expected

    def test_every_modality_has_an_entry(self) -> None:
        """No silent fallback: a missing entry must be a loud mistake, not a
        guess that reads as 'the detector found nothing'."""
        for modality in Modality:
            assert default_polarity(modality) in tuple(Polarity)

    def test_an_unknown_modality_raises(self) -> None:
        with pytest.raises(ValueError, match="not a valid Modality"):
            default_polarity("xrd")


class TestTheDetector:
    def test_dark_particles_are_found_when_the_detector_is_told(self) -> None:
        """The reproduction, in miniature: the same four particles, imaged the
        other way round."""
        dets = LogDetector(threshold=0.05, polarity=Polarity.DARK_ON_BRIGHT).detect(
            _dark_on_bright(), 1.0, sizes=SIZES
        )
        assert len(dets) == 4

    def test_and_are_missed_when_it_is_not(self) -> None:
        """D-12 itself, kept as a test so the defect cannot come back quietly.

        On the audit's 22-particle TEM phantom the wrong polarity finds **0**.
        On four caps in the corners of a 64 px square it finds two blobs — the
        *gaps between* the particles, which is the same failure wearing a
        smaller image: what it locates is the background, so the assertion is
        about where the detections are, not how many."""
        dets = LogDetector(threshold=0.05, polarity=Polarity.BRIGHT_ON_DARK).detect(
            _dark_on_bright(), 1.0, sizes=SIZES
        )
        centres = [(16, 16), (16, 48), (48, 16), (48, 48)]
        for d in dets:
            assert all(abs(d.x_px - cx) + abs(d.y_px - cy) > 5 for cy, cx in centres)

    def test_the_bright_path_is_untouched(self) -> None:
        """Everything AFM depends on runs through this branch."""
        before = LogDetector(threshold=0.05).detect(_bright_on_dark(), 1.0, sizes=SIZES)
        assert len(before) == 4
        assert LogDetector().polarity is Polarity.BRIGHT_ON_DARK

    def test_inverting_twice_returns_the_same_detections(self) -> None:
        """`max - z` is its own inverse, so a dark-on-bright detector on an
        inverted image must agree with a bright-on-dark one on the original —
        the property that makes one inversion at the entrance sufficient."""
        bright = LogDetector(threshold=0.05, polarity=Polarity.BRIGHT_ON_DARK).detect(
            _bright_on_dark(), 1.0, sizes=SIZES
        )
        dark = LogDetector(threshold=0.05, polarity=Polarity.DARK_ON_BRIGHT).detect(
            _dark_on_bright(), 1.0, sizes=SIZES
        )
        assert [(d.x_px, d.y_px) for d in bright] == [(d.x_px, d.y_px) for d in dark]


class TestThePipeline:
    def test_tem_no_longer_finds_the_background(self) -> None:
        from nanoscope.application.use_cases import run_pipeline

        data = MicroscopyData(image=_dark_on_bright(), nm_per_pixel=1.0, modality="tem")
        # An explicit threshold, so the test fails for D-12's reason and not
        # because the adaptive one found nothing in a four-cap toy image.
        cfg = PipelineConfig(detector="log", mode="detect", log_threshold=0.05)
        assert len(run_pipeline(data, cfg).detections) == 4

    def test_an_explicit_polarity_overrides_the_modality(self) -> None:
        """A TEM image that breaks the convention — stained the other way, or
        already inverted by the acquisition software. The operator says so."""
        from nanoscope.application.use_cases import run_pipeline

        data = MicroscopyData(image=_bright_on_dark(), nm_per_pixel=1.0, modality="tem")
        default_run = run_pipeline(
            data, PipelineConfig(detector="log", mode="detect", log_threshold=0.05)
        )
        override = run_pipeline(
            data,
            PipelineConfig(
                detector="log",
                mode="detect",
                log_threshold=0.05,
                polarity=Polarity.BRIGHT_ON_DARK,
            ),
        )
        assert len(override.detections) == 4
        assert len(default_run.detections) != 4


class TestTheYoloInput:
    """The model was trained on inverted AFM height maps, so it looks for *dark*
    particles. Inverting a TEM image hands it the background."""

    def test_a_bright_on_dark_image_is_inverted(self) -> None:
        from nanoscope.infrastructure.models import YoloDetector

        det = YoloDetector(polarity=Polarity.BRIGHT_ON_DARK)
        prepared = det._prepare_image(_bright_on_dark())
        # The particles occupied the high end; after inversion they are the low one.
        assert prepared[:, :, 0].mean() > 127

    def test_a_dark_on_bright_image_is_not(self) -> None:
        from nanoscope.infrastructure.models import YoloDetector

        det = YoloDetector(polarity=Polarity.DARK_ON_BRIGHT)
        prepared = det._prepare_image(_dark_on_bright())
        assert prepared[:, :, 0].mean() > 127

    def test_both_polarities_give_the_model_the_same_picture(self) -> None:
        """The point of the whole exercise: however the sample was imaged, the
        network sees dark particles on a bright field."""
        from nanoscope.infrastructure.models import YoloDetector

        bright = YoloDetector(polarity=Polarity.BRIGHT_ON_DARK)._prepare_image(_bright_on_dark())
        dark = YoloDetector(polarity=Polarity.DARK_ON_BRIGHT)._prepare_image(_dark_on_bright())
        # Within one grey level: `bitwise_not` is `255 - x` on the *rounded*
        # value, while inverting before the min-max stretch rounds afterwards.
        assert np.abs(bright.astype(int) - dark.astype(int)).max() <= 1
