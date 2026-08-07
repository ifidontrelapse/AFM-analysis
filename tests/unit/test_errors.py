"""One answer to "this input cannot be used" (D-15, ADR-0030).

The audit's table for D-15 is five inputs and five different behaviours, none of
them a typed error naming the offending parameter. The harness's own matrix was
worse: eleven degenerate inputs against five entry points produced `ValueError`,
`TypeError`, `IndexError`, `LinAlgError` and `RuntimeError` — and, for a 1-D
array, a 3-D array, a NaN map and an infinite map, `detect_particles` returned a
clean empty result, so unusable input and an empty sample were the same answer.

These tests pin the three properties that replaced it: every entry point refuses
the same things, every refusal is catchable both as a project error and as the
builtin it replaced, and a valid image is not refused by any of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.application.capabilities import validate_request
from nanoscope.application.use_cases.pipeline import run_pipeline
from nanoscope.core.entities import PipelineConfig
from nanoscope.core.errors import (
    AnalysisFailedError,
    DataFormatError,
    InvalidImageError,
    InvalidInputError,
    InvalidParameterError,
    MissingFileError,
    NanoscopeError,
    UnsupportedRequestError,
)
from nanoscope.core.science.detection.log import (
    detect_particles,
    estimate_log_threshold,
    estimate_log_threshold_adaptive,
)
from nanoscope.core.science.measurement import measure_all_baseline, measure_geometry_from_mask
from nanoscope.core.science.preprocessing import (
    build_substrate_map,
    estimate_radius_otsu,
    estimate_rough_radius,
    flatten_lines,
    flatten_plane,
    get_substrate_map,
)
from nanoscope.infrastructure.storage import load_afm

SIZES = {"radii_px": np.array([2.0, 4.0])}


def _valid(size: int = 32) -> np.ndarray:
    """A small, ordinary height map: finite, 2-D, float32, with structure."""
    ys, xs = np.mgrid[0:size, 0:size]
    z = np.zeros((size, size), dtype=np.float32)
    for cy, cx in ((8, 8), (8, 24), (24, 8), (24, 24)):
        z += 5.0 * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * 3.0**2))
    return z


#: Every entry point that takes a height map, with the rest of its arguments
#: filled in. The point of the parametrization is that the list is complete:
#: a new entry point added without validation shows up as a missing row.
ENTRY_POINTS = {
    "flatten_plane": lambda z: flatten_plane(z),
    "flatten_lines": lambda z: flatten_lines(z),
    "get_substrate_map": lambda z: get_substrate_map(z, 3),
    "estimate_radius_otsu": lambda z: estimate_radius_otsu(z, 1.0, 1.0),
    "estimate_rough_radius": lambda z: estimate_rough_radius(z, 1.0, 1.0),
    "build_substrate_map": lambda z: build_substrate_map(z, 1.0, min_size_nm=1.0),
    "estimate_log_threshold": lambda z: estimate_log_threshold(z),
    "estimate_log_threshold_adaptive": lambda z: estimate_log_threshold_adaptive(
        z, {"min_sigma": 1.0, "max_sigma": 8.0}
    ),
    "detect_particles": lambda z: detect_particles(z, 1.0, SIZES),
    "measure_all_baseline": lambda z: measure_all_baseline(z, z, np.empty((0, 4))),
}

#: What a height map is not. Each of these produced a different exception type —
#: or no exception — depending on which door it went through.
BAD_IMAGES = {
    "not_an_array": "a string",
    "one_dimensional": np.arange(64, dtype=np.float32),
    "three_dimensional": np.zeros((8, 8, 3), dtype=np.float32),
    "empty": np.zeros((0, 0), dtype=np.float32),
    "with_nan": np.where(np.arange(64).reshape(8, 8) == 3, np.nan, 1.0).astype(np.float32),
    "with_inf": np.where(np.arange(64).reshape(8, 8) == 3, np.inf, 1.0).astype(np.float32),
    "boolean": np.ones((8, 8), dtype=bool),
}


class TestEveryEntryPointRefusesTheSameThings:
    @pytest.mark.parametrize("entry", sorted(ENTRY_POINTS))
    @pytest.mark.parametrize("case", sorted(BAD_IMAGES))
    def test_the_answer_is_one_error_type(self, entry: str, case: str) -> None:
        """The whole task in one assertion: 70 combinations, one answer."""
        with pytest.raises(InvalidImageError):
            ENTRY_POINTS[entry](BAD_IMAGES[case])

    @pytest.mark.parametrize("case", sorted(BAD_IMAGES))
    def test_and_the_message_names_the_parameter(self, case: str) -> None:
        """PROJECT_RULES §3: an error names the offending parameter and its
        value. `z_above` is what `detect_particles` calls its argument, so that
        is the word the message uses — not `z`, and not `array`."""
        with pytest.raises(InvalidImageError, match="z_above"):
            detect_particles(BAD_IMAGES[case], 1.0, SIZES)


class TestTheAuditsTable:
    """D-15's five rows, in the order the audit wrote them."""

    def test_run_pipeline_on_a_string(self) -> None:
        """Was: `AttributeError: 'str' object has no attribute 'image'` — the
        name of a field on the class the caller did not pass."""
        with pytest.raises(InvalidInputError, match="got str"):
            run_pipeline("not-data", PipelineConfig())

    def test_a_three_dimensional_array(self) -> None:
        """Was: `too many values to unpack (expected 2)`, from `h, w = z.shape`
        inside `flatten_plane` — CPython's sentence, in our file, which is the
        exact thing that made the golden interpreter-dependent (B-058)."""
        with pytest.raises(InvalidImageError, match="3 dimensions"):
            flatten_plane(np.zeros((8, 8, 3), dtype=np.float32))

    def test_an_array_containing_nan(self) -> None:
        """Was: `array must not contain infs or NaNs`, from `scipy.lstsq` — a
        true sentence about a matrix the caller never saw."""
        z = _valid()
        z[4, 4] = np.nan

        with pytest.raises(InvalidImageError, match="1 nan"):
            flatten_plane(z)

    def test_a_one_by_one_array(self) -> None:
        """Was: a `ValueError` in Russian, from Otsu. The message is now ours,
        in English, and says what the fit needs — `poly_order + 1` points."""
        with pytest.raises(InvalidParameterError, match="at least 2 points per row"):
            flatten_lines(np.array([[1.0]], dtype=np.float32))

    def test_an_all_zero_array_is_still_not_an_error(self) -> None:
        """Was: "no error; silently returns zero detections" — and it stays that
        way, deliberately. A flat map is *valid data with nothing in it*, which
        ADR-0018 settled; refusing it here would make "no particles" unsayable.
        The taxonomy exists to separate that case from the ones above, not to
        turn every quiet answer into an exception."""
        blobs = detect_particles(np.zeros((64, 64), dtype=np.float32), 1.0, SIZES)

        assert blobs.shape == (0, 4)


class TestTheTaxonomyIsCatchable:
    @pytest.mark.parametrize(
        ("call", "expected", "builtin"),
        [
            (lambda: flatten_plane("nope"), InvalidImageError, ValueError),
            (lambda: flatten_lines(_valid(), poly_order=-1), InvalidParameterError, ValueError),
            (lambda: get_substrate_map(_valid(), -1), InvalidParameterError, ValueError),
            (
                lambda: estimate_radius_otsu(_valid(), 1.0, 10_000.0),
                AnalysisFailedError,
                ValueError,
            ),
            (
                lambda: validate_request("sem", "log", "baseline", False),
                UnsupportedRequestError,
                ValueError,
            ),
            (lambda: load_afm("x.gwy", fmt="gwy"), DataFormatError, ValueError),
            (lambda: run_pipeline(42, PipelineConfig()), InvalidInputError, ValueError),
        ],
        ids=[
            "image",
            "parameter",
            "negative-radius",
            "analysis",
            "capability",
            "format",
            "pipeline-argument",
        ],
    )
    def test_a_caller_catching_the_builtin_still_catches_it(
        self, call, expected: type, builtin: type
    ) -> None:
        """The migration property. Every class inherits the builtin it replaced
        at its site, so the notebooks — the only callers this library has — keep
        working with no edit, and a caller who wants the distinction can ask for
        it."""
        with pytest.raises(expected):
            call()
        with pytest.raises(builtin):
            call()
        with pytest.raises(NanoscopeError):
            call()

    def test_a_missing_file_is_still_a_file_not_found_error(self, tmp_path) -> None:
        """`except FileNotFoundError` is what a caller writes around a loader,
        and it is not a `ValueError`, so the rule is per class rather than one
        base for everything."""
        from nanoscope.infrastructure.storage import load_microscopy_image

        with pytest.raises(MissingFileError) as exc:
            load_microscopy_image(str(tmp_path / "absent.png"), "sem")

        assert isinstance(exc.value, FileNotFoundError)
        assert not isinstance(exc.value, ValueError)

    def test_an_unsupported_request_is_not_an_invalid_input(self) -> None:
        """The distinction the taxonomy exists to make: nothing about
        `(sem, log, baseline)` is malformed. It is a request this version has no
        path for, and a caller may want to offer the user something else rather
        than tell them their data is wrong."""
        with pytest.raises(UnsupportedRequestError) as exc:
            validate_request("sem", "log", "baseline", False)

        assert not isinstance(exc.value, InvalidInputError)


class TestValidInputIsNotRefused:
    @pytest.mark.parametrize("entry", sorted(ENTRY_POINTS))
    def test_an_ordinary_map_goes_through_every_entry_point(self, entry: str) -> None:
        """The other half of the task: validation that rejects real data is a
        worse defect than the one it fixes. Every phantom in the golden is a
        map like this one, which is why none of them moved."""
        ENTRY_POINTS[entry](_valid())

    @pytest.mark.parametrize("dtype", ["uint8", "int32", "float32", "float64"])
    def test_an_integer_image_is_a_height_map(self, dtype: str) -> None:
        """`load_microscopy_image` returns `uint8`. Refusing integers would take
        the SEM/TEM path out of the library one commit after M3-T08 fixed it."""
        flatten_plane((_valid() * 10).astype(dtype))

    def test_a_zero_radius_is_allowed_although_it_is_a_degenerate_opening(self) -> None:
        """`disk(0)` is one pixel, so the opening is the identity and the
        "substrate" comes back equal to the image. It is reachable today — from
        `estimate_rough_radius` on an unscaled noisy scan — and it is what
        ADR-0025 measured and recorded. Refusing it here would move a number,
        and that is not this task's to move. Filed as **B-061**."""
        z = _valid()

        assert np.array_equal(get_substrate_map(z, 0), z)

    def test_an_unknown_scale_is_still_a_state_and_not_a_rejection(self) -> None:
        """ADR-0019/0025: `None` means unknown and is a supported argument.
        `ensure_positive(..., allow_none=True)` is what keeps the validator from
        undoing three tasks' worth of work."""
        z = _valid()

        substrate, _, _, sizes = build_substrate_map(z, None, min_size_nm=1.0)

        assert sizes["radii_nm"] is None
        assert substrate.shape == z.shape


class TestMasksAreCheckedToo:
    def test_a_float_array_is_not_a_mask(self) -> None:
        """`mask.astype(bool)` on a float array is a silent threshold at zero.
        The mirror of refusing a boolean height map, and the same reason."""
        with pytest.raises(InvalidImageError, match="boolean"):
            measure_geometry_from_mask(np.ones((8, 8), dtype=np.float32), 1.0)

    def test_a_real_mask_is_measured(self) -> None:
        mask = np.zeros((16, 16), dtype=bool)
        mask[6:10, 6:10] = True

        assert measure_geometry_from_mask(mask, 2.0)["area_px"] == 16
