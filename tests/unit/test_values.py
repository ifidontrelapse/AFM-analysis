"""The value objects added in M2-T02.

Only `PixelScale` carries logic — a guard and two conversions. The enums are
tested for the one property the rest of the code will lean on: they compare equal
to the string literals already written across `src/`, which is what makes M2-T10's
adoption possible one call site at a time instead of all at once.
"""

from __future__ import annotations

import pytest

from nanoscope.core.values import DeviceKind, Modality, PixelScale, Polarity


class TestPixelScale:
    def test_converts_length_and_area(self) -> None:
        scale = PixelScale(9.77)
        assert scale.to_nm(3.0) == pytest.approx(29.31)
        # Area scales with the square — the bug this type exists to prevent is
        # someone writing `area_px * nm_per_px`.
        assert scale.area_to_nm2(4.0) == pytest.approx(4.0 * 9.77**2)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
    def test_rejects_a_scale_that_is_not_positive(self, bad: float) -> None:
        # NaN included deliberately: `not (nan > 0)` is True, and a NaN scale
        # would otherwise propagate silently into every measurement.
        with pytest.raises(ValueError, match="pixel scale must be positive"):
            PixelScale(bad)

    def test_is_frozen_and_compared_by_value(self) -> None:
        assert PixelScale(2.0) == PixelScale(2.0)
        with pytest.raises(AttributeError):
            PixelScale(2.0).nm_per_px = 3.0  # type: ignore[misc]


class TestEnums:
    def test_modality_equals_the_literals_already_used_in_src(self) -> None:
        assert Modality.AFM == "afm"
        assert {m.value for m in Modality} == {"afm", "sem", "tem"}
        assert f"{Modality.SEM}" == "sem"

    def test_polarity_names_both_sides(self) -> None:
        assert {p.value for p in Polarity} == {"bright_on_dark", "dark_on_bright"}
        assert Polarity.DARK_ON_BRIGHT != Polarity.BRIGHT_ON_DARK

    def test_device_kinds(self) -> None:
        assert {d.value for d in DeviceKind} == {"cpu", "cuda", "rocm", "mps"}
        assert f"{DeviceKind.CUDA}" == "cuda"
