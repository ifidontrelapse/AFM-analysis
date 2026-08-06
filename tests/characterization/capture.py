"""
Characterization harness: record what the code does TODAY.

    python tests/characterization/capture.py --write     # regenerate goldens
    python tests/characterization/capture.py             # compare against goldens

This is a safety net for refactoring, not a correctness test. A golden value
that is scientifically wrong is still recorded faithfully — the audit says which
ones are wrong. The rule is: refactors must not move these numbers unless the
change is declared, justified in an ADR, and the golden is updated in the same
commit.

Tolerance policy
----------------
Floats are compared with ``rtol=1e-6, atol=1e-9``. That is far tighter than any
scientifically meaningful difference and is chosen to catch *accidental* change,
not to assert physical accuracy. Counts and error types must match exactly.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import phantoms  # noqa: E402

GOLDEN_DIR = Path(__file__).parent / "golden"
RTOL = 1e-6
ATOL = 1e-9

#: Keys ending in this suffix are recorded and never compared (ADR-0022).
UNCHECKED = "_unchecked"


# ── helpers ───────────────────────────────────────────────────────────────────


def _num(x: Any) -> Any:
    """JSON-safe scalar, preserving nan/inf as sentinel strings."""
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        if math.isnan(f):
            return "__nan__"
        if math.isinf(f):
            return "__inf__" if f > 0 else "__-inf__"
        return f
    return x


def _array_digest(a: np.ndarray, k: int = 6) -> dict:
    """Shape/dtype plus order-independent summary statistics.

    Detection outputs are compared by their sorted distribution rather than by
    row order, so that a refactor which legitimately reorders results (e.g. a
    different iteration order) does not read as a numerical change while a real
    change in the values still does.
    """
    a = np.asarray(a)
    out: dict[str, Any] = {"shape": list(a.shape), "dtype": str(a.dtype), "size": int(a.size)}
    if a.size == 0:
        return out
    flat = a.astype(np.float64).ravel()
    finite = flat[np.isfinite(flat)]
    out["n_nonfinite"] = int(flat.size - finite.size)
    if finite.size == 0:
        return out
    out["min"] = _num(finite.min())
    out["max"] = _num(finite.max())
    out["mean"] = _num(finite.mean())
    out["std"] = _num(finite.std())
    out["sum"] = _num(finite.sum())
    qs = np.percentile(finite, [10, 25, 50, 75, 90])
    out["percentiles_10_25_50_75_90"] = [_num(q) for q in qs]
    if finite.size <= k:
        out["values_sorted"] = [_num(v) for v in np.sort(finite)]
    return out


def _df_digest(df) -> dict:
    out = {"n_rows": len(df), "columns": sorted(map(str, df.columns))}
    for col in sorted(df.columns):
        s = df[col]
        if s.dtype.kind in "ifb":
            out[f"col::{col}"] = _array_digest(s.to_numpy())
        else:
            vals, counts = np.unique(s.astype(str).to_numpy(), return_counts=True)
            out[f"col::{col}"] = {
                "value_counts": dict(zip(map(str, vals), map(int, counts), strict=True))
            }
    return out


@contextlib.contextmanager
def _quiet():
    """The library prints progress to stdout; keep goldens clean."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


def _we_wrote_this_message(frame: traceback.FrameSummary) -> bool:
    """Did *this project* author the exception text (ADR-0022 / B-058)?

    Two signals, and both are needed. The frame must be inside `nanoscope`, or
    the wording belongs to numpy, scipy or scikit-image — `Only 2-D and 3-D
    images supported.` is skimage's sentence, not ours. And the raising line
    must be an explicit `raise`, or the wording belongs to CPython: `h, w =
    z.shape` in our own file produces *its* "too many values to unpack", which
    3.14 reworded and CI reported as characterization drift (M1-T08).
    """
    return "nanoscope" in Path(frame.filename).parts and "raise " in (frame.line or "")


def _record(fn, *args, **kwargs) -> dict:
    """Run fn, capturing either its digest-able result or the exception it
    raises. Recording the *error* is as important as recording the value:
    several inputs are supposed to fail, and how they fail is part of the
    contract we are about to change.

    The exception *type* and the function it came out of are always compared.
    The *message* is compared only when we wrote it; a message written by CPython
    or by a library is recorded under `..._unchecked`, which `compare` skips —
    it is evidence for a reader, not a promise about somebody else's wording
    (ADR-0022).
    """
    try:
        with np.errstate(all="ignore"), _quiet() as buf:
            value = fn(*args, **kwargs)
        return {"ok": True, "value": value, "stdout_lines": len(buf.getvalue().splitlines())}
    except Exception as exc:
        frame = traceback.extract_tb(exc.__traceback__)[-1]
        key = "error_message" if _we_wrote_this_message(frame) else "error_message" + UNCHECKED
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            key: str(exc)[:300],
            "raised_in": frame.name,
        }


# ── per-stage capture ─────────────────────────────────────────────────────────


def capture_preprocessing(ph: phantoms.Phantom) -> dict:
    from nanoscope.core.science.preprocessing import (
        build_substrate_map,
        estimate_radius_otsu,
        flatten_lines,
        flatten_plane,
    )

    out: dict[str, Any] = {}

    r = _record(flatten_plane, ph.image)
    out["flatten_plane"] = {**r, "value": _array_digest(r["value"]) if r["ok"] else None}
    if not r["ok"]:
        return out
    z_plane = r["value"]

    r = _record(flatten_lines, z_plane)
    out["flatten_lines"] = {**r, "value": _array_digest(r["value"]) if r["ok"] else None}
    if not r["ok"]:
        return out
    z_flat = r["value"]

    # Idempotence probe: flattening an already-flat map should be a no-op.
    r2 = _record(flatten_lines, z_flat)
    out["flatten_lines_idempotence_max_abs_delta"] = (
        _num(np.abs(r2["value"] - z_flat).max()) if r2["ok"] else None
    )

    r = _record(build_substrate_map, z_flat, ph.pixel_size_nm)
    if r["ok"]:
        substrate, z_above, opening_radius, sizes = r["value"]
        out["build_substrate_map"] = {
            "ok": True,
            "substrate": _array_digest(substrate),
            "z_above": _array_digest(z_above),
            "opening_radius": _num(opening_radius),
            "opening_radius_type": type(opening_radius).__name__,
            "sizes": {
                "typical_radius_px": _num(sizes["typical_radius_px"]),
                "typical_radius_nm": _num(sizes["typical_radius_nm"]),
                "n_objects_reported": _num(sizes["n_objects"]),
                "n_radii_kept": len(sizes["radii_px"]),
                "otsu_threshold": _num(sizes["otsu_threshold"]),
                "radii_px": _array_digest(sizes["radii_px"]),
            },
            # D-04's measuring stick (ADR-0024). `min_size_px_floored` is what
            # the code used to compute — 0 whenever the scan is coarser than the
            # minimum — beside the physical threshold that replaced it.
            "min_size_nm_used": 5,
            "min_size_px_equivalent": _num(5 / ph.pixel_size_nm),
            "min_size_px_floored": int(5 / ph.pixel_size_nm),
        }
    else:
        out["build_substrate_map"] = r

    # The manual-radius branch raised UnboundLocalError on 100% of calls until
    # M3-T01 (D-01, ADR-0014). It returns now, so record what it returns —
    # otherwise fixing the defect would leave this branch less characterized than
    # it was while broken.
    rm = _record(build_substrate_map, z_flat, ph.pixel_size_nm, 5, 15)
    if rm["ok"]:
        substrate_m, z_above_m, opening_radius_m, sizes_m = rm["value"]
        out["build_substrate_map_manual"] = {
            "ok": True,
            "substrate": _array_digest(substrate_m),
            "z_above": _array_digest(z_above_m),
            "opening_radius": _num(opening_radius_m),
            "opening_radius_type": type(opening_radius_m).__name__,
            "sizes": {
                "typical_radius_px": _num(sizes_m["typical_radius_px"]),
                "n_radii_kept": len(sizes_m["radii_px"]),
                "otsu_threshold": _num(sizes_m["otsu_threshold"]),
            },
            "manual_radius_px_requested": 15,
        }
    else:
        out["build_substrate_map_manual"] = rm

    # M3-T20 / ADR-0025: the same call with no scale at all — reachable from
    # `load_afm(fmt="npy")` since the fabricated 1.0 nm/px was removed. The
    # pixel-space fields must equal the scaled run's; only the `_nm` ones go
    # absent, and the size filter is skipped because it cannot be expressed.
    rn = _record(build_substrate_map, z_flat, None)
    if rn["ok"]:
        substrate_n, z_above_n, opening_radius_n, sizes_n = rn["value"]
        out["build_substrate_map_no_scale"] = {
            "ok": True,
            "substrate": _array_digest(substrate_n),
            "z_above": _array_digest(z_above_n),
            "opening_radius": _num(opening_radius_n),
            "sizes": {
                "typical_radius_px": _num(sizes_n["typical_radius_px"]),
                "typical_radius_nm": _num(sizes_n["typical_radius_nm"]),
                "n_objects_reported": _num(sizes_n["n_objects"]),
                "n_radii_kept": len(sizes_n["radii_px"]),
                "otsu_threshold": _num(sizes_n["otsu_threshold"]),
                "radii_px": _array_digest(sizes_n["radii_px"]),
                "radii_nm_is_none": sizes_n["radii_nm"] is None,
            },
        }
    else:
        out["build_substrate_map_no_scale"] = rn

    # D-05's own reproduction: a size filter no object can pass. This returned
    # `{"typical_radius_px": nan, ...}` until M3-T06 (ADR-0017) and the nan
    # surfaced several calls later, inside estimate_log_params, as "zero-size
    # array to reduction operation minimum". It raises here now, and the golden
    # records which — the failure is part of the contract.
    out["estimate_radius_otsu_all_filtered"] = _record(
        estimate_radius_otsu, z_flat, ph.pixel_size_nm, 500
    )
    return out


def capture_log_detection(ph: phantoms.Phantom) -> dict:
    from nanoscope.core.science.detection.log import (
        detect_particles,
        estimate_log_params,
        estimate_log_threshold,
        estimate_log_threshold_adaptive,
    )
    from nanoscope.core.science.preprocessing import (
        build_substrate_map,
        flatten_lines,
        flatten_plane,
    )

    out: dict[str, Any] = {}
    pre = _record(build_substrate_map, flatten_lines(flatten_plane(ph.image)), ph.pixel_size_nm)
    if not pre["ok"]:
        return {"preprocessing_failed": pre}
    _, z_above, _, sizes = pre["value"]

    r = _record(estimate_log_params, sizes)
    out["estimate_log_params"] = (
        {"ok": True, **{k: _num(v) for k, v in r["value"].items()}} if r["ok"] else r
    )

    out["estimate_log_threshold_static"] = {
        k: _num(v) for k, v in [("value", _record(estimate_log_threshold, z_above).get("value"))]
    }

    if r["ok"]:
        ra = _record(estimate_log_threshold_adaptive, z_above, r["value"], 20.0)
        out["estimate_log_threshold_adaptive_p20"] = (
            {"ok": True, "value": _num(ra["value"])} if ra["ok"] else ra
        )

    for pct in (10.0, 20.0, 40.0):
        rb = _record(detect_particles, z_above, ph.pixel_size_nm, sizes, 0.3, None, pct)
        key = f"detect_particles_p{int(pct)}"
        if rb["ok"]:
            blobs = rb["value"]
            out[key] = {
                "ok": True,
                "n_blobs": len(blobs),
                "n_true_particles": ph.n_particles,
                "y_px": _array_digest(blobs[:, 0]),
                "x_px": _array_digest(blobs[:, 1]),
                "sigma_px": _array_digest(blobs[:, 2]),
                "radius_nm": _array_digest(blobs[:, 3]),
            }
        else:
            out[key] = rb

    # D-07 (ADR-0019): an unknown pixel scale is a supported state, and before
    # M3-T11 this call raised `TypeError: unsupported operand type(s) for *`.
    # Recording it takes the invariant "no scale, no nanometres" out of prose.
    from nanoscope.core.science.detection import BaseDetector

    rn = _record(detect_particles, z_above, None, sizes, 0.3, None, 20.0)
    if rn["ok"]:
        dets = BaseDetector._blobs_to_detections(rn["value"])
        out["detect_particles_no_scale"] = {
            "ok": True,
            "n_blobs": len(rn["value"]),
            "sigma_px": _array_digest(rn["value"][:, 2]),
            "radius_nm": _array_digest(rn["value"][:, 3]),
            "n_detections": len(dets),
            "n_detections_with_radius_nm": sum(d.radius_nm is not None for d in dets),
        }
    else:
        out["detect_particles_no_scale"] = rn
    return out


def capture_baseline_measurement(ph: phantoms.Phantom) -> dict:
    from nanoscope.core.science.detection.log import detect_particles
    from nanoscope.core.science.measurement import measure_all_baseline
    from nanoscope.core.science.preprocessing import (
        build_substrate_map,
        flatten_lines,
        flatten_plane,
    )

    pre = _record(build_substrate_map, flatten_lines(flatten_plane(ph.image)), ph.pixel_size_nm)
    if not pre["ok"]:
        return {"preprocessing_failed": pre}
    _, z_above, _, sizes = pre["value"]
    z_flat = flatten_lines(flatten_plane(ph.image))

    det = _record(detect_particles, z_above, ph.pixel_size_nm, sizes, 0.3, None, 20.0)
    if not det["ok"]:
        return {"detection_failed": det}
    blobs = det["value"]

    out: dict[str, Any] = {"n_blobs_in": len(blobs)}
    r = _record(measure_all_baseline, z_flat, z_above, blobs)
    out["measure_all_baseline"] = {"ok": True, **_df_digest(r["value"])} if r["ok"] else r

    # Constant height offset invariance: heights must not move when the whole
    # map is shifted. Recorded now so the property test in Phase 5 has a datum.
    r2 = _record(measure_all_baseline, z_flat + 100.0, z_above, blobs)
    if r["ok"] and r2["ok"] and len(r["value"]) and "height_nm" in r["value"]:
        d = np.abs(r2["value"]["height_nm"].to_numpy() - r["value"]["height_nm"].to_numpy()).max()
        out["height_invariance_under_100nm_offset_max_delta"] = _num(d)

    out["measure_all_baseline_empty_blobs"] = (
        lambda rr: {"ok": True, **_df_digest(rr["value"])} if rr["ok"] else rr
    )(_record(measure_all_baseline, z_flat, z_above, np.empty((0, 4))))
    return out


def capture_yolo_preprocessing(ph: phantoms.Phantom) -> dict:
    """Image preparation only — no weights, no inference, CPU-safe.

    `correct` below is the normalise-then-cast reference this block was written
    to measure the distance from (D-03). M3-T03 made `_prepare_image` equal to
    it, so the two derived numbers are now an invariant rather than a defect
    size: `mean_abs_diff_vs_normalize_first` is 0.0 and the two level counts
    agree. Reordering the cast moves both, on all 7 phantoms.

    Every phantom is square, so the letterbox of M3-T04 (D-21) leaves all of the
    above byte-identical — which is exactly why `non_square_half_height` exists.
    It prepares the top half of the same image, the cheapest way to characterize
    the aspect-ratio path without inventing a phantom; the old code squashed it
    2:1, the letterbox pads it instead.
    """
    import cv2

    from nanoscope.core.values import Polarity, default_polarity
    from nanoscope.infrastructure.models import YoloDetector

    det = YoloDetector.__new__(YoloDetector)
    det.yolo_size = 640
    # `__new__` skips `__init__`, so the polarity has to be set by hand — and it
    # is now what decides whether `_prepare_image` inverts (ADR-0023).
    det.polarity = default_polarity(ph.name.split("_")[0])
    r = _record(det._prepare_image, ph.image)
    if not r["ok"]:
        return r
    current = r["value"]
    normalised = cv2.normalize(
        cv2.resize(ph.image, (640, 640)), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    if det.polarity is Polarity.BRIGHT_ON_DARK:
        normalised = cv2.bitwise_not(normalised)
    correct = cv2.cvtColor(normalised, cv2.COLOR_GRAY2RGB)
    out = {
        "ok": True,
        "current": _array_digest(current[:, :, 0]),
        "current_unique_levels": int(np.unique(current[:, :, 0]).size),
        "normalize_first_unique_levels": int(np.unique(correct[:, :, 0]).size),
        "mean_abs_diff_vs_normalize_first": _num(
            np.abs(current[:, :, 0].astype(int) - correct[:, :, 0].astype(int)).mean()
        ),
    }

    r = _record(det._prepare_image, ph.image[: ph.image.shape[0] // 2])
    if r["ok"]:
        grey = r["value"][:, :, 0]
        rows_of_border = int((grey == 255).all(axis=1).sum())
        out["non_square_half_height"] = {
            "ok": True,
            "grey": _array_digest(grey),
            "unique_levels": int(np.unique(grey).size),
            "fully_255_rows": rows_of_border,
        }
    else:
        out["non_square_half_height"] = r

    # The YOLO half of D-07 (ADR-0019). `_boxes_to_detections` needs no weights,
    # so the box → Detection conversion is recordable even though inference is
    # not: with a scale, and with the scale unknown, which used to raise.
    boxes = np.array([[10.0, 10.0, 30.0, 34.0], [100.0, 100.0, 140.0, 138.0]])
    for label, scale in (("scaled", ph.pixel_size_nm), ("no_scale", None)):
        rb = _record(YoloDetector._boxes_to_detections, boxes, scale)
        out[f"boxes_to_detections_{label}"] = (
            {
                "ok": True,
                "radius_px": [_num(d.radius_px) for d in rb["value"]],
                "radius_nm": [_num(d.radius_nm) for d in rb["value"]],
                # D-09 (ADR-0028): with no scores passed, none are reported.
                # This used to read [1.0, 1.0] — the dataclass default, which is
                # what every YOLO detection carried.
                "confidence": [_num(d.confidence) for d in rb["value"]],
            }
            if rb["ok"]
            else rb
        )

    # The scores the model actually produces, converted at the same seam. 0.0 is
    # included on purpose: it is falsy, and a `confidence or 1.0` phrasing would
    # erase exactly the least confident detection.
    rc = _record(YoloDetector._boxes_to_detections, boxes, ph.pixel_size_nm, np.array([0.91, 0.0]))
    out["boxes_to_detections_with_scores"] = (
        {"ok": True, "confidence": [_num(d.confidence) for d in rc["value"]]} if rc["ok"] else rc
    )

    # Length mismatch: an error rather than a `zip` that drops the tail.
    out["boxes_to_detections_confidence_mismatch"] = _record(
        YoloDetector._boxes_to_detections, boxes, ph.pixel_size_nm, np.array([0.9])
    )
    return out


def capture_degenerate() -> dict:
    from nanoscope.core.science.detection.log import (
        detect_particles,
        estimate_log_threshold_adaptive,
    )
    from nanoscope.core.science.preprocessing import (
        build_substrate_map,
        flatten_lines,
        flatten_plane,
    )

    out: dict[str, Any] = {}
    for name, arr in phantoms.degenerate_inputs().items():
        entry: dict[str, Any] = {}
        for label, fn, args in [
            ("flatten_plane", flatten_plane, (arr,)),
            ("flatten_lines", flatten_lines, (arr,)),
            ("build_substrate_map", build_substrate_map, (arr, 1.0)),
            (
                "detect_particles",
                detect_particles,
                (arr, 1.0, {"radii_px": np.array([2.0, 4.0])}, 0.3, 0.1, 20.0),
            ),
            # D-11 lives here, not in detect_particles: dividing by a
            # non-positive maximum produced a threshold outside [0, 1] — 2.4997
            # on `negative_with_structure` — while detect_particles returned an
            # empty array either way and so looked innocent (ADR-0018).
            (
                "estimate_log_threshold_adaptive",
                estimate_log_threshold_adaptive,
                (arr, {"min_sigma": 1.0, "max_sigma": 8.0}, 20.0),
            ),
        ]:
            r = _record(fn, *args)
            if not r["ok"]:
                entry[label] = r
            elif isinstance(r["value"], np.ndarray):
                entry[label] = {"ok": True, "result": _array_digest(r["value"])}
            elif isinstance(r["value"], (int, float, np.number)):
                # Scalars used to be recorded as the string "non-array", which
                # is how a threshold of 2.4997 stayed invisible (D-11).
                entry[label] = {"ok": True, "result": _num(r["value"])}
            else:
                entry[label] = {"ok": True, "result": "non-array"}
        out[name] = entry
    return out


def capture_contracts() -> dict:
    """Serialization boundary: what a naive JSON encode does to a result."""
    import dataclasses

    import pandas as pd

    from nanoscope.core.entities import Detection, PipelineConfig, PipelineResult

    det = Detection(x_px=1.0, y_px=2.0, radius_px=3.0, radius_nm=4.0)
    res = PipelineResult(
        detections=[det],
        masks=[{"mask": np.zeros((4, 4), bool), "score": 0.9}],
        measurements=pd.DataFrame([{"x_px": 1.0}]),
        pixel_size_nm=2.0,
        detector_name="log",
        mode="segment",
        modality="afm",
    )
    try:
        json.dumps(dataclasses.asdict(res))
        serializable = True
        err = None
    except Exception as exc:
        serializable = False
        err = f"{type(exc).__name__}: {str(exc)[:120]}"
    return {
        "default_detection_bbox": list(det.bbox),
        "default_detection_bbox_len": len(det.bbox),
        # Was 1.0 until M3-T05 (D-09, ADR-0028): a detector that computes no
        # score reports none, rather than reporting certainty.
        "default_detection_confidence": _num(det.confidence),
        "pipeline_result_json_serializable": serializable,
        "pipeline_result_json_error": err,
        "config_fields": sorted(f.name for f in dataclasses.fields(PipelineConfig)),
        "result_fields": sorted(f.name for f in dataclasses.fields(PipelineResult)),
    }


# ── driver ────────────────────────────────────────────────────────────────────


def build_all() -> dict:
    snapshot: dict[str, Any] = {
        "_meta": {
            "purpose": "Characterization baseline — records CURRENT behaviour, not correct behaviour.",
            "tolerance": {"rtol": RTOL, "atol": ATOL},
            "numpy": np.__version__,
            "python": ".".join(map(str, sys.version_info[:3])),
        }
    }
    import scipy
    import skimage

    snapshot["_meta"]["scikit_image"] = skimage.__version__
    snapshot["_meta"]["scipy"] = scipy.__version__

    for factory in phantoms.ALL_AFM_PHANTOMS:
        ph = factory()
        snapshot[ph.name] = {
            "ground_truth": {
                "n_particles": ph.n_particles,
                "pixel_size_nm": ph.pixel_size_nm,
                "scan_size_nm": ph.scan_size_nm,
                "radii_px": _array_digest(ph.radii_px),
                "heights_nm": _array_digest(ph.heights_nm),
            },
            "image": _array_digest(ph.image),
            "preprocessing": capture_preprocessing(ph),
            "log_detection": capture_log_detection(ph),
            "baseline_measurement": capture_baseline_measurement(ph),
            "yolo_input_preparation": capture_yolo_preprocessing(ph),
        }

    for factory in phantoms.ALL_IMAGE_PHANTOMS:
        ph = factory()
        snapshot[ph.name] = {
            "ground_truth": {
                "n_particles": ph.n_particles,
                "pixel_size_nm": ph.pixel_size_nm,
                "radii_px": _array_digest(ph.radii_px),
            },
            "image": _array_digest(ph.image),
            # SEM/TEM enter run_pipeline with the RAW image as the detector input.
            "log_detection_on_raw_image": _record(_log_on_raw, ph),
            "yolo_input_preparation": capture_yolo_preprocessing(ph),
        }
        r = snapshot[ph.name]["log_detection_on_raw_image"]
        if r.get("ok"):
            r["value"] = _array_digest(np.asarray(r["value"]))

    snapshot["degenerate_inputs"] = capture_degenerate()
    snapshot["contracts"] = capture_contracts()
    return snapshot


def _log_on_raw(ph: phantoms.Phantom) -> np.ndarray:
    """Reproduce exactly what run_pipeline does for SEM/TEM: hand the raw image
    to LogDetector with sizes=None — and, since M3-T10, with the polarity the
    modality implies (ADR-0023). Without that argument this reproduction would
    stop reproducing the pipeline the moment D-12 was fixed."""
    from nanoscope.core.science.detection import LogDetector
    from nanoscope.core.values import default_polarity

    modality = ph.name.split("_")[0]
    d = LogDetector(
        overlap=0.3, percentile=20.0, threshold=None, polarity=default_polarity(modality)
    )
    d.detect(ph.image, ph.pixel_size_nm, sizes=None)
    return d.last_blobs


def compare(new: Any, old: Any, path: str, diffs: list[str]) -> None:
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(set(old) | set(new)):
            if k.endswith(UNCHECKED):
                continue  # recorded for the reader, never a promise (ADR-0022)
            if k not in old:
                diffs.append(f"{path}.{k}: ADDED")
            elif k not in new:
                diffs.append(f"{path}.{k}: REMOVED")
            else:
                compare(new[k], old[k], f"{path}.{k}", diffs)
        return
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            diffs.append(f"{path}: length {len(old)} -> {len(new)}")
            return
        for i, (a, b) in enumerate(zip(new, old, strict=True)):
            compare(a, b, f"{path}[{i}]", diffs)
        return
    if (
        isinstance(old, (int, float))
        and isinstance(new, (int, float))
        and not isinstance(old, bool)
    ):
        if not math.isclose(float(new), float(old), rel_tol=RTOL, abs_tol=ATOL):
            diffs.append(f"{path}: {old!r} -> {new!r}")
        return
    if old != new:
        diffs.append(f"{path}: {old!r} -> {new!r}")


def diff_against_golden() -> list[str]:
    """Capture the pipeline again and compare it with the committed golden file.

    The callable seam between "measure" and "print/exit", so that the same
    comparison can be driven from the CLI below and from
    ``tests/characterization/test_golden.py``. It performs no I/O beyond reading
    the golden file and never rewrites it.

    Returns:
        One path-addressed string per difference, in the same format the CLI
        prints (``group.stage.quantity: old -> new``). Empty when stable.

    Raises:
        FileNotFoundError: if the golden file has not been generated yet.
    """
    old = json.loads((GOLDEN_DIR / "baseline.json").read_text())
    snapshot = build_all()
    diffs: list[str] = []
    for key in sorted(set(old) | set(snapshot)):
        if key == "_meta":
            continue
        compare(snapshot.get(key), old.get(key), key, diffs)
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="regenerate the golden file")
    args = ap.parse_args()

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden_path = GOLDEN_DIR / "baseline.json"

    if args.write or not golden_path.exists():
        golden_path.write_text(json.dumps(build_all(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {golden_path.relative_to(REPO_ROOT)}")
        return 0

    diffs = diff_against_golden()

    if diffs:
        print(f"CHARACTERIZATION DRIFT: {len(diffs)} difference(s)\n")
        for d in diffs[:80]:
            print(f"  {d}")
        if len(diffs) > 80:
            print(f"  ... and {len(diffs) - 80} more")
        return 1
    n_groups = len(json.loads(golden_path.read_text())) - 1  # minus `_meta`
    print(f"characterization baseline stable ({n_groups} groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
