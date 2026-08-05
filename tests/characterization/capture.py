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


def _record(fn, *args, **kwargs) -> dict:
    """Run fn, capturing either its digest-able result or the exception it
    raises. Recording the *error* is as important as recording the value:
    several inputs are supposed to fail, and how they fail is part of the
    contract we are about to change."""
    try:
        with np.errstate(all="ignore"), _quiet() as buf:
            value = fn(*args, **kwargs)
        return {"ok": True, "value": value, "stdout_lines": len(buf.getvalue().splitlines())}
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:300],
            "raised_in": traceback.extract_tb(exc.__traceback__)[-1].name,
        }


# ── per-stage capture ─────────────────────────────────────────────────────────


def capture_preprocessing(ph: phantoms.Phantom) -> dict:
    from nanoscope.core.science.preprocessing import (
        build_substrate_map,
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
            "min_size_pixel_used": int(5 / ph.pixel_size_nm),
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

    from nanoscope.infrastructure.models import YoloDetector

    det = YoloDetector.__new__(YoloDetector)
    det.yolo_size = 640
    r = _record(det._prepare_image, ph.image)
    if not r["ok"]:
        return r
    current = r["value"]
    correct = cv2.cvtColor(
        cv2.bitwise_not(
            cv2.normalize(cv2.resize(ph.image, (640, 640)), None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )
        ),
        cv2.COLOR_GRAY2RGB,
    )
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
    return out


def capture_degenerate() -> dict:
    from nanoscope.core.science.detection.log import detect_particles
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
        ]:
            r = _record(fn, *args)
            entry[label] = (
                {"ok": True, "result": _array_digest(r["value"])}
                if r["ok"] and isinstance(r["value"], np.ndarray)
                else ({"ok": True, "result": "non-array"} if r["ok"] else r)
            )
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
    to LogDetector with sizes=None."""
    from nanoscope.core.science.detection import LogDetector

    d = LogDetector(overlap=0.3, percentile=20.0, threshold=None)
    d.detect(ph.image, ph.pixel_size_nm, sizes=None)
    return d.last_blobs


def compare(new: Any, old: Any, path: str, diffs: list[str]) -> None:
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(set(old) | set(new)):
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
