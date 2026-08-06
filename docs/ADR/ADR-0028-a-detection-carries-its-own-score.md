# ADR-0028 — A detection carries its own score, or none

- **Status:** Accepted
- **Date:** 2026-08-06
- **Affects:** `nanoscope/core/entities/detection.py`,
  `nanoscope/infrastructure/models/yolo.py`,
  `nanoscope/core/science/detection/base.py` · audit **D-09** · M3-T05
- **Numerical impact:** **29 golden keys added, 0 values changed.** Inference is outside the
  gate, so nothing recorded could move; what is new is the conversion seam's scores — and the
  fact that `Detection.confidence` had never been recorded at all.

## Context

```python
Detection(x_px=cx, y_px=cy, radius_px=radius_px, radius_nm=..., bbox=...)
#                                             confidence never assigned -> 1.0
```

The model scores every box. `cfg.yolo_conf` *filters* on those scores. And then the conversion to
entities dropped them, so every YOLO detection reported **1.0** — including a box that had only
just cleared the threshold. A reader sorting detections by confidence got the input order; a
reader filtering on it got everything.

The default was the real problem. `confidence: float = 1.0` is a **substitute value**, and this
milestone has now spent four ADRs removing substitute values: a fabricated pixel scale
(ADR-0019, ADR-0025, ADR-0026), a fabricated minimum particle size (ADR-0024), a fabricated empty
table (ADR-0027). Each time the argument was the same — a value that is indistinguishable from a
measured one, standing where nothing was measured.

It applies to the LoG detector too, and that is the part the audit did not say. LoG computes no
score at all; the old default made it claim certainty in the same field the YOLO detector was
about to fill with a real number.

## Decision

**`Detection.confidence` is `float | None`, defaulting to `None`.** No score is not a score of
1.0.

**Both YOLO backends pass their own scores.** The direct one from `results[0].boxes.conf`, the
tiled one from `CombineDetections.filtered_confidences` — after NMS, so they are the scores of
the boxes that survived, in the order they survived in. A fix that reached one backend would be
half a fix, and ADR-0021 left both backends alive.

**A length mismatch is an error, not a `zip`.** `zip` would drop the tail and return a shorter,
entirely plausible list; worse, a misaligned score reads as a measurement *of that box*. The
error names both counts.

```python
if confidences is not None and len(confidences) != len(boxes):
    raise ValueError(f"got {len(confidences)} confidences for {len(boxes)} boxes; ...")
```

**The LoG detector reports `None`, and no confidence is invented for it.** Its blob response is a
filter response, not a probability: it is unnormalised, it scales with particle contrast, and
turning it into a `[0, 1]` score would be a scientific claim. **M3-T15** — the evaluation harness
— is the task that could validate such a claim, and it does not exist yet.

## Consequences

**Positive**

- D-09 is closed: a YOLO detection reports what the model said about it.
- A consumer can now tell the two detectors apart by asking whether a score exists, instead of
  reading 1.0 from both and believing one of them.
- `0.0` survives being reported, because the field is `float | None` and nothing spells the
  fallback with `or`. That is the same trap ADR-0025 removed from the loaders, one field over.

**Negative**

- **A consumer that read `confidence` as a `float` now gets `None` from the LoG path.** Nothing
  in the repository reads it — the field's only consumer was the deleted React client, which
  rendered it and would have always shown 100 % — but it is a type change in a public entity.
- The two backends now have a second thing they must keep aligned with their boxes. The
  length check makes a mismatch loud rather than silent, which is the most that can be done
  without running inference in the gate.

**Neutral**

- **mypy 14 → 12.** Not because this defect had a static shadow — it had none, an unassigned
  default is perfectly typed — but because passing a second array through
  `_detect_tiled` would have *added* a third `"None" has no attribute ...` error on
  `self._last_result`. Annotating that field `Any`, which its own comment already described,
  removed all three. A change that would have made the baseline worse ended up making it better,
  and the annotation is the honest one: the two possible result types come from optional heavy
  dependencies that must not be imported at module level.
- Inference stays outside the gate (PROJECT_RULES §6). What is tested is the **conversion**, at
  the same weight-free seam M3-T04, M3-T10 and M3-T11 used.

## The measured delta

**29 keys added, 0 values changed.** Four per phantom on all seven — `confidence` on both
`boxes_to_detections_*` conversions, the new `boxes_to_detections_with_scores` (which includes a
`0.0`), and the length-mismatch error — plus `contracts.default_detection_confidence`.

**That last key is `ADDED`, not changed, and that is the finding.** The harness recorded
`default_detection_bbox` and `default_detection_bbox_len` — the defaults of the field the audit
filed as **D-16** — and did not record `confidence`, the field it filed as **D-09**, one line
below in the same dataclass. So the golden could never have caught this defect: the number 1.0
that every YOLO detection carried was not written down anywhere.

M3 has now hit this three times. M3-T07 found that the harness recorded every scalar as the
string `"non-array"`; M3-T12 found it recording `columns: []` for a real phantom without anyone
reading it as a defect; and here it simply had no entry. **A characterization baseline is only a
gate for the values it happens to record**, and the audit's defect list is not the same document
as the harness.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| Propagate YOLO's score, leave the default at `1.0` | Fixes the reported half and leaves the LoG detector claiming certainty it never computed. The two detectors are compared against each other constantly — that is what M3-T15 exists for | Only one detector existed |
| Give LoG a confidence derived from its blob response | The response is unnormalised and contrast-dependent; presenting it as a confidence is a scientific claim with nothing behind it. It would also be *unfalsifiable* until an evaluation harness exists | M3-T15 had measured the response against ground truth |
| Keep `float` and use `nan` for "no score" | Puts a not-a-number in a field typed as a number, and every consumer needs `math.isnan` instead of `is None`. ADR-0019 made exactly this call for `radius_nm`: dataclasses can express absence, arrays cannot | `Detection` were stored column-wise |
| `zip(boxes, confidences)` and accept whatever length comes back | Silently truncates, and a shifted score is worse than a missing one because it is attributed to a specific box | The two came from one array |
| Wait for M3-T15 and do scores then | T15 measures detector *quality*; it needs a confidence that means something to do that. This is its precondition, not its consequence | — |

## Compliance

- `tests/unit/test_confidence.py` — 7 tests. Each box carries its own score; the scores stay with
  their own boxes (checked by matching against box size, not by index); a length mismatch raises
  and names both counts; no scores given means none reported; a bare `Detection` has `None`; LoG
  detections have `None`; and **`0.0` survives**, because it is falsy and any `or`-spelled
  fallback would erase precisely the least confident detection. **Restoring the drop turns 6 red.**
- Golden: `boxes_to_detections_*` gains a `confidence` list, plus a new
  `boxes_to_detections_with_scores` (including a `0.0`) and the length-mismatch error;
  `contracts.default_detection_confidence` is recorded for the first time, as `null`.

## References

- `docs/audit/2026-07-28-baseline-audit.md` §D-09
- `ADR-0019`, `ADR-0025` — absent is `None`, never a substitute value; and the `or` trap
- `ADR-0021` — why both YOLO backends are still alive, and therefore why both must carry scores
- **M3-T15** — the evaluation harness, which is what would license a derived LoG confidence
