# CURRENT TASK

**ID:** `M3-T05`
**Title:** A detection carries the score its detector gave it, or none at all
**Milestone:** M3 — Numerical correctness, twelfth task
**Defect:** **D-09** (medium) · **ADR:** **ADR-0028**
**Branch:** `sci/yolo-confidence` (stacked on `sci/empty-measurements-keep-their-schema`)
**Status:** planned — no code written yet.

---

## Why this task is next

Every `critical` and `high` defect in M3 is closed. D-09 is the first `medium` one in the task
list, and it is the one that produces a **wrong number in a user-facing field** rather than a
crash or a shape.

```python
Detection(x_px=cx, y_px=cy, radius_px=radius_px, radius_nm=..., bbox=...)
#                                                  confidence never assigned -> 1.0
```

The model scores every box, `cfg.yolo_conf` uses those scores to *filter*, and then
`_boxes_to_detections` throws them away. Every YOLO detection reports **100 % confidence**,
including the ones that only just cleared the threshold.

---

## The decision this task has to make

`confidence: float = 1.0` is a **substitute value**, and this milestone has spent four ADRs
deleting substitute values: a fabricated pixel scale (ADR-0019, ADR-0025, ADR-0026), a fabricated
minimum size (ADR-0024), a fabricated empty table (ADR-0027). The same argument applies here, and
it applies to the LoG detector as much as to YOLO — LoG has no score to give, and 1.0 says it
has one.

| | |
|---|---|
| Propagate YOLO's score, leave the default at `1.0` | Fixes the reported defect and leaves LoG claiming certainty it never computed |
| Propagate YOLO's score, and make "no score" **`None`** ✅ | Absent is absent, in the one field where a reader compares detectors against each other |
| Invent a LoG confidence from the blob response | The response is not a probability and is not normalised; inventing one is a scientific claim this task has no basis for — and M3-T15 is the task that would have to validate it |

**In scope:** both YOLO backends must pass their own scores — the direct one from
`results[0].boxes.conf`, the tiled one from `CombineDetections.filtered_confidences`. A fix that
only reaches one backend is half a fix.

---

## Scope

**Out of scope**

- Any LoG-side score. `Detection.confidence` is `None` there, which is the honest reading of
  "this detector does not produce one"
- `cfg.yolo_conf`'s filtering behaviour. The threshold already works; this task is about what
  survives it carrying its own number
- The `bbox` default (`D-16`) sitting two lines below in the same dataclass. It is **M3-T14**,
  and its `type: ignore` is annotated to expire itself

---

## Definition of done

- [ ] `Detection.confidence` is `float | None`, defaulting to `None`
- [ ] Both YOLO backends pass real per-box scores; a mismatch in length is an error, not a zip
      that silently truncates
- [ ] LoG detections report `None`
- [ ] Tests, including: the score reaches the entity, and restoring the drop turns them red
- [ ] `make check` green; delta quantified (the harness records `Detection`'s defaults and the
      `boxes_to_detections_*` conversions)
- [ ] ADR-0028; `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`, ADR index
- [ ] Commit: `M3-T05: a detection carries the score its detector gave it`

---

## Notes

Inference is outside the gate (PROJECT_RULES §6), so the golden cannot execute either backend.
`_boxes_to_detections` is a `staticmethod` needing no weights — the harness already records it
for D-07 — so the conversion is testable even though the inference around it is not. That is the
same seam M3-T04 and M3-T10 used.
