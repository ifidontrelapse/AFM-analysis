# ADR-0023 — Polarity is configured, with a per-modality default

- **Status:** Accepted
- **Date:** 2026-08-05
- **Affects:** `nanoscope/core/values/modality.py` · `core/science/detection/log.py` ·
  `infrastructure/models/yolo.py` · `core/entities/pipeline.py` ·
  `application/use_cases/pipeline.py` · audit **D-12** · M3-T10 · decision **B3**
- **Numerical impact:** **`tem_dark_particles` goes from 0 detections to 22 of 22.** 19 golden
  values change and 12 keys are added; every AFM and SEM recording is byte-identical.

## Context

`LogDetector.detect` Otsu-thresholds the image and keeps the **bright** side:

```python
thresh = threshold_otsu(z_above)
binary = z_above > thresh          # particles are assumed to be the high side
```

`blob_log` then looks for bright blobs, and `YoloDetector._prepare_image` inverts
unconditionally, because the weights were trained on inverted AFM height maps and therefore
expect *dark* particles.

TEM images particles by absorption: they come out **dark on bright**. So on TEM the LoG path
characterised the background — measured on the audit's phantom, **0 of 22** — and the YOLO path
inverted an image that was already the right way round, handing the network the background too.
The audit's summary of the cause is the whole problem: *"There is no polarity concept anywhere in
the codebase, and no test covers it."*

The vocabulary has existed since M2-T02 — `Polarity.BRIGHT_ON_DARK` / `DARK_ON_BRIGHT`, written
and deliberately adopted by nothing, with a note naming this task. B3 asked which way it should
be decided.

## Decision

**Polarity is a setting. Its default comes from the modality; the operator can override it.**

```python
_DEFAULT_POLARITY = {
    Modality.AFM: Polarity.BRIGHT_ON_DARK,
    Modality.SEM: Polarity.BRIGHT_ON_DARK,
    Modality.TEM: Polarity.DARK_ON_BRIGHT,
}
```

`PipelineConfig.polarity: Polarity | None = None` — `None` means "this modality's convention",
resolved once in `run_pipeline` and passed to whichever detector is built.

**Configured, not detected.** An auto-detector would be a heuristic over the intensity
distribution, and its failure mode is the failure mode this defect already has: *zero particles,
no error*. An operator who has to distinguish "my sample is empty" from "the heuristic guessed
wrong" is in the same position D-12 put them in. A default that is wrong is visible in the
configuration and can be overridden in one line.

**One inversion, at the detector's entrance.**

```python
if self.polarity is Polarity.DARK_ON_BRIGHT:
    z_above = z_above.max() - z_above
```

Everything downstream keeps the single convention it was written for — particles are the high
side. `max - z` rather than `-z` because the LoG path normalises by the maximum and needs it
positive (ADR-0018), and because `max - z` is its own inverse, which a test uses: a
`DARK_ON_BRIGHT` detector on an inverted image returns the same centres as a `BRIGHT_ON_DARK`
detector on the original.

**Both detectors, one commit.** The YOLO half is the same defect mirrored: `_prepare_image`
inverts only when the image is bright-on-dark, so the network sees dark particles either way.
Fixing one and filing the other would leave the concept half-introduced and require B3 to be
executed twice; ADR-0010 separates *defects*, and this is one.

## Consequences

**Positive**

- **TEM works.** 22 of 22 on the phantom that has returned 0 since the audit. TEM is one of the
  three first-class modalities.
- Both detectors now state an assumption they used to make silently, in a type that already
  existed for the purpose.
- The AFM and SEM paths are byte-identical, which the golden proves rather than asserts.

**Negative**

- A fourth modality, or a sample that breaks its instrument's convention, needs the override.
  That is the deliberate cost of not guessing.
- `PipelineConfig` gains a field, so `config_fields` in the golden shifts. Recorded and expected.
- The YOLO change is not measurable here: inference is outside the gate (§6), so what the golden
  shows is that the *input* is now right, not that detections improved. On TEM they can hardly be
  worse — the model was being handed the background — but **no claim is made**, and M3-T15 owns
  the question.

**Neutral**

- `Polarity` stops being dead code. `core/values/__init__.py` has warned since M2-T02 that
  M2-T13 must not delete it; that warning has now paid for itself.

## The measured delta

| what | before | after |
|---|---|---|
| `tem_dark_particles` · `log_detection_on_raw_image` | **0** blobs | **22** blobs (22 true) |
| `tem_dark_particles` · prepared YOLO input, mean grey | 43.3 | **211.7** (= 255 − 43.3) |
| `contracts.config_fields` | 12 fields | 13, `polarity` inserted |
| `sem_bright_particles`, all 5 AFM phantoms | — | **unchanged** |

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| Auto-detect from the intensity distribution (skew, or median vs mean) | Fails silently and in exactly the shape of the original defect — zero particles, no error — and leaves the operator unable to tell a guess from an empty sample | Combined with a configured override *and* a way to report which way it guessed |
| Invert inside `detect_particles` instead of at the detector's entrance | The Otsu sizing branch above it would still keep the bright side, so `sizes` and the blob search would disagree about which way is up | The sizing lived elsewhere |
| Take the absolute value of the LoG response, finding bright and dark blobs at once | Finds both kinds in every image, so a bright artefact in a TEM scan becomes a particle. It also doubles the false positives on noise, which `afm_sparse_low_snr` already shows is the weak point | Mixed-polarity samples were a real use case |
| Fix the LoG path only, file YOLO separately | Half a concept: `run_pipeline` would resolve a polarity that one of its two detectors ignores, and B3 would have to be executed twice | The two paths did not share the assumption |

## Compliance

- `tests/unit/test_polarity.py` — 14 tests (3 parametrised): each modality's default, and an
  unknown modality raising rather than falling back; dark particles found when the detector is
  told and *not located* when it is not; the bright path untouched; the double-inversion property;
  the TEM route through `run_pipeline`; an explicit override beating the modality default; and,
  for YOLO, that both polarities hand the model the same picture to within one grey level.
- Golden: `_log_on_raw` and `capture_yolo_preprocessing` now resolve polarity the way
  `run_pipeline` does — without that, the harness would have stopped reproducing the pipeline at
  the moment D-12 was fixed.

## References

- `docs/audit/2026-07-28-baseline-audit.md` §D-12, and `docs/audit/characterization-baseline.md`
  §3.2 for the 22-versus-0 table
- `nanoscope/core/values/modality.py` — `Polarity`, written in M2-T02 for this task
- Decision **B3**, answered by the operator 2026-08-05
