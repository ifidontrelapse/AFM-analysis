# ADR-0004 — A single Device Manager owns backend selection

- **Status:** Accepted
- **Date:** 2026-08-03
- **Affects:** `core/ports/device.py`, `infrastructure/device/` · M4-T12

## Context

The application runs PyTorch models — YOLO for detection, SAM2 for segmentation, and
training in M8 — on whatever hardware the operator has: NVIDIA (CUDA), AMD (ROCm), Apple
(MPS), or CPU only. `pyproject.toml` currently pins PyTorch to the CUDA 11.8 index, which
is a build-time assumption about the user's machine.

Today no component decides anything: device selection is whatever torch defaults to
inside ultralytics and sam2. There is no way to tell the user what will be used, no way
to override it, no way to fall back gracefully when CUDA initialisation fails, and no way
to report out-of-memory in terms a user can act on.

If each provider decides for itself, the answer will differ between detection,
segmentation and training, and the reason will be spread across three files.

## Decision

Introduce a **Device Manager** — one component, one authority.

- It **probes** available backends at startup: CUDA, ROCm, MPS, CPU, with device names,
  memory and driver versions where obtainable.
- It **selects** according to an explicit policy: user preference → best available →
  CPU fallback.
- It **hands providers a resolved device.** Providers accept it; they never choose.
- It **reports** capability and the reason for a fallback in language a user can read
  ("CUDA present but unusable: driver too old — using CPU").
- It is defined as a port in `core/ports/device.py` and implemented in
  `infrastructure/device/`.

**No module outside the Device Manager may call `torch.cuda.is_available()`, read
`CUDA_VISIBLE_DEVICES`, or hardcode `"cuda"` / `"cpu"` / `"mps"`.**

## Consequences

**Positive**

- One place to add a backend, fix a probe, or change the fallback policy.
- The device becomes visible and overridable in the UI, and recorded with every result
  and training run — which matters for reproducibility.
- Providers become testable on CPU with an injected device, without patching torch.
- Failures become explainable instead of appearing as an ultralytics stack trace.

**Negative**

- One more indirection between a provider and its runtime.
- Probing ROCm and MPS reliably is genuinely fiddly, and it cannot be tested on hardware
  we do not have — some code paths will ship unexercised.
- Torch's build is chosen at install time; the Device Manager can report a mismatch but
  cannot fix a CPU-only wheel on a CUDA machine.

**Neutral**

- Introduces a `DeviceKind` value object in the domain, since results record where they
  were computed.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Let each provider call `torch.cuda.is_available()` | The status quo. Three inconsistent answers, no user control, no diagnosis, and the check duplicated wherever a model is loaded. |
| A global `DEVICE` module constant | Global mutable state, decided at import time, untestable, and cannot express per-job overrides. |
| An environment variable only | Works for one power user, invisible in the UI, and silently unset for everyone else. |
| Rely on ultralytics/sam2 defaults | Different defaults per library, no fallback policy, and no way to report what happened. |

## Compliance

- Grep gate: `torch.cuda`, `"cuda"`, `"mps"`, `"rocm"` appear only under
  `infrastructure/device/`.
- Every provider constructor takes a resolved device argument; none has a default of
  `"cuda"`.
- `DeviceManager.describe()` output is included in the diagnostics bundle (M9-T04) and
  in every persisted training run.

## References

- `systempromt.md` (Device Manager)
- `docs/Architecture.md` §4.2
- `docs/TASKS.md` M4-T12
