# ADR-0049 — No torch is a CPU, and ROCm is not CUDA

- **Status:** Accepted
- **Date:** 2026-08-12
- **Deciders:** operator + agent (M4-T12)
- **Affects:** `core/entities`, `core/ports`, `infrastructure/device` · M4 · every provider in M4-T13

## Context

ADR-0004 decided that one component owns device selection, and named the shape: probe, select by
policy, hand providers a resolved device, report the reason for a fallback in language a user can
read. It has been waiting three months for a milestone with an application layer in it.

What it did not settle is what happens on the two machines this project actually has to survive:
one with no torch installed at all, and one with an AMD card. Implementing it settled both.

## Decision

### 1. No torch means the CPU, reported rather than raised

`import torch` happens inside the probe, and an `ImportError` produces a device list of exactly
`[CPU]` plus an informational log line.

CI installs no torch on purpose — the `ci` dependency group exists to skip a CUDA wheel that its
tests never execute — so a device manager that raises on the import is one no test can run. It is
also the honest answer for the machine: the processor is there, and the analysis will run on it.

### 2. ROCm is told apart from CUDA by `torch.version.hip`

A torch built for ROCm answers `torch.cuda.is_available()` with **True** and serves AMD cards
through the `torch.cuda` API. A probe that trusts the function name reports a Radeon as CUDA, and
the operator reads "CUDA" in a dialog about an AMD card — the sort of wrongness that survives for
years because it never crashes.

`torch.version.hip` is what distinguishes them. The `torch_name` stays `cuda:N` for both, because
that is what a ROCm torch expects; the **kind** is what says which hardware it is.

### 3. A fallback says why, in a sentence

`select()` never raises for hardware that is not there. It falls back and fills `DeviceSelection.
reason` with something readable — *"CUDA was requested but no CUDA device is available — using
CPU"* — and logs it at `WARNING`.

ADR-0004 asked for this in those words, and it is the part that is easy to skip because the code
works without it. A fallback nobody is told about is a forty-fold slowdown that reads as the
application being slow.

### 4. The order of preference is a stated convention, not a measurement

CUDA, ROCm, MPS, CPU. There is no AMD card and no Mac here to measure with, so pretending the order
is empirical would be worse than admitting it is not: a discrete NVIDIA card is what this project's
operator has, ROCm is the same shape of hardware through a younger stack, and MPS is unified memory
on a laptop. It lives in one tuple, `PREFERENCE_ORDER`, and reordering it is one line.

### 5. The probe is cached, with an explicit `refresh()`

Probing imports torch and queries a driver. A settings dialog that lists devices must not do that on
every repaint. `refresh()` exists for the case that genuinely changes the answer — a driver fixed, or
an eGPU plugged in, without restarting.

## Consequences

**Positive**

- W8 is closed: something decides, and `DeviceKind` — in `core` since M2-T02 and unadopted since —
  finally has a resolver.
- `application` and `gui` can offer a device choice without importing torch, which is what the port
  is for.
- The ROCm branch is *tested*, on a machine with no AMD hardware, because the probe is driven by a
  fake torch module.
- Verified on the operator's own machine: "NVIDIA GeForce GTX 1070 (cuda)", selected without a
  fallback.

**Negative**

- The fake-torch tests assert against an API that torch could change. That is the cost of testing a
  branch no CI machine can exercise, and the alternative is not testing it.
- Memory and driver versions are not probed, though ADR-0004 mentions them "where obtainable". A
  name and a kind are what a *selection* needs; the rest is a dialog's problem in M5.
- Nothing consumes the selected device yet. `YoloDetector` and the SAM2 wrapper take no device
  argument, and threading one through belongs with the registry that constructs them (M4-T13) —
  named here so the gap is a plan rather than an oversight.

**Neutral**

- Multi-GPU is listed but not chosen between: every CUDA device appears, and the policy takes the
  first. An operator with two cards has a preference nobody has expressed yet.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Import torch at module scope | A second of import and CUDA libraries loaded to answer a question that may be "no torch" |
| Raise when torch is missing | No test in CI could run, and the machine still has a CPU |
| Trust `torch.cuda.is_available()` for the kind | Reports every AMD card as CUDA — silently, forever |
| Raise when the preferred device is unavailable | The operator wants their analysis to run; they want to be *told* it ran slowly |
| Probe on every call | A driver query per repaint of a settings dialog |
| Rank devices by measured throughput | There is no AMD card or Mac here to measure; a fabricated ranking is worse than a stated convention |

## Compliance

- `tests/unit/test_device_manager.py` drives the probe with a fake torch: absent, CPU-only, CUDA,
  ROCm, MPS, two cards, the ordering, the cache, `refresh`, and every selection path including the
  logged fallback.
- No module outside `infrastructure/device` calls `torch.cuda.is_available()` or writes a device
  string (PROJECT_RULES §2.6); `Device.torch_name` is produced only here.
- `DeviceProvider` is the third row of `core/ports/__init__.py`'s table to pay out.

## References

- ADR-0004 (a single Device Manager owns backend selection) — the decision this implements
- `docs/Architecture.md` §2.3 W8, §4.2 · `docs/Roadmap.md` M4 exit criteria
- M2-T02, which put `DeviceKind` in `core` and left it unadopted for this task
