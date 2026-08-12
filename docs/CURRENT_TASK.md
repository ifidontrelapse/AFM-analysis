# CURRENT TASK

**ID:** `M4-T12`
**Title:** `DeviceManager` — one component decides where inference runs
**Milestone:** M4 — Application layer, twelfth task
**Defect:** W8 (no device management) · **ADR:** ADR-0004 is accepted already; **ADR-0049** records
what implementing it decided
**Branch:** `feat/m4-application-layer`
**Status:** **done 2026-08-12.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

W8: *"Nothing decides CPU vs CUDA vs ROCm vs MPS; it is implicit in torch defaults."* It still is —
grepping `infrastructure/models` for `cuda` returns nothing at all, so every inference this project
has run went wherever torch felt like putting it.

ADR-0004 decided the shape three months ago and has been waiting for a milestone with an
application layer in it. This is that milestone, and this is one of its exit criteria.

---

## The decisions implementing it forces

**1. Where does the probe run, and what happens without torch?** In `infrastructure/device/`, with
torch imported **inside the function**, and **no torch means CPU** — reported, not raised.

CI installs no torch (the `ci` dependency group exists precisely to skip the CUDA wheel), so a
device manager that raises on `import torch` is one no test can run. It is also the honest answer
for a machine without it: the CPU is there.

**2. How are ROCm and CUDA told apart?** By `torch.version.hip`.

A torch built for ROCm answers `torch.cuda.is_available()` with **True** and exposes its devices
through the same `torch.cuda` API — so a naive probe reports an AMD card as CUDA, and a user reads
"CUDA" in a dialog about a Radeon.

**3. What does selection do when the preference is unavailable?** Falls back, and **says why in a
sentence a person can read** — ADR-0004 asked for exactly that. A fallback nobody is told about is
a silent 40× slowdown.

**4. Is the port worth it?** Yes, and it is `DeviceProvider` from `core/ports/__init__.py`'s table,
which dates it to this task. `application` must be able to say "run this on the selected device"
without importing torch — the reason `DeviceKind` went into `core` in M2-T02 and has sat unadopted
since.

**5. What order is "best available"?** CUDA, ROCm, MPS, CPU. Not measured — nobody here has an AMD
card or a Mac to measure with — so it is stated as a **convention with its reason** and made
trivial to reorder.

---

## Scope

**In scope**

1. `core/entities/device.py` — `Device` and `DeviceSelection`, the second carrying the reason
2. `core/ports/device.py` — the `DeviceProvider` port, adopting `DeviceKind` at last
3. `infrastructure/device/manager.py` — probing, the policy, the readable fallback
4. **ADR-0049** — no torch is CPU, ROCm told apart by `version.hip`, the fallback speaks
5. Tests against a **fake torch module**: absent, CPU-only, CUDA, ROCm, MPS, and an unavailable
   preference

**Out of scope**

- **Handing the device to the providers** — M4-T13, where the registry that constructs them lands
- **Memory and driver introspection.** ADR-0004 says "where obtainable"; a name and a kind are what
  a selection needs, and the rest is a dialog's problem in M5

---

## Definition of done

- [x] `DeviceManager` probing without a GPU, and without torch at all
- [x] The policy: explicit choice → best available → CPU, with a readable reason
- [x] ADR-0049
- [x] Tests over a fake torch for every backend
- [x] `make check` green — golden byte-identical (verified across this session's three tasks)
- [x] Docs and the ADR index
- [x] Commit: `M4-T12: one component decides where inference runs`

---

## What it turned up

**Running the real probe on this machine was worth doing.** It reports *"NVIDIA GeForce GTX 1070
(cuda)"* and selects it without a fallback — which is the exit criterion's own wording ("on this
machine") and the one thing a fake torch cannot demonstrate.

**The ROCm branch is the reason the fake exists.** There is no AMD card here and there never will
be, so the only way to test the branch that distinguishes a Radeon from an NVIDIA card is to build
a torch that claims to be one. Without that test, the branch would be written and never executed.
