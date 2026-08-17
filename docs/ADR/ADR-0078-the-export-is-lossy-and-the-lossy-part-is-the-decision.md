# ADR-0078 — The export is lossy, and the lossy part is the decision

- **Status:** Accepted
- **Date:** 2026-08-17
- **Deciders:** operator + agent (M7-T09)
- **Affects:** `application/use_cases`, `core/ports`, `infrastructure/storage`, `gui` · M7 · M8

## Context

M7's fourth exit criterion: *"annotations export to a format the M8 dataset builder consumes"*. Six
milestones of hand work live in one project directory, and nothing can take it out or bring labels in
from the tools an operator already uses.

The shape of the task is the mismatch between the two things: a row here carries an outline
(ADR-0072), a painted mask (ADR-0073), a note, an id, two timestamps and the `source` that says
whether a person or a detector produced the box (ADR-0044). A label file carries a class index and
four numbers.

## Decision

### 1. The format is the trainer's — YOLO labels and a class list

`labels/<image stem>.txt`, one line per box, `class cx cy w h` normalised to the image, beside a
`classes.txt` whose line numbers **are** the class indices. It is what M8-T03 trains with, and what
labelImg, CVAT and Roboflow read and write — which is what makes export and import one decision
rather than two formats that drift.

### 2. `data.yaml` and the train/val split belong to M8-T02

A split is a *dataset* decision: how much to hold out, stratified by what, and against which
evaluation. Writing one here means M8-T02 arrives to find its central decision already made by the
task that happened to write the labels first. What ships is the labels and the mapping that gives
their indices meaning.

### 3. What the format cannot carry, the caller chooses — starting with `source`

An outline exports as its bounding box, which is **already what the row stores** beside it
(ADR-0072), so nothing is lost that the row did not lose first. A note, an id and the timestamps are
project bookkeeping.

`source` is different, because ADR-0044 made it load-bearing: *a model trained on its own output is
confirming itself*. An export that silently included `from_detection` boxes is exactly how a training
set stops being able to tell, so **the scope is named in the menu** — *hand-drawn only* and
*everything* — which is ADR-0067's rule one milestone on.

**The scope is in the directory name.** Found by a test: two exports a second apart landed in the
same timestamped directory, so a hand-drawn label set and an everything set were mixed in the file an
operator would then have trained on. `annotations_manual_<timestamp>` and `annotations_all_<timestamp>`
cannot collide.

### 4. An import is told where the labels came from

A `.txt` says nothing about who drew the box, so the import asks — the same shape as M5-T07's dialog,
which asks modality and pixel size and *invents neither* — and writes the directory it came from into
the annotation's note, so the provenance survives a trip the format cannot describe.

### 5. An import is one edit, on the main thread; an export is a job

Two hundred labels go through one `Composite` (ADR-0077 §3), so an import is one `Ctrl+Z`. It does
**not** run in the background: the command stack is deliberately not thread-safe, because undo is one
person's sequence of actions (ADR-0045), and a worker thread pushing onto it is two people editing one
project through one history. An export touches no history and reads every scan, so it is a job.

### 6. What is refused, and what is only reported

A class index no list has, a coordinate outside `[0, 1]`, a line that is not five fields: **refusals**,
because such a file is not describing this image and importing it anyway puts boxes where nobody drew
one. A label file naming no image of this project: **reported and skipped** (ADR-0040), because a
directory of labels for a larger dataset is a normal thing to import from.

A box that ran off the edge of the scan is **clamped, not refused**: a drag past the border is an
ordinary thing an operator does, and the part on the image is the part that was ever measurable.

### 7. The GUI does not know which format it is

The name guard from M6-T02 (PROJECT_RULES §2.5) caught the first draft: the viewmodel's docstrings
said *YOLO*. Which trainer reads these files is `application`'s business; the window offers
*annotations*, in and out.

## Consequences

**Positive** — hand work leaves the project in the format the rest of the world labels in, and comes
back; M8-T02 has an input it did not have to invent; the distinction M8 depends on is a named choice
rather than a default.

**Negative** — the export reads every annotated scan to normalise, so a project of forty large scans
pays forty reads; and the import blocks the window while it does the same. Both are bounded by the
number of annotated images and neither writes anything until it has succeeded. A round trip loses the
outline, the mask, the note and the ids: an exported polygon comes back a rectangle, which is why the
project — not the export — remains where the work lives.

**Neutral** — segmentation labels (the polygon form of the same format) are not written. The shapes
are stored; when a trainer asks for them, the file that writes a box is the file that grows a second
line format.

## Alternatives considered

| Alternative | Why not |
|---|---|
| COCO or VOC XML as well | A second format with no caller (ADR-0041) |
| Write `data.yaml` and a split here | M8-T02's decision, made by the wrong task |
| One CSV of every annotation | Nothing in the training ecosystem reads it; the round trip would be ours alone |
| Export everything, always | The training set silently contains the model's own output (ADR-0044) |
| Guess an imported label's source | The one field M8 cannot recover, invented at the only moment somebody knew |
| Import as a background job | A job that edits pushes onto a stack that is single-threaded by decision (ADR-0045) |
| Refuse a box that overhangs the scan | A drag past the edge is normal; the clamped box is what was drawn |

## Compliance

`tests/gui/test_label_exchange.py` asserts the line as **text**
(`0 0.200000 0.300000 0.200000 0.200000` for a known box on a 100-by-50 scan), the sorted class list,
the clamp, that a polygon exports as its stored box, that *hand-drawn only* leaves an adopted box out
and lands in **a different directory**, that a project with nothing drawn is a sentence rather than an
empty export, the **round trip** back to the same coordinates, that the stated source and the source
directory reach the row, that a whole import is one undo, that labels land on the scan they name, and
each refusal and the report.

## References

- ADR-0044 — `source`, and why a training set has to be able to tell
- ADR-0072 / ADR-0073 — the shapes whose box this exports
- ADR-0077 — `Composite`, whose second caller this is
- ADR-0067 — a scope is named in the menu, never implied
- ADR-0048 — an export is a snapshot, and it does not replace yesterday's
