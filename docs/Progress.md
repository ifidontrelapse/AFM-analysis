# Progress

Append-only session log. Newest entry on top. Every session adds one entry, finished or not.

**Entry format:** date · milestone · task IDs · what changed · what was learned · what is next.
A session that changes scientific output states the numerical delta explicitly.

---

## 2026-08-09 — M4-T01 · **the project format is a versioned contract**

**Task:** `M4-T01`, the first of M4. **Branch:** `feat/m4-application-layer` — M4 changes no
scientific output, so `sci/` no longer applies. **ADR:** **ADR-0038**.

### Why first

Everything else in M4 writes into a project — the schema (T02), the repositories (T03), the
lifecycle use cases (T04/T05), CSV export (T11), the log sinks (T14). Left implicit, the format
becomes *whatever the first task to land happens to do*, and unlike every other contract in this
repository **the operator's data is on the far side of this one**. ADR-0003 fixed the layout and
deferred the contract here by name; the deferral was blocking.

### The three decisions

**Two independent version numbers.** `format_version` describes the directory and lives in the
manifest; `schema_version` describes the tables and lives in the database, as SQLite's own
`PRAGMA user_version` (M4-T02 owns it). A single shared number would make every schema bump claim
the layout had changed — and, decisively, **the layout has to be readable without opening the
database**.

**`project.json` is the identity file.** A directory is a project if and only if it carries one;
never inferred from `images/` or `database.sqlite` being present. Not in the database, because
ADR-0003's own consequence — *corruption is contained* — is a slogan if a project with a damaged
database cannot say what it is.

**Refuse newer, accept older.** A forward migration cannot be written by the past: opening a
project written by a later version means guessing what its fields mean, and the guess gets written
back to disk. One integer rather than semver — one reader, shipped with the writer, asking one
question.

### What writing the tests turned up

**A rule that reads like politeness is a data-loss guard.** "A reader must ignore fields it does
not know" is unremarkable until you write the *writer*: the manifest was a three-field dataclass,
so an older application rewriting a newer project's `project.json` would have **deleted every
field it did not recognise, silently**. Unknown fields are now carried through, with a test. *An
additive change is only additive if it survives the round trip.*

**A non-integer version is refused, not coerced.** `"2"` compares as neither newer nor older than
`1`, and `True` **is** an `int` in Python — both would sail past a naive check as compatible.

### Delta

**Zero, and no numerical code was touched at all** — the golden is byte-identical, which is M4's
risk profile working as stated. New: one module (`infrastructure/storage/project_format.py`), one
error class (`ProjectFormatError`, inside the ADR-0030 taxonomy, one error for four refusals), one
document (`docs/ProjectFormat.md`), one ADR. **21 tests** over every row of the matrix, each
refusal checked for naming the path or both versions.

### Gate

`make check` green — **500 tests** (478 → 500: 21 new, plus one the `no print in library code`
sweep adds automatically for the new module), zero golden differences. mypy unchanged at 6.

### What is next

**M4-T02** — SQLite schema v1 and the forward-migration mechanism. It owns `schema_version`, the
number this task decided the home and the rules for.

---

## M3 — milestone summary

**Closed 2026-08-09** on the operator's decision, with `M3-T16` left open and blocked on **B6**.
Twenty-five of twenty-six tasks done; **twenty-four ADRs**, ADR-0014 through ADR-0037.

| Exit criterion | Result |
|---|---|
| Every critical and high defect closed, each with commit + ADR + golden update | ✅ 2026-08-06. Critical D-01, D-02, D-03, D-04, D-19; high D-05/D-06, D-07 (three faces), D-08, D-12, D-18 |
| Degenerate-input contract documented and tested at every numerical entry point | ✅ M3-T13, ADR-0030. Seven error classes, one `ensure_height_map` at fourteen entry points, 7 bad inputs × 10 entry points proven to give one error type |
| One measurement schema across all four producers | ✅ M3-T14, ADR-0031. Blocks present-in-full or absent-in-full, `method` naming the producer |
| Evaluation harness reports precision / recall / localisation per phantom | ✅ M3-T15, ADR-0032. One-to-one optimal assignment, a scale-free match radius |
| Operator sign-off on D-04 semantics and D-12 polarity | ✅ 2026-08-05, both executed |

**Every defect the July audit reproduced is closed except `D-24`** — the stale README, which M9
owns. Twenty-three of twenty-four, across M1, M2 and M3.

**What M3 actually bought.** The pipeline computes what it claims to compute, and every change to
what it computes is *stated*: 26 tasks, each with its own commit, its own ADR and a measured
before/after delta against seeded phantoms. Tests 119 → **478**. mypy **20 → 6**. The two changes
that moved the most numbers moved them for reasons written down before they were measured —
M3-T09's 696 values (opening radii round up) and M3-T24's 730 (the rough estimate stops
truncating itself).

**The pattern that repeated, and is the milestone's real finding.** Five times, closing a defect
meant first extending the harness that had missed it — M3-T05, T07, T12, T22, T25 each shipped a
golden probe for behaviour nobody had recorded. **A characterization suite pins what you thought
to record.** The corollary showed up in M3-T26: `scale=1.7` survived five months of review because
its value made no measurable difference, and *a constant whose value does not matter does not get
audited*.

**Eight substitute values deleted.** `1.0` for a missing confidence, `1.0` for an unknown pixel
scale, a fabricated scan size, `nan` where a raise belonged, an empty tuple for an absent bbox,
`2.4997` for a threshold, a `NaN` height presented as a measurement, and zero-fill for a dropped
scan line. The rule that emerged and held: **absent is a state, not a number.**

**What M3 got wrong, and corrected in the open.** M3-T24's plan predicted a delta ~70× smaller
than it measured — the rough radius also feeds the LoG sigma range, which nobody had traced.
ADR-0025's diagnosis of `afm_sparse_low_snr` was corrected by M3-T23: losing the scale did two
things and the filter was the smaller one. Both corrections sit where the original claim was made.

**What M3 did not do, and named instead of hiding.** `M3-T16` needs real multi-version Nanoscope
files (**B6**). Four findings are open because each is an algorithm choice with an operator's view
attached: **B-062** (recall 0.000 on `afm_sparse_low_snr` — six particles, none found), **B-065**
(a gap-tolerant *pipeline*, not just levelling), **B-066** (deliberate interpolation), **B-067**
(an opening margin from the radius distribution's upper tail). They stay in M3's list; the
roadmap allows M4 to run in parallel.

**Not claimed, and said throughout: a phantom is not a sample.** Every quality number in this
milestone is measured against synthetic ground truth. What it licenses is *comparison* — this
version against that one — not an accuracy claim about a real scan.

### Next

**M4 — application layer**, and the risk profile inverts: the domain is called, not modified, so
no task in M4 should move a golden number at all. `M4-T01` is first — the project directory
format as a versioned public contract, which ADR-0003 already sketched and explicitly deferred
to this task.

---

## 2026-08-09 — M3-T19 · **the annotations stop lying about `None`; M3's exit criteria are met**

**Task:** `M3-T19`. **Branch:** `sci/m3-numerical-correctness`. **ADR:** none — an annotation is
not executed and no decision was made (the **M3-T18** precedent).
**Defect:** M3-T19, found by mypy in M1-T04 and the last unblocked engineering item in M3.

### What was filed, and what was there

Filed: three mypy errors in `estimate_log_threshold_adaptive`, where `responses` starts as a
`list[float]` and is rebound to an ndarray before `.min()`/`.max()` are called on it.

Found while reading for it, twice, stated the other way round:

```python
def detect_particles(..., threshold: float = None, ...)            # log.py:192
def build_substrate_map(..., manual_radius_px: float = None, ...)  # substrate.py:301
```

Both bodies test the parameter for `None` two dozen lines later (`log.py:237`,
`substrate.py:334`) — "not supplied, work it out" is the *documented* meaning of the default, and
the annotation denied that it could happen. **One defect class — an annotation that does not
describe the value the code carries — and six of mypy's twelve errors.** Fixing only what was
filed would have left half of it.

### Delta: zero, by construction

Not zero as measured, zero as a property: **no executable line changed.** The golden was left
untouched rather than rewritten. The measurement this task owes the gate is mypy's count instead —
**12 → 6** — and the six that remain are not annotation drift: four are `pipeline.py` (three
`ndarray | None` arguments and the `if/elif` detector dispatch, both M4's questions) and two are
third-party overloads (`cv2.normalize`, `Axes.imshow`).

The two new tests pin the *meaning* the annotation now claims: an explicit `None` at either entry
point gives the same answer as omitting the argument. `float | None` would otherwise be a claim
nobody checks — the golden harness passes `threshold=None` and no unit test ever did.

### Recorded, not filed

`r = max(int(sigma), 1)` at `log.py:165` is a truncation of exactly the family M3-T24 hunted. It
sizes a **peak-lookup window**, not a physical radius, so a pixel of it is not a defect anyone can
demonstrate. It is written down in `CURRENT_TASK.md` rather than turned into a backlog entry,
because the backlog is for findings with a consequence.

### M3's exit criteria

Three of the five checkboxes in `Roadmap.md` were still unticked while the tasks behind them had
closed on 2026-08-07 — the degenerate-input contract (M3-T13), one measurement schema (M3-T14),
and the evaluation harness (M3-T15). Ticked in this commit with the evidence. **All five criteria
are met.**

### Gate

`make check` green — **478 tests** (2 new), **zero golden differences**. mypy **12 → 6**.

### What is next

**M3 has nothing left that does not need a decision.** `M3-T16` is blocked on **B6**. The four open
findings — **B-062** (recall 0.000 on `afm_sparse_low_snr`), **B-065** (a gap-tolerant *pipeline*),
**B-066** (deliberate interpolation), **B-067** (a margin from the radius distribution's upper
tail) — are each an algorithm choice with an operator's view attached, not an afternoon. The two
real options are to take one of them with a decision made first, or to close M3 and open **M4**,
which the roadmap explicitly allows to run in parallel.

---

## 2026-08-08 — M3-T26 · **B-064 closed: the opening-radius constants are measured**

**Task:** `M3-T26`. **Branch:** `sci/m3-numerical-correctness`. **ADR:** **ADR-0037**.
**Defect:** B-064 — filed by M3-T24, the last engineering finding open in M3.

### What was unknown

Two numbers set every opening radius in the project — `scale=1.7`, documented with one line, and a
bare `2.5` inside a branch. Neither derived anywhere, and **both chosen while ADR-0035's `int()`
truncation was in place**: the effective margin was `1.7 × int(r)/r`, which at r = 4.9 is **1.39**.

Until M3-T15 there was no way to ask whether they were right.

### The sweep

**The rough factor barely matters.** From 1.3 to 2.4, mean recall and precision are *identical*
(0.7686 / 0.9958) and the radius error moves in the third decimal. The second stage re-estimates
from Otsu and absorbs it — M3-T24's finding approached from the other side, and **the explanation
for why the truncation survived five months: a constant whose value does not measurably matter
does not get audited.**

**The final factor is a genuine trade-off:**

| factor | dense recall | mean radius error |
|---|---:|---:|
| 1.5 | **0.886** | 0.890 px |
| 2.5 | 0.843 | **0.494 px** |
| 4.0 | 0.800 | 0.579 px |

A smaller opening finds more particles; a larger one measures their radii better. The recall cost
falls entirely on `afm_dense_overlapping` — a bigger disk steps *over* two touching particles
instead of into the gap between them — and ×1.5 buys three detections there for an 80 % worse
radius error across the set.

### The decision, and why "keep them" is a result

**2.5 minimises the radius error on both hard phantoms and is the only value in the sweep not
beaten on the metric it leads.** 1.7 is kept because nothing in range distinguishes it. So both
values stay — and stop being anonymous: `DEFAULT_ROUGH_SCALE`, `DEFAULT_OPENING_SCALE`,
`MIN_OPENING_RADIUS_PX`, and the literal becomes an `opening_scale` parameter.

The plumbing *is* the deliverable. A magic number inside a branch is not a decision anyone can
revisit, which is exactly why this finding needed two tasks to surface. "The constants were
already right" is the difference between a number nobody has checked and one someone has.

### Delta: zero

The golden is byte-identical and was left untouched rather than rewritten, so the commit carries
no phantom change at all. The plan predicted zero; naming a constant and exposing it as a
parameter must not move a number, and if it had, a default was mistyped.

The measurement itself is deliberately **not** in the golden — 25 detection runs against a file
that already costs ten minutes. It lives in the ADR, in the constants' docstrings, and in a test
that pins the trade-off's *direction* rather than its numbers.

### Two things the sweep turned up

**`afm_sparse_low_snr` scores 0.000 at every factor**, which is more evidence that **B-062** is a
detector-threshold question and not a substrate one.

**B-067, filed:** the margin comes from `typical_radius_px`, a **median**. On a polydisperse
sample the median particle is by definition too small for half of them — and the sweep shows the
symptom, because `afm_tilted_polydisperse` is the *only* phantom that loses a detection as the
factor grows, and it is the polydisperse one. A margin derived from the distribution's upper tail
is a different algorithm, and the harness can now score it.

### Gate

`make check` green — **476 tests** (12 new), **zero golden differences**. mypy unchanged at 12.

### What is next

M3's engineering queue is empty. What remains needs a decision: **B6** blocks `M3-T16`, **B-062**
wants an operator's view of a sensitivity trade-off, and **B-065**, **B-066**, **B-067** are each
a real algorithm choice. `M3-T19` is a `low` typing finding. The milestone's exit criteria are the
thing to review.

---

## 2026-08-08 — M3-T25 · **B-060 closed: levelling can fit around a gap**

**Task:** `M3-T25`. **Branch:** `sci/m3-numerical-correctness`. **ADR:** **ADR-0036**.
**Defect:** B-060 — filed by M3-T13, whose rejection this completes rather than reverses.

### What was wrong with being right

M3-T13 made a non-finite value a rejection and that was correct: three functions had three
answers, and one contract replaced them. ADR-0030 also wrote, in its own text, that rejecting was
not the best behaviour available. **A dropped scan line is a real artefact, not malformed input** —
two rows of NaN and four thousand good ones, and the whole scan refused.

### The decision, and the number under it

`flatten_plane(z, *, allow_gaps=False)` and `flatten_lines(z, poly_order=1, *, allow_gaps=False)`.
**Opt-in**, because accepting NaN silently would put the library back where D-15 found it —
levelling tolerating what detection refuses. The gap stays **absent**, never interpolated: an
interpolated value is a measurement nobody made, and that would have been the eighth substitute
value this milestone declined to add. A row with too few finite points comes back absent and the
count is **warned**.

Measured against levelling the same scan with no gap at all, on a 64 px synthetic scene: masked
fit **0.029 nm**, `nan_to_num(z, 0.0)` **0.134 nm**. The tilt coefficient is where the difference
lives — 0.0496 filled against 0.0511 true. **Zero-filling does not add noise; it tells the fit
that the sample dips to zero along two lines, and the plane leans to accommodate it.**

### The delta — 5 differences, exactly as predicted

All of them the new `gapped_levelling` block. Nothing recorded moves, because the default path is
untouched by construction — the plan said so before the measurement and the measurement agreed.

The block carries the *evidence*, not only the result: masked fit, ungapped reference and the
zero-filled alternative, per phantom. **The finding is that the advantage tracks the tilt** —
4.2× on `afm_tilted_polydisperse`, 1.2–1.7× on the flat ones. That is the mechanism confirming
itself: the fill corrupts the *plane*, so its damage is proportional to how much plane there is to
get wrong. A real AFM scan is tilted by construction.

### The honest headline

**This does not make the pipeline gap-tolerant.** The levelled output still carries NaN, so
`build_substrate_map` and both detectors still refuse it. What the caller gains is a levelled map
to crop, inspect or fill deliberately, instead of an exception. It is pinned by a test so the
limitation cannot be forgotten, and filed as **B-065** — which needs a decision rather than
plumbing: the morphological opening propagates NaN across the whole structuring element, so
passing it through turns two dropped rows into a band `2 × opening_radius` wide, and masking the
opening instead means the substrate is interpolated across the gap by a morphology operator.
**B-066** files deliberate interpolation, with the note that whatever fills a gap must mark the
pixels it invented or it recreates B-059 one array along.

### Gate

`make check` green — **464 tests** (12 new), 5 declared golden differences and nothing else.
mypy unchanged at 12.

---

## 2026-08-08 — M3-T24 · **B-063 closed: the rough estimate stops truncating its own radius**

**Task:** `M3-T24`. **Branch:** `sci/m3-numerical-correctness`. **ADR:** **ADR-0035**.
**Defect:** B-063 — filed by M3-T23, which fixed its consequence and left the cause.

### Why it was engineering rather than an operator decision

The rule was decided three times before this task existed. **ADR-0020**: `_integer_radius` is the
one funnel and radii round *up*. **ADR-0024**: this exact `int()` pattern, deleted as D-04's
mechanism. And the parameter's own docstring, from the March 2026 commit that introduced both the
truncation and `scale=1.7` — *"a multiplier so the disk is safely **larger** than a particle"*.
The truncation makes it smaller. It contradicted the line below it from the day both were written.

### The delta — 730 differences, and why the plan said ~10

Four phantoms. 320 of the differences are under 1 %, 301 between 1 and 5 %, 21 over 20 % (baseline
percentiles near zero, where a small absolute move is a large relative one). **Mean measured
height moves ≤ 0.09 %**; the largest scientifically meaningful move is 2.5 %, at one phantom's
90th height percentile.

**The plan predicted "one phantom's substrate and heights" and was wrong by ~70×.** Both of its
specific predictions held — the substrate moves on one phantom, the heights on one — but the
simulation modelled only what `build_substrate_map` *returns*. What it missed is that `sizes`
travels onward:

```
rough radius → Otsu on the roughly-opened map → sizes
                                                  ├→ final radius   (absorbed: moves on 1 of 5)
                                                  └→ estimate_log_params → sigma range → every blob
```

379 of the 730 differences are `log_detection`. **`afm_dense_overlapping` detects one more
particle (59 → 60) with a byte-identical substrate** — the clean demonstration that the sigma
range, not the topography, is what moved.

The lesson is about the two-stage design, not about this defect: the second stage is robust to the
first, but the *diagnostics* the first stage emits are wired straight into the detector, and
nothing in the code says so.

### Detection quality, answerable for the first time

**Recall unchanged on every phantom.** On `afm_tilted_polydisperse`, the only one whose substrate
moved: mean radius error 0.765 → **0.718 px**, signed −0.704 → **−0.669 px**, mean localisation
0.6137 → 0.6156 px, median localisation 0.4802 → **0.4770 px**.

Radius error improves 6 %, localisation degrades 0.3 % in the mean and improves in the median.
**A wash, and reported as one.** The signed error moving toward zero is consistent with the
mechanism — a larger opening leaves slightly more of each particle above the substrate — but 6 %
on one phantom is not evidence, and no claim is made. Before M3-T15 this paragraph could not have
been written at all.

### Filed, not fixed

**B-064 — the provenance of `1.7` and `2.5`.** Neither is derived anywhere, and both were chosen
*with* the truncation in place, so whatever tuning they got was done against an estimate that was
systematically small: the effective margin was `1.7 × int(r)/r`, which at r = 4.9 is **1.39**, not
1.7. What makes it a task rather than a preference is that M3-T15 can now score a candidate
against ground truth. Its honest scope includes asking whether one constant per image is right at
all — a polydisperse sample arguably needs the margin to track the largest particle, not the
median.

### Gate

`make check` green — **452 tests** (18 new), 730 declared golden differences and nothing else.
Restoring `int()` turns 11 of the 18 new tests red. mypy unchanged at 12.

**One test caught my own overstatement while I was writing it:** the first draft asserted that
truncation and the correct value differ on every radius tried. At an equivalent radius of 3.432
they are both 6 — the fraction was too small to cross an integer. The test now says so explicitly,
because that is also why the defect was hard to see.

### What is next

M3 has one blocked task and three open findings. **B6** blocks `M3-T16`; **B-062** wants an
operator's view of a sensitivity trade-off; **B-060** and **B-064** are engineering with their own
ADRs to write. `M3-T19` is a `low` typing finding and is the only unblocked item left inside the
milestone's task list.

---

## 2026-08-07 — M3-T23 · **B-061 closed: a rough radius below one pixel is not an estimate**

**Task:** `M3-T23`. **Branch:** `sci/m3-numerical-correctness`. **ADR:** **ADR-0034**.
**Defect:** B-061 — filed by M3-T13, which found it by writing a validation rule that was *too
strict*.

### The defect

`estimate_rough_radius` could return **0**. `disk(0)` is a single pixel, so the rough opening is
the **identity**: the substrate comes back equal to the image and `z_above` is zero everywhere.
Nothing raises, nothing warns, and the result has the shape of an answer.

The condition is that `median + std` selected single-pixel noise. Measured: the median object area
on `afm_sparse_low_snr` is **1.0 px** in *both* the scaled and the unscaled run. The scaled one
survives only because `min_size_px = 5 / 1.95 = 2.56` floors it — **the estimate is equally
worthless there and the floor hides it.**

### It corrects ADR-0025's diagnosis

That ADR recorded **17 objects → 3351** on this exact path and read it as *"losing the scale is
losing the filter"*. Losing the scale did **two** things, and the filter was the smaller one: the
rough radius collapsed to zero, so Otsu ran on a map that had never been opened. Fixing only that
half moves **3351 → 627** — roughly four fifths of the inflation.

ADR-0025 is not edited. Accepted ADRs are immutable; its number stands, with ADR-0034 as the
correction.

### The delta — 11 differences, all in one cell

Everything is inside
`afm_sparse_low_snr.preprocessing.build_substrate_map_no_scale.sizes`: the object count
**3351 → 627**, the Otsu threshold **7.7e-09 → 1.459**, and the radius distribution with them.

**`opening_radius`, `substrate` and `z_above` are unchanged** — the *median* radius drives the
final radius and it is 0.798 px either way. So the fix changes what the function **reports about
the sample**, not what it **returns as the substrate**, on this image. That is a coincidence of
this median, not a property, and the ADR says so.

**The golden had the evidence all along.** An Otsu threshold of **7.7e-09** is Otsu applied to an
all-zero map — exactly what `z − z` is when the opening is the identity. It sat in the file next
to "3351 objects" on a phantom with six particles, and read as normal, because nothing in a
characterization baseline says what a plausible value looks like. The previous four findings were
the harness *not recording* something; this one is the harness recording it and nobody looking.

**It does not make the unscaled run good.** The median radius is still 0.798 px, `radii_px.max`
rises to 93.5 px as one large background component clears the new threshold, and 627 objects on a
six-particle phantom is still wrong. It is now wrong for the single reason ADR-0025 named. The
remainder is **B-062**.

### Filed with measurements, not fixed

**B-063 — the `int()` inside the estimate.** `radius_px = int(np.sqrt(median_area / np.pi))` is a
second, undeclared rounding in a function whose only rounding is supposed to be `_integer_radius`
(ADR-0020), the same pattern ADR-0024 deleted as D-04's mechanism, and *how* the estimate reaches
exactly 0 rather than 0.96. Deleting it moves the rough radius on **every** phantom — 14 → 15,
12 → 14, 11 → 12, 7 → 9 — and therefore every height. Different blast radius, different review,
and it needs someone to ask whether the `× 1.7` scale factor was tuned around the truncation.

### Gate

`make check` green — **434 tests** (9 new), 11 declared golden differences and nothing else.
M3-T20's `test_and_that_costs_the_substrate_on_a_noisy_scan` still passes: the unscaled run still
counts more objects and still uses a different opening radius. Removing the branch turns 5 of the
9 new tests red. mypy unchanged at 12.

### What is next

M3 has two open findings — **B-062** (recall 0.000) and **B-063** — plus `M3-T16` (blocked on
**B6**) and `M3-T19` (`low`). B-062 wants an operator's view of the sensitivity trade-off; B-063
is engineering with a wide delta.

---

## 2026-08-07 — M3-T22 · **B-059 closed: a height that is not a number is not a measurement**

**Task:** `M3-T22`. **Branch:** `sci/m3-numerical-correctness`. **ADR:** **ADR-0033**.
**Defect:** B-059 — found while writing M3-T12's tests on 2026-08-06, deferred twice rather than
bundled into a schema change (ADR-0010).

### The defect

```python
if metrics["height_nm"] <= 0:   # "discard negative heights — they are artefacts"
    continue
```

**`nan <= 0` is `False`**, so the guard written to discard artefacts kept the most artefactual
value there is. On a constant map: two rows, `height_nm` and `baseline_nm` both `NaN`,
`baseline_source` "global", and nothing said so.

The route is four steps and every one of them is reasonable on its own. A map with one value has
no Otsu split, so the substrate mask is **empty**; `np.median` of nothing is `nan`; a particle
whose own ring is too small falls back to that global baseline; and the guard lets it through.

### What needed deciding, and what did not

**The comparison did not.** ADR-0018 ruled on `not x > 0` versus `x <= 0` five days ago, in this
milestone, for exactly this reason. This is its third site.

**The silence did.** The fix on its own turns two `NaN` rows into zero rows — which reads exactly
like "there was nothing here", the sentence that let this survive a whole milestone. So the empty
substrate mask now warns, naming the cause and the consequence, the same call ADR-0025 made for
the skipped `min_size_nm` filter.

### The delta — 5 differences, all of them the probe

The fix moves **nothing recorded**. That is the finding: `not h > 0` and `h <= 0` agree on every
number, and **no phantom has an empty substrate**, so the golden had no way to see this. The probe
ships in the same commit — `measure_all_baseline_empty_substrate`, five AFM phantoms, recording
`{n_rows: 0, columns: [13 names]}` where the same probe would have recorded two rows of `NaN`
before.

**Fifth time in M3 that closing a defect meant extending the harness that missed it**, after
M3-T07's `"non-array"` scalars, M3-T12's `columns: []`, M3-T05's never-recorded field and
M3-T14's `list(det.bbox)`.

### Found while testing: the empty substrate is all-or-nothing

The planned test for "partial success — some particles have their own ring" could not be written,
because that case does not exist. `get_clean_ring` intersects the ring with the substrate mask,
so **an empty substrate leaves every particle without a ring**; all of them fall back to the
baseline that is `nan`, and the whole table goes. The rows are never a subset — which is why the
warning names the substrate rather than the missing rows. Pinned by a test instead of a comment.

### Gate

`make check` green — **425 tests** (10 new), 5 declared golden differences and nothing else. mypy
unchanged at 12. Restoring `<= 0` turns 2 of the 10 red; the other eight cover the warning and the
behaviour this task must not change, and a mutant of the comparison correctly leaves them green.

### What is next

M3's numerical work is complete. `M3-T16` is blocked on **B6**, `M3-T19` is a `low` typing
finding, and **B-061** and **B-062** are the two findings this session filed — each needs a
decision. The milestone's exit criteria are the thing to review.

**Third time `x <= 0` has been the wrong way to write it** (ADR-0018, ADR-0025, here). A fourth
and the rule belongs in `PROJECT_RULES` §3 beside the unit conventions, not in three ADRs.

---

## 2026-08-07 — M3-T15 · **The project can measure detection quality for the first time**

**Task:** `M3-T15`. **Branch:** `sci/m3-numerical-correctness`. **ADR:** **ADR-0032**.
**Defect:** none — this is the gap five tasks in this milestone wrote "not claimed" for.

### What was missing

The golden catches a number that moved and says nothing about whether the number is any good.
M3-T03, M3-T10, M3-T21, M3-T05 and M3-T14 each had to write some version of "not claimed: better
detections". Meanwhile `phantoms.py` has carried exact ground truth since the audit and says so in
its own first paragraph — *"so that a future evaluation harness can score detection against it"*.
The data to answer the question had been in the repository the whole time.

### The two rules that make the numbers mean something

**A match is a centre inside the particle** — `distance <= match_factor × radius`, scale-free
because the tolerance is the particle's own size. A fixed pixel threshold would be two different
physical tolerances across a phantom set spanning 1.95 to 29.3 nm/px.

**One detection per particle, assigned optimally.** Ten boxes on one particle are one hit and nine
false positives; that is what makes precision mean anything. The pairing minimises total distance
over the admissible pairs rather than taking nearest-first, because greedy gets the same counts
and can pair the wrong two — a test pins a case where greedy costs 6.0 and the optimum 4.0.

Ratios with a zero denominator are `None`. A detector that reported nothing on an empty image has
no precision, and 1.0 would have been the seventh substitute value this milestone deleted.

### The numbers — 7 golden differences, all `ADDED`

| Phantom | TP | FP | FN | precision | recall | localisation |
|---|---:|---:|---:|---:|---:|---:|
| `afm_flat_monodisperse` | 24 | 0 | 0 | 1.000 | 1.000 | 0.43 px · 0.86 nm |
| `afm_tilted_polydisperse` | 30 | 0 | 0 | 1.000 | 1.000 | 0.61 px · 1.23 nm |
| `afm_coarse_pixels` | 14 | 0 | 0 | 1.000 | 1.000 | 0.41 px · 4.05 nm |
| `afm_dense_overlapping` | 59 | 1 | 11 | **0.983** | **0.843** | 0.83 px · 1.65 nm |
| `afm_sparse_low_snr` | 0 | 0 | 6 | — | **0.000** | — |
| `sem_bright_particles` | 22 | 0 | 0 | 1.000 | 1.000 | 0.44 px · 0.66 nm |
| `tem_dark_particles` | 22 | 0 | 0 | 1.000 | 1.000 | 0.36 px · 0.18 nm |

**`tem_dark_particles` is the one that closes a loop.** ADR-0023 fixed D-12 four days ago and
could only report "0 → 22 blobs", a count from a detector nobody had scored. It is now a
measurement: every particle found, none invented, a third of a pixel from the truth.

**`afm_sparse_low_snr` scores recall 0.000** — six particles, none found. Not new behaviour;
M3-T12 had already noticed the phantom produces zero blobs, and the golden had been recording a
zero-column measurement table for it since the baseline. But "0 blobs" and "recall 0.0 against six
known particles" are different sentences, and only the second is a defect report. **Filed as
B-062**, not fixed here: it moves numbers, so it needs its own ADR (ADR-0010).

**Every AFM radius is biased small and both image radii large** — −0.19 to −0.70 px, +0.19 and
+0.29 px. Consistent within a modality, which is what a calibration offset looks like rather than
scatter, and precisely the distinction the *signed* error was reported for alongside the absolute
one.

### What it does not license, written before the numbers existed

**A phantom is not a sample.** Seven synthetic images license "this change improved detection on
the phantom set" and nothing about real scans — that is **B6 / M3-T16**, still waiting on the
operator. And these are baselines, not a before/after: the "before" was never recorded and cannot
be recovered without re-running four superseded code paths.

### Gate

`make check` green — **415 tests**, 7 declared golden differences and nothing else. 21 of the new
tests are this task's; the other two arrived by themselves, because `test_logging.py` and
`test_import_graph.py` parametrize over the package's modules and a new module is a new case in
each. mypy unchanged at 12.

### What is next

M3's numerical work is done. `M3-T16` is blocked on **B6**, and `M3-T19` is a `low` mypy finding.
The milestone's exit criteria are the thing to review next, along with three findings this session
filed rather than fixed: **B-060**, **B-061**, **B-062**.

---

## 2026-08-07 — M3-T14 · **D-16 and D-17 fixed: one measurement schema**

**Task:** `M3-T14`. **Branch:** `sci/m3-numerical-correctness`. **ADR:** **ADR-0031**.
**Defects:** D-16, D-17, medium — the last two the audit reproduced.

### Three faults where the audit named one

The audit counted columns and found four producers with four schemas. Reading them found:

1. **One quantity under two names.** `score`/`sam_score` — the copy-paste drift the audit did
   catch — and `mask_area_px`/`area_px`, which it did not.
2. **Two quantities under one name**, which is worse. `radius_nm` was the *detector's blob radius*
   in `measure_all_baseline` and the *measured mask's* equivalent radius in the SEM/TEM SAM2 path.
   The first fault makes a consumer write more code; this one makes it compute the wrong number,
   silently, the moment it concatenates two tables.
3. **Columns that varied per row**, because both SAM2 producers assembled records with
   `if k in res`, so two particles in one call could disagree about what was measured.

### The shape that replaced it

A **core** every producer emits, plus **blocks present in full or absent in full**, with `method`
naming the producer so a reader knows which blocks to expect. Not one wide table with NaN where a
producer cannot fill a column: that says SEM/TEM *has* heights and they are all missing. It has no
heights. Six ADRs this milestone have turned on absent versus substituted, and a column of NaN is
a substitution with better manners.

`detector_radius_nm` is where we looked; `radius_nm` is what we found.

### The delta — 62 differences, and the rename is provably a rename

Sixty in the baseline table (five phantoms × the populated run and the empty probe × six
differences each) and two in the `Detection` defaults. Comparing the two golden files column by
column: **`col::radius_nm`'s digest before equals `col::detector_radius_nm`'s digest after** on all
five phantoms; `x_px`/`y_px` are identical in every statistic with only the dtype moving;
**35 column digests unchanged, 0 changed**. `peak_nm`, the one added number, satisfies
`peak_nm == height_nm + baseline_nm` — the definition `height_nm` was already computed from.

**The SAM2 producers contribute zero differences, and that is not evidence.** There are no weights
here or in CI, so the golden cannot execute either of them; their delta is zero *by construction*.
The 31 tests, driven by a stub predictor that returns three candidate masks and their scores the
way `SAM2ImagePredictor` does, are the entire safety net for that half of D-17.

### The harness had the same bug the code did

`capture_contracts` did `list(det.bbox)` — a `TypeError` the moment a bbox can be absent. **That is
D-16's assumption living inside the tool built to catch D-16.** It now records `None`, and
`default_detection_bbox_len` is kept rather than deleted: `0` was the defect, `None` is the absence
that replaced it, and the two should read differently in the file.

Fourth time in M3 the harness itself was part of the finding, after M3-T07's `"non-array"`,
M3-T12's `columns: []` and M3-T05's never-recorded field.

### mypy caught a comparison written five minutes earlier

Adding the detect-mode empty table, I wrote `segmentation=cfg.mode == "segment"` inside the branch
that only runs when `cfg.mode == "detect"`. mypy called it a non-overlapping equality check —
correct, and it can only ever be False. Two new errors appeared in this change and both were fixed
rather than annotated; the count is unchanged at 12.

### Gate

`make check` green — **392 tests** (31 new), 62 declared golden differences and nothing else.

### What is next

**M3-T15**, the evaluation harness — precision, recall and localisation against phantom ground
truth. It is all that is left of M3's numerical work, and **five tasks have now had to write "not
claimed"** for want of it.

---

## 2026-08-07 — M3-T13 · **D-15 fixed: one answer to "this input cannot be used"**

**Task:** `M3-T13`. **Branch:** `sci/m3-numerical-correctness`. **ADR:** **ADR-0030**.
**Defect:** D-15, medium — and the task five others deferred a rejection to by name (T06, T07,
T08, T17, T20).

### The defect, as the harness had been recording it all along

The audit printed five inputs and five behaviours. The matrix underneath was worse: eleven
degenerate inputs against five entry points, producing **`ValueError`, `TypeError`, `IndexError`,
`LinAlgError` and `RuntimeError`** — four of them from libraries the caller never named — and
`detect_particles` answering a 1-D array, a 3-D array, a NaN map and an infinite map with a clean
empty result.

**That last part is the scientific defect, not the cosmetic one.** A run that finds nothing is an
ordinary outcome, and it was indistinguishable from a run that never had a chance. TEM finding 0
of 22 particles (D-12) went unquestioned for exactly as long as it did because zero is a
believable answer.

### The delta — 129 differences, not one of them a value

| | |
|---|---|
| `error_type` changed | **32** |
| `error_message` now compared, having been recorded-but-unchecked | **28** |
| `raised_in` changed | 15 |
| cells that used to succeed and now raise | **13** |
| `error_type` / `raised_in` added on those | 26 |
| `result` removed on the 11 that used to return one | 11 |
| `stdout_lines` removed, `value` changed (`flatten_dtypes.bool`) | 4 |

`TypeError`, `IndexError`, `LinAlgError` and `RuntimeError` each appear exactly once in the
transition table, collapsing into `InvalidImageError` or `InvalidParameterError`. The matrix
became one column.

**Foreign exception messages in the golden: 15 → 0.** ADR-0022 built the `error_message_unchecked`
category last week because CPython 3.14 reworded a message and the golden called it drift. The
category is now empty — every message the harness records is one this project wrote, compared
exactly, on 45 keys where there were 17. The mechanism stays: the policy is right and the next
library upgrade can refill it.

**Twelve differences sit under phantoms and none is a measurement.** They are the exception types
of two probes the harness added to record *failures* — M3-T06's empty-after-filter case and
M3-T05's length-mismatch case. Every phantom is a valid image, so validation is a no-op on all
seven, which is the property that had to hold.

### What was decided

**Every class inherits the builtin it replaced at its site.** `InvalidImageError` is a
`ValueError`; `MissingFileError` is a `FileNotFoundError` and deliberately *not* a `ValueError`.
The notebooks are the only callers this library has, and a silent change to what their `except`
clauses catch is the worst possible way to deliver an error taxonomy.

**A height map is 2-D, non-empty, integer-or-real, and finite.** Finiteness is the half that is a
choice: `flatten_plane` has always rejected NaN through `scipy.lstsq`, while `flatten_lines`
propagated it. Rejecting at the entry makes the existing contract the library's instead of the
first step's — and it **supersedes ADR-0018 on that one input**, which is written into both ADRs
rather than left for someone to discover. A flat or negative map is still valid data with nothing
in it, and still answers "no particles".

**A boolean array is not a height map** — which supersedes part of M3-T08, one commit later. That
task made levelling promote instead of storing residuals in a `bool` array, where they became a
mask; the promotion rule stands for every dtype that *is* a height map, and the mask is now
refused at the entry, so the pathology is unreachable rather than corrected.

### Two things found and filed, not fixed

**B-061 — a rough opening radius of 0 is reachable and looks like a result.** The first version of
this change validated `radius_px` as *positive*, and one of M3-T20's tests went red:
`estimate_rough_radius` returns 0 on an unscaled noisy scan, `disk(0)` is a single pixel, and the
opening is then the identity — the "substrate" comes back equal to the image. That is exactly the
degenerate path **ADR-0025 measured and recorded**. Refusing it here would have moved a number
inside a validation task, so the check is non-negative and the real question is filed.

**B-060 — levelling that fits around a dropped scan line.** Refusing NaN is the honest reading of
what the code already did. It is not the best behaviour available, and a masked least-squares fit
would be better; it changes what levelling *computes*, so it is a numerical task with its own ADR.

### Gate

`make check` green — **359 tests** (109 new), 129 declared golden differences and nothing else.
mypy unchanged at 12: a missing check has no static shadow either.

### What is next

**M3-T14** — one measurement schema across the four producers (D-16/D-17), the last `medium` — and
then **M3-T15**, the evaluation harness that four ADRs have had to write "not claimed" for.

---

## 2026-08-07 — M3-T08 · **D-13 fixed: levelling returns the residuals it computed**

**Task:** `M3-T08`. **Branch:** `sci/m3-numerical-correctness`. **ADR:** **ADR-0029**.
**Defect:** D-13, medium — the first of the three `medium` tasks `STATE.md` named as next.

Session note: this was worked from a laptop with **no model weights and no `data/`**, so the
task was chosen to be one nothing outside the gate could verify. `flatten_lines` is pure
NumPy/SciPy; the CI-shaped environment (`uv sync --only-group ci`, Python 3.12, no torch)
reproduces the golden here exactly, which was confirmed *before* anything was edited.

### The defect

```python
result = np.empty_like(z)                      # keeps the input's dtype
result[i] = row - np.polyval(coeffs, xi)       # float64 residual, cast on assignment
```

The quantity being stored is what is left of a row after its *own* best fit is removed, so it is
fractional by construction. An array that cannot hold fractions rounds every value away.
`flatten_plane` never had this — it returns `z - plane` and lets NumPy promote — so the two halves
of "flattening" disagreed about the dtype of the same map.

### The delta — 13 differences, and not one of them under a phantom

8 dtype changes (`float32` → `float64`) and 4 sums in `degenerate_inputs`, plus the added
`flatten_dtypes` group. **No phantom moves**, because `flatten_plane` hands `flatten_lines`
float64 on every recorded chain — which is exactly why the golden could not see this defect, and
why the audit's own remediation note said *"golden covers float; add an integer case"*. This
commit adds it.

**The four sums are the fix as a physical property.** A least-squares residual sums to zero over
the range it was fitted on; that is what "the trend was removed" means. Stored in float32 the sum
sat at **1e-6**; it now lands at float64 round-off, 1e-13. The fit never changed — only where the
answer was written down.

Thirteen is what the *comparison* saw. The storage error touches every value in those eight
arrays, and at `rtol=1e-6` the harness correctly judged `min`, `max`, `std` and the percentiles
unchanged — 7.3e-06 on values of order 100. It surfaces in `sum` alone because the true sum is
zero, where any absolute error is an infinite relative one. `_meta.python` also moves
3.12.13 → 3.12.0, this laptop's interpreter; it is recorded and never compared, and the numbers
are comparable because **the golden was verified stable here before anything was edited** — same
numpy, scipy and scikit-image, zero drift.

### The audit understated it, in two directions

The audit measured a `uint8` ramp whose residuals were all under 1, got "all zeros", and filed it
as truncation. Truncation is the small case. On an image with real structure the residuals are
tens of nanometres and **an integer output wraps the negative ones**: on the newly recorded 8-bit
phantom the levelled map is wrong by up to **257**, 100 % of pixels differ, and every pit came
back rendered as a peak. A reader looking at that map would not have seen a degraded result; they
would have seen features that are not there.

**Boolean input the audit did not measure at all.** `result[i] = <float array>` into a `bool`
array stores `!= 0`, so levelling a mask returned *a mask of where the residual was non-zero* —
65 % of pixels wrong, max error 1.44, and the array still has the shape and the name of
topography.

### Who was actually exposed

Not the documented chain. **`load_microscopy_image` returns `uint8`** — it is `cv2.imread(...,
IMREAD_GRAYSCALE)` and the only file entry point SEM/TEM has — so the exposed caller is the
modality the project has been fixing all week (D-12/M3-T10 was the last one). `load_afm(fmt="npy")`
passes through whatever the file holds, and three documents advertise `flatten_lines` as a
function you may call on its own.

### The decision, which is smaller than it looks

`np.promote_types(z.dtype, np.float64)` rather than a hardcoded `float64`. For every dtype
`np.polyfit` accepts today the two are the same expression, so this is a choice about which *rule*
is stated: the hardcoded one agrees with `flatten_plane` by coincidence, and one rule in both
halves of flattening is what D-13 is about. float32 in becomes float64 out — declared drift, and
already what `flatten_plane` does with the same input.

**Deferred on purpose:** every rejection. `np.promote_types` raises its own `TypeError` on a
string array one line earlier than `np.polyfit` did, and neither names the parameter. Non-numeric
dtypes, 1-D, 3-D and NaN are **M3-T13**, which takes all the entry points in one pass — a taxonomy
of one is not a taxonomy. The three degenerate inputs that raise today raise exactly what they
raised before.

### Gate

`make check` green — 249 tests (17 new), golden 13 declared differences and nothing else, ruff
clean, format clean. **CI green as run #61, 407 s**, on the machine that has no torch and a
different Python patch release, which is the run that matters for a change whose whole subject is
dtype. **mypy unchanged at 12**: a dtype that is right for one input and wrong for
another has no static shadow, which is the second time this milestone that a real defect was
invisible to the type checker (M3-T02 was the first, a unit error).

### What is next

**M3-T13** and **M3-T14**, the two remaining `medium` tasks, both of which touch every entry
point — and **M3-T15**, the evaluation harness, which four ADRs have now had to write "not
claimed" for.

---

## 2026-08-06 — M3-T05 · **D-09 fixed: a detection carries its own score, or none**

**Task:** `M3-T05`. **Branch:** `sci/yolo-confidence`. **ADR:** **ADR-0028**.
**Defect:** D-09, medium — the first `medium` one, every `critical` and `high` being closed.

### The defect

The model scores every box. `cfg.yolo_conf` *filters* on those scores. The conversion to entities
then dropped them, so every YOLO detection reported **1.0** — including a box that had only just
cleared the threshold. Sorting by confidence gave the input order; filtering on it kept
everything.

The default was the real defect. `confidence: float = 1.0` is a **substitute value**, the fifth
this milestone has deleted after a fabricated pixel scale (ADR-0019/0025/0026), a fabricated
minimum size (ADR-0024) and a fabricated empty table (ADR-0027). And it reached further than the
audit said: **the LoG detector claimed 1.0 too**, in the same field, having computed nothing.

### The delta — 29 keys added, 0 values changed

Inference is outside the gate, so nothing recorded could move; what is new is the conversion
seam's scores, recorded on all seven phantoms, plus a `0.0` case and the length-mismatch error.

**The finding is `contracts.default_detection_confidence: ADDED`.** The harness recorded
`default_detection_bbox` and `default_detection_bbox_len` — the defaults of the field the audit
filed as **D-16** — and never recorded `confidence`, the field it filed as **D-09**, one line
below in the same dataclass. **The golden could not have caught this defect**, because the 1.0
every detection carried was not written down anywhere.

That is the third time in M3. M3-T07 found the harness recording every scalar as the string
`"non-array"`; M3-T12 found it recording `columns: []` for a real phantom and nobody reading it;
this one had no entry at all. **A characterization baseline is a gate only for the values it
happens to record** — and the audit's defect list and the harness are two different documents,
written by the same person in the same week.

### What was decided, beyond propagating a number

**`None`, not 1.0, for a detector that has no score.** LoG's blob response is a filter response:
unnormalised, contrast-dependent, and not a probability. Turning it into a confidence would be a
scientific claim, and **M3-T15** — the evaluation harness — is the only thing that could license
one. It still does not exist.

**A length mismatch raises.** `zip` would drop the tail and return a shorter, plausible list;
worse, a shifted score reads as a measurement *of that box*. And `0.0` is kept as `0.0`: it is
falsy, and an `or`-spelled fallback would erase exactly the least confident detection — the same
trap ADR-0025 pulled out of the loaders.

### mypy 14 → 12, by accident

This defect had no static shadow — an unassigned default is perfectly typed. But threading a
second array through `_detect_tiled` would have **added** a third `"None" has no attribute ...`
error on `self._last_result`. Annotating that field `Any` — which its own comment already
described, and which is honest because both possible result types live in optional heavy
dependencies — removed all three. A change that would have made the baseline worse made it
better.

### CI

Runs **#54** (M3-T17), **#55** (M3-T12) and **#56** (M3-T05) — all `success`. Fourteen branches
pushed, fourteen green.

### Branch consolidation

The 32 task branches were merged down to **one**, `sci/m3-numerical-correctness`, at the
operator's instruction — the stack was strictly linear, so every branch was an ancestor of the
tip and no commit was lost; that was verified branch by branch before anything was deleted,
locally and on `origin`. All of them were green on CI first. **A declared deviation from
PROJECT_RULES §7 ("one task per branch"), recorded in `STATE.md` rather than left silent:** what
makes a task attributable is one commit with its own ADR, golden update and quantified delta —
ADR-0010's requirement — not the branch it sat on.

### Session close, 2026-08-06

Five tasks in one day: **M3-T02, T20, T17, T12, T05**, plus **M3-T18** closed as a side effect
and **B-059** filed. **Every `critical` and `high` defect the audit reproduced is now closed**,
which is the first of M3's five exit criteria. mypy 15 → 12; 199 → 232 tests.

Three of the five turned up something the plan did not have. D-04's "90 % of scans" costs nothing
on 58 % of them. Losing the pixel scale means losing the size filter, which reaches the substrate.
And D-09 was invisible to the golden, because the harness recorded the defaults of the field next
to it and not of the field itself — the third time this milestone that **the harness, not the
code, was the blind spot**.

### Next

**M3-T08** (`flatten_lines` dtype promotion, D-13), then **M3-T13** (error taxonomy) and
**M3-T14** (one measurement schema, and the `bbox` default whose `type: ignore` is written to
expire itself). **M3-T15** is the one that unblocks every claim about detection quality.

---

## 2026-08-06 — M3-T12 · **D-08 fixed: an empty measurement table keeps its columns**

**Task:** `M3-T12`. **Branch:** `sci/empty-measurements-keep-their-schema`.
**ADR:** **ADR-0027**. **Defect:** D-08, high — the last unblocked `high` one.

### The defect

```python
df = pd.DataFrame(results)     # results == [] -> a DataFrame with zero columns
```

Two ordinary outcomes drop a row: a mask running past the image edge, and a non-positive height.
When they take the last one, "no particles" and "no such column" become the same object, and the
caller cannot ask the first question without handling the second:

```python
>>> plot_pipeline_result(result_with_no_particles, z, scan)
KeyError: 'height_nm'
```

### The delta — 78 differences, 0 values moved

Twelve column names appear, plus `columns: length 0 -> 12`, in six places: the
`measure_all_baseline_empty_blobs` probe on all five AFM phantoms, and — the entry worth reading
— **`afm_sparse_low_snr`'s ordinary `measure_all_baseline` run**.

**D-08 was live on a real phantom's normal path.** That phantom detects 0 blobs at the harness's
threshold, so its measurement table was the zero-column one in the run the phantom exists to
represent, not in a probe written to provoke the defect. The golden had been recording
`columns: []` for it since the baseline was taken in M0, and nobody had read it as a defect —
including the audit, which reproduced D-08 from the code rather than from the golden.

No populated table changes. The four phantoms that measure particles keep every value and every
column, which is the evidence that the **declared** schema is the one the code already emitted.

### The declaration has to be checked, not asserted

`BASELINE_COLUMNS` names twelve columns and their dtypes; `empty_baseline_table()` builds the
zero-row frame from it. The schema now lives in two places — the declaration and the row literal
— so the test that matters is the one on the **populated** path: the emitted columns and dtypes
must equal the declaration. Without it, a future column added to the row dict would silently stop
matching the empty table, and the golden could not catch it, because its empty case has no
columns to compare against.

Dtypes are part of the promise. `df["height_nm"].mean()` on an empty `str` column is not the
answer it is on an empty `float64` one, and pandas 3 infers `str` rather than `object` for text —
which the first draft of the declaration got wrong and the populated-path test caught immediately.

### Found while testing, filed not fixed

`if metrics["height_nm"] <= 0: continue` — and **`nan <= 0` is `False`**, so a NaN height reaches
the table. Reachable on a constant map: `substrate_mask` is empty, `np.median` of nothing is
`nan`, and `global_baseline` carries it into every height. **ADR-0018 already ruled on this exact
comparison** — the guard has to be `not height > 0`, because that and `<= 0` differ precisely on
`nan`. Filed as **B-059**; it moves a number, and ADR-0010 keeps one defect to one commit.

### What was deliberately left

The two SAM2 producers build each record with `if k in res`, so their columns vary **per row** —
declaring a schema for them means deciding what it is, which is D-16/D-17 and **M3-T14**. Same
for `run_pipeline`'s detect-mode empty frame, where the right columns depend on the modality.

### Next

**Every `critical` and `high` defect in M3 is now closed.** What remains is `medium`: T05
(YOLO confidence), T08 (`flatten_lines` dtype promotion), T13 (error taxonomy), T14 (one
measurement schema). **B6 → M3-T16** is the last operator answer waiting; **B-040** goes last.

---

## 2026-08-06 — M3-T17 · **D-07's third face: a header without a scan size parses**

**Task:** `M3-T17`. **Branch:** `sci/spm-header-without-scan-size`. **ADR:** **ADR-0026**.
**Defect:** high, found by mypy in M1-T04.

### The defect

```python
else:
    scan_size_nm = None

pixel_size_nm = scan_size_nm / samps      # TypeError, one line later
```

The `else` exists **specifically** to tolerate a header with no `Scan Size:`, and the next line
divides by `samps` unconditionally. **The fallback crashed on the branch it had just taken**, and
had done since the code was written.

### The delta — none, and none was possible

`afm_io` has no phantom: the characterization set is synthetic *arrays*, not synthetic *files*.
Nothing this module does can move a golden number, so its 28 unit tests are the entire safety
net. That is worth stating rather than leaving as a blank line in a table — it is also why the
M1-T06 suite was validated by killing four mutants of this parser, and why this fix was found by
that suite in the first place.

**mypy 15 → 14, and the error that went was this defect's**: the function was annotated
`-> np.ndarray` and returned a three-tuple. Third task in M3 where the static shadow of a defect
was sitting in the tolerated baseline (M3-T09, M3-T11, this).

### Two failure modes, one expression

`samps` is the divisor, so `Samps/line: 0` was a `ZeroDivisionError` out of the same line — an
error naming nothing. Both are now guarded, and a *stated* `Scan Size: 0` is rejected as well:
**absent and wrong are different**, and only one of them is recoverable. That distinction is
ADR-0025's, one commit old, now applied to the second loader so the two agree about what a scale
is. A file is not a caller, but the principle transfers exactly.

### D-07 is closed

Three faces, three tasks, three ADRs: the detectors (M3-T11, ADR-0019), the npy loader (M3-T20,
ADR-0025), the SPM header (this, ADR-0026). Every route to "no scale" now ends in the same state.

### Next

**M3-T12** — D-08, empty measurements must return a schema-stable DataFrame. Then the `medium`
ones in order. **B6 → M3-T16** is the last operator answer waiting; **B-040** goes last of all.

---

## 2026-08-06 — M3-T20 · **D-07 closed on both sides: an unknown AFM scale is not a fabricated 1.0**

**Task:** `M3-T20`. **Branch:** `sci/npy-no-invented-scale`. **ADR:** **ADR-0025**.
**Defect:** the other half of D-07, high, found by the M1-T06 tests.

### The defect

```python
pixel_size_nm=pixel_size_nm or 1.0,
scan_size_nm=scan_size_nm or float(z.shape[0]),
```

Three defects in two lines: a fabricated scale is indistinguishable from a measured one, `or`
swallows an explicit `0.0`, and a row count is not dimensionally a length in nanometres.

ADR-0019 wrote the order this had to happen in — *"that task makes `None` survivable; M3-T20
makes it honest"* — and it held: the detectors have accepted `None` since M3-T11, so the
fabrication could be removed without introducing a crash.

### The delta — 5 keys added, 0 values changed

Every phantom has a scale, so nothing recorded moves. The new keys are
`build_substrate_map_no_scale`, one per AFM phantom — a path that could not be reached from
`load_afm` until the fabrication was removed. What they record is the finding:

| phantom | opening radius | objects kept | typical radius px | substrate |
|---|---|---|---|---|
| `afm_flat_monodisperse` | 19 → 19 | 24 → 24 | 7.2031 → 7.2031 | identical |
| `afm_coarse_pixels` | 11 → 11 | 14 → 14 | 4.0093 → 4.0093 | identical |
| `afm_dense_overlapping` | 16 → 16 | 51 → 51 | 6.0239 → 6.0239 | identical |
| `afm_tilted_polydisperse` | 18 → 18 | 29 → **30** | 6.8868 → 6.8520 | identical |
| `afm_sparse_low_snr` | 8 → **5** | 17 → **3351** | 2.9854 → **0.7979** | **differs** |

### What it turned up: losing the scale is losing the filter

Without a scale, `min_size_nm` cannot be expressed, so the filter does not run — and **an
unscaled run is exactly a scaled run with `min_size_nm=0`**, which a test now pins. On four of
the five phantoms that costs nothing and the substrate is bit-identical. On the noisy one it
costs the radius estimate: 3351 objects instead of 17, a median radius of 0.80 px instead of
2.99, and therefore an opening radius of 5 instead of 8. **The substrate is not
scale-independent in general**, because the filtered radii feed the radius that opens it.

That is **D-04's mechanism arriving by a different road** — one commit after D-04 was closed. It
is why the skip is logged at `WARNING` naming both the minimum it could not apply and the
consequence. A silently disabled size filter is precisely what the previous ADR spent its length
removing.

The draft of ADR-0025 claimed the pixel-space result was "produced exactly as before". The
regenerated golden said otherwise on one phantom in five, and the ADR now says what was measured.
**A comfortable sentence in an ADR is a hypothesis until the golden agrees with it.**

### The contract, and who inherits it

- `None` is unknown and passes through; a scale that *is* given must be positive, so `0.0`, `-1`
  and `nan` raise instead of being swallowed by `or`. Same rule as `PixelScale.__post_init__`,
  restated at the boundary — the value object is not adopted as the field type, and the ADR says
  why.
- `AFMRawData` and `PreprocessingResult` carry `float | None`, so the AFM and SEM/TEM branches of
  `run_pipeline` finally have the same type.
- `scan_size_nm` is **not** derived from `pixel_size_nm * z.shape[0]`: the SPM path derives from
  columns, the deleted line used rows, and nothing settles that axis convention yet.
- **M3-T17 now inherits a contract rather than a question.** When `_read_nanoscope_z` stops
  dividing `None` by `samps`, the state it produces already means something everywhere
  downstream.

### Tests

10, of which 6 turn red if `pixel_size_nm or 1.0` comes back. The end-to-end one is
`run_preprocessing` on a bare `.npy` — the route the defect actually travels.

### CI

Run **#52**, `success`. Eleven branches pushed, eleven green.

### Next

**M3-T17** — the same state arriving from the SPM header, and the contract for it now exists.
**M3-T12** is the other unblocked `high` one. **B6 → M3-T16** is the last operator answer
waiting; **B-040** goes last of all.

---

## 2026-08-06 — M3-T02 · **D-04 fixed: the minimum particle size is a physical size (B2)**

**Task:** `M3-T02`. **Branch:** `sci/min-size-in-nm`. **ADR:** **ADR-0024**.
**Defect:** D-04, critical — the last of the four. **Decision:** B2 — filter in nanometres,
delete the `int()`.

### The defect

```python
min_size_pixel = int(min_size_nm / pixel_size_nm)      # three call sites
radii_px = radii_px[radii_px >= min_size_pixel]        # one comparison
```

`int(5 / 9.77) == 0`, and 9.77 nm/px is the median of the operator's scans. A threshold of zero
admits every connected component Otsu produced — including single-pixel noise, whose radii then
set the opening radius and the LoG sigma range.

The unit trail is the whole story: `min_size_nm` was converted **to** pixels to be compared
against `radii_px`, and three lines later `radii_px` was converted **back** to `radii_nm` for the
result — twice, identically, the audit's §Duplication entry. The nanometre values the comparison
wanted already existed.

### The delta — 47 differences: 27 changed, 15 added, 5 removed

| phantom | nm/px | old | new | objects kept | typical radius px |
|---|---|---|---|---|---|
| `afm_flat_monodisperse` | 2.00 | 2 px | 2.5 px | 24 → 24 | 7.203 → 7.203 |
| `afm_coarse_pixels` | 9.77 | **0 px** | 0.512 px | 14 → 14 | 4.009 → 4.009 |
| `afm_dense_overlapping` | 2.00 | 2 px | 2.5 px | 51 → 51 | 6.024 → 6.024 |
| `afm_tilted_polydisperse` | 2.00 | 2 px | 2.5 px | 29 → 29 | 6.887 → 6.887 |
| `afm_sparse_low_snr` | 2.00 | 2 px | 2.5 px | **75 → 17** | 2.877 → **2.985** |

Everything else that moves follows from those 58 removals: the radius distribution
(`min` 2.03 → 2.52 px, `sum` 369 → 150), the rough stage's Otsu threshold (1.168 → 1.459), and
`max_sigma` downstream (86.8 → 132.3). The final opening radius is **8 on both sides**, so
`substrate` and `z_above` are byte-identical and **no measured height moves on any phantom**.
Five keys removed and fifteen added are the harness swapping `min_size_pixel_used` for the
physical threshold, its pixel equivalent, and the floored value — the arithmetic this commit
deletes, kept as the measuring stick.

### What the delta turned up: the phantom built for D-04 does not move

`afm_coarse_pixels` exists *because* `int(5 / 9.77) == 0`. Its numbers are unchanged. The
smallest object a labelling can produce is one pixel, equivalent radius `sqrt(4/π)/2 = 0.564 px`
— **5.51 nm at that scale**, already above the 5 nm minimum. The broken filter and the correct
one agree there because there is nothing either could remove.

So the headline was re-measured, on all **628** scan headers in `data/` rather than the audit's
120 sample. The 90 % reproduces exactly (**568 / 628**), and splits into three regimes:

| pixel scale | scans | what the `int()` did | what the fix changes at the 5 nm default |
|---|---|---|---|
| ≥ 8.86 nm/px | **365 (58 %)** | floored to 0 | nothing — one pixel is already over 5 nm |
| 5 – 8.86 nm/px | **203 (32 %)** | floored to 0 | the filter starts working |
| ≤ 5 nm/px | 60 (10 %) | quantised down | the filter stops being lenient |

**The finest 10 % were harmed by a mechanism the audit did not name.** Not the floor to zero —
truncation. `afm_sparse_low_snr` is in that band: at 2 nm/px `int()` turned a 2.5 px threshold
into 2 px, and **58 of its 75 "objects" were noise living in that half-pixel**. The other three
2 nm/px phantoms are clean, so nothing of theirs sits between 2 and 2.5 px and nothing of theirs
moves.

D-04's honest size: real on 90 % of scans, worth nothing on 58 % of them, and worth **77 % of the
object count** where it bit.

### mypy is unchanged at 15, and that is the point

M3-T09 and M3-T11 each removed mypy errors that were their defect's static shadow. This one has
none: `int(float) -> int` is impeccably typed. **A unit error is invisible to a type checker that
cannot tell a nanometre from a pixel** — the `_nm` / `_px` suffix convention in PROJECT_RULES §3
is the only checker this class of defect has, and it is read by people, which is exactly how this
one was found.

### Tests

5 new, over a fixture of three blocks with **exact** pixel areas (64, 16, 1 px²), so the
equivalent radii are arithmetic rather than an artefact of a Gaussian's tail meeting Otsu. One
test per regime above, plus the scale-invariance of the stated threshold and the error message's
units. **Restoring the `int()` turns 3 of the 5 red**; the two that stay green document regimes
where the two arithmetics agree, and say so.

Three tests written earlier in this task were replaced rather than kept: each asserted something
the *old* code also satisfied. A test that cannot fail on the defect it names is documentation
with an assert in it.

### The duplicated `radii_nm`

Fixed here, after M3-T01, M3-T06 and M3-T09 each left it alone on purpose (ADR-0010 keeps tidying
out of numerical commits). This change *forces* it: the filter needs `radii_nm` before it runs,
so the assignment moves above the filter and the second copy has nowhere left to be.

### CI: seven branches, seven greens

The seven `sci/` branches that had never left this machine were pushed together and all seven
passed — runs **#44–#50**, 139–459 s. That is the first CI reading of **M3-T07, T09, T10, T11,
T21, T02 and B-058**, six of which move golden numbers, so the regenerated baselines are now
confirmed reproducible on a machine that is not this one. Ten of ten branches are pushed and
green.

### Next

**B6 → M3-T16** (header-only SPM fixtures), then **B-040** last, because it rewrites every SHA
above it. The unblocked `high` tasks are **M3-T20**, **M3-T12** and **M3-T17**.

---

## 2026-08-05 — M3-T10 · **D-12 fixed: TEM finds 22 of 22 instead of 0 (B3)**

**Task:** `M3-T10`. **Branch:** `sci/detection-polarity`. **ADR:** **ADR-0023**.
**Defect:** D-12, high. **Decision:** B3 — configured, with a per-modality default.

### The defect

`LogDetector` Otsu-thresholded and kept the **bright** side; `blob_log` looks for bright blobs;
`YoloDetector._prepare_image` inverted unconditionally because the weights expect dark particles.
TEM images by absorption, so its particles are **dark on bright** — and both detectors were
therefore working on the background. Measured on the audit's phantom: **0 of 22**.

The audit named the cause exactly: *"There is no polarity concept anywhere in the codebase, and
no test covers it."* The vocabulary has existed since M2-T02 — `Polarity`, written for this task
and adopted by nothing, with a note telling M2-T13 not to delete it. It has now paid for itself.

### The delta — 19 values changed, 12 keys added

| what | before | after |
|---|---|---|
| `tem_dark_particles` · `log_detection_on_raw_image` | **0** blobs | **22** blobs (22 true) |
| `tem_dark_particles` · prepared YOLO input, mean grey | 43.3 | **211.7** (= 255 − 43.3) |
| `contracts.config_fields` | 12 fields | 13, `polarity` inserted |
| `sem_bright_particles` and all five AFM phantoms | — | **unchanged** |

### Configured, not detected

An auto-detector would be a heuristic whose failure mode is D-12's own: **zero particles, no
error**. The operator would be back to not knowing whether the sample is empty or the guess was
wrong. A wrong default is visible in the configuration and overridden in one line;
`PipelineConfig.polarity=None` means "this modality's convention", resolved once in
`run_pipeline`.

**One inversion, at the detector's entrance** — `z_above.max() - z_above`, so everything
downstream keeps the single convention it was written for. `max - z` and not `-z` because the LoG
path normalises by the maximum and needs it positive (ADR-0018); it is also its own inverse,
which one test uses: dark-on-bright detection on an inverted image returns the same centres as
bright-on-dark on the original.

**Both detectors in one commit.** The YOLO half is the same defect mirrored — `_prepare_image`
now inverts only a bright-on-dark image, so the network sees dark particles either way. ADR-0010
separates *defects*, and this is one; splitting it would have left `run_pipeline` resolving a
polarity that one of its two detectors ignored.

### Not claimed

That YOLO detects better on TEM. Inference is outside the gate — what the golden shows is that
the **input** is now right. It can hardly be worse than handing the model the background, but
M3-T15 owns that question.

### Tests

14 tests (3 parametrised), including the one that keeps D-12 from returning quietly: with the
wrong polarity the detector's blobs must not sit on the particles. On the 22-particle phantom
that reads as 0 found; on a four-cap toy image it finds the *gaps between* them, which is the
same failure at a smaller size — so the assertion is about **where** the detections are.

### Next

**B2 → M3-T02**, the critical one. Then **B6 → M3-T16** and **B-040** last.

---

## 2026-08-05 — B-058 · **The golden compares the messages we wrote, and only those**

**Branch:** `sci/golden-exception-text`. **ADR:** **ADR-0022**. **Backlog:** B-058, which
specified that this needed an ADR rather than a quiet edit.

### The fragility

`capture.py` compared recorded exception messages exactly, and most of those sentences were never
ours: `too many values to unpack (expected 2)` is CPython's, `Only 2-D and 3-D images supported.`
is scikit-image's. CPython 3.14 reworded the first, the first real CI run resolved 3.14, and the
gate went red as **characterization drift with no scientific change behind it** (M1-T08). CI was
pinned to 3.12; the fragility stayed.

### The rule

The exception **type** and the **function it came out of** are always compared. The **message**
is compared only when this project wrote it:

```python
"nanoscope" in Path(frame.filename).parts and "raise " in (frame.line or "")
```

**Both signals, because either alone is wrong.** Filename alone claims `h, w = z.shape` in our
own `flatten_plane` — the M1-T08 case exactly, our file, CPython's wording. `raise` alone claims
`raise TypeError('Only 2-D and 3-D images supported.')`, which is skimage's sentence in skimage's
file.

### The delta — 15 keys renamed, 0 values changed

`error_message` → `error_message_unchecked` on 15 recordings; `compare` skips any key ending in
`_unchecked`. **7 messages remain compared, all of them `estimate_radius_otsu`'s** — the ones
PROJECT_RULES §3 governs, which must name the parameter and its value, and whose wording M3-T06
changed on purpose.

Nothing is dropped: the text is still recorded and still regenerated, because a reader diagnosing
a failure needs it. The harness simply stops promising that somebody else's wording is stable.

### What it unblocks

**A Python upgrade no longer reads as drift** — `STATE.md` listed that as the precondition for
touching the interpreter version at all. The contract also got *sharper* rather than looser: what
failed and where is still compared exactly.

### Tests

6 tests: our own `raise` is ours, a CPython message in our own file is not, a library's explicit
`raise` is not — so neither signal is redundant — and `compare` ignores a reworded `_unchecked`
value, reports a reworded one of ours, and still reports a changed `error_type` either way.

### Next

**B3 → M3-T10** (polarity), **B2 → M3-T02** (the critical one), **B6 → M3-T16**, then **B-040**.

---

## 2026-08-05 — M3-T21 · **The tiled backend is not the default (B7)**

**Task:** `M3-T21`. **Branch:** `sci/tiling-default`. **ADR:** **ADR-0021**.
**Decision:** B7, answered by the operator — keep the backend, stop defaulting to it.

### The finding

`use_tiling=True` was the default and **has never tiled**. `_prepare_image` returns exactly one
640 px square, the crop shape is also 640, so `get_crops_xy` computes
`int((640 - 640) / (640 * 0.75)) + 1 = 1` step per axis. One crop, the whole image. The tiled
backend ran the direct backend's work through an extra library, more slowly, and the only reason
tiling exists — small particles at native resolution instead of downscaled into 640 px — never
happened.

**The overlap is not the lever.** `int((side - shape) / step) + 1` is 1 for *any* step when
`side == shape`; a test asserts it at 0, 25, 50 and 75 % overlap. Only the input size is: real
tiling needs `shape * (2 - overlap/100)` = **1120 px**, and a 512 px scan cannot reach it.

### The delta — zero, and for a stated reason

Inference is outside the gate (PROJECT_RULES §6): only `_prepare_image` is recorded, and it is
untouched. **`git diff` on the golden is empty.**

That is not the same as "nothing changed on real data". The two backends are **not
bit-identical** even at one crop — `MakeCropsDetectThem` preprocesses its own way and
`CombineDetections` runs a second NMS at 0.25 on top of ultralytics' `iou=0.7`. Detections on
real scans may differ slightly, nothing in the gate can see it, and **no claim is made that
either is better.** M3-T15 owns that.

### Kept, not deleted

Deleting the backend was the shortest diff and the wrong one: the question is input size, not
backend. Choosing between "upsample to 1120 and tile" (the model then examines interpolated
pixels) and "crops smaller than 640" (inference cost rises with crop count) is a
resolution-versus-cost trade-off, and the project **cannot yet measure detection quality** —
M3-T15, the evaluation harness, does not exist. Deleting now means rewriting from history later.

### Self-reporting

Asking for tiling anyway now logs that it will produce one crop and resolve nothing extra. It
lives in `_warn_if_single_crop`, separate from `_detect_tiled`, because that function imports
`patched_yolo_infer` on its first line and then runs a model — a test through it would need
weights and would run inference inside the gate. The first attempt did exactly that and took
5.7 s before failing; the seam is the fix.

### Next

**B-058** (an ADR for the golden storing CPython exception text), then **B3 → M3-T10**,
**B2 → M3-T02**, **B6 → M3-T16**, and **B-040** last.

---

## 2026-08-05 — M3-T09 · **D-10 fixed: opening radii are integers, rounded up (B4)**

**Task:** `M3-T09`. **Branch:** `sci/opening-radius-ceil`. **ADR:** **ADR-0020**.
**Defect:** D-10, medium. **Decision:** B4, answered by the operator — round up.

### The defect

`disk(8.5)` is an 18x18 element: an even side, no centre pixel, and a morphological opening
biased by half a pixel. Three sites fed `disk()` a float and each did something different — one
floored, one passed the caller's value through untouched (ADR-0014 left it that way on purpose,
pending this decision), and one was annotated `-> int` while returning a float.

Two facts narrowed the choice past the audit's table: **any integer radius is already centred**
(`disk(r)` is `2r+1` on a side), so "round to the nearest odd" solves nothing extra; and the
three sites disagreeing *is* the defect.

### The delta — 696 golden values, 0 keys added or removed

| phantom | opening radius | blobs (true) | mean height nm |
|---|---|---|---|
| `afm_flat_monodisperse` | 17 → **19** | 24 → 24 (24) | 16.1202 → 16.1194 |
| `afm_coarse_pixels` | 9 → **11** | 14 → 14 (14) | 17.8636 → 17.8664 |
| `afm_dense_overlapping` | 14 → **16** | 59 → 59 (70) | 13.3297 → **13.3791** |
| `afm_tilted_polydisperse` | 17 → **18** | 30 → 30 (30) | 16.1175 → 16.1030 |
| `afm_sparse_low_snr` | 7 → **8** | 0 → 0 (6) | — |

**No particle count moves. The largest height change is 0.049 nm — 0.37 %, on the phantom whose
particles touch.** That is the size of D-10 on data, and saying it plainly matters: this was
worth fixing because it was silent and systematic, not because it was large. The 696 changed
values are the *propagation* — radius → substrate → `z_above` → detection → every measurement —
not the defect's magnitude.

**Why +2 and not +1 on three phantoms.** The radius is estimated twice: the rough radius is
rounded up, which changes the rough substrate, which changes the Otsu radii the final radius
comes from. Same two-stage estimate as always, now with one rounding rule at both stages.

### One line, not three

The guard sits in `get_substrate_map` — the funnel every caller passes through — so one `ceil`
fixes all three sites and the fourth caller cannot forget it. `build_substrate_map` also rounds
the value it *reports*: ADR-0014 made the manual branch return the radius it actually uses, and
opening with 9 while reporting 8.5 would reinstate that lie one field along.

**mypy 18 → 15.** Three of the removed errors are this defect's static shadow — the return-type
lie and the `float` passed where an `int` was declared — sitting in the baseline since M1-T04.

### Tests

11 tests (6 parametrised). Restoring the floor turns 4 red; the centring property stays green,
because floor also yields an integer — it is the *direction* the other four pin down.

### Next

**B7 → M3-T21**, then **B-058**, **B3 → M3-T10**, **B2 → M3-T02**.

---

## 2026-08-05 — M3-T11 · **D-07 fixed: an unknown pixel scale is a state, not a crash**

**Task:** `M3-T11`. **Branch:** `sci/unknown-scale`. **ADR:** **ADR-0019**.
**Defect:** D-07, high.

### The defect

`MicroscopyData.nm_per_pixel` is `float | None` — "scale unknown" is a typed, supported state —
and `run_pipeline` hands that value straight to the detector. Both detectors multiplied by it
without looking:

```
TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'
```

An SEM or TEM image with no scale metadata therefore had exactly one outcome, and it was an
exception. The invariant D-07 states is that a physical value is **either physical or absent** —
never zero, never a pixel count wearing nanometre units, never a crash. The project already kept
it in `measure_geometry_from_mask`, which has returned `radius_nm=None` since M2-T06. The
detectors never got the same treatment.

### The delta — 168 golden keys added, 0 changed

| what | before | after |
|---|---|---|
| `detect_particles_no_scale` (5 AFM phantoms) | `TypeError` | 24 blobs, `radius_nm` all NaN, **0** detections carrying a radius |
| `boxes_to_detections_{scaled,no_scale}` (7 phantoms) | not recorded | `[5.0, 9.5]` vs `[null, null]`, same `radius_px` |
| mypy errors | 19 | **18** |

**No number moves.** Every phantom has a scale, so every existing recorded value is
byte-identical; the new keys record a path that used to raise and therefore had nothing to
record.

### Where the missing value lives, and why it is two different things

`Detection.radius_nm` becomes `float | None`. The `(N, 4)` blob array cannot hold `None` — one
dtype for the whole column — so `detect_particles` writes `NaN` there, and
`_blobs_to_detections` turns it into `None` at the entity boundary. NaN is the float convention
for "no measurement", and it is what pandas would coerce a `None` into one step downstream
anyway.

**This is not the NaN ADR-0018 deleted an hour ago.** That one came out of arithmetic
(`0 / 0`) and propagated into decisions — a threshold comparison, a sigma range, an `int()` two
calls away. This one is a marker in a reporting column, written on purpose, read by exactly one
line, and never compared against anything. The distinction is not the value but whether anything
downstream is allowed to compute with it, and it is commented at the site.

### mypy had already found it

`pipeline.py:62 — Incompatible types in assignment (expression has type "float | None", variable
has type "float")`. That error has been in the baseline since M1-T04. It is D-07, reported at
the assignment instead of at the crash, and it was read as noise from a legacy file. Annotating
the variable `float | None` — which is what the pipeline actually carries — removes it.

### Tests

8 tests. The interesting mutation is not restoring the crash, it is the *tempting wrong fix*:
`pixel_size_nm or 1.0`, which is what the npy loader does today (**D-20 / M3-T20**). It turns 4
red, including the assertion that the nm column is NaN rather than a pixel count. The pixel-space
columns are asserted **bit-identical** with and without a scale, which is what makes "only the
physical value is missing" a fact rather than a claim.

### Next

**M3-T20** is now the natural successor: this task makes `None` survivable, M3-T20 makes the
npy loader stop fabricating `1.0` — in that order, because the reverse introduces a `None` into
a path that still crashes on it. **M3-T12** and **M3-T17** are the other unblocked `high` ones.

---

## 2026-08-05 — M3-T07 · **D-11 fixed: the LoG normalisation requires a positive maximum**

**Task:** `M3-T07`. **Branch:** `sci/log-zero-max`. **ADR:** **ADR-0018**.
**Defect:** D-11, medium.

### The defect

`z_norm = z_above / z_above.max()`, at two call sites — `estimate_log_threshold_adaptive` and
`detect_particles`. Neither checked the divisor:

- **`max() == 0`** — a flat map. `0/0` makes every pixel `nan`, `blob_log` finds nothing, and
  `detect_particles` logs *"no particles found; try lowering the threshold"*. The operator is
  sent to tune a knob that cannot help.
- **`max() < 0`** — a map negative everywhere. Dividing by a negative number **flips the
  topography**: the substrate ends up brighter than the peaks. Measured, on caps sitting at
  −10 nm with peaks at −4 nm: the adaptive threshold came out **2.4997**, a number compared
  against a `[0, 1]`-normalised response, so nothing could ever exceed it.

A third site, `estimate_log_threshold`, has carried the guard since it was written. The module
already knew the answer in one of three places.

### The delta — 65 golden keys added, 0 changed

| what | before | after |
|---|---|---|
| `negative_with_structure` · `estimate_log_threshold_adaptive` | **2.4997** (never recorded) | **0.05** |
| `estimate_log_threshold_adaptive` | not recorded at all | recorded for all 11 degenerate inputs |
| `negative_with_structure` | — | added |

**Nothing changed, and that is the finding.** No number moved because the harness had never
recorded the one that was wrong. `build_substrate_map` guarantees `z_above >= 0`, so every
phantom and every scan through the normal path has a positive maximum and is byte-identical.
The negative case is reachable only through `LogDetector.detect` on a raw SEM/TEM image —
which is **D-12**, still waiting on **B3**.

### Why the harness could not see it, and the two changes that fixed that

D-11 was recorded on ten degenerate inputs and **invisible in every one**: `detect_particles`
returned an empty `(0, 4)` array before and after, because a `nan` image and a correctly
refused image both yield no blobs. So:

1. **`negative_with_structure` was added.** The existing `all_negative` is a *constant* −5, and
   dividing a constant by its own maximum gives a constant — the flip has nothing to flip.
   Structure is what makes the inversion observable.
2. **Scalars are recorded instead of being written down as the string `"non-array"`.** That one
   line in `capture.py` is why a threshold of 2.4997 sat in the harness's output, unrecorded,
   since Phase 0.

M3-T01's principle a third time: a fix that leaves its own path uncharacterized is not
finished. The harness change is the larger half of this commit.

### `not z_max > 0`, not `z_max <= 0`

They differ on `nan`, and `nan` is the case that matters — `nan <= 0` is `False`, so the
arithmetic comparison lets a `nan` maximum through and the division spreads it across every
pixel. The awkward negation is the point, and it is commented as such.

### Zero particles is an answer, not an error

The opposite call from ADR-0017 four days of commits ago, and the difference is who is wrong.
There the *caller* asked for a filter no object could pass. Here the *data* has no signal above
the substrate, and "no particles above the substrate" is true and useful about a legitimate
input — an empty region of a scan. Raising would force every caller to tell "flat" from
"broken" inside a `try`.

`DEFAULT_THRESHOLD = 0.05` is now named rather than written three times. The value does not
change.

### Tests

11 tests (5 parametrised); restoring the raw division turns **3** red. The other two guard
cases — a `nan` maximum, and `sizes` still being validated before the image — pass either way,
by construction: a `nan` image also produces no blobs, which is exactly why this defect
survived Phase 0.

### Next

**M3-T11**, **M3-T12**, **M3-T17** or **M3-T20** — the `high` unblocked defects left. M3-T21 is
still blocked on **B7**, and B2/B3/B4 remain open. **M3-T19** (low, mypy) lives in the function
this task guarded and was deliberately not folded in: it is a typing defect, not a numerical
one, and ADR-0010 forbids the bundle.

---

## 2026-08-05 — M3-T06 · **D-05 / D-06 fixed: the sizing stops lying about how many**

**Task:** `M3-T06`. **Branch:** `sci/otsu-sizing`. **ADR:** **ADR-0017**.
**Defects:** D-05 (high), D-06 (medium) — same eight lines, so one commit.

### The defects

`estimate_radius_otsu` guards `len(props) == 0` *before* the size filter and never after. When
the filter removed everything, `np.median([])` returned `nan` with a `RuntimeWarning`, and the
`nan` travelled into the LoG sigma range to surface two calls later as `zero-size array to
reduction operation minimum` — a message naming neither the parameter nor the stage. And
`n_objects` returned the *pre-filter* count while `radii_px` was already filtered.

### The delta — 8 golden differences

| what | before | after |
|---|---|---|
| `afm_sparse_low_snr` · `n_objects_reported` | **1023** | **75** |
| `degenerate_inputs.extreme_aspect` · error | `cannot convert float NaN to integer`, raised in `build_substrate_map` | the sizing's own message, raised in `estimate_radius_otsu` |
| `estimate_radius_otsu_all_filtered` | — | added, 5 phantoms |

**1023 against 75 is a 13.6× over-count** on the noisiest phantom — every one of those 948
extra "objects" is a single-pixel noise blob that the filter had already discarded. The other
four phantoms do not move, and *why* they do not is worth stating: **D-04** floors
`min_size_pixel` to 0 on coarse scans, so on most inputs the filter removes nothing and the
two counts already agreed. When **B2 / M3-T02** answers D-04, this fix starts mattering on
real data.

The `extreme_aspect` line is D-05 caught in the wild by the harness: the `nan` had reached
`max(int(nan * 2.5), 5)` in `build_substrate_map`, one call away from where it was created.

### The error message says three things, on purpose

`Otsu found 804 objects, none with a radius of at least min_size_pixel=5 px (the largest is
3.48 px).` PROJECT_RULES §3 requires the parameter and its value. The largest object is there
because without it "this image has no particles" and "your minimum size is 100× too large"
read identically, and the caller cannot tell which. It stays a plain `ValueError`: the typed
taxonomy is M3-T13, and half a taxonomy is worse than none.

### What the tests found

**An M3-T01 test started failing** — `test_a_different_radius_produces_a_different_substrate`,
written four commits ago. Its fixture is four 4.7 px particles and it passed `min_size_nm=5`
at 1 nm/px, so the filter removed all four. It had been passing *because* the sizing silently
returned `nan`: the test compared `z_above` arrays and never looked at `sizes`. That is D-05's
blast radius in miniature — the defect is invisible exactly because the `nan` sits in a field
nobody reads until something far away divides by it. The test now passes `min_size_nm=1` and
says why.

4 new tests; restoring the old behaviour turns 3 red. The fourth — the filter removing nothing
— passes either way by design, and is what guarantees the four unmoved phantoms stay unmoved.

### Next

**M3-T21** (the single-crop tiling) or **M3-T07** (D-11, LoG normalisation against a zero
maximum). M3-T21 needs a decision first — see `STATE.md`.

---

## 2026-08-05 — M3-T04 · **D-21 fixed: the scan is letterboxed, not squashed**

**Task:** `M3-T04`. **Branch:** `sci/yolo-letterbox` (off `sci/yolo-normalise-then-cast`).
**ADR:** **ADR-0016**. **Defect:** D-21, medium. Same three lines as M3-T03, the other half
of "the detector is fed correctly" — and a separate commit, because it is a separate defect.

### The defect

`_prepare_image` resized every scan to `640 × 640`, and `_scale_boxes` undid it with two
factors. The two agreed, so boxes came back where they belonged — the defect is not
misplaced detections, it is that **on a non-square scan the model never saw the sample**. At
2:1 a circular particle is an ellipse of aspect 2, and `radius_px = min(w, h) / 2` then
reports its smaller half-axis as the radius, in nanometres, to the measurement table.

### The delta

**0 golden differences. 7 keys added.** A square scan gives `scale = 640/side` and zero
padding — the old arithmetic exactly — so the 7 existing `yolo_input_preparation` blocks are
byte-identical, which the regenerated baseline proves rather than assumes.

That is also the problem: **every phantom is square (256 × 256, one 128 × 128), so the
harness could not see this fix at all.** It gains `non_square_half_height` in the same
commit — the top half of each phantom, prepared — on the same reasoning as M3-T01's harness
change: a fix that leaves its own path uncharacterized is not finished. A 128 × 256 half now
records 320 fully-255 border rows and 252–256 grey levels; under the old code it recorded a
2:1 squash and no border at all.

### Three things the reading turned up

1. **Deleting the resize would not have fixed it.** Both backends return boxes in the
   coordinates of the image handed to them, so `_scale_boxes` is only the inverse of our own
   resize — tempting to delete both. But `MakeCropsDetectThem` resizes whatever it receives
   to a multiple of the crop size with a plain `cv2.resize`. The squash would have moved
   into a dependency, where no ADR of ours governs it.
2. **The padding value is a scientific choice, not a convention.** ultralytics pads with 114
   grey. In an inverted height map that means "a particle of middling height", so it would
   draw a large one around the sample. 255 is what the *lowest* point looks like after the
   inversion, so the border reads as more substrate. It is applied after the normalisation —
   padding first would let the border join the min-max stretch and shift every grey level,
   which is D-03 again in a different disguise.
3. **`use_tiling=True`, the default, produces exactly one crop.** With a 640 × 640 input and
   a 640 × 640 crop shape, `get_crops_xy` computes `int((640-640)/(640*0.75)) + 1 = 1` step
   per axis. The sliding window covers the whole image in a single tile: the tiled backend
   does the direct backend's work, more slowly, and small particles are never seen at native
   resolution — the only reason tiling exists. Real tiling needs ≥ 1120 px of input and a
   512 × 512 scan cannot reach it. **Filed as M3-T21, not fixed here** (ADR-0010): it changes
   what inference *does*, not what it is *fed*.

### Tests

5 geometry tests beside the 6 from M3-T03: a circle on a 2:1 scan stays a circle, the border
is exactly 255, an awkward 37 × 91 shape round-trips through forward-then-inverse, a square
scan is not padded, a 4:64 strip still yields the model square. **Restoring the squash turns
4 of the 5 red.** The fifth — the square-scan invariant — passes either way by design; it is
what guarantees the golden does not move.

### Next

**M3-T06 / D-05, D-06** (Otsu sizing) is the next unblocked defect, or **M3-T21** if the
YOLO path is to be finished while it is fresh. The two remaining criticals, D-04 and D-12,
still need operator answers (B2, B3).

---

## 2026-08-05 — M3-T03 · **D-03 fixed: YOLO finally sees the data**

**Task:** `M3-T03`. **Branch:** `sci/yolo-normalise-then-cast`.
**ADR:** **ADR-0015**. **Defect:** D-03, critical.
Second numerical change of the project, and the first one the golden can measure as a
*number* rather than as an exception disappearing.

### The defect

`_prepare_image` cast the float height map to `uint8` and normalised afterwards:

```python
img = cv2.resize(z_above, (640, 640)).astype(np.uint8)   # keeps the integers in range,
img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)  # wraps the rest; too late
```

A 0–18 nm scan therefore reached the network as 19 grey levels, because those are the
integers between 0 and 18. Normalising afterwards stretched the survivors back across
0–255, so the corrupted image looked correctly exposed.

### The delta

**67 golden differences, every one under `yolo_input_preparation`, on all 7 phantoms.**
Nothing else in the baseline moved — no preprocessing, no LoG detection, no measurement.

| phantom | range (nm) | levels before | after | corr(before, after) |
|---|---|---:|---:|---:|
| `afm_flat_monodisperse` | −0.5 … 18.1 | 19 | 256 | 0.997 |
| `afm_tilted_polydisperse` | −1.8 … 45.6 | 47 | 255 | 0.914 |
| `afm_dense_overlapping` | −0.6 … 19.0 | 19 | 256 | 0.997 |
| `afm_sparse_low_snr` | −4.3 … 5.0 | **8** | 239 | **−0.499** |
| `afm_coarse_pixels` | −0.6 … 20.0 | 21 | 256 | 0.997 |
| `sem_bright_particles` | 14.5 … 230.6 | 208 | 255 | 1.000 |
| `tem_dark_particles` | 23.7 … 234.3 | 200 | 254 | 1.000 |

The audit's headline was 12.6% retention on one realistic map. Across the phantoms the
range is **3.1%–81.2%**, and the shape of that spread is the finding: **the cleaner the
sample, the worse the corruption.** A quiet 5 nm scan — a *good* measurement — kept 8 levels
of 256 and its negative heights wrapped to near-white, which is why its correlation with the
correctly prepared image is **negative**. The SEM/TEM phantoms, which are already images
spanning most of 0–255, barely moved. So the defect was invisible exactly where the code was
most likely to be eyeballed.

### What the fix is, and one thing it deliberately is not

Three lines: normalise in float, then `.astype(np.uint8)`, then invert. `bitwise_not` needs
an integer array, which is what pinned the cast too early in the first place.

The cast **truncates rather than rounds**, matching the reference implementation the
characterization harness has compared against since the baseline was recorded. That choice
is worth one sentence because of what it buys: `mean_abs_diff_vs_normalize_first`, the field
the harness added in Phase 0 purely to *size* this defect, now reads **0.0** on every
phantom. The defect's own measuring stick is now a regression guard. `cv2.CV_8U` rounding
would be ≤1/256 more faithful and would have left that number at ~0.5 forever.

**It is not bundled with D-21** (`M3-T04`), the aspect-ratio distortion two lines above —
same function, different defect, ADR-0010.

### What this does not claim

**Detections are now computed from the data. They are not thereby "better".** The weights in
`checkpoints/best12x.pt` were trained on images prepared by the old path; if the training set
came through this same function, the model learned to read 8-to-47-level posterised inputs,
and correct inputs are a distribution shift. The gate cannot say either way — inference is
outside it by PROJECT_RULES §6. Answering it needs M3-T15 (evaluation harness) and possibly
M7 (retraining). Any stored YOLO result predating this commit is not comparable to one after
it.

### Tests

`tests/unit/test_yolo_input.py` — 6 tests, written against the defect rather than the
implementation: full dynamic range, a sub-unit range (0–0.8 nm collapsed to one value
before), monotonicity, no wraparound past 255 nm, invariance under `z → a·z + b`, and the
constant-map degenerate case. **Restoring the old order turns 5 of 6 red**; the sixth is the
constant map, which is genuinely unaffected.

### Next

**M3-T04 / D-21** — the other defect in these three lines: `cv2.resize` squashes a
non-square scan to 640×640, and `_scale_boxes` stretches the boxes back with two different
factors. Same file, same context, but its own commit and its own ADR.

---

## 2026-08-04 — M3-T01 · **D-01 fixed: the manual-radius branch runs for the first time**

**Task:** `M3-T01`. **Branch:** `feat/delete-src-shims` (shared with M2's last two).
**ADR:** **ADR-0014**. **Defect:** D-01, critical.
**This is the first numerical change in the project**, and M3's rules apply: its own commit,
its own ADR, its own golden update, and the quantified delta below.

### The defect

`build_substrate_map` has two branches. The automatic one assigns `opening_radius`; the
manual one — taken whenever a caller passes `manual_radius_px` — computes the substrate and
then falls into a shared `return` that reads a variable it never bound.

**`UnboundLocalError`, on 100% of calls, on every input, since the branch was written.** Not
an edge case: the parameter has never worked. The golden already recorded the exception for
all five AFM phantoms, which is why the fix shows up as a declared change rather than a
surprise.

### The fix, and the two decisions it refuses to make

One line: `opening_radius = manual_radius_px`, the value actually passed to
`get_substrate_map`. The field is documented as "the radius finally used", and on this
branch that is the caller's value by definition.

**No rounding and no floor**, though the automatic branch applies both (`max(int(...), 5)`):

- Rounding a half-integer radius is **open decision B4 / M3-T09** — `disk(4.5)` has no
  centre pixel and shifts `z_result` by half a pixel. Choosing here would pre-empt a physics
  decision that belongs to the operator.
- Applying the floor would silently override an explicit request. A caller who asks for 3
  and receives 5 has been lied to; if 3 is wrong, that is a validation error (M3-T13).

ADR-0014 records both, including the cost: `opening_radius` is now `int` on one branch and
the caller's type on the other, which is untidy and is precisely what M3-T09 resolves.

### The quantified delta

**50 golden differences, every one of them under `build_substrate_map_manual`** — 10 fields
× 5 AFM phantoms. Nothing else in the file moved, which is the claim that matters: the path
100% of real callers use is untouched.

| Phantom | `opening_radius` | substrate mean / std |
|---|---|---|
| `afm_coarse_pixels` | 15 (int) | −1.7860 / 0.9750 |
| `afm_dense_overlapping` | 15 (int) | −1.9941 / 0.8875 |
| `afm_flat_monodisperse` | 15 (int) | −1.2760 / 0.6384 |
| `afm_sparse_low_snr` | 15 (int) | −2.6203 / 0.1917 |
| `afm_tilted_polydisperse` | 15 (int) | −2.1305 / 1.3381 |

Before: `{"ok": false, "error_type": "UnboundLocalError"}` for each.

**The harness was extended in the same commit.** It used to record only the failure and
discard the value; now it records the returned arrays, the radius and its type. Without
that, fixing the defect would have left this branch *less* characterized than it was while
broken — the golden would have said `ok: true` and nothing else.

### Verified

`tests/unit/test_substrate.py`, 6 tests, and the ones that matter are not "it returns":

- **a different radius produces a different substrate** — guards against the fix being
  cosmetic, i.e. binding the variable without the radius reaching the opening;
- **the automatic branch is untouched** — the path everyone actually uses;
- **both branches agree when given the same radius**, to `rtol=1e-6`, which is what makes
  them the same operation differing only in how the radius is chosen.

Restoring the bug turns **5 of 6 red**.

**CI green, run 22** — the first CI run with no `src/` at all, and the first to verify a
numerical fix. The re-baselined golden reproduces on a machine that is not the author's,
which is what makes the 50-difference delta a fact rather than a local observation.

### Next

`M3-T03` — YOLO input preparation normalises *after* casting to `uint8`, so only 12.6% of
the dynamic range survives (D-03, critical). Unlike this one it is covered by the golden's
`yolo_input_preparation` block on all 8 phantoms, so the before/after is a numeric delta
rather than an exception disappearing.

**Three M3 tasks are blocked on operator decisions** and cannot be started: `min_size_nm`
semantics (B2/M3-T02), opening-radius rounding (B4/M3-T09), TEM detection polarity
(B3/M3-T10). They are physics questions.

---

## 2026-08-04 — M2-T15 · M2-T16 · **`src/` is gone, and M2 closes**

**Tasks:** `M2-T15` delete the shims · `M2-T16` refresh `PROJECT_CONTEXT.md`.
**Branch:** `feat/delete-src-shims`. **Scientific impact:** none. 119 tests, golden zero
drift.

### M2-T15 — the task was bigger than its title

"Delete the shims" could not be done by deleting shims. Three modules had never had one —
`src/pipeline.py`, `src/preprocessing_pipeline.py`, `src/visualization.py` were still real
code importing `src.types` — so the exit criterion *"the pytest path hack is deleted"* was
unreachable until they moved as well:

| From | To | Why there |
|---|---|---|
| `src/pipeline.py` | `application/use_cases/pipeline.py` | It coordinates: picks a detector, sequences science and adapters, owns neither |
| `src/preprocessing_pipeline.py` | `application/use_cases/preprocessing.py` | Same |
| `src/visualization.py` | `infrastructure/imaging/plots.py` | matplotlib is a rendering dependency; the domain is defined by not having one |

Callers rewired: the characterization harness (7 import sites), three test modules, and both
notebooks. **`pythonpath` is deleted outright, not shrunk** — `nanoscope` is imported the
same way in tests, in CI, and by someone who installed the wheel. mypy points at one package
instead of two.

**A test caught a naming trap.** `use_cases/run_pipeline.py` containing `run_pipeline()`
shadows itself through the package `__init__`: `import ...use_cases.run_pipeline` hands back
the *function*, and `monkeypatch.setattr` on it fails with a confusing `AttributeError`. The
modules are `pipeline` and `preprocessing` now. A review would not have found that.

Two mypy errors followed the moved code under the strict override and are declared rather
than fixed. One is worth naming: `use_cases/pipeline.py` assigns a `YoloDetector` to a name
inferred as `LogDetector` — **that is exactly the `if/elif` dispatch the `Detector` port
exists to remove**, so the type error is the design note. M4 removes it.

### M2-T16 — the map had drifted past usefulness

`PROJECT_CONTEXT.md` described `src/`, a React frontend deleted by ADR-0012, a `pytest.ini`
deleted in M1-T05, and `preprocess_batch.py` — a script that had been broken since the
commit introducing `AFMRawData`. Rewritten: repository map, layer diagram, dependency
direction, every module path in §5–§10, dependencies, quality gates, known gaps, agent
guidance.

Three things are now said differently on purpose: the dependency rule is described as a
**test** with the file that enforces it and the measured import weight; the gaps section
says which defects M2 resolved and which it deliberately did not, with an audit ID and M3
task for each; and the agent guidance leads with *read `STATE.md` first* and with the rule
that a numerical change gets its own commit, ADR, golden update and quantified delta.

---

## M2 — milestone summary

Sixteen tasks, `M2-T01`…`M2-T16`, all closed. Against `docs/Roadmap.md`:

| Exit criterion | Result |
|---|---|
| Zero golden drift after every move | ✅ Sixteen relocations. **The only golden change in the entire milestone was six non-numeric lines**, declared in M2-T12: four translated exception messages and two `stdout_lines` counts |
| Import-graph test passes | ✅ Static over the AST, proven to fail on a real violation |
| Domain imports nothing heavy | ✅ Asserted by name. **185 modules, 0.07 s** (was 626 with matplotlib and pandas). The criterion's "< 100" was itself wrong and was corrected |
| All ports defined, behaviour reachable through them | ⚠️ **Partly, and deliberately.** One port exists because one is implemented; two of the seven were *removed* rather than deferred; behaviour still reaches `YoloDetector` by name. **M4 owns it** |
| Zero `print`, zero non-English strings | ✅ Both asserted, not just done |
| `src/` and the path hack deleted | ✅ Entirely |

**What M2 actually bought.** 2 021 lines of scientific code moved out of a directory called
`src` into four named layers, and **not one number changed**. That claim is mechanical: 8
seeded phantoms compared at `rtol=1e-6` on every commit, on a machine that is not the
author's. Tests went 23 → 119. Ruff findings in the code we own: 109 → 13, each remaining
one behind an ignore that names the task deleting it.

**What M2 got wrong, and corrected in the open.** Three plans written before the code
existed did not survive it: **"< 100 modules"** was unachievable (numpy is 141);
**seven ports** became one, with two removed outright (ADR-0013); **"10 unreachable
functions"** was four, because six were load-bearing in ways a caller count cannot see.
Each is recorded where the original claim was made, not quietly reinterpreted.

**What M2 did not touch, on purpose:** 28 open defects, 20 mypy errors, the three physics
questions blocking M3. Moving code and changing numbers in one commit makes a red golden
ambiguous — that rule held for all sixteen tasks and is the reason the milestone is
believable.

### Next

**M3 — numerical correctness**, and the rules change: every task gets its own commit, its
own ADR, its own golden update, and a quantified before/after delta. `M3-T01` is first —
`build_substrate_map(manual_radius_px=...)` raises `UnboundLocalError` on 100% of calls
(D-01), and the golden already records that exception, so the fix will show up as a declared
change rather than a surprise.

---

## 2026-08-04 — M2-T11…T14 · **The library stops shouting, speaks English, and installs**

**Tasks:** `M2-T11` logging · `M2-T12` English-only · `M2-T13` dead code · `M2-T14`
packaging. **Branch:** `feat/logging-english-deadcode-install`.
**Scientific impact:** **zero numbers moved**, and for the first time that is a claim the
golden had to be *re-baselined* to make. 115 tests, `make check` green.

### M2-T11 — and the port that should not exist (ADR-0013)

The 13 `print` calls are gone. Each module has `logging.getLogger(__name__)`, messages use
lazy `%`-formatting so the template survives into the record, and no library module
configures logging — that decision belongs to the application.

**The task asked for a `LogSink` port. There isn't one, and `ADR-0013` says why.** It would
only ever wrap `logging`, whose `Handler` is already the extension point it was going to
provide, whose `LogRecord` is already the structured payload, and whose handlers are already
attached by the application rather than the library. The SQLite destination
`Architecture.md` §3.1 describes becomes a `logging.Handler` in M6 and needs no abstraction
in `core`. `ADR-0001`'s port list is amended.

That is the second port removed from a plan written before the code existed. The pattern is
worth naming: **the ports table was designed top-down, and two of its seven entries dissolve
on contact with what the standard library already provides or with the fact that nothing
implements them.**

`tests/unit/test_logging.py`, 41 tests. The AST sweep is per-module and is the half that
stays true for code nobody thought to test; the rest exercise real call paths through
`caplog`, including the assertion that a caller who configures nothing gets silence.

### M2-T12 — and the first declared golden change

197 Russian lines across six modules translated. `grep -rn "[а-яА-ЯёЁ]"` over
`nanoscope/`, `src/` and `tests/` returns **nothing**.

`src/visualization.py` was translated in place even though it has not moved yet: it is one
of the five modules the task names, and "it will move later" is not a reason to leave
Russian in library code.

**The golden diff is the argument for having a golden at all — six lines, not one of them a
number:**

```
4× degenerate_inputs.*.build_substrate_map.error_message
   'Otsu не нашёл ни одного объекта…' -> 'Otsu found no objects…'
sem_bright_particles.log_detection_on_raw_image.stdout_lines: 8 -> 0
tem_dark_particles.log_detection_on_raw_image.stdout_lines: 4 -> 0
```

The `stdout_lines` pair is M2-T11 arriving: **the golden records how much a function
prints**, so replacing `print` with a logger is visible to it. Re-baselined with
`capture.py --write`; the re-compare is clean.

**The golden also caught a bug in M2-T11 before any human did.** The first version of the
fallback warning read `"1%% of the image width"` — `%%` is only an escape when `logging`
actually formats, which it does not when no arguments are passed, so users would have seen
the doubled sign.

### M2-T13 — four functions, not ten

Deleted: `run_full_pipeline` (a wrapper that only forwarded), `plot_pipeline_result`,
`plot_detections_histogram`, `make_synthetic_afm` (a `pass` stub). None had a caller in
code, tests, notebooks or the golden.

**The other six are kept, and reporting four is the finding rather than a shortfall.** The
audit's "10 unreachable functions" counted callers; six of them are load-bearing for
reasons a caller count cannot see:

| Kept | Why |
|---|---|
| `load_microscopy_image` | 4 tests exercise it, and the audit itself calls it structural — it is the **only** file-loading entry point for SEM and TEM |
| `estimate_log_threshold` | **The golden records its value on every phantom.** It is the baseline the adaptive variant was adopted against; deleting it deletes the comparison M3 needs if that adoption turns out wrong |
| `run_preprocessing` | No caller, but it is the documented preprocessing entry point in `README` and `Development.md`, and M4 wires it into a use case |
| `afm_viewer`, `plot_detections`, `YoloDetector.last_result` | Used by the notebooks M1-T09 deliberately kept |

### M2-T14 — installable, and one honest half-measure

`[build-system] hatchling`, `packages = ["nanoscope"]`. `src/` is deliberately not packaged:
shipping it would publish `import src`, the exact collision ADR-0011 renamed the package to
avoid. Verified against the built wheel — `py.typed` included, no `src/`, 37 modules.

**The `pythonpath` hack is half gone, and the remaining half is now written down rather than
inherited.** The `"src"` entry was the real hack: it put `src/` itself on the path, so
`import types` and `import pipeline` resolved to project modules and shadowed the standard
library. Deleted. `"."` stays because the shims and the characterization harness still
import `src.*`; M2-T15 removes both together.

Worth knowing before M2-T15: **CI does not install the project.** `uv sync --only-group ci`
installs the group and not the project — deliberately, since installing it would resolve
torch and undo M1-T08's CPU-only environment — so in CI `nanoscope` also resolves through
that `"."` entry. Confirmed by building the CI environment in a scratch venv: `nanoscope`
absent from its `site-packages`, torch absent, suite passing.

### The ignore list shrank, which was the point of dating it

`nanoscope/` ruff findings with the ignores switched off: **64 → 13**. `T201` (M2-T11),
`RUF001`/`RUF003` (M2-T12) and the whole `loaders.py` entry are deleted from
`pyproject.toml`. What remains is one real defect signature (`RUF013`, implicit Optional)
and three cosmetics that M3 takes when it touches the code anyway.

**CI green, run 21** — including the re-baselined golden, which is the run that matters:
the six declared changes reproduce on a machine that is not the author's, so the new
baseline is a fact rather than a local artefact.

### Next

`M2-T15` — delete the `src/` shims. Every caller must move to `nanoscope` first, including
the characterization harness and the two notebooks. Then `M2-T16` refreshes
`PROJECT_CONTEXT.md` and M2 is done.

---

## 2026-08-04 — M2-T09 · M2-T10 · **The layout is now enforced, and the rules are checked first**

**Tasks:** `M2-T09` import cycles + import-graph test · `M2-T10` the capability matrix.
**Branch:** `feat/import-graph-and-capabilities`.
**Scientific impact:** none for M2-T09. **M2-T10 changes behaviour on purpose** — invalid
requests now fail before inference instead of after — and the golden cannot see it, because
it never calls `run_pipeline`. `make check` green, 74 passed, golden zero drift.

### M2-T09 — five cycles, one cause

All five had the same root: `src/__init__.py` re-exported `run_pipeline`, the detectors and
the entities, and Python runs a package's `__init__` before any submodule. So
`import src.types` — the module `PROJECT_CONTEXT.md` called "the dependency root" — pulled
in the pipeline, SAM2 and matplotlib before it could hand you a dataclass.

Nothing in the repository or the notebooks ever wrote `from src import X`. Every caller
already imported the submodule. **Emptying one file broke all five cycles and cost no
caller anything.**

| | before | after |
|---|---|---|
| `import src.types` | 1198 modules, 0.77 s | **187 modules, 0.07 s** |
| `import nanoscope.core.entities` | 626 modules, matplotlib+pandas | **185 modules, neither** |

The second half came from putting `import pandas` behind `TYPE_CHECKING` in
`core/entities/pipeline.py`. `from __future__ import annotations` means `"pd.DataFrame"` is
never evaluated, so the run-time import bought nothing but ~380 modules — and
`dataclasses.fields(...).type` is that same string either way.

**`tests/unit/test_import_graph.py` checks two things, two ways, on purpose.** Direction is
static, over the AST, so a forbidden edge is caught in a module no test executes. Weight is
dynamic, in a subprocess, because pytest has already polluted `sys.modules` and because a
function-local `import torch` is fine while a module-level one is not.

Both halves were proven to fail — one infrastructure import added to a core module turns 3
tests red. There is also a guard against the glob matching nothing (a parametrised test
over an empty list passes vacuously) and an explicit test that the weight check can detect
matplotlib.

### M2-T10 — the matrix, and the word "before"

The rules lived in three places: `if` statements through `src/pipeline.py`, a table in
`PROJECT_CONTEXT.md`, and — until ADR-0012 deleted it — the React client, where the audit
found they had **already drifted** (D-19). `nanoscope/application/capabilities.py` is now
the copy that runs; the prose table documents it instead of restating it.

The half that matters is *when*. Validation used to sit after detection, so AFM + YOLO +
baseline ran a complete YOLO pass and then raised — minutes of GPU work for a request that
was invalid before any compute started (D-14). `validate_request` is the first thing
`run_pipeline` does now, and **every rejection message is byte-identical** to the one it
raised before, in the same most-specific-first order.

12 tests carry this change alone. The important ones monkeypatch both detector classes to
raise on construction, then assert the `ValueError` still arrives — they fail the moment
validation drifts back behind inference. Verified by commenting out the call: 2 red.

### A criterion that could not be met as written

M2's exit criteria say `import nanoscope.core.entities` should load **"< 100 modules"**. It
loads 185, and it cannot do better: **numpy alone is 141 modules**, and the domain is
explicitly allowed to use numpy (`Architecture.md` §3, "pure Python + NumPy"). Any module
holding an `np.ndarray` annotation pays that.

The number predates measuring it. What it was protecting is real — *don't let the domain
get expensive* — and that is now asserted directly, by name: no torch, ultralytics, sam2,
`patched_yolo_infer`, matplotlib, PySide6, cv2 or pandas. The module count stays as a
secondary bound at 250, to catch a new dependency hiding under numpy's noise floor.
`docs/Roadmap.md` records the change rather than the criterion being quietly reinterpreted.

**CI green, run 20.** 74 tests, including the new import-graph checks — which are worth
running there specifically: CI has no torch, so a module-level heavy import would fail the
job at import time rather than at the assertion, and both paths are red.

### Next

`M2-T11` — structured logging, replacing 13 `print` calls in library code, and the first
port to ship with its adapter (`LogSink`). Then `M2-T12` (English-only library code) —
between them they delete most of the ruff ignore list the science subtree carries.

---

## 2026-08-04 — M2-T07 · M2-T08 · **The dependency rule stops being a diagram**

**Tasks:** `M2-T07` model-backed code to infrastructure · `M2-T08` the ports.
**Branch:** `feat/infrastructure-models-and-ports`.
**Scientific impact:** none. `make check` green, **golden zero drift**, 34 passed.

### M2-T07 — what could never live in `core`

| From | To |
|---|---|
| `src/detection/yolo_detector.py` | `infrastructure/models/yolo.py` |
| `src/segmentation.py` — the SAM2 runners | `infrastructure/models/sam2.py` |
| `src/segmentation.py` — `afm_to_rgb`, `overlay_masks` | `infrastructure/imaging/colormap.py` |

The imaging split is the same accident M2-T06 untangled in `measure.py`. Neither function
has anything to do with SAM2 — one applies a matplotlib colormap, the other blends colours
over an RGB image — they were simply first needed there.

**This is the move that makes the dependency rule true rather than aspirational.** `core`
is *defined* by not importing torch, ultralytics, sam2 or `patched_yolo_infer`. After this
commit, nothing under `core` does. Every heavy import stays function-local, which is not a
style preference: CI installs none of those packages (M1-T08), so a module-level import
turns the job red — and that is the outcome to want, not a problem to paper over with a
`try: import torch`.

AST-verified: 4 of 6 definitions code-identical. The two that are not are named — ruff
sorted a function-local import in `YoloDetector`, and `_run_sam2_single` lost a `RET505`
`else` and had its `src.measure` import rewired to
`nanoscope.core.science.measurement`. That last one is deliberate: leaving it would have
pointed an adapter at the legacy shim, which is backwards for the layer and a cycle waiting
for M2-T09.

**ruff earned its keep.** Moving `afm_to_rgb` out left two dangling references in
`sam2.py`, and `F821` caught them before a single test ran. That is the concrete argument
for keeping `ruff check` blocking on moved code instead of excluding it wholesale the way
`src/` is.

mypy 21 → 21, but not for free: landing `yolo` and `sam2` under the strict `nanoscope.*`
override produced **16 errors in one commit** — untyped `predictor` parameters, bare `dict`
and `tuple`, `Returning Any`. Both modules join `core.science` at default strictness.
`infrastructure.storage.loaders` is deliberately **not** listed: it passes strict as it
stands, which keeps the exemption about the code rather than about the directory.

### M2-T08 — one port, and the reason for the other six

The task called for seven: `Detector`, `Segmenter`, `ImageLoader`, `ProjectRepository`,
`TrainingProvider`, `DeviceProvider`, `LogSink`. **One is written.**

Six of them have no implementation, no caller, and no second candidate implementation
anywhere in the repository. An interface written before its first adapter is a guess about
a shape. It gets rewritten the moment real code has to fit through it — except that by then
it is quoted in a document and looks decided. `core/ports/__init__.py` carries the table of
which task brings each one (M2-T11 `LogSink`, M4-T12 `DeviceProvider`, M6
`ProjectRepository`, M7 `TrainingProvider`, …). **That table is the commitment; an empty
`Protocol` would only have been the appearance of one.**

`Detector` is different, which is why it exists today: `LogDetector` in `core.science` and
`YoloDetector` in `infrastructure.models` both satisfy it *right now*, from opposite
layers, and neither imports `core.ports` to do it. That is precisely the situation an
abstraction is for.

It is a `Protocol`, not a replacement for `BaseDetector`. They are different things and the
docstring says so: `BaseDetector` is inherited and exists to share `_blobs_to_detections`
and the `radius_px = sigma * sqrt(2)` relation it carries; `Detector` is structural, so an
adapter conforms without a dependency edge back into the domain.

`tests/unit/test_ports.py`, 3 tests — and the assertions are the weaker half. mypy checks
the *signature* structurally through a typed helper; `runtime_checkable` only proves a
method of that name exists. The negative case is there too, because a port that accepts
anything is decoration. The third test asserts that importing `nanoscope.core.science`
leaves torch and ultralytics out of `sys.modules`: the dependency rule as a fact rather
than a diagram, and a down payment on M2-T09.

**CI green, run 19** — and this run carries more weight than usual: CI is the environment
with no torch, ultralytics, sam2 or patched_yolo_infer, so it is the machine that proves
the moved adapters can be imported without them. `test_ports.py`'s `sys.modules` assertion
passed there, not just here.

### Next

`M2-T09` — break the five import cycles and add the import-graph test. `src/__init__.py`
imports `pipeline`, which imports `src.types`, which is now a shim into `nanoscope`; the
cycles are all still there, and `import src.types` still loads 1179 modules. This is the
task that turns the layout into something a machine refuses to let you break.

---

## 2026-08-04 — M2-T04 · M2-T05 · M2-T06 · **Three moves, one branch, zero drift**

**Tasks:** `M2-T04` I/O · `M2-T05` the LoG detector · `M2-T06` measurement.
**Branch:** `feat/core-io-detection-measurement`. Batched at the operator's request; they
touch overlapping shims, so splitting them across branches would have meant three merges
of the same files.
**Scientific impact:** none. `make check` green, **golden zero drift**, 31 passed.
**16 top-level definitions moved.**

### What moved, and along which line

| Task | From | To | The line drawn |
|---|---|---|---|
| M2-T04 | `afm_io.py` | `core/science/io/nanoscope_spm.py` + `infrastructure/storage/loaders.py` | Parsing versus the world. Every function in `loaders.py` takes a path and opens it — `cv2` and `np.load` are adapters, not domain |
| M2-T05 | `detection/log_detector.py`, `detection/base.py` | `core/science/detection/` | Pure NumPy stays; **`yolo_detector.py` deliberately does not move** — it imports torch, so it is infrastructure, and M2-T07 owns it |
| M2-T06 | `measure.py` | `core/science/measurement/height.py` + `geometry.py` | AFM versus any modality. `height` needs a Z map; `geometry` needs only a binary mask, which is what SEM and TEM have |

M2-T06 is the one that was worth doing for its own sake rather than for tidiness: the
mask-geometry code — area, radius, circularity, aspect ratio — was trapped inside an
AFM-named module, which is why `src/segmentation.py` reaches into `src.measure` to get
shape metrics for SEM and TEM. Now it is where it belongs, and the SEM/TEM path stops
depending on an AFM module by accident.

Four `src/` modules are now shims that define nothing: `afm_io`, `measure`,
`detection/base`, `detection/log_detector`.

### Verified before the gate ran

The AST comparison from M2-T03, now over three moves at once: definitions matched by name,
docstrings stripped, bodies compared as trees. **Detection (7 definitions) and measurement
(5) are code-identical**, with docstrings differing only in trailing whitespace.

**Three functions in `loaders.py` are not verbatim, and the honest thing is to name them
rather than round the claim up.** `ruff check --fix` applied three of its *safe* fixes on
the way through:

- `RET505` — `elif` became `if`, after a branch that returns
- `UP037` — two return annotations lost their quotes, which `from __future__ import
  annotations` had already made lazy
- `PIE790` — a `pass` disappeared from a function whose docstring is its body

Each is semantics-preserving, none is in the numerical core, and the alternative — keeping
them verbatim — meant either a red commit hook or inventing three more ignores for rules
that should apply to an adapter. Recorded here, in the commit message, and provable from
the AST diff.

mypy: **21 before, 21 after**, in new locations. The two that landed in moved code are both
pre-existing: `_read_nanoscope_z` is annotated `-> np.ndarray` and returns a 3-tuple, and
`make_synthetic_afm` has an empty body.

### The finding worth keeping: ruff was wrong about the science

`RUF046` fired on `int(round(y))` in `measure_all_baseline` — "value being cast to `int` is
already an integer". It is not. `round()` on a `np.float64` returns a `np.float64`, and the
detector feeds this loop numpy scalars, so dropping the `int()` would change the dtype of
the `x_px` and `y_px` columns in every measurement DataFrame the project produces.

Ruff is reasoning about builtins; this code is fed numpy. The rule is now on the science
ignore list with that explanation attached, alongside two genuinely cosmetic ones (`N806`,
`SIM108`) — and it is the concrete argument for why "harmless style fixes" and "verbatim
move" cannot be the same commit. A linter is a good reason to look; it is not a mandate.

**CI green, run 18** — and it is worth noting *which* environment went green: CI has no
torch, no ultralytics, no sam2, so it exercised the moved code with the heavy dependencies
absent. That is the configuration M2-T07 has to keep working.

### Next

`M2-T07` — YOLO and SAM2 wrappers to `infrastructure/models/`. They import torch, which
CI does not install, so this is also the first move where the CI environment and the local
one see different code.

---

## 2026-08-04 — M2-T03 · **Behaviour moves, and the transit rules get written down**

**Task:** `M2-T03` — move preprocessing. **Branch:** `feat/core-preprocessing`.
**Scientific impact:** none. `make check` green, **golden zero drift** — and unlike M2-T02
this was real behaviour: least-squares plane fitting, per-line detrending, morphological
opening, Otsu radius estimation.

### What changed

`src/preprocess.py` → `nanoscope/core/science/preprocessing/`, split into `flatten.py`
(levelling) and `substrate.py` (what is underneath). They share no state and run at
different stages, so the split cost nothing and the directory is a real package rather
than scaffolding. `src/preprocess.py` is now a shim that defines nothing.

**The move was not eyeballed.** Every function was parsed before and after and compared as
an abstract syntax tree: **all six are code-identical**, and the only docstring differences
are trailing whitespace the formatter trimmed. The five mypy errors that live in this code
moved with it — 21 before, 21 after, none gained, none lost. That is what a move should
look like on paper before the golden is even asked.

### The real work: what happens when legacy meets a strict package

`nanoscope.*` is strict (M1-T04) and `ruff check` blocks on it. Verbatim legacy satisfies
neither — this module alone had 34 ruff findings and 5 mypy errors — 8 of the ruff ones were
whitespace the formatter fixes, leaving 22 that it cannot. Three ways out, and
only one of them survives contact with fifteen more moves:

| Option | Why not |
|---|---|
| Fix the defects during the move | They are numbers. D-04, the unbound `opening_radius`, the wrong return type — each moves a value the golden records. PROJECT_RULES §4.1, and it would make a red golden ambiguous |
| `type: ignore` / `noqa` on every line | Fifteen modules of it, burying the errors that matter in the ones we already know about, and every line has to be unpicked in M3 anyway |
| **Declare the transit status once, in configuration** | Chosen |

So: mypy runs `nanoscope.core.science.*` at **default** strictness rather than strict, and
ruff ignores **six named rules** for that subtree — the three Russian-text rules (M2-T12),
`print` (M2-T11), implicit-optional (M3) and `RET504` (cosmetic). Everything else — `E`,
`F`, `B`, `UP`, `SIM` — blocks there as normal, which is the difference between this and
the `src/` carve-out, where nothing is checked at all.

**Nothing is silenced.** The same errors report, exactly as they do for `src/` today, and
CI publishes the counts. The ledger after the move: `src/` **109 → 74**, and **22** in
the science subtree, each covered by a named ignore with an owner. Reformatting alone
removed 8 whitespace findings that will never come back. Each ignore names the task that deletes it, and both blocks shrink
to nothing as M2-T11, M2-T12 and M3 land. Declaring legacy status is not the same as hiding
it — the test is whether the declaration has an expiry date, and these do.

### Learned

- **`ruff format` on the science was a decision, not a side effect.** M1-T07 kept the
  formatter off `src/` deliberately. Here the moved file is formatted, because it lands in
  a package where formatting is blocking — and it is safe for exactly one reason: the AST
  comparison above proves it, and the golden confirms it. The rule "do not reformat the
  science" was about unverified rewriting, not about whitespace with a proof attached.
- **The audit's defects are now readable in place.** `substrate.py`'s docstring names the
  three that travel with it — the unbound `opening_radius`, D-04's `int(5 / 9.77) == 0`,
  and the `-> int` that returns a float — beside the code instead of only in a document.

**CI green, run 17.**

### Next

`M2-T04` — `afm_io.py` splits: pure parsing into `core/science/io/`, and an `ImageLoader`
port implemented in `infrastructure/storage/`. The first move that is also a **shape**
change, and the first to touch the 22 unit tests from M1-T06.

---

## 2026-08-04 — M2-T02 · **First scientific code moved, zero drift**

**Task:** `M2-T02` — extract entities and value objects. **Branch:** `feat/core-entities`.
**Scientific impact:** **none, and this time that had to be proved.** `make check` green
after the move alone (23 passed) and again with the new types (31 passed). Golden: zero
drift.

### Three commits, deliberately

The milestone rule is "any drift is a bug in the move". A single commit mixing a move with
new code would mean bisecting to find out which half moved a number.

**1 — the move.** The six dataclasses in `src/types.py` → `nanoscope/core/entities/`
(`image.py`, `detection.py`, `pipeline.py`, re-exported from `__init__`). `src/types.py` is
now a shim that **defines nothing**. `types.py` was the dependency root of the old package
— five modules import from it, it imports from none — which is why it could move first
without dragging anything with it.

Verified mechanically before the gate ran, not by eye: the pre-move module was loaded
side by side with the new one and compared field by field — identical names, order,
annotations, defaults and default factories across all six classes — and
`src.types.X is nanoscope.core.entities.X` asserted for each. **One `Detection` class in
the process, not two**; two would make `isinstance` lie across the boundary for as long as
both packages exist, and nothing in the test suite would have noticed.

**2 — the strict override, which the move walked straight into.** M1-T04 configured
`nanoscope.*` strict; legacy code arriving verbatim does not satisfy it. Three errors:

- `sizes: dict` and `masks: list[dict]` → `dict[str, Any]`. `Any` is honest, not lazy —
  both really do mix ndarrays, floats and ints under string keys. Annotations are not
  numbers (the golden records field *names*), and the gate confirmed it.
- `Detection.bbox` got a scoped `type: ignore[assignment]`, because mypy complaining there
  **is** audit defect D-16. Fixing it changes `default_detection_bbox_len`, which the
  golden records — that is M3's job, with a declared delta. `warn_unused_ignores = true`
  means the ignore becomes an error the day M3 fixes the defect, so it expires itself.

`nanoscope` is back to **0 mypy errors**. That bright line is the point: a real error in
new code cannot hide behind a known one.

**3 — the value objects.** `Modality`, `Polarity`, `PixelScale`, `DeviceKind`, plus 8 tests.

### Decision: defined, not adopted

The new types are wired to nothing. Replacing `modality: str` with `modality: Modality`
changes what `dataclasses.asdict` produces, and the golden serializes that field — so
adoption is a behavioural change and belongs to the task that has a consumer for it:
`Modality` → M2-T10, `Polarity` → M3-T10 (D-12, open decision B3), `PixelScale` →
M2-T03…T07, `DeviceKind` → M4-T12. They are unused on purpose, and the package docstring
says so, because **M2-T13 retires dead code** and would otherwise be right to delete them.

None of the four is invented. Each spells out something the code already says badly:
`Modality` the `"afm"/"sem"/"tem"` literals, `Polarity` the bright-on-dark assumption the
LoG detector makes silently, `PixelScale` the `radius_px * nm_per_pixel` and
`area_px * nm_per_pixel ** 2` arithmetic written by hand in five modules — with the guard
none of those call sites has, since a zero or NaN scale currently propagates into every
measurement instead of failing.

### Learned

- **`ruff` knew the stdlib better than I did.** `(str, Enum)` tripped UP042 and pointed at
  `enum.StrEnum`, which has existed since 3.11 and made three `__str__` overrides
  unnecessary. Less code, and the linter found it, not a review.
- **The tests were validated by mutation, as in M1-T06.** Dropping the square from
  `area_to_nm2` and weakening the guard to `< 0` both turn the suite red — 3 failures,
  including the NaN case. A test suite that has never failed is a decoration.
- **The golden's reach was checked before trusting it.** It records
  `sorted(f.name for f in fields(...))` for `PipelineConfig` and `PipelineResult` — field
  names, not types — which is exactly why the annotation tightening in commit 2 is safe and
  why adding a field would not have been.

**CI green, run 16** — the golden is reproduced on a machine that is not the author's,
which is the whole reason M1 built it.

### Next

`M2-T03` — move `preprocess.py` into `core/science/preprocessing/`. The first move of
*behaviour* rather than declarations, so the golden stops being a formality.

---

## 2026-08-04 — M2-T01 · **`nanoscope` exists**

**Task:** `M2-T01` — create the package skeleton. **Branch:** `feat/nanoscope-skeleton`.
**Scientific impact:** none — **zero lines of code moved.** `make check` green, golden zero
drift. Nothing under `src/` was opened.

### What changed

```
nanoscope/
├── py.typed
├── __init__.py
├── app/            composition root — the only layer that knows every other
├── core/           entities, values, ports, science — no Qt, no torch, no I/O
├── application/    use cases, DTOs, capabilities, jobs
├── infrastructure/ adapters; everything that touches a file, a GPU or a framework
├── gui/            PySide6 (M5), no business logic
└── resources/      assets — a package so `importlib.resources` can find them
```

Each `__init__.py` carries one paragraph: that layer's half of the dependency rule. The
rule is the only reason the directory exists, so it is written where someone adding a file
will actually be standing. M2-T09 then enforces it mechanically with an import-graph test.

Also: distribution `afm-analysis` → **`nanoscope`**, mypy's `files` extended to
`["src", "nanoscope"]`, and ruff's isort `known-first-party` to `["nanoscope", "src"]`.

### Verified

- **The lock diff was read, not trusted.** Renaming the distribution rewrites `uv.lock`,
  and CI runs `uv sync --locked`, which fails on a stale lock — but re-locking can also
  quietly re-resolve dependencies, and the golden is sensitive to numpy/scipy versions.
  Parsed both files and compared: **119 shared packages, 0 version changes**, the only
  difference being the project's own entry moving from `afm-analysis` to `nanoscope`.
  `uv lock --check` passes.
- **The strict override binds for the first time.** M1-T04 wrote
  `[[tool.mypy.overrides]] module = "nanoscope.*"` with `disallow_untyped_defs` and five
  more, four months of sessions before any such package existed — mypy had been printing
  `unused section(s): module = ['nanoscope.*']` ever since. That note is now gone, mypy
  checks **20 files instead of 13**, and `nanoscope` contributes **0 errors**. New code is
  strict from its first line, which was the point of writing the override early.
- `import nanoscope` and all six layer packages import from the repository root; the
  existing `pythonpath = ["."]` already covers it, so no packaging work was needed here.
  The editable install is M2-T14.
- **CI green, run 15.** The step that mattered was `uv sync --locked`: it accepts the
  renamed lock, which is the half of a distribution rename that fails silently locally
  and loudly on a runner.

### Decisions

- **Only the six layers ADR-0011 names — no deeper directories.** `core/entities/`,
  `core/ports/`, `core/science/io/` and the rest arrive in M2-T02…T08, each with the code
  that fills it. An empty directory tree tests nothing, and `Architecture.md` §3.1 already
  holds the plan; a second copy of a plan is a thing to keep in sync, not a skeleton.
- **No `[build-system]`, no console script.** The project is a uv *virtual* project
  (`source = { virtual = "." }`), nothing is built or installed today, and the entry point
  would have nothing to launch until M5. M2-T14 owns the install.
- **`src` stays first-party in both tools.** Two names for as long as both exist; M2-T15
  deletes the second.

### Next

`M2-T02` — extract entities and value objects from `types.py`. The first task that moves
scientific code, and therefore the first real exercise of the golden as a mechanical gate
rather than a promise.

---

## 2026-08-04 — M1-T10 · **One command is the gate — and M1 closes**

**Task:** `M1-T10` — add a one-command gate. **Branch:** `chore/make-check`.
**Scientific impact:** none. `make check` green: 23 passed, golden zero drift. No file
under `src/` was touched.

### What changed

A 53-line `Makefile` at the repository root, and `.github/workflows/ci.yml` rewritten to
call its targets.

| Target | Runs | In `check` |
|---|---|---|
| `check` | `format` → `lint` → `test` | — it *is* the gate |
| `format` | `ruff format --check .` | yes |
| `lint` | `ruff check . --no-fix` | yes |
| `test` | `pytest` (~190 s, golden included) | yes |
| `fast` | `pytest -m "not slow"` (1.4 s) | no — inner loop |
| `golden` | the golden alone | part of `test` |
| `types` | `mypy --no-pretty` | **no** |
| `lint-legacy` | `ruff check src --no-force-exclude --statistics` | **no** |

Bare `make` prints that list and runs nothing. Recipes are not silenced with `@`, so
`make check` shows the commands it runs rather than replacing them with a spinner.

**The point of the task was the second half, not the Makefile.** CI no longer contains a
copy of any command: its Format, Lint, Tests and legacy-report steps are `make format`,
`make lint`, `make test`, `make lint-legacy` + `make types`. The workflow still owns what
only it can own — the interpreter pin, `--only-group ci`, `UV_NO_SYNC`, the CPU-only
assertion, the failure report — and nothing else. The steps stay separate rather than
collapsing into one `make check` so a red job names the stage in the UI without anyone
needing log access, which M1-T08 established they do not have.

### Proven, not assumed

- **Fails closed, at the first step.** A deliberately misformatted file stopped `check`
  during `format` after **0.04 s** with exit 2 — it never reached lint or the 190 s test
  step. Reverted.
- **A failing test fails the target.** A temporary `assert False` turned `make fast` red
  with exit 2. Reverted. Both checks matter because every target is a single command:
  make's per-line shell semantics, the classic way to build a gate that cannot fail, have
  nothing to hide behind here.
- Every target was run alone. `types` reports 22 errors and `lint-legacy` 109 findings —
  both exit non-zero, both unchanged from M1-T04 and M1-T07.
- **CI run 14 is green, first try** — 216 s total: install 8 s, environment assertion 1 s,
  `make format` and `make lint` under a second each, `make test` 194 s, legacy report 6 s.
  The refactor did not disturb what only the workflow can own: the CPU-only + Python 3.12
  assertion still passes, so the green is still green for the right reason. Unlike M1-T08,
  which needed four runs, nothing here had to be discovered on the runner — the targets
  were the same ones already proven locally.

### Decisions

- **`make`, not `just`** — present on every Linux machine, which is the stated target
  platform. One line, as the task asked; no survey.
- **`types` and `lint-legacy` are outside `check`.** The legacy baseline is non-zero *by
  design* (M1-T04: reported, never silenced, never blocking). A `check` that included them
  could not pass today, and a gate that cannot pass is a gate people learn to skip. They
  are targets so the awkward flags — `--no-force-exclude` especially — are not retyped
  from memory, and CI publishes both to the run summary.
- **`.NOTPARALLEL:`** — one line, because `make -j check` would interleave the gate's
  stages and the order is the point.
- **`pretty = true` deleted from `[tool.mypy]`.** The target passes `--no-pretty` so one
  wording reads the same in a terminal and in a job summary; leaving the setting would
  have made it configuration contradicted by a flag — exactly the two-descriptions
  problem this task exists to remove.

### Learned

- **Three descriptions of one gate had already drifted.** `docs/Development.md` §4 listed
  bare `mypy`; `PROJECT_RULES` §6 listed `mypy nanoscope`, a package that does not exist
  yet, and `python tests/characterization/capture.py`, which M1-T05 folded into `pytest`.
  Nobody noticed, because nobody executes a document. Both now point at targets.
- **A gate is only single-sourced if the *slow* runner uses it too.** A Makefile that CI
  ignores is a fourth description, not a consolidation.

### M1 — milestone summary

Eleven tasks, `M1-T01`…`M1-T11`, all closed. Against the exit criteria in
`docs/Roadmap.md`:

| Exit criterion | Result |
|---|---|
| `git ls-files \| wc -l` < 100 | ✅ **64** (was 2 877) |
| lint, format, types, tests runnable via one command | ✅ `make check` — **with one deviation**: `mypy` is `make types`, deliberately outside the blocking gate while the legacy core is `src/`. It joins `check` in M2-T01, where the package is strict and blocking from its first line |
| `pytest` executes the golden and passes | ✅ 23 tests, golden inside the run (M1-T05) |
| CI runs the full gate on every push, CPU-only, no weights, no network | ✅ green, ~160 s warm, CPU-only asserted rather than assumed |
| No file over 1 MB is tracked | ❌ **two remain** — `images/yolo_sam2_comparison.png` (3.2 MB) and `images/log.png` (3.0 MB). Known and filed as **B-054**, deferred to the README rewrite (M9-T01) because recompressing published figures is a content decision. The pre-commit limit stops *new* ones; these predate it |

Also delivered beyond the stated scope: 22 unit tests for the SPM parser where there had
been one fake test, 9 commit hooks each demonstrated failing, four new defects found by
the tooling (M3-T17…T20) and two answered decisions executed (B1, B5 / ADR-0011, ADR-0012).

**What M1 actually bought.** Before: 2 877 tracked files, no linter, no type checker, one
test that could not pass, a golden enforced by discipline. After: a change is reviewable,
and "it works" is a command anyone can run — including a machine that is not the author's.
Every one of M2's sixteen relocation tasks has to prove it moved no number, and that proof
now exists mechanically.

**What M1 did not fix, on purpose:** 109 ruff findings and 22 mypy errors in `src/`, 28
open defects, 5 import cycles, 13 `print` calls, 197 non-English lines. These are M2 and
M3. M1 built the instrument; it did not use it.

### Next

`M2-T01` — the `nanoscope` package skeleton. Unblocked since B1 was answered; every other
M2 task depends on it. Before any Python upgrade, **B-058**: the golden compares CPython
exception text, so a new interpreter reads as characterization drift.

---

## 2026-08-04 — Decisions · **B1 and B5 answered; ADR-0012**

**Tasks:** none — this is the operator closing two open decisions, executed under
`PROJECT_RULES` §8 (a decision gets an ADR; reversing one gets a new ADR).
**Branch:** `chore/notebooks`
**Scientific impact:** none. `pytest` 23 passed, golden zero drift. No file under `src/`
was touched.

### B1 — the package is `nanoscope`

ADR-0011 moves from **Proposed** to **Accepted**. It had been open since M0 and was the
single remaining blocker on M2: every one of the sixteen relocation tasks needs the name.
The distribution is renamed with the package in M2-T01 — not here, because renaming without
the skeleton would leave the repository in a state neither old nor new.

### B5 — delete the parked client and the broken batch script

**ADR-0012**, superseding ADR-0007. That earlier ADR had parked `frontend/` rather than
deleting it, explicitly because *"deleting the directory outright is a separate decision
that needs the operator"*. The operator has now made it.

- **`frontend/`** — 21 tracked files, a React client written against a FastAPI backend that
  was never written. Everything ADR-0007 said about why it was never finished stands; only
  the disposal changed.
- **`preprocess_batch.py`** — broken on **every** input since `e8caf25` (**D-02**), reporting
  its own failure as `0 converted, N failed`. It had been broken for the entire period the
  audit covers and nobody noticed, which is the strongest available evidence that nothing
  depended on it. Verified by grep before deleting, not assumed.

Neither is destroyed: both are in git history, and ADR-0012 records the commit to recover
them from.

### The second-order effect, which is the real win

`preprocess_batch.py` was the **only file outside `src/` excluded from the blocking lint and
format checks**. It was carved out in three places — `pyproject.toml`,
`.pre-commit-config.yaml`, and the CI workflow — and each was a place the exclusion could
drift. All three now name exactly one path:

| | before | after |
|---|---|---|
| legacy exclusion | `["src", "preprocess_batch.py"]` | `["src"]` |
| ruff findings in the carve-out | 117 | **109**, all in `src/` |
| tracked files | 78 | **63** |
| tracked working tree | 7.8 MB | **7.6 MB** |

The exclusion is now a single, temporary, well-understood thing that M2 dissolves entirely,
rather than a list that grows whenever something else turns out to be unfixable.

### Documents brought in line

`README.md`, `PROJECT_CONTEXT.md`, `docs/Architecture.md` (§2.1, W2, W5, W16, §6, §7),
`docs/Development.md`, `docs/Backlog.md` (B-041, B-042 closed; B-046 rejected), `TASKS.md`
(M2-T13 narrowed to its remaining half; M2-T01 no longer says "confirm the name").

`docs/audit/` and ADR-0009 keep every old path: they are records of what was true then, and
PROJECT_RULES §0 freezes them. ADR-0007 is marked superseded but otherwise untouched —
its reasoning is still why the client was never finished.

### Learned

- **"Parked" is a way of not deciding, and it has a running cost.** ADR-0007 listed
  "a parked directory rots, and it will confuse future readers unless the documentation
  keeps saying it is parked" as an accepted cost. Ten months of that cost was paid in one
  afternoon of edits across eight documents.
- **A carve-out with two entries is a list; with one entry it is a deadline.** The second
  entry is what made the exclusion feel permanent, and it was there for a file nobody could
  run.

### Next

`M1-T10` — the one-command gate — then M1 closes and **M2-T01 starts**.

---

## 2026-08-04 — M1 · `M1-T09` Notebooks · **8.3 MB → 32 KB, and the gate is green everywhere**

**Task:** M1-T09 (complete)
**Branch:** `chore/notebooks`
**Scientific impact:** none. `pytest` 23 passed, golden zero drift. Nothing importable moved
— no production path referenced a notebook, which was checked rather than assumed.

### What changed

| | before | after |
|---|---:|---:|
| `afm_gold_nanoparticles.ipynb` | 6.24 MB | **0.027 MB** |
| `preprocessing.ipynb` | 2.07 MB | **0.005 MB** |
| tracked working tree | 17 MB | **7.8 MB** |

All 45 code cells survive intact — what left was 12 embedded PNGs and their base64 padding.
Both notebooks now live in `notebooks/` beside a README that states what they are, that
nothing may import them, and how to get the outputs back.

Stripping was done with the **already-configured hook**, not a separate tool: `nbstripout`
is not a project CLI, it lives inside pre-commit's isolated environment, so
`pre-commit run nbstripout --files …` runs exactly the code that will run on every future
commit. One mechanism, not two.

### `main.ipynb` deleted

A tracked **0-byte file that was not valid JSON** (audit §330). It had been deleted in the
working tree since before this session and was never committed as such; the deletion is now
recorded. Nothing referenced it.

### The gate is green across the whole repository for the first time

`pre-commit run --all-files` was red from the day the hooks were added (M1-T07), and that
red was *correct* — which is the worst state for a gate to be in, because a check known to
fail teaches people to skim past it. It now passes end to end.

The last obstacle was not the notebooks. It was a missing final newline in
`docs/archive/plan-frontend-react-client.md`, which I had twice reverted as "archived
material should not be touched". One byte. Reverting it kept the whole gate red, which was
the wrong trade — `docs/audit/` is the frozen directory, `docs/archive/` is not.

### Nothing was destroyed

The one irreversible-looking step was stripping plots of real experiments that may not be
reproducible without `data/` — 628 local scans that are not in the repository. Checked
first: **both notebooks were committed with their outputs**, so every image is still in git
history and the README carries the command to retrieve it. No export to `images/` was
needed, and none was made — that would have re-added megabytes to fix a problem git had
already solved.

### Stale references fixed

`README.md` already documented the notebooks as living in `notebooks/`, so the move made it
*less* wrong, not more. Removed its line for `sam2.ipynb`, which does not exist.
`project.md` listed four notebooks of which two exist; corrected. `docs/audit/` and
ADR-0009 mention the old paths and were **not** touched — they are historical records of
what was true then.

### Not done, deliberately

The notebooks were not executed, so whether they still run against today's `src/` is
unverified — and per the task, unverified is the correct outcome: they predate the
`AFMRawData` change (`e8caf25`, **D-02**) that silently broke `preprocess_batch.py` the same
way. That is M2/M3 information, recorded in `notebooks/README.md`, not this task's problem.

The 8.3 MB is still in git history; `.git` is unchanged. Working-tree size is what moved.
History rewriting remains **B-040**.

### Next

`M1-T10` — a one-command gate (`make check`). It is the last task in M1.

**B1, the package name, is the only thing blocking M2.**

---

## 2026-08-04 — M1 · `M1-T08` CI · **green, after four runs taught a lesson**

**Task:** M1-T08 (complete)
**Branch:** `chore/ci`
**Scientific impact:** none. Golden zero drift, in two environments.

### What was added

`.github/workflows/ci.yml` — format → lint → tests+golden → legacy report, on push and
pull request. About four minutes, three of which are the golden.

### The hard part was the environment, and it went the opposite way to expectations

`CURRENT_TASK.md` flagged the risk that `uv sync` would drag torch off the CUDA 11.8 index
and clone SAM2 from GitHub. Tracing the imports first showed something better than a
workaround: **`src/` never imports torch at all.** `ultralytics` and `patched_yolo_infer`
appear only inside two function bodies in `yolo_detector.py`, SAM2 only inside
`segmentation.py`'s call paths, and `import src.pipeline` pulls in nothing heavier than
matplotlib, pandas, scikit-image and scipy.

So CI installs a `ci` dependency group — numpy, scipy, scikit-image, pandas, matplotlib,
headless OpenCV, plus dev — and **nothing else**. No CUDA wheel, no git clone, no weights.
Versions are not repeated in the group; uv resolves every group against one lock, so CI
gets exactly what the dev machine has. That last point is not cosmetic: the golden is
sensitive to numpy/scipy/scikit-image versions, and it was checked explicitly before
anything else.

| | golden baseline | dev `.venv` | what CI installs |
|---|---|---|---|
| numpy | 2.4.4 | 2.4.4 | 2.4.4 |
| scipy | 1.17.1 | 1.17.1 | 1.17.1 |
| scikit-image | 0.26.0 | 0.26.0 | 0.26.0 |

The claim is also asserted in the workflow rather than trusted: a step fails the job if
`torch`, `ultralytics`, `sam2` or `patched_yolo_infer` is importable. A CI run that is
green for the wrong reason is worse than a red one.

### Two traps found by running it instead of writing it

**`uv run` re-syncs the project environment by default.** Every `uv run ruff …` step would
have silently reinstalled the full runtime — torch included — undoing the `--only-group ci`
install one step earlier, and the job would still have gone green. `UV_NO_SYNC: "1"` is set
at job level, and the CPU-only assertion above is what would catch a regression.

**`ruff format --check .` failed on three Markdown files.** Ruff formats Python inside
Markdown code blocks, and `docs/Architecture.md` contains illustrations, not source — a
`Protocol` sketch with `...` bodies and a registry call aligned by hand for reading. A
formatter must not be authoritative over prose examples, so `*.md` is now excluded.
Nothing found this by inspection; it took running the command that CI would run.

### One exclusion, three consumers

`src/` and `preprocess_batch.py` were excluded in `.pre-commit-config.yaml` (M1-T07) and
would have needed excluding again in the workflow — a third place to drift. They now live
in `[tool.ruff] extend-exclude` with `force-exclude = true`, declared once:

- `ruff check .` / `ruff format --check .` → the code we own; blocking in hooks and CI
- `ruff check src preprocess_batch.py --no-force-exclude` → the legacy baseline, on demand

`force-exclude` is what extends the exclusion to paths named explicitly on the command
line, which is what pre-commit passes. Verified afterwards that the hooks still skip `src/`.

### mypy reports a different number in CI, and it is not a bug

**21 errors in CI, 22 locally.** The missing one is
`yolo_detector.py:99 Incompatible types in assignment (list[Any] vs None)` — part of
M3-T18. With `ultralytics` absent, mypy types `YOLO(...)` as `Any`, and assigning `Any` to
a `None`-typed attribute is legal, so the defect becomes invisible.

A type checker's output depends on which third-party packages are installed. Worth
remembering when M2 makes `nanoscope/` strict: the strictness is only as good as the stubs
present. Documented in `Development.md` §4 with both numbers, because a CI summary that
disagrees with `STATE.md` and explains nothing is how people learn to ignore CI.

### Verification before pushing

The exact command sequence was executed against a scratch environment built the way CI
builds it (`uv sync --only-group ci --locked` into a separate `UV_PROJECT_ENVIRONMENT`,
leaving the dev venv untouched). All of it passed — and the section after this one is about
why that was not enough.

| Step, run in the CI-shaped environment | Result |
|---|---|
| `uv sync --only-group ci --locked` | 42 packages, no torch/ultralytics/sam2/patched-yolo-infer |
| CPU-only assertion | passes |
| `ruff format --check .` | clean |
| `ruff check . --no-fix` | **All checks passed** |
| `pytest -q` | 23 passed, 195 s, zero golden drift |
| legacy report | 117 ruff findings, 21 mypy errors |
| **broken parser** (z divisor mutated) | 3 failed → **exit 1** |
| **drifted golden** (`height_nm.mean` +0.1%) | 1 failed, quantity named → **exit 1** |

Both rejection cases are the DoD's real question — a CI that cannot fail is the same
non-gate as a test that cannot fail (M1-T05, M1-T06). Confirmed red on both.

### Decisions recorded

- **No `pre-commit run --all-files` in CI.** Its blocking content already runs directly
  with the same configuration, and `--all-files` is red on the two committed notebooks,
  which are M1-T09's property. CI must not be the thing that forces an unrelated task.
- **No README badge.** `README.md` is stale until M9 (D-24). A green badge on a document
  that misdescribes the project claims health it does not have.
- **mypy is non-blocking**, because `files = ["src"]` and `src/` is the legacy core. M2-T01
  points it at `nanoscope/`, where it is strict and blocking from the first line.

### Learned

- **Read the imports before designing around them.** The expensive mitigation this task
  budgeted for — a CPU torch index, a trimmed install — turned out to be unnecessary
  because the code already isolates its heavy dependencies. The cheapest answer was
  available only after tracing what actually gets imported.
- **A CI step that cannot fail is invisible.** `uv run`'s implicit sync would have produced
  a green job in a wrong environment. The assertion step exists because of it.
- **Tooling has opinions outside its remit.** Ruff reformatting documentation prose is the
  same class of surprise as `fix = true` rewriting the scientific core (M1-T03).

### Then it was pushed, and the local verification turned out to be insufficient

Everything above was true and none of it was enough. Four runs on `ubuntu-latest`:

| Run | Result | Cause |
|---|---|---|
| 1 | red at `pytest` | **Unreadable.** The annotation said `Process completed with exit code 1`; job logs need admin rights on the repository. A gate that cannot explain itself is barely a gate — pytest output now goes to `::error::` annotations and the run summary, both readable without admin. |
| 2 | red at `Install uv` | **My error.** `releases/latest` returned `v9.0.0` and I assumed a floating `@v9` tag. `actions/checkout` publishes floating majors; `astral-sh/setup-uv` does not. Both now pinned to exact releases, checked against the tag list rather than inferred. |
| 3 | red at `pytest` | **The real one.** Exactly one golden difference, and not a number. |
| 4 | **green** | Python 3.12 pinned and asserted. |

### The finding — the golden is pinned to the interpreter, and nobody knew

```
degenerate_inputs.three_dimensional.flatten_plane.error_message:
  'too many values to unpack (expected 2)'          ← golden
  'too many values to unpack (expected 2, got 3)'   ← CI
```

`capture.py:_record` stores `str(exc)` verbatim, and `degenerate_inputs` compares it
exactly. CPython 3.14 reworded that message. Reproduced locally in seconds once the
comparison was visible: 3.12.13 gives the short form, 3.14.6 the long one.

CI was on 3.14 because `requires-python = ">=3.12"` let uv take the newest interpreter on
the runner — `.python-version` was not enough. Fixed with `uv python install 3.12`,
`uv sync --python 3.12`, and an assertion on `sys.version_info` beside the CPU-only one, so
an environment mismatch fails where it happens instead of surfacing three minutes later as
mystery drift.

**The fragility itself is not fixed and should not be fixed quietly.** `Development.md`
warned that upgrading scikit-image or SciPy may move golden numbers; nobody knew the same
was true of the Python patch version, for values that are not numbers at all. M2 rests on
"zero drift" holding on a machine that is not the author's. Filed as **B-058** with three
candidate fixes; changing what the harness records is a characterization-contract change
and needs an ADR (PROJECT_RULES §4.3).

### Learned

- **"Verified locally" is a weaker claim than it sounds.** Every step was run here, in an
  environment built the way CI builds it, and it passed. It still missed the one variable
  the local environment held constant: the interpreter. The only thing that found it was
  running it somewhere else.
- **A gate that cannot say why it failed trains people to ignore it.** Two of the four runs
  were spent making the failure legible rather than fixing anything.
- **Do not infer a tag from a release.** `v9.0.0` existing says nothing about `v9` existing.

### Timings, from the green run on `main`

**236 s end to end**, comfortably inside the 8-minute budget:

| step | |
|---|---:|
| checkout + uv + Python | 4 s |
| install the CI environment | **3 s** |
| assert environment (3.12, CPU-only) | 0 s |
| ruff format + check | 0 s |
| **tests and characterization golden** | **202 s** |
| legacy baseline report | 7 s |

The install being 3 s is the `ci` dependency group paying for itself — a full `uv sync`
would have fetched a CUDA torch wheel and cloned SAM2 before running a single test. The
golden is 96% of the run, which is the right shape: everything cheap happens first and fails
fast, and the one expensive check is the one M2 depends on.

### Next

`M1-T09` (notebooks), then `M1-T10` (`make check`), and M1 closes.

**B1, the package name, is still the only thing between here and M2.**

---

## 2026-08-04 — M1 · `M1-T07` Pre-commit — the first mechanism that can refuse

**Task:** M1-T07 (complete)
**Branch:** `chore/pre-commit`
**Scientific impact:** none. No file under `src/` is modified; the golden reports zero
drift. The characterization harness was reformatted — see below, with the proof.

### What was added

`.pre-commit-config.yaml`, nine hooks, and `pre-commit 4.6.1` in the dev group.

**ruff runs as a `repo: local` hook**, calling the project's own `uv run ruff`. The
conventional `astral-sh/ruff-pre-commit` mirror declares a second ruff version in a second
file, and the two drift until local and CI disagree about what counts as a finding.
`pyproject.toml` is now the only place a version is stated.

**`ruff check --no-fix`, never `--fix`.** Formatting is not an opinion (PROJECT_RULES §3:
`ruff format` decides), but a lint autofix rewrites logic — and M1-T03 removed `fix = true`
for exactly this reason. Format rewrites; check reports.

**pytest and mypy are not hooks.** The golden alone is 200 s. A hook that slow is a hook
people bypass with `--no-verify`, and a gate that gets routed around is worse than no gate.
They go to CI (M1-T08).

### Every hook was proven to fire

Nine hooks, each given a deliberately bad staged file:

| Probe | Result |
|---|---|
| 2 MB binary | `check-added-large-files` **refused** |
| unformatted Python | `ruff format` rewrote it, commit **aborted** |
| unused import | `ruff check` **refused** |
| trailing whitespace / no final newline | both fixers **rewrote**, commit **aborted** |
| broken YAML, broken TOML | both **refused** |
| notebook with outputs | `nbstripout` **stripped** them, commit **aborted** |

An accident along the way: the first large-file probe used a `.pt` file and nothing
happened, because M1-T01's `.gitignore` had already excluded it. The hook is the second
line of defence, not the first.

### What `--all-files` revealed — the reason this task nearly shipped a bug

The sweep modified **`src/measure.py`, `src/preprocess.py`, `src/visualization.py` and
`preprocess_batch.py`**. The ruff hooks were excluded from `^src/`, but two things were not
caught by that:

- `end-of-file-fixer` and `trailing-whitespace` had **no exclusion at all** and trimmed
  inside the scientific core;
- `preprocess_batch.py` lives at the repository root, not under `src/`, so `ruff format`
  reformatted it — it is core code that the path-based exclusion simply missed.

Everything was reverted and the config now uses one named exclusion, `^(src/|preprocess_batch\.py)`,
applied to every hook that **rewrites** a file. Hooks that only **refuse** — large files,
merge conflicts, YAML/TOML — still apply everywhere, `src/` included. Nothing is exempt
from being stopped; the core is only exempt from being edited.

The posture is deliberate and matches mypy's from M1-T04: the core is reported, not
silenced, and not rewritten. Two reasons, neither of them taste. `ruff check` reports 109
findings in `src/`, so a blocking hook there would make every commit that touches the core
impossible — M2 is sixteen such tasks, and the gate would be bypassed on day one. And
PROJECT_RULES §4.1 forbids rewriting the science to make the architecture prettier: a
whitespace trim riding inside an M2 relocation commit is noise in the one diff that has to
be readable as a pure move.

### The characterization harness was cleaned, deliberately

`--all-files` also flagged 8 ruff findings and formatting in `tests/characterization/`.
Those were applied by hand rather than reverted, because a gate that is red on the day it
arrives gets ignored. All eight are behaviour-identical — `int(len(x))` → `len(x)`, two
dead `noqa: BLE001` directives (the `S`/`BLE` families are not selected), import order, and
line joins. **The golden was run afterwards and reports zero drift**, which is the only
argument that counts for a file that generates the baseline.

### Damage report — an uncommitted file was rewritten

`pre-commit run --all-files` ignores the index and rewrites the working tree. The tree held
an uncommitted `project.md` from before this session; the sweep restored its missing final
newline, and the file is now byte-identical to `HEAD`. Nothing was lost beyond that one
newline — 11752 bytes before, 11753 after, no textual difference — but the hazard is real
and is now a warning in `docs/Development.md` §4: commit or stash before running
`--all-files`.

### Measurements

| | |
|---|---|
| Hooks configured / proven to fire | 9 / 9 |
| `pytest` | 23 passed, 188 s |
| Characterization golden | zero drift, after the harness reformat |
| Files under `src/` modified by this task | **0** |
| `--all-files` still failing on | the two committed notebooks (M1-T09) and one archived doc — knowingly, both are other tasks' property |

### Learned

- **A path-based exclusion is only as good as the paths.** `^src/` looks like "the
  scientific core" and is not: `preprocess_batch.py` sits at the root and imports it. The
  sweep is what showed this; a config review would not have.
- **Hooks that rewrite and hooks that refuse need different scopes.** Conflating them
  either blocks legitimate commits or edits code nobody asked to touch. Splitting the two
  made the whole configuration obvious.
- **`--all-files` is not a dry run.** It edited a file that was not staged, not committed,
  and not part of this task.
- The `.gitignore` from M1-T01 already stopped the model-weight probe before pre-commit saw
  it. Layered defences are working as intended, and worth remembering when reading a green
  hook run: it may be green because something earlier said no.

### Next

`M1-T08` — CI. The slow half of the gate that pre-commit deliberately refuses to run:
`pytest` including the golden, plus ruff and mypy reporting on `src/` without blocking.

**B1 remains the only thing blocking M2.**

---

## 2026-08-04 — M1 · `M1-T06` A real test for the SPM parser · **the suite is green**

**Task:** M1-T06 (complete)
**Branch:** `test/spm-io`
**Scientific impact:** none — `src/` is not edited. The golden reports zero drift.

### What was there

Eleven lines that tested nothing: no assertion, `z` assigned and never read, `ImportError`
caught for `pyfmreader` (a package this project does not depend on — the parser is
hand-written) while the actual failure is `FileNotFoundError`, and a read of `data/5.011`,
which is git-ignored and absent from any clean checkout. It failed on every machine, and
had the file been present it would have passed regardless of what the parser returned.

### What replaces it

`tests/unit/test_afm_io.py` — **22 tests**, no binary fixture, no `data/`, no network.

The fixture is a synthetic Nanoscope SPM byte stream built in the test module: preamble,
a decoy image block, the Height block, `0x1A`, padding to the declared data offset, then
an `int16`/`int32` payload. Field names and formats were taken from a **real** local file
(`data/pvp8k/2-6-dmfa-pvp.039` — read, not committed), including the two details that
matter: Nanoscope writes micrometres as `~m`, and every header carries a second
sensitivity, `@Sens. ZsensSens`, thirty times the real one.

| Group | Covers |
|---|---|
| Round trip | shape, `float32`, `[y, x]` orientation, values (non-square 6×4, so a transpose cannot survive), Height-block selection over a decoy, `int32` when `Bytes/pixel != 2` |
| Calibration | `pixel_size_nm == scan_size_nm / samps`, the full LSB → volts → nm chain, and `~m` / `um` / `µm` / `nm` conversion |
| Failure modes | missing file · no Ciao blocks · missing header field · no Z scale · no Zsens · truncated payload · no `Scan Size` (M3-T17) · unsupported format |
| Other entry points | `fmt="npy"` with and without metadata; `load_microscopy_image` greyscale round trip, unknown scale, missing file |

### The suite was tested, not just written

A test suite that has never failed is a hypothesis. Four mutations of `src/afm_io.py`, run
and reverted:

| Mutation | Result |
|---|---|
| `pixel_size_nm = scan_size_nm / lines` instead of `/ samps` | **5 failed** |
| Z scale divisor `65536 → 32768` | **4 failed** |
| Height-block selection replaced by "take the first block" | **13 failed** |
| Zsens regex loosened to `Zsens\w*`, so it also matches `ZsensSens` | **survived** |

The fourth is the one worth recording. The decoy `ZsensSens` line was only being written
when the correct `Zsens` line was also present and earlier in the file, so `re.search`
found the right one either way and the test proved nothing. The real hazard is a header
that has `ZsensSens` but no `Zsens`: a loosened pattern would then silently scale every
height in the scan by ~30 and raise nothing. The fixture now always carries the decoy, and
`test_spm_without_zsens_is_rejected_and_zsenssens_is_not_a_substitute` kills the mutant.

Written and passed, that test was decoration. Only the mutation showed it.

### New defect found — M3-T20

`load_afm(fmt="npy")` fabricates a physical scale: `pixel_size_nm or 1.0` and
`scan_size_nm or float(z.shape[0])`. PROJECT_RULES §3 and D-07 both say an unknown scale is
`None` — never a stand-in. So every downstream `_nm` on that path is a pixel count wearing
nanometre units; the row count is used as a length in nanometres, which is not even
dimensionally a size; and because it is written with `or`, a caller who explicitly passes
`0.0` is overruled too. Not in the audit, not previously filed → **M3-T20**, high.

Both this and M3-T17 are pinned by assertions that name the task, so the fix flips a
documented expectation instead of breaking a surprise.

### Measurements

| | |
|---|---|
| `pytest` | **23 passed**, 200 s — first green run in the project's history |
| `pytest -m "not slow"` | 22 passed, **0.88 s** |
| Characterization golden | zero drift |
| mypy | 22 errors, unchanged — `files = ["src"]`, tests are not checked |
| ruff check / format on the new file | clean |
| Binary fixtures added | none |

### Learned

- **The parser is more testable than it looks.** It needs six header fields and two regex
  matches; a faithful fixture is ~60 lines. The reason it had no tests was not difficulty.
- **Deriving the fixture from a real file paid for itself immediately.** `~m` for
  micrometres and the `ZsensSens` twin are not things one invents at a desk, and both are
  now covered.
- **Mutation testing found the one worthless test out of 23.** Cheap — four edits and four
  runs — and it is the only reason the Zsens guard is real. Worth repeating whenever a test
  claims to defend a subtle regex or a unit conversion.
- The fast loop is 0.88 s, of which **1.4 s of import cost is D-18** — `import src.afm_io`
  pulls 1209 modules through `src/__init__.py`. It is inside pytest's startup rather than
  the test time, but it is the same defect, and M2-T09 removes it.

### Next

`M1-T07` — pre-commit hooks. With the suite green, the gate can start refusing bad commits
instead of reporting them afterwards.

**B1 remains the only thing blocking M2.**

---

## 2026-08-04 — M1 · `M1-T05` The golden runs under pytest

**Task:** M1-T05 (complete)
**Branch:** `chore/golden-in-pytest`
**Scientific impact:** none — no golden value changed, no numerical code touched.
`capture.py`'s comparison, tolerances and digests are byte-for-byte the same; the CLI
prints the same line it printed before (`characterization baseline stable (9 groups)`).

### What changed

- **One seam in `capture.py`**: `diff_against_golden() -> list[str]` — read the golden,
  `build_all()`, `compare()`, return the path-addressed diff. `main()` now calls it and
  keeps sole ownership of printing and exit codes. Nothing else in the file moved.
- **`tests/characterization/test_golden.py`** — one `@pytest.mark.slow` test that asserts
  the diff is empty and puts it in the assertion message. It reimplements nothing; if the
  test and the CLI ever disagreed it would be because they share a code path, and they do.
- **`pytest.ini` deleted, configuration folded into `pyproject.toml`** (scope item 7).
  This was the one open decision in the task, and the deciding fact is not tidiness: while
  a `pytest.ini` exists pytest ignores `[tool.pytest.ini_options]` **entirely and
  silently**. Two files that can shadow each other is exactly the failure mode this task
  exists to remove. The `pythonpath = [".", "src"]` hack moved across unchanged and still
  dies in M2-T14.
- `docs/Development.md` §4 and §5 document both invocations.

### The proof that matters

A test that cannot fail is not a safety net, so the negative case was run rather than
assumed. `afm_flat_monodisperse…detect_particles_p20.n_blobs` was edited 24 → 23 in the
golden file:

```
E   AssertionError: CHARACTERIZATION DRIFT: 1 difference(s)
E       afm_flat_monodisperse.log_detection.detect_particles_p20.n_blobs: 23 -> 24
```

Red, one line, the quantity named with both values. The golden was then restored;
`git diff` on `baseline.json` is empty.

### Measurements

| | |
|---|---|
| `pytest tests/characterization/test_golden.py` | **passed**, 192 s |
| `pytest -m "not slow"` | 1.4 s, golden deselected, `test_io.py` fails as expected (M1-T06) |
| `python tests/characterization/capture.py` | unchanged output, exit 0 |
| Marker warnings | none |
| ruff on the new file | clean, formatted |

### Learned

- **The task estimated ~100 s; it is 192 s.** The figure in `Development.md` was inherited
  and never measured. Corrected there. It matters: this is the number that decides whether
  people keep running the full suite, and `-m "not slow"` is the answer to it.
- **`pytest.ini` + `[tool.pytest.ini_options]` is a silent-override trap.** Had the marker
  been registered in `pyproject.toml` while `pytest.ini` still existed, the registration
  would have done nothing and the warning would have stayed — with no error to explain it.
- **The harness was already test-shaped.** `build_all()` and `compare()` were pure; only
  `main()` mixed in printing. One extracted function was the whole job — no restructuring,
  hence no risk to the numbers.
- The two `RuntimeWarning`s pytest now surfaces (`Mean of empty slice`, `Degrees of freedom
  <= 0`) are not new. They come from the degenerate-input phantoms and always went to
  stderr; the CLI just made them easy to overlook. They are characterized behaviour.

### Next

`M1-T06` — replace `tests/test_io.py`. It is the only thing keeping `pytest` red: no
assertions, catches `ImportError` while the real failure is `FileNotFoundError`, and it
reads `data/5.011`, a path that does not exist in a clean checkout.

**M2 is no longer blocked by the safety net** — the golden is mechanically enforced. It is
still blocked by **B1**, the package name.

---

## 2026-08-04 — M1 · `M1-T04` Mypy configuration

**Task:** M1-T04 (complete)
**Branch:** `chore/mypy-config`
**Scientific impact:** none — configuration only, no source file edited

### The 30 errors, classified before writing any configuration

The rule for this task was that configuration must not silence a real bug. So all 30
default-run errors were read against the source first.

| Class | Count | Disposition |
|---|---:|---|
| Missing third-party stubs | 8 | silenced per module — pandas, scipy, patched_yolo_infer, ultralytics |
| **Static confirmation of known audit defects** | **13** | kept visible; already have M3 tasks |
| **Real typing defects not previously filed** | **5** | kept visible; **filed as M3-T17…T19** |
| Stub strictness / known stub | 4 | kept visible, harmless |

**The 13 that confirm the audit.** mypy independently reproduces, statically, defects the
Phase 0 audit found by execution:

| Error | Defect |
|---|---|
| `preprocess.py:202  Cannot determine type of "opening_radius"` | **D-01** — the critical `UnboundLocalError`. The manual-radius branch never assigns it; mypy sees the same missing assignment the runtime does. |
| `preprocess.py:149,158  return float, expected int` + `:184 arg-type float→int` | **D-10** — `estimate_rough_radius` is annotated `-> int` and returns a float, which reaches `disk()` and produces even-sized structuring elements. The whole chain is visible in three errors. |
| `types.py:63  tuple[Never, ...] vs tuple[int, int, int, int]` | **D-16** — `bbox` defaults to `()` against a four-element annotation. |
| `preprocess.py:164`, `log_detector.py:125,257`, `pipeline.py:52` | **D-07** — implicit `Optional` and the unknown-pixel-scale contract. |
| `pipeline.py:94 ×2, :110  ndarray \| None where ndarray expected` | the SEM/TEM path, where `z_flat` is `None` by construction. |
| `afm_io.py:100  returns tuple, annotated ndarray` | **D-02** — the return-convention change that silently broke `preprocess_batch.py`. |

A configured type checker would have caught the project's single critical defect before
it was ever committed.

**The 5 that are new.** Filed as tasks, not suppressed:

- **`afm_io.py:98` — new defect, not in the audit.** When the header carries no
  `Scan Size:` field the code sets `scan_size_nm = None` and then immediately evaluates
  `pixel_size_nm = scan_size_nm / samps` → `TypeError`. The `else` branch exists
  specifically to handle that header, and it crashes on the next line. → **M3-T17**
- `yolo_detector.py:50,87,99` — `_last_result` is initialised to `None`, so its inferred
  type is `None`; `.filtered_boxes` is then accessed unguarded. → **M3-T18**
- `log_detector.py:111,116` — `responses` is annotated `list[float]` and then rebound to
  an ndarray before `.min()`/`.max()`. Works at runtime, wrong as a contract. → **M3-T19**

### Configuration

- `[tool.mypy]`: `python_version = "3.12"`, `files = ["src"]`, `warn_unused_configs`,
  `warn_redundant_casts`, `warn_unused_ignores`
- **`nanoscope.*` is strict from its first line** — `disallow_untyped_defs`,
  `disallow_incomplete_defs`, `disallow_untyped_calls`, `disallow_any_generics`,
  `check_untyped_defs`, `no_implicit_optional`, `warn_return_any`, `strict_equality`.
  Retrofitting strictness later is far more expensive than starting with it.
- **`src/` posture: checked, not silenced.** No `ignore_errors`. It carries 22 errors,
  13 of which are the most valuable output this tool has produced; hiding them to make a
  number green would be the opposite of the point. The package is deleted in M2-T15, so
  the errors are a documented baseline, and CI reports them without blocking (M1-T08).
- `ignore_missing_imports` scoped **per module**, never globally — a blanket setting would
  also hide a typo in a first-party import.

`mypy` now runs with no command-line flags: **22 errors in 7 files** (30 minus the 8
stub gaps). It emits one note — `unused section(s): module = ['nanoscope.*']` — which is
deliberate: it is a visible reminder that M2-T01 has not happened yet, and it disappears
the moment the package exists. Verified non-fatal: mypy exits 0 on a clean file with that
note present.

### Considered and rejected

Installing `pandas-stubs` and `scipy-stubs` instead of silencing those imports. It would
give real coverage, but pandas-stubs against pandas 2.x typically produces a fresh wave of
errors in code that is scheduled for deletion in M2. Revisit when `nanoscope` exists —
backlog **B-057**.

### Next

`M1-T05` — wire the characterization golden into `pytest`. It is the only check that
passes today and the only protection M2 has, and it currently runs only when someone
remembers to type the command.

---

## 2026-08-03 — M1 · `M1-T03` Ruff configuration repair

**Task:** M1-T03 (complete)
**Branch:** `chore/ruff-config`
**Scientific impact:** none — `capture.py` reports `characterization baseline stable (9 groups)`

### Done

- **Removed `fix = true` / `show-fixes`.** `ruff check .` was rewriting source files as a
  side effect of being asked a question — 66 automatic edits to the scientific core, from
  a command the documentation told people to run. Fixing is now explicit: `ruff check --fix`.
- Moved `select` / `ignore` under `[tool.ruff.lint]`; the deprecation warning is gone
  (stderr is now empty).
- `target-version` `py311` → `py312`; the project requires `>=3.12`.
- `known-first-party` `["your_package_name"]` → `["src"]` (unedited template value; it
  becomes the real package in M2-T01).
- `classmethod-decorators`: dropped `pydantic.validator` — pydantic is not a dependency.
- `per-file-ignores`: dropped `S101`; the `S` (bandit) family is not selected, so the
  entry was dead configuration. Backlog **B-056**.
- Excluded `*.ipynb` from lint. Notebooks are experiments, not interfaces
  (PROJECT_RULES §7); their 68 findings are import order and prints in exploratory cells.
  Notebook hygiene is M1-T09.

### Verification

| Check | Result |
|---|---|
| Config deprecation warning | gone — stderr empty |
| `ruff check .` modifies the tree | **no** — `git diff --exit-code` clean after every run |
| `src/` findings before vs after | **identical** — `--statistics` diff is empty |
| Total findings | 196 → **128** (the 68 excluded are all notebooks) |
| `ruff format --check .` | runs; 18 files would be reformatted (not fixed here — M2) |
| Characterization | zero drift |

### Correction to the M1-T02 baseline

The `src/` figure recorded in M1-T02 was **109, not 108**. I produced the 108 by summing
a `--statistics` listing that I had truncated with `head -20`, dropping the last row
(`W291 trailing-whitespace`, 1). The commit message of `13857e5`, and `STATE.md` /
`TASKS.md` before this entry, carry the wrong number.

The distribution is otherwise unchanged, and the invariant this task was checked against
still holds: the configuration repair changed **nothing** about what is reported — the
before/after statistics diff is empty. Corrected everywhere in the living documents;
the commit message of `13857e5` is history and stays as written.

Burn-down target for M2 is therefore **109 findings in `src/`**, of which 44 are the
ambiguous-unicode signature of the Russian text (D-22) and 13 are `print` (D-23).

### Next

`M1-T04` — mypy configuration. 30 errors today with default settings; the task is to
choose strictness for new code and a baseline exclusion for `src/` until M2 lands, not to
fix the errors.

---

## 2026-08-03 — M1 · `M1-T02` Dev dependencies and quality baseline

**Task:** M1-T02 (complete)
**Branch:** `chore/dev-dependencies`
**Scientific impact:** none — `capture.py` reports `characterization baseline stable (9 groups)`
after the environment change

### Done

- `[dependency-groups] dev` added to `pyproject.toml`: pytest, pytest-cov, ruff, mypy
- `uv sync`; `uv.lock` gained **only** tooling packages — no runtime version moved.
  Verified against the golden `_meta` pins: torch 2.7.1+cu118, NumPy 2.4.4, SciPy 1.17.1,
  scikit-image 0.26.0 — all unchanged, which is why the golden is still stable

| Tool | Version |
|---|---|
| pytest | 9.1.1 |
| pytest-cov | 7.1.0 |
| ruff | 0.16.1 |
| mypy | 2.3.0 |

### Baseline — the M2 burn-down target

Measured, nothing fixed. These are the numbers M2 has to drive to zero.

**ruff — 196 findings total, 109 in `src/`** (66 auto-fixable)
*(corrected in the M1-T03 entry above; this session recorded 108 from a truncated listing)*

| Rule | src/ | What it is |
|---|---:|---|
| RUF002/003/001 | **44** | ambiguous unicode in docstrings, comments, strings — this is the Russian text of **D-22**, found mechanically |
| T201 | **13** | `print` in library code — **exactly the 13 the audit counted by hand** |
| F401 | 11 | unused imports |
| I001 | 11 | unsorted imports |
| W293/W292/W291 | 10 | whitespace |
| RET504/505 | 7 | unnecessary assign / superfluous else |
| RUF046 | 2 | unnecessary `int()` cast — **adjacent to D-10**, the opening-radius rounding defect |
| RUF013 | 2 | implicit `Optional` — the unknown-scale contract (**D-07**) |
| A005 | 1 | `src/types.py` shadows the stdlib `types` module — a real M2-T02 constraint |
| others | 8 | B007, C408, N806, PIE790, RUF022, SIM108, UP037 ×2 |

The remaining 88 findings are in notebooks, `preprocess_batch.py` and
`tests/characterization/capture.py`.

**mypy — 30 errors in 9 files** (default settings, no configuration yet)

| Code | Count |
|---|---:|
| import-untyped | 8 |
| assignment | 7 |
| arg-type | 6 |
| return-value | 3 |
| attr-defined | 3 |
| has-type, empty-body, call-overload | 3 |

The most interesting one is static confirmation of a known runtime defect:

```
src/pipeline.py:110: error: Argument 4 to "run_sam2_from_blobs" has incompatible type
"ndarray[...] | None"; expected "ndarray[...]"
```

That is the SEM/TEM path, where `z_flat` is `None` by construction (`pipeline.py:53`).
A type checker would have caught it before it ever ran.

**pytest — 1 test, 1 failed**

```
FAILED tests/test_io.py::test_load_spm - FileNotFoundError: 'data/5.011'
```

Correction to the prediction in `CURRENT_TASK.md`: the test does not pass vacuously, it
**fails**. It catches `ImportError` while `load_afm` raises `FileNotFoundError` (audit
D-20), and the path is absent from a clean checkout. The suite has been red the whole
time; nobody could see it because pytest was never installed. M1-T06 replaces it.

### Side effect worth knowing about

`uv sync` uninstalled three packages that were in the environment but not in `uv.lock`:
`clip` (from the ultralytics CLIP repo), `ftfy`, `regex`. They were installed outside uv,
so `uv sync` removed them to match the lock — expected behaviour, but not something I
intended.

Nothing under `src/` imports them; they are needed only for **YOLO-World** models, and
`checkpoints/yolov8s-world.pt` is such a model (it is not the configured default —
`PipelineConfig.yolo_model_path` points at `best12x.pt`). If YOLO-World is wanted, the
fix is to declare it as a real dependency rather than let it be installed ad hoc:

```bash
uv add "clip @ git+https://github.com/ultralytics/CLIP.git"
```

Recorded as backlog **B-055**.

### Not done, deliberately

No finding was fixed. Repairing the ruff configuration is M1-T03, the mypy configuration
is M1-T04, and the 109 `src/` findings are M2 work — under the protection of the golden
file, not before it.

### Next

`M1-T03` — repair the ruff configuration. It currently emits a deprecation warning for
top-level `select`/`ignore`, targets `py311` on a 3.12 project, still carries
`known-first-party = ["your_package_name"]`, and — most importantly — sets `fix = true`,
so `ruff check .` rewrites source files as a side effect of being asked a question.

---

## 2026-08-03 — M1 · `M1-T01` Repository hygiene

**Task:** M1-T01 (complete), M1-T11 (complete — absorbed)
**Branch:** `chore/repo-hygiene`
**Scientific impact:** none — `capture.py` reports `characterization baseline stable (9 groups)`

### Result

| Metric | Before | After |
|---|---:|---:|
| Tracked files | 2 877 | **77** |
| Tracked under `frontend/node_modules` | 2 800 | **0** |
| Tracked model weights | 1 (`yolov8s-world.pt`, 26 MB, staged) | **0** |
| Largest tracked non-notebook file | 6.5 MB | 3.2 MB (README figure) |

### Done

- `git rm -r --cached frontend/node_modules` — 2 800 files untracked, all still on disk
- `git rm --cached yolov8s-world.pt` — the 26 MB blob (`2fa1b38`) was staged for addition
  and would have entered history on the next `git commit` without a pathspec. Removed
  from the index before that happened; the file is not on disk and the four real
  checkpoints under `checkpoints/` are untouched
- `.gitignore` rewritten: added `node_modules/`, `output/`, `*.pt`, `*.pth`, `*.onnx`,
  `*.safetensors`, `*.zip`, `*.tar.gz`, `build/`, `dist/`, `*.egg-info/`, `.mypy_cache/`,
  `.coverage`, `htmlcov/`; grouped and commented by rationale
- `.claude/settings.json` now **tracked** — agent configuration is shared (PROJECT_RULES §7).
  `.claude/settings.local.json` (per-machine permissions) stays ignored. This completes
  the `.gitignore` edit that was already sitting uncommitted in the working tree
- **Junk removed from disk:**
  - `.zip` — a 22-byte *empty* zip archive, tracked since February
  - `__pycache__/` × 4, including `.pyc` files for modules that no longer exist
    (`sam2_pipeline`, `config`, `detection`) and bytecode from CPython 3.14 while the
    venv is 3.12 — stale in two independent ways
  - `.pytest_cache/`, `.ruff_cache/`
  - `output/`, `notebooks/` — both empty directories
  - root `package-lock.json` — an empty stray from an accidental `npm install` at the
    repository root (the real one is `frontend/package-lock.json`)
- `plan.md` → `docs/archive/plan-frontend-react-client.md`, un-ignored and now tracked,
  with an ARCHIVED header pointing at ADR-0007. It was gitignored, so the only record of
  the intended HTTP contract was unshareable. Path references in ADR-0007 updated
  (editorial only — the decision is unchanged)

### Deviation from the stated Definition of Done

"Largest tracked file, excluding the pre-existing notebooks, < 1 MB" is **not met**:
`images/yolo_sam2_comparison.png` (3.2 MB) and `images/log.png` (3.0 MB) are README
figures. They are legitimate content, not junk, and untracking them would break the
README. Recorded as backlog **B-054** (optimise figures) rather than silently ignored.

The notebooks (6.5 MB + 2.2 MB, committed with outputs) are untouched — that is M1-T09.

### Not done, deliberately

History still carries the 78 MB: `git rm --cached` stops the growth, it does not shrink
`.git` (still 81 MB). Rewriting history invalidates every clone and needs the operator's
approval — backlog **B-040**.

### Next

`M1-T02` — declare `pytest`, `pytest-cov`, `ruff`, `mypy` as dev dependencies. None of
them is installed today, so the quality gate does not exist yet; the characterization
runner is currently the only working check.

---

## 2026-08-03 — M0 · Engineering foundation

**Tasks:** M0-T01 … M0-T08 (all complete)
**Branch:** `docs/engineering-infrastructure`
**Base:** `11e0ecc` (frontend init)
**Code changed:** none — documentation only, by design

### Done

- Read `systempromt.md`, `PROJECT_CONTEXT.md`, `README.md`, plus `project.md`, `plan.md`,
  the Phase 0 audit and the characterization baseline
- Analysed the repository directly: 12 modules / 2 021 LOC under `src/`, 13 frontend
  source files, 8 characterization phantoms, 2 854 tracked files
- Recorded 7 strengths and 16 weaknesses with measured evidence → `Architecture.md` §2
- Defined the target Clean Architecture: `app` / `core` / `application` / `infrastructure` /
  `gui` / `resources`, with an enforced dependency rule and a layer-contract table
- Wrote the constitution → `PROJECT_RULES.md`
- Broke the project into 10 milestones (M0–M9) with exit criteria → `Roadmap.md`
- Broke the milestones into 110 tasks → `TASKS.md`
- Wrote 11 ADRs (0001–0011) → `docs/ADR/`
- Established the state protocol → `STATE.md`, `CURRENT_TASK.md`, this file
- Selected `M1-T01` as the first task

### Learned

- **The starting position is better than a greenfield.** A completed, *reproduced* Phase 0
  audit and a committed golden baseline over 8 seeded phantoms already exist. That changes
  the strategy: the domain can be moved aggressively, because drift is detectable to
  `rtol=1e-6`.
- **The stack pivoted.** The previous direction was React + a FastAPI backend that was
  never written; the target is a Qt6 desktop application. The React client is the only
  work made obsolete by the pivot, and it is parked rather than deleted (ADR-0007).
- **The domain layer is genuinely worth preserving.** A modality-neutral `Detection`, a
  `BaseDetector` ABC, lazily imported SAM2 and a deliberate dependency root mean the
  Clean Architecture target is an extraction, not a rewrite.
- **Two problems are urgent for different reasons.** `node_modules` (98% of tracked files)
  makes review impossible; the staged `yolov8s-world.pt` has a closing window before it
  is permanently in history. Both are M1-T01.
- **Structure must precede correctness.** Fixing D-03 or D-04 today would change numbers
  inside a codebase with 5 import cycles and no test gate. M2 before M3 is not
  bureaucracy — it is the only way the deltas stay attributable.

### Open questions raised

B1 package name · B2 `min_size_nm` semantics · B3 detection polarity · B4 opening-radius
rounding · B5 fate of `frontend/` and the notebooks · B6 real sample data in git.
Full text in `STATE.md`. None blocks M1.

### Next

Execute `M1-T01` on branch `chore/repo-hygiene`: untrack `frontend/node_modules` and
`yolov8s-world.pt`, rewrite `.gitignore`. See `CURRENT_TASK.md`.

---

## Before 2026-08-03 — inherited context

Not a session log; recorded so the history is not lost.

| When | What |
|---|---|
| 2026-07-28 | Phase 0 audit: 24 defects reproduced by execution, 5 import cycles, 10 dead functions → `docs/audit/2026-07-28-baseline-audit.md` |
| 2026-07-28 | Characterization baseline: 8 seeded phantoms, golden file at `rtol=1e-6` → `tests/characterization/`, `docs/audit/characterization-baseline.md` |
| `11e0ecc` | React + Vite frontend scaffolded against an unimplemented `/analyze` backend |
| `e8caf25` | `afm_io` reworked to return `AFMRawData` (silently broke `preprocess_batch.py` — D-02) |
| `cd360aa` | Generalisation to SEM/TEM: `MicroscopyData`, `load_microscopy_image` |
| `f1cf175` | `types.py` and `pipeline.py` introduced — the first deliberate layering |
| `0ef8c50` | Detection refactored to `BaseDetector` + `LogDetector` / `YoloDetector` |
| earlier | SAM2 integration, tiled YOLO, LoG baseline, morphological substrate estimation |
