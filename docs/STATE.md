# STATE

**Last updated:** 2026-08-13 · **Branch:** `feat/m5-gui-shell` · **Base commit:** `aceb5c7`

> This file is mandatory and must be updated at the end of **every** development session.
> Read it first when a session starts.

---

## Current milestone

**M5 — GUI shell**

A Qt6 application that starts, opens a project made in M4, and shows a scan — with dockable
panels, a dark theme, and no business logic in a single widget. **The layer underneath it is
finished**, so M5 constructs objects rather than inventing them; the guard added in M4-T15 says
nothing outside `gui/` may import Qt, and `gui/` may not reach into `core.science` or
`infrastructure` (Architecture §3.2).

**M4 closed 2026-08-12 — all fifteen tasks, all six exit criteria, ADR-0038…ADR-0051, and the
golden byte-identical throughout.** It built what W1 said did not exist. Three of its scheduled
tasks turned out not to need building and were closed with arguments instead of code (autosave,
three of five lifecycle use cases, the SQLite log sink) — the same judgement M2-T08 made about the
ports, and by now a pattern worth naming: *a task written before the layer beneath it existed is a
hypothesis, and checking it is part of doing it.* Tests 478 → **828**, schema version **5**, mypy
unchanged at 6. Left open on purpose: **W10 made closable rather than closed** (the registry
exists; `PipelineConfig` keeps its path until M5's composition root uses it) and **B-068**, which
needs an operator's view. Milestone summary in `docs/Progress.md`.

**M3 closed 2026-08-09** on the operator's decision — 25 of 26 tasks, ADR-0014…ADR-0037, all five
exit criteria met, tests 119 → 478, mypy 20 → 6, and **every defect the July audit reproduced
closed except D-24** (the stale README, M9). Left open on purpose: `M3-T16` (blocked on **B6**)
and four findings that are each an algorithm choice needing an operator's view — **B-062**,
**B-065**, **B-066**, **B-067**. They stay in M3's list.

**M2 closed 2026-08-04** — sixteen tasks, 2 021 lines of science moved into four named layers, and
**not one number changed**. **M1 closed 2026-08-04** — all eleven tasks done, four of five exit
criteria met; the fifth has two known exceptions filed as **B-054**. Milestone summaries in
`docs/Progress.md`.

## Current task

**None selected. `M5-T03` done 2026-08-13 (ADR-0054); `M5-T04` — the project explorer — is next**,
and it carries one of M4's outstanding obligations: counting annotations before `remove_image`
discards them (ADR-0044).

**`M5-T03` done 2026-08-13 (ADR-0054, implementing ADR-0002) — one source of colour truth, and a
contrast floor that can fail.** The stylesheet carries `@{token}` placeholders and **no colour of
its own**, enforced by a test rather than by review — *the rule and its enforcement ship together,
or only the rule does*. A placeholder with no token **raises** instead of substituting an empty
string. **Palette *and* stylesheet from one table**, because QSS does not reach tooltips or Qt's
disabled states, whose default on a dark palette is near-black on dark. **Contrast is a floor:**
every text pair clears 4.5:1 (WCAG AA), recomputed by the test, including `TEXT_MUTED` — *"muted"
must not come to mean "unreadable"* — and a test checks the measure itself first, since a contrast
check that cannot fail is decoration. **Found:** the substitution ran over the stylesheet's own
comment explaining what a token is (comments are stripped first now), and `@space_mdpx` parsed as
one name, so placeholders are braced. A wheel was built and inspected: the stylesheet ships. 19
tests, **888** in the suite, **golden byte-identical**.

**`M5-T02` done 2026-08-13 (ADR-0053) — a window that holds the container and nothing else. W2 is
no longer "no UI at all":** `nanoscope --gui` opens a window with menus, a toolbar, three docks and
a status bar, and opens a project into it. **`QApplication` lives in `gui/launcher.py`**, imported
inside the `--gui` branch, because M4-T15's guard fails if anything outside `gui/` imports Qt — and
it is right, since the headless entry point is the one CI runs. **The layout is a setting, not a
`QSettings`** (ADR-0047's scope rule: it follows the operator, not the work), and an unreadable one
is ignored rather than fatal. **Every dock names the task that fills it** and carries an
`objectName`, without which `restoreState` silently drops it. **Found: an existing test hung
instead of failing** when `--gui` stopped being a stub — a hang is the worse of the two, so the
entry-point test now asserts the handover and a new GUI test enters the loop with a timer that
knows how to leave. Also found: `from conftest import` became ambiguous the moment a second
conftest existed (the helper is now a module of its own), and the whole-layer test's
`"PySide6" not in sys.modules` was a weaker copy of the subprocess guard — deleted. 17 GUI tests,
**867** in the suite, **golden byte-identical**.

**`M5-T01` done 2026-08-12 (ADR-0052) — one place that constructs everything.** Until it, **every
caller of M4's eight components was a test building them by hand**. **The entry point works
headless today:** `nanoscope --project PATH` opens, prints and exits; `--devices` reports the
hardware; `--gui` says the window arrives in M5-T02 — because an entry point that only works once
a window exists cannot run in CI and cannot help the operator whose project will not open. **A
container, not a framework.** Our errors are messages, anything else keeps its traceback (ADR-0030,
first surface to use it). **ADR-0040's obligation is discharged:** the integrity report is *shown*.
**Found:** reading the version at import time cost eleven modules and tripped M2-T09's weight
guard — it is now lazy through a PEP 562 `__getattr__`. `print` got one scoped exception, stated in
`pyproject.toml` and in M2-T11's test, because a CLI that logs instead of printing has no output.
16 tests, **golden byte-identical**.

Two obligations from M4 are still outstanding, both needing a widget: **count annotations before
`remove_image`** (ADR-0044) and **marshal a job's listener onto the main thread** (ADR-0043).

## Completed

### M4 — Application layer ✅ (closed 2026-08-12)

**`M4-T15` done 2026-08-12 (no ADR — a test that only exercises existing decisions makes none).**
**One long test on purpose**, because the *sequence* is the subject: configure logging, create,
state a preference, register a model, select a device, import **as a job with progress**, analyse,
annotate, undo, export, close — then **reopen and find all of it**, which is the assertion that
matters. A second test states the same promise as **files** (manifest, `images/`,
`results/run_*/measurements.csv`, `exports/`, no `-wal`). The **Qt guard is phrased for a world
where `gui/` exists** — nothing *outside* it imports Qt, statically and in a subprocess — and was
added while trivially true, because a guard added after the first violation has already failed
once. **Found: B-068** — `PipelineConfig`'s default mode is `"segment"`, so the most natural call
in the project raises *predictor must be provided* and the default configuration is one CI can
never execute; filed rather than fixed, since changing a default changes every caller who omits it.
Also repaired: `test_ports.py`'s in-process `sys.modules` assertion, which any earlier test could
break — and today one legitimately did.

**`M4-T14` done 2026-08-12 (ADR-0051) — a log that only works when everything works is not a log.**
ADR-0013 deferred the SQLite sink by name; the database exists now, and this task **refuses it with
the argument written down**: *a log must not depend on the thing whose failure it records*. The most
valuable lines this application will write are about a database it could not open, and a handler
inside that database has nowhere to put them. **Two rotating files instead** —
`$XDG_STATE_HOME/nanoscope/nanoscope.log` for what the application did, `<project>/logs/` for what
happened to that work — in **JSON Lines**, so `extra=` survives as fields. Only `app/` attaches
handlers, and they are named, so configuring twice replaces rather than duplicates. 12 tests.

**`M4-T13` done 2026-08-12 (ADR-0050, implementing ADR-0005) — a model is a record, not a path in a
default argument.** Exit criterion met **without a weight file in sight**, because the registry
hands back **factories, never instances**: constructing a detector loads weights, so instantiating
on lookup makes "what models does this project have?" expensive and impossible in CI. Weights may
be **shared**, with the consequence stated rather than prevented — `models` is the one table without
the absolute-path check, because refusing would force duplicating gigabytes or lying about where
the file is. Identity is a name somebody chose; re-registering replaces. `provenance` is free text,
since provenance that must fit a schema stops being recorded. **The device from M4-T12 arrives
here**, closing ADR-0049's named gap. **`run_pipeline` is deliberately untouched** (ADR-0010), so
**W10 is not closed but made closable**, with M5 as the payer. Schema **v5**, 13 tests. **Found:**
the `TABLES_BY_VERSION` guard written last session went red on the very next migration — exactly
what it was for.

**`M4-T12` done 2026-08-12 (ADR-0049, implementing ADR-0004) — one component decides where
inference runs. W8 closed**, exit criterion met, verified on the operator's machine (*"NVIDIA
GeForce GTX 1070 (cuda)"*). Before it, `grep -r cuda infrastructure/models` returned **nothing**.
Three decisions: **no torch means the CPU**, reported not raised, since CI installs none on purpose;
**ROCm is told apart by `torch.version.hip`**, because a ROCm build answers
`torch.cuda.is_available()` with True and a naive probe calls a Radeon "CUDA" — wrongness that
survives for years because it never crashes; and **a fallback says why in a sentence**, which
ADR-0004 asked for and which is the easy part to skip. 14 tests over a **fake torch**, the only way
the ROCm branch can be tested at all. `DeviceKind`, in `core` since M2-T02, finally has a resolver.

**`M4-T11` done 2026-08-12 (ADR-0048) — an export is not a copy of the stored table.** ADR-0042
predicted this would be nearly free; **the format was free and the export was not.** Three things a
copy would not have: **provenance in front** (`image`, `run_id`, `detector`, … — the stored table
is *filed under* its run, and a CSV on a desktop has no context, so a column of heights with no
scan name is a column of numbers), **more than one run in one file** (statistics across a dataset
is why the measurements exist), and **a timestamped name** (an export is a snapshot; replacing
yesterday's loses work). Nothing dishonest is written: a detect-only selection **raises** rather
than producing headers with no rows, and a missing stored table fails loudly. 10 tests.

**`M4-T10` done 2026-08-12 (ADR-0047) — a preference belongs either to the operator or to the
work.** Two stores: `~/.config/nanoscope/settings.json` (XDG) for what follows the operator, a
`settings` table (schema **v4**) for what belongs to the project and travels with its directory.
**Reads merge, project first; writes name their scope** — "save this" without saying where is a
question, and guessing leaks one project's choice into every other or hides a global preference in
one directory; a project-scope write with no project open **raises**. Values are **JSON**, so a
boolean stays a boolean, and a stored `None` is an *answer*. Neither store validates keys — a
registry would be edited by every feature, including ones in `gui/`. The file is written by
replacement, and a corrupt one reads as empty rather than stopping the application. 26 tests.

**`M4-T09` done 2026-08-12 (ADR-0046) — there is no dirty state to save. Closed by understanding
rather than by code:** no production code was written. Architecture §4.5 scheduled autosave before
there was storage to autosave; **every mutating repository method commits before it returns**, so
a service would be **a timer that flushes nothing** — worse than useless, because it creates the
impression of protection where the protection actually lives. What ships is the proof:
`test_durability.py` abandons repositories without `close()`, reads from a second connection while
the writer lives, checks every write path, and **kills a process with `SIGKILL` between writes**,
finding its rows intact. Two named triggers would reverse it (GUI-only view state; any batching
write path), the second as a red test. `Architecture.md` §4.5 corrected in the same commit.

**`M4-T08` done 2026-08-12 (ADR-0045) — undo is a session, and it says so. M4's undo/redo exit
criterion is met**, against a real database. **The stack knows nothing but order** — it never learns
what a command is, so it will not grow when a table arrives; redo is `do()` again. **Undo is a
session and persisting it is a promise deliberately not made:** replaying edits against a directory
that may have changed on disk is an undo that can be silently wrong, which is worse than one
honestly short (M4-T14's log is history, not reversibility). A **failing undo propagates and the
history stands still**, because a swallowed error makes the *next* undo wrong too. **The plan was
reversed by its own test:** it said a redo should insert a fresh row, and
`test_a_sequence_undoes_and_redoes_whole` showed that a new id leaves every command above pointing
at nothing — **undo one command deep, which is not undo**. `restore_annotation` puts a row back *as
itself*, separate from `add_annotation` because creating and restoring are different acts; safe
under LIFO, refused by `UNIQUE` outside it. 18 tests, **golden byte-identical**, mypy unchanged
at 6.

**`M4-T07` done 2026-08-12 (ADR-0044) — an annotation is a row, because it cannot be recomputed.**
The first data an operator *makes* rather than the application deriving it. A **table**, two tasks
after ADR-0042 sent the measurement table to a *file* — so the rule is written down: **not "files
versus database" but "does the shape vary, and is it derived?"** A measurement table's columns
depend on its producer and it re-runs; an annotation is fixed in shape, edited one at a time with
undo behind it, and irreplaceable. **One shape, the box** (what training consumes, what a drag
produces); masks deferred a third time. **`source` (`manual` / `from_detection`) is load-bearing:**
a model trained on its own output is confirming itself. Annotations **cascade** with their image,
which hands M6 an obligation the ADR states rather than a `force=True` flag — `annotations_for` is
to be **counted before** the confirmation dialog, because `remove_image` now destroys hand work.
Floats, not integers, and a zero-area box refused twice. **No use case** (ADR-0041, fourth
application). Schema **v3**, 20 tests, **golden byte-identical**, mypy unchanged at 6.

**`M4-T06` done 2026-08-12 (ADR-0043) — a job reports, and stops when it is asked.** Thin over
`ThreadPoolExecutor`, which already does submit, results, exceptions and cancelling a job that has
not started; **progress and stopping one that has** are the whole module. **Cancellation is
cooperative and there is no version of it that is not:** a queued job is dropped, a running one
stops at its next `raise_if_cancelled()`, and **one with no checkpoint finishes anyway** — so the
button means *stop at the next checkpoint*, which the GUI must say, or it is a button that appears
to do nothing. **A job is a handle, not a base class** (ADR-0041's rule, third application, in the
place the pull is strongest). Threads over processes — NumPy and torch release the GIL, and a
predictor cannot be pickled. Progress is counts; `total = 0` means *cannot say*. The listener fires
**on the worker thread**, stated three times, because M5 must marshal. **Found by the first test
that ran an import under the runner:** `sqlite3` binds a connection to its creating thread, so a
project opened on the main thread was **unusable inside every job** — it would have surfaced in M5
as a crash in a background worker. Fixed with `check_same_thread=False` *and* a reentrant lock
around every repository method, since `save_analysis` writes three statements another thread could
commit half of. 19 tests, none synchronising on wall-clock time. **Golden byte-identical**, mypy
unchanged at 6.

**`M4-T05` done 2026-08-12 (ADR-0042) — what the analysis found outlives the session. M4's second
exit criterion is met.** The decision the criterion itself names: **the index is relational, the
tabular product is a file.** `analysis_runs` and `detections` are schema v2; the measurement table
goes to `results/run_<id>/measurements.csv`, because **ADR-0031 made that table variable by
construction** and a relational shape for it is one wide grid with NULLs — rejected in ADR-0031's
own words — or an EAV pivot that loses the declared dtypes. It also reconciles ADR-0003 with
itself. A `detect` run writes no table; a *missing* table raises rather than reading as empty;
results **cascade** with their image, which is where M4-T02's `PRAGMA foreign_keys` finally becomes
load-bearing. **One use case, not three** (ADR-0041's rule again). Masks are not persisted — SAM2's
weights are outside the gate, so the format would be written blind. **Found by a test, and live:**
`run_analysis` was analysing an npy *without* the scale the project recorded at import — every
`radius_nm` `None`, the physical size filter silently skipped, **the D-07 family of defect M3 spent
a milestone removing, reintroduced one layer up**. Also learned: the LoG detector *does* emit a
synthetic bbox, so ADR-0031's absence case has no producer in the gate. 15 tests; **schema v1 → v2
is the first migration applied to a database with rows in it**. **Golden byte-identical**, mypy
unchanged at 6.

**`M4-T04` done 2026-08-12 (ADR-0041) — a project can be created, opened and populated.** **M4's
first exit criterion is met**, headless and end to end. **Two of the five named use cases were
written, and the other three are the decision:** `CloseProject` and `ListImages` would be
`repo.close()` and `repo.list_images()` under new names, and **a function that forwards one call to
one object is not a layer — it is a second name for the same method**; `CreateProject` went to
`SqliteProjectRepository.create`, because it is `mkdir` + manifest + SQLite and `application` may
import none of them (Architecture §3.2, PROJECT_RULES §2.7). Same judgement as M2-T08's
seven-ports-one-written, recorded so the absence reads as a decision later. The two survivors carry
policy: `open_project` **reads** the integrity report and hands it over with the images (ADR-0040's
closing obligation), and `import_images` **does not abort the batch** — forty scans do not lose
thirty-nine to the fortieth, and only `NanoscopeError` is caught, because a bug that keeps going
for another thirty-nine files is a worse bug. **Found by mypy, not by a caller:** the port did not
declare `name`, which `open_project` reads — the use case would have worked anyway on the SQLite
class, and the second implementation would have found it at run time in the GUI. A colliding name
is disambiguated against the *filesystem* (`scan.spm` → `scan_1.spm`), so an untracked file is
never overwritten. 25 tests, 11 of them through a **second implementation of the port**. **Golden
byte-identical**, mypy unchanged at 6.

**`M4-T03` done 2026-08-12 (ADR-0040) — the repository reports what it finds, and never reconciles
by deleting.** The debt ADR-0003 wrote down and never collected: *"deleting a file behind the
application's back produces a dangling row; the repository layer must reconcile"*. The check had
no home until there was a repository to put it in, and it is the piece most likely to be skipped,
because everything works in a test where nobody deletes anything. **It reports, in both
directions, and changes nothing** — a missing file is as likely to be an unmounted drive as a
deletion, and **the row carries the annotations and measurements the file does not**, so the
obvious cleanup destroys the more expensive half of the pair on *open*, unasked. An untracked file
is not imported either: that means guessing it was meant to be here and inventing a modality.
**Existence, not contents** — verifying checksums would read every scan on every open. `add_image`
computes the checksum itself (one passed in can describe a different file) and refuses a file that
is not there. Relative paths enforced here with M4-T02's `CHECK` as the backstop; queries return
`ImageRecord`, never a `sqlite3.Row`. **The `ProjectRepository` port arrived with its first
adapter**, the first row of M2-T08's table to pay out — load-bearing, since M4-T04's use cases may
not import `infrastructure`. 30 integration tests, including **ADR-0003's own compliance clause at
last**: a project moved, a project copied, `cache/` deleted between two opens. **Golden
byte-identical**, mypy unchanged at 6.

**`M4-T02` done 2026-08-12 (ADR-0039) — the schema has a version and a way forward.** The
mechanism before the tables, because creating them is its first job: version 0 is an empty file
and every table in existence was made by a migration step. `MIGRATIONS` is an ordered list of
`(version, statements)` with **`SCHEMA_VERSION` derived from its last entry** — a constant that
can disagree with the list eventually does. **The finding is a stdlib default:** Python's
`sqlite3` opens a transaction implicitly before **DML only**, so `CREATE TABLE` runs in autocommit
and a step failing partway would leave tables behind *at the old version*, the one state a
migration must never produce; each step runs under an explicit `BEGIN` with `PRAGMA user_version`
inside it, and a test breaks a step to prove nothing survives. **v1 holds one table, `images`** —
a table with no reader is columns designed before their caller, and "no destructive migrations"
makes a wrong one expensive to remove. **No WAL**, which is a layout decision: `-wal` and `-shm`
would be two files in a published contract, and a project copied mid-write would leave committed
data behind. `PRAGMA foreign_keys` is set on the connection (off by default, per connection, and a
silent no-op inside a transaction). Two rules became `CHECK` clauses — a relative path, and a
`modality` the enum still recognises. 23 tests, **golden byte-identical**, mypy unchanged at 6,
no new dependency.

**`M4-T01` done 2026-08-09 (ADR-0038) — the project format is a versioned contract.** First of the
milestone because everything else in M4 writes into a project, and left implicit the format
becomes whatever the first task to land happens to do. Three decisions: **two independent version
numbers** (`format_version` in the manifest, `schema_version` as the database's `PRAGMA
user_version` — the layout must be readable *without opening the database*); **`project.json` is
the identity file**, not a database row, or ADR-0003's "corruption is contained" is a slogan; and
**refuse newer, accept older**, because a forward migration cannot be written by the past. The
spec (`docs/ProjectFormat.md`) ships with the executable half
(`infrastructure/storage/project_format.py`), since a contract nothing executes drifts within two
tasks. **Found while writing the tests:** "a reader ignores fields it does not know" is a
data-loss guard, not politeness — an older application rewriting a newer manifest would have
deleted every unrecognised field silently; they are now carried through, with a test. 21 tests,
delta **zero, golden byte-identical**, mypy unchanged at 6.

**`M3-T19` done 2026-08-09, the last task of M3** (no ADR — an annotation is not executed; the
M3-T18 precedent). Filed as three mypy errors about a `list[float]` rebound to an ndarray; reading the file for it found
the same class twice more, stated the other way round — `threshold: float = None` and
`manual_radius_px: float = None`, **implicit Optionals whose bodies branch on `None` two dozen
lines later**. Six of mypy's twelve errors, one defect class: *an annotation that does not describe
the value the code carries*. Delta: **zero, by construction — no executable line changed**; the
golden was left untouched. mypy **12 → 6**, and the remainder is not annotation drift (four in
`pipeline.py`, M4's; two third-party overloads). 2 tests pin what `float | None` now claims: an
explicit `None` equals the omitted argument. Recorded, not filed: `r = max(int(sigma), 1)` at
`log.py:165` sizes a peak-lookup window, not a physical radius.

**The operator's decision, 2026-08-09: close M3 and open M4** rather than take one of the four
open findings. The roadmap allows the two to run in parallel (§"Sequencing", point 3).

The state carried forward: every numerical defect the audit reproduced is closed, detection
quality is measured, and **B-059, B-060, B-061, B-063 and B-064** are closed with them. Inside
M3's task list only **M3-T16** (blocked on **B6**) remains.

**Everything else open needs a decision, not an afternoon:** **B-062** (recall 0.000 — wants an
operator's view of a sensitivity trade-off), **B-065** (a gap-tolerant *pipeline* — needs an
answer to "what is a substrate under a gap?"), **B-066** (deliberate interpolation, and a method
choice), **B-067** (a margin from the radius distribution's upper tail rather than its median).
**B-040** goes last of everything because it rewrites every SHA above it.

**`M3-T26` done 2026-08-08 (ADR-0037)** — **B-064 closed: the opening-radius constants are
measured.** `scale=1.7` and a bare `2.5` set every opening radius, neither derived anywhere, both
chosen while ADR-0035's truncation was in place (effective margin `1.7 × int(r)/r` = **1.39** at
r = 4.9). Swept both over the five AFM phantoms against ground truth. **The rough factor barely
matters** — recall and precision identical from 1.3 to 2.4, because the second stage re-estimates
from Otsu, which is *why the truncation survived five months*. **The final factor is a real
trade-off:** dense recall 0.886 at ×1.5 → 0.800 at ×4.0 as a bigger disk steps over touching
particles, against a radius error that is best at ×2.5. Decision: **keep both, name them, expose
the literal** — `DEFAULT_ROUGH_SCALE`, `DEFAULT_OPENING_SCALE`, `MIN_OPENING_RADIUS_PX`, and
`opening_scale` as a parameter, because a magic number inside a branch is not a decision anyone
can revisit. Delta: **zero, golden byte-identical.** `afm_sparse_low_snr` scores 0.000 at every
factor — more evidence B-062 is a detector question. **B-067** filed.

**`M3-T25` done 2026-08-08 (ADR-0036)** — **B-060 closed: levelling can fit around a gap.**
M3-T13's rejection was right as a uniform contract and ADR-0030 said in its own text it was not
the best behaviour available. Both levelling functions now take **`allow_gaps=False`** and fit
over the finite pixels when asked — **opt-in**, because accepting NaN silently would put the
library back where D-15 found it. The gap stays **absent**, never interpolated, and rows that
cannot be fitted are counted out loud. Measured against the ungapped answer: masked fit
**0.029 nm** against `nan_to_num`'s **0.134 nm**, and the difference lives in the *tilt*
coefficient — zero-filling tells the fit the sample dips to zero along two lines. Delta:
**5 differences, all the new block; nothing recorded moves**, as predicted. The block records the
evidence rather than the outcome, and the finding is that **the advantage tracks the tilt** (4.2×
on the tilted phantom, 1.2–1.7× on the flat ones). **Honest headline: the pipeline is still not
gap-tolerant** — the output carries NaN and the substrate step refuses it, pinned by a test.
**B-065** and **B-066** filed.

**`M3-T24` done 2026-08-08 (ADR-0035)** — **B-063 closed.** Two roundings in three lines, in
opposite directions, only one documented. **Not an operator decision:** ADR-0020 already made
`_integer_radius` the one funnel and the direction *up*, ADR-0024 deleted this exact `int()` as
D-04's mechanism, and the parameter's own docstring calls `scale=1.7` a multiplier making the disk
*safely larger* — the truncation made it smaller, from the day both lines were written. Delta:
**730 differences across four phantoms**, large in count and small in magnitude — 320 under 1 %,
mean measured height **≤ 0.09 %**, largest meaningful move 2.5 % at one 90th percentile. **The
plan predicted ~70× less, and the correction is the finding:** the two-stage design absorbs the
rough radius for the *final* radius (moves on 1 of 5), but `sizes` also feeds
`estimate_log_params`, so the LoG **sigma range** shifts wherever the rough radius did — 379 of
the differences are `log_detection`, and **`afm_dense_overlapping` detects one more particle with
a byte-identical substrate**. Quality, checkable for the first time: **recall unchanged
everywhere**; radius error −6 %, localisation +0.3 % in the mean on the one phantom that moved —
**a wash, reported as one**. 18 tests; restoring `int()` turns 11 red. **B-064** filed.

**`M3-T23` done 2026-08-07 (ADR-0034)** — **B-061 closed.** `estimate_rough_radius` could return
**0**, and `disk(0)` makes the opening the **identity**: the substrate came back equal to the
image, with the shape of an answer. The condition is that `median + std` selected single-pixel
noise — the median object area is **1.0 px** on `afm_sparse_low_snr` in *both* runs, and the
scaled one survives only because `min_size_px = 2.56` floors it, so **the estimate is equally
worthless there and the floor hides it**. A sub-pixel estimate now takes the fallback the function
already had, checked before `_integer_radius` because `ceil(0.96)` is 1; it lands on **3 px**,
which is what the scaled run of the same image computes. **It corrects ADR-0025's diagnosis:**
losing the scale did two things and the filter was the smaller one — **3351 → 627 objects**.
Delta: **11 differences, all in one cell**; `opening_radius`, `substrate` and `z_above` unchanged.
**The golden had the evidence all along** — an Otsu threshold of **7.7e-09**, which is Otsu on the
all-zero map an identity opening produces. 9 tests; removing the branch turns 5 red. **B-063**
filed with measurements.

**`M3-T22` done 2026-08-07 (ADR-0033)** — **B-059 closed.** `if height <= 0` was written to
discard artefacts and **`nan <= 0` is `False`**, so a `NaN` height was the one value it let
through: a constant map has no Otsu split, the substrate mask comes back empty, `np.median` of
nothing is `nan`, and every particle falling back to the global baseline inherited it — two rows
of `NaN` in a table of measurements. The comparison needed no new decision (ADR-0018's rule, third
site); **the silence did**, because the fix alone turns two `NaN` rows into zero rows, which reads
like "there was nothing here" — so an empty substrate mask now warns, naming cause and
consequence. Delta: **5 differences, all the new probe; the fix moves nothing recorded**, because
no phantom has an empty substrate — the fifth time in M3 that closing a defect meant extending the
harness that missed it. **Found while testing: the empty substrate is all-or-nothing** —
`get_clean_ring` intersects the ring with the substrate mask, so no particle can be measured at
all and the "partial success" case does not exist. 10 tests; restoring `<= 0` turns 2 red.

**`M3-T15` done 2026-08-07 (ADR-0032)** — **the project can measure detection quality.** Five
tasks had written "not claimed" for want of this; `core/science/evaluation.py` scores detections
against the ground truth the phantoms have carried since the audit. A match is a centre within
`match_factor × the particle's own radius` — scale-free, where a fixed pixel threshold would be
two different physical tolerances across 1.95–29.3 nm/px — and the assignment is **one-to-one and
optimal**, because ten boxes on one particle are 1 TP and 9 FP and greedy can pair the wrong two.
Delta: **7 differences, all `detection_quality: ADDED`.** The numbers: four AFM phantoms and both
image phantoms at **precision and recall 1.000**, localisation 0.36–0.61 px;
`afm_dense_overlapping` at **0.983 / 0.843**; and **`tem_dark_particles` 22 of 22 at 0.36 px**,
which turns ADR-0023's "0 → 22 blobs" from a count into a measurement. **`afm_sparse_low_snr`
scores recall 0.000** — six particles, none found — filed as **B-062**. Radius error is
consistently negative on AFM and positive on the image phantoms, which is a calibration offset
rather than scatter. 21 tests. **Not licensed, and said before the numbers existed: a phantom is
not a sample.**

**`M3-T14` done 2026-08-07 (ADR-0031)** — **D-16 and D-17 fixed: one measurement schema.**
`schema.py` declares a **core** every producer emits plus **blocks present in full or absent in
full** (detector, height, geometry, segmentation), with `method` naming the producer so a reader
knows which blocks to expect — not one wide table with NaN, because SEM/TEM does not have heights
that are all missing, it has no heights. Reading the producers found **three** faults where the
audit named one: two names for one quantity (`score`/`sam_score`, `mask_area_px`/`area_px`),
columns that varied **per row** (`if k in res`), and — the one the audit missed, and the worst —
**`radius_nm` was two quantities under one name**, so concatenating the baseline table with the
SEM/TEM one produced a column holding two different measurements. Now `detector_radius_nm` and
`radius_nm`. `bbox` is `| None`, and the `type: ignore` M2-T02 wrote to expire itself expired.
Delta: **62 differences — names, dtypes and added columns; 35 column digests unchanged, 0
changed**, and the renamed column's digest is byte-identical to the one it replaced. **The
harness needed the same fix the code did**: `list(det.bbox)` is a `TypeError` once a bbox can be
absent — D-16's assumption living inside the tool meant to catch it. 31 tests over five tables,
the SAM2 pair driven by a **stub predictor**, because there are no weights here or in CI — which
is also why their golden delta is zero *by construction* and the tests are the whole safety net.
mypy unchanged at 12.

**`M3-T13` done 2026-08-07 (ADR-0030)** — **D-15 fixed: one answer to "this input cannot be
used".** Seven classes in `core/errors.py`, **each also inheriting the builtin it replaced at its
site**, so every `except ValueError` in the notebooks keeps catching what it caught and the
taxonomy lands in one commit instead of a migration; and one `ensure_height_map` called at
**fourteen** entry points. A height map is 2-D, non-empty, integer-or-real, and **finite** — that
last one being the decision, since `flatten_plane` already enforced it through `scipy.lstsq` while
`flatten_lines` propagated NaN and `detect_particles` answered a NaN map with "no particles".
Delta: **129 differences and not one measured value** — 32 exception types, 28 messages that
became ours, 15 `raised_in`, and **13 cells that used to answer an input they could not use**.
`TypeError`, `IndexError`, `LinAlgError` and `RuntimeError` each collapse into one of ours, and
**foreign messages in the golden go 15 → 0**, emptying the category ADR-0022 created for them.
The twelve phantom-level differences are exception types on two probes that were already failing.
**Supersedes ADR-0018 on non-finite input only** — a flat or negative map still answers "no
particles" — and M3-T08 on boolean input, now refused rather than corrected. 109 tests; the centre
is 7 bad inputs × 10 entry points, **70 combinations and one error type**, with the same sweep
proving a valid map passes all ten. **B-060** and **B-061** filed rather than smuggled in. mypy
unchanged at 12.

**`M3-T08` done 2026-08-07 (ADR-0029)** — **D-13 fixed: levelling returns the residuals it
computed.** `np.empty_like(z)` kept the input's dtype, so float64 residuals were cast back on
assignment; the allocation now promotes with `np.promote_types(z.dtype, np.float64)`, which is
`flatten_plane`'s own rule rather than a hardcoded float64 that would agree with it by
coincidence. Delta: **13 differences — 8 dtypes, 4 sums, 1 added group — and no phantom moves**,
because `flatten_plane` hands float64 on in every recorded chain. **The audit understated the
defect in two directions:** it measured a ramp with sub-1 residuals and called it truncation, but
an integer output **wraps** a negative residual — on the newly recorded 8-bit phantom the levelled
map is wrong by up to **257** and every pit comes back as a peak; and boolean input, unmeasured,
returned a *mask* of where the residual was non-zero. The exposed caller is
`load_microscopy_image`, which returns `uint8` from `cv2.imread` and is the only file entry point
SEM/TEM has. The four moved sums are the fix as a physical property: a least-squares residual sums
to zero, and float32 storage left it at 1e-6 instead of 1e-13. 17 tests; restoring
`np.empty_like(z)` turns **14** red, the three survivors being the float64 cases. **mypy unchanged
at 12** — a dtype right for one input and wrong for another has no static shadow.

**`M3-T05` done 2026-08-06 (ADR-0028)** — **D-09 fixed.** Both YOLO backends now pass their own
per-box scores, and a length mismatch raises rather than being `zip`ped away. **`confidence` is
`float | None`, defaulting to `None`**: `1.0` was a substitute value — the fifth this milestone
has deleted — and it made the **LoG** detector claim certainty it never computed, which the audit
had not said. No confidence is invented for LoG: its blob response is not a probability, and
**M3-T15** is the only thing that could license one. Delta: **29 keys added, 0 values changed**,
and the finding is that `default_detection_confidence` is *added*: the harness recorded the
defaults of D-16's field and not of D-09's, one line below, **so the golden could never have
caught this defect**. Third time in M3 that the harness was the blind spot. 7 tests; restoring
the drop turns 6 red. **mypy 14 → 12**: threading a second array through would have added a third
`_last_result` error, and annotating that field removed all three.

**`M3-T12` done 2026-08-06 (ADR-0027)** — **D-08 fixed.** `pd.DataFrame([])` has zero columns, so
a scan with nothing measurable answered every read by name with `KeyError`. The baseline schema
is declared — twelve columns with dtypes — and returned whether or not a row survived, with a
test proving the declaration still describes what the **populated** path emits. Delta: **78
golden differences, all columns appearing where there were none, 0 values moved**. The finding is
the sixth block: **`afm_sparse_low_snr` detects 0 blobs on its ordinary path**, so D-08 was live
on a real phantom's normal run, and the golden had been recording `columns: []` for it since the
baseline was taken. **Found while testing, filed not fixed: `nan <= 0` is `False`, so a NaN height
reaches the table — B-059**, the same comparison ADR-0018 already ruled on. 7 tests; restoring
`pd.DataFrame(results)` turns 3 red.

**`M3-T17` done 2026-08-06 (ADR-0026)** — the SPM parser's `else` branch, written to tolerate a
header with no `Scan Size`, divided `None` by `samps` on the very next line: **the fallback
crashed on the branch it had just taken**. It now returns `(None, None, z)` — the height map
decodes as always, only the metadata is absent, and ADR-0025 gave that state a meaning everywhere
downstream one commit earlier. The same expression's other failure mode went with it
(`Samps/line: 0` was a `ZeroDivisionError` naming nothing), and a *stated* non-positive
`Scan Size` is rejected as well: **absent and wrong are different**. Delta: **0 golden
differences, and none was possible** — `afm_io` has no phantom, so its 28 unit tests are the whole
safety net. **mypy 15 → 14**, the removed error being this defect's own (`-> np.ndarray` on a
function returning a three-tuple). 3 tests; restoring the division turns 3 red.

**`M3-T20` done 2026-08-06 (ADR-0025)** — the npy loader no longer invents `1.0` nm/px and a scan
size equal to the row count. `None` is unknown and passes through to the entity; a scale that
*is* given must be positive, so `0.0`, `-1` and `nan` raise instead of being swallowed by `or`.
`build_substrate_map` accepts `None`: the `_nm` outputs are absent and the `min_size_nm` filter
cannot be applied, which is **warned**. Delta: **5 golden keys added, 0 values changed** — every
phantom has a scale. The new keys carry the finding: an unscaled run is exactly a scaled run with
`min_size_nm=0`, so on four phantoms the substrate is bit-identical and on `afm_sparse_low_snr`
it is not — **17 objects become 3351**, the typical radius falls 2.99 px → 0.80 and the opening
radius 8 → 5. **Losing the scale is losing the filter**, which is D-04's mechanism one commit
after D-04 was closed. 10 tests; restoring `pixel_size_nm or 1.0` turns 6 red.

**`M3-T02` done 2026-08-06 (ADR-0024, decision B2)** — **D-04 fixed, the last `critical`.**
`int(min_size_nm / pixel_size_nm)` is gone; the filter compares `radii_nm >= min_size_nm`, so a
physical minimum stays physical. Delta: **47 differences — 27 changed, 15 added, 5 removed**.
`afm_sparse_low_snr` drops **75 objects to 17** and everything derived from its radii moves with
it; the other four AFM phantoms are byte-identical and **no measured height moves anywhere**,
because the final opening radius is 8 on both sides. **Re-measuring all 628 scan headers**
reproduces the audit's 90 % (568/628) and adds what it did not measure: the zero threshold cost
**nothing** on the 365 scans (58 %) coarser than 8.86 nm/px, where one pixel is already 5.5 nm;
it disabled the filter on the 203 (32 %) in the 5–8.86 band; and the finest 60 (10 %) were hurt
by **truncation**, not by the floor — `afm_sparse_low_snr`'s 2.5 px threshold became 2 px, and 58
of its 75 "objects" were noise living in that half-pixel. **mypy unchanged at 15**: a unit error
has no static shadow, which is why the suffix convention is the only check this class has.
5 tests; restoring the `int()` turns 3 red.

**`M3-T07` done 2026-08-05 (ADR-0018)** — D-11 fixed: `z_above / z_above.max()` at two sites
never checked its divisor. A flat map made every pixel `nan` and the code blamed the threshold;
a negative map flipped the topography and produced an adaptive threshold of **2.4997** against a
`[0, 1]`-normalised response. Both sites now stop on a non-positive or `nan` maximum —
`DEFAULT_THRESHOLD = 0.05` from the estimator, an empty `(0, 4)` from `detect_particles`.
**65 golden keys added, 0 changed**: the working path is byte-identical, and the wrong number
had never been recorded at all, because the harness wrote every scalar down as the string
`"non-array"` and its only negative degenerate input was *constant*. Fixing that is the larger
half of the commit.

**`M3-T06` done 2026-08-05 (ADR-0017)** — D-05/D-06 fixed: the empty-after-filter case raised
nothing and returned `nan`; it now raises with the parameter, its value and the largest object
measured, and `n_objects` counts survivors. **8 golden differences**, of which the headline is
`n_objects_reported` **1023 → 75** on `afm_sparse_low_snr` — a 13.6× over-count of
single-pixel noise. It also broke a test written in M3-T01 that had been passing *because* of
the `nan`.

**`M3-T04` done 2026-08-05 (ADR-0016)** — D-21 fixed: the scan is scaled isotropically and
padded to the model square instead of squashed into it, and `_scale_boxes` inverts exactly
that. **0 golden differences, 7 keys added**: a square scan is byte-identical, and every
phantom is square — which is why the harness gained `non_square_half_height` in the same
commit. Reading for it turned up **M3-T21**: `use_tiling=True`, the default, produces exactly
one crop, so the tiled backend has never tiled.

**`M3-T03` done 2026-08-05 (ADR-0015)** — D-03 fixed: the cast to `uint8` now happens
*after* the normalisation. **67 golden differences, all under `yolo_input_preparation`**;
grey levels reaching the network went from 8–208 to 239–256. The retention spread is
**3.1%–81.2%**, and the cleaner the scan the worse the loss: the quiet 5 nm phantom kept 8
levels of 256 and came out **anti-correlated** (−0.499) with a correctly prepared image.
**This does not mean detections improved** — the weights were trained on images the old path
produced; see the ADR's Consequences.

**All five operator decisions were answered on 2026-08-05** — B2, B3, B4, B6, B7 — and four are
executed: B4/M3-T09, B7/M3-T21, B3/M3-T10 and B2/M3-T02. **B6/M3-T16** is the one left, plus
**B-040** (purging `node_modules` and the weights from git history), which goes last because it
rewrites every SHA. **B-058** is done (ADR-0022).
**B-054** is closed: the operator deferred the two oversized README figures to the M9-T01 rewrite.

---

### M3 — Numerical correctness (in progress)

- **M3-T26** ✅ (2026-08-08, **ADR-0037**) — **B-064 closed: the opening-radius constants are
  named, exposed and measured.** Two numbers set every opening radius in the project and neither
  was derived anywhere; both were chosen while ADR-0035's `int()` truncation was in place, so the
  effective margin was `1.7 × int(r)/r` — **1.39 at r = 4.9**, not 1.7. Both swept over the five
  AFM phantoms against ground truth, which M3-T15 made possible. **The rough factor barely
  matters:** mean recall and precision are *identical* from 1.3 to 2.4 and the radius error moves
  in the third decimal, because the second stage re-estimates from Otsu — M3-T24's finding from
  the other side, and **the explanation for why the truncation went unaudited for five months**.
  **The final factor is a genuine trade-off:** `afm_dense_overlapping`'s recall falls 0.886 →
  0.800 as the factor grows, because a bigger disk steps *over* two touching particles instead of
  into the gap, while the radius error is minimised at ×2.5 on both hard phantoms. ×1.5 buys three
  detections on one phantom for an 80 % worse radius error across the set. **Decision: keep both,
  and stop them being anonymous** — `DEFAULT_ROUGH_SCALE`, `DEFAULT_OPENING_SCALE`,
  `MIN_OPENING_RADIUS_PX`, and the literal becomes an `opening_scale` parameter, because the
  plumbing *is* the deliverable: a magic number inside a branch is not a decision anyone can
  revisit, which is why this took two tasks to surface. Delta: **zero — the golden is
  byte-identical and was left untouched rather than rewritten.** The sweep deliberately does not
  go in the golden (25 detection runs against a file that already costs ten minutes); it lives in
  the ADR, the docstrings, and a test pinning the trade-off's *direction*. Two things it turned
  up: `afm_sparse_low_snr` scores **0.000 at every factor**, which is more evidence **B-062** is a
  detector-threshold question; and **B-067**, because the margin comes from the *median* particle
  and `afm_tilted_polydisperse` is the only phantom that loses a detection as the factor grows.
  12 tests.

- **M3-T25** ✅ (2026-08-08, **ADR-0036**) — **B-060 closed: levelling can fit around a gap.**
  M3-T13 made a non-finite value a rejection, which was right — three functions had three answers
  and one contract replaced them — and ADR-0030 wrote in its own text that it was not the best
  behaviour available. **A dropped scan line is a real artefact**: two rows of NaN and four
  thousand good ones, and the scan was refused. `flatten_plane` and `flatten_lines` now take
  **`allow_gaps=False`** and fit over the finite pixels when asked. **Opt-in on purpose** —
  accepting NaN silently would restore D-15's disagreement, levelling tolerating what detection
  refuses. The gap stays **absent**, never interpolated (the eighth substitute value this
  milestone declined to add), and a row with too few finite points comes back absent with the
  count **warned**, because rows vanishing unexplained is how B-059 survived. Measured against
  levelling the same scan intact: masked fit **0.029 nm**, `nan_to_num(z, 0.0)` **0.134 nm** — and
  the difference is in the *tilt* coefficient (0.0496 against 0.0511 true), because zero-filling
  does not add noise, it tells the fit the sample dips to zero along two lines. Delta: **5
  differences, all the new `gapped_levelling` block; nothing recorded moves**, which the plan
  predicted before measuring. The block carries the comparison rather than only the result, and
  **the advantage tracks the tilt** — 4.2× on `afm_tilted_polydisperse`, 1.2–1.7× on the flat
  phantoms, which is the mechanism confirming itself. **The honest headline is a negative one:
  this does not make the pipeline gap-tolerant** — the levelled output still carries NaN and
  `build_substrate_map` still refuses it, asserted as a test. 12 tests, including that an intact
  map levels **byte-identically** with and without the flag. **B-065** (the pipeline; needs a real
  decision about a substrate under a gap) and **B-066** (interpolation) filed.

- **M3-T24** ✅ (2026-08-08, **ADR-0035**) — **B-063 closed: the rough estimate stops truncating
  its own radius.** `radius_px = int(np.sqrt(median_area / np.pi))` was a second, undeclared
  rounding, downward, three lines above the declared one that rounds up. **The rule had already
  been decided three times:** ADR-0020 (one funnel, direction up), ADR-0024 (this exact `int()`,
  deleted as D-04's mechanism), and the parameter's own March-2026 docstring — `scale=1.7` is *"a
  multiplier so the disk is safely larger than a particle"*, and the truncation made it smaller.
  Applying an existing rule, not making a new one, which is why it did not need the operator.
  Delta: **730 differences on four phantoms** — 320 under 1 %, mean measured height **≤ 0.09 %**,
  the largest meaningful move 2.5 % at one phantom's 90th height percentile. **The plan predicted
  ~70× less and the correction is the finding:** both of its specific predictions held, but it
  modelled only what `build_substrate_map` *returns* — `sizes` also feeds `estimate_log_params`,
  so the LoG **sigma range** moves wherever the rough radius did. 379 differences are
  `log_detection`, and **`afm_dense_overlapping` gains a detected particle (59 → 60) with a
  byte-identical substrate**, which isolates the mechanism. The lesson is about the two-stage
  design: the second stage is robust to the first, but the *diagnostics* the first emits are wired
  straight into the detector and nothing says so. **Detection quality, answerable for the first
  time (M3-T15): recall unchanged on every phantom**; on the one that moved, radius error 0.765 →
  0.718 px and localisation 0.6137 → 0.6156 px — **a wash, and reported as one**. 18 tests;
  restoring `int()` turns 11 red. One of them caught my own overstatement mid-draft: at an
  equivalent radius of 3.432 truncation and the correct value are both 6, and the test now says so
  rather than glossing it. **B-064** filed — the constants themselves, now measurable.

- **M3-T23** ✅ (2026-08-07, **ADR-0034**) — **B-061 closed: a rough radius below one pixel is not
  an estimate.** `estimate_rough_radius` could return **0**; `disk(0)` is a single pixel, so the
  rough opening was the **identity** — substrate equal to the image, `z_above` zero everywhere,
  and nothing said so. The condition behind it: `median + std` selected single-pixel noise, and
  the median object area is **1.0 px** on `afm_sparse_low_snr` in *both* the scaled and unscaled
  runs. The scaled one survives only because `min_size_px = 5/1.95 = 2.56` floors the answer —
  **the estimate is equally worthless there and the floor hides it**, which is the part worth
  remembering. A sub-pixel estimate now takes the "too flat or too noisy" fallback the function
  already had (1 % of the image width), checked **before** the rounding because `ceil(0.96)` is 1;
  it returns **3 px**, exactly what the scaled run of the same image computes — the one case with
  a known-good answer agrees. **It corrects ADR-0025**, which recorded 17 → 3351 objects and read
  it as "losing the scale is losing the filter": losing the scale did two things, and this was the
  bigger — **3351 → 627**. Delta: **11 differences, all inside one cell**, with `opening_radius`,
  `substrate` and `z_above` **unchanged**, because the median radius that drives the final radius
  is 0.798 px either way — the fix changes what the function *reports about the sample*, not what
  it *returns as the substrate*, on this image. **The golden had the evidence all along:** an Otsu
  threshold of **7.7e-09** is Otsu on the all-zero map an identity opening produces, and it sat
  next to "3351 objects" on a six-particle phantom reading as normal. First finding of this kind
  in M3 that was recorded rather than missing. 9 tests; removing the branch turns 5 red. **B-063**
  filed with the measured effect of its own fix (14 → 15, 12 → 14, 11 → 12, 7 → 9).

- **M3-T22** ✅ (2026-08-07, **ADR-0033**) — **B-059 closed: a height that is not a number is not
  a measurement.** The guard `if metrics["height_nm"] <= 0` exists to discard artefacts, and
  **`nan <= 0` is `False`**, so it kept the most artefactual value there is. Four reasonable steps
  produce it: a constant map has no Otsu split → the substrate mask is empty → `np.median` of
  nothing is `nan` → every particle whose own ring is too small inherits it. Two rows of `NaN` in
  a table of measurements, with nothing said. **The comparison needed no decision** — ADR-0018
  ruled on `not x > 0` five days earlier, in this milestone, for this exact reason, and this is
  its third site. **The silence did:** the fix on its own turns two `NaN` rows into zero rows,
  which reads exactly like "there was nothing here", so an empty substrate mask now warns and
  names both cause and consequence (ADR-0025's call, applied again). Delta: **5 differences, every
  one of them the new probe** — the fix moves nothing recorded, because **no phantom has an empty
  substrate**, which is precisely why the golden could not catch it; fifth time in M3 that closing
  a defect meant extending the harness that missed it. **Found while testing:** the planned
  "partial success" test could not be written, because that case does not exist —
  `get_clean_ring` intersects the ring with the substrate mask, so an empty substrate leaves
  *every* particle without one and the whole table goes. The rows are never a subset, which is why
  the warning names the substrate. 10 tests; restoring `<= 0` turns 2 red, and the other eight
  correctly stay green because they cover what this task must not change. **mypy unchanged at 12.**

- **M3-T15** ✅ (2026-08-07, **ADR-0032**) — **the first measurement of detection *quality* this
  project has ever taken.** Not a defect: the gap M3-T03, T10, T21, T05 and T14 each wrote "not
  claimed" for. `core/science/evaluation.py` — in `core`, not `tests/`, because M4's annotation
  flow and M8's training loop need it — scores detections against the ground truth `phantoms.py`
  has carried since the audit and whose own docstring asked for exactly this. **A match is a
  centre inside the particle** (`match_factor × radius`), scale-free where a fixed pixel threshold
  would be two different physical tolerances across a set spanning 1.95–29.3 nm/px; **the
  assignment is one-to-one and optimal** (`linear_sum_assignment`), because ten boxes on one
  particle are 1 TP and 9 FP, and greedy nearest-first can pair the wrong two and then measure the
  localisation error between the wrong objects — a test pins a case where greedy costs 6.0 against
  the optimum's 4.0. Ratios with a zero denominator are `None`, not a substituted 1.0. Delta:
  **7 differences, all `detection_quality: ADDED`; nothing moved.** The numbers: `flat`,
  `tilted`, `coarse`, `sem` and `tem` all at **precision 1.000 / recall 1.000** with 0.36–0.61 px
  localisation; `afm_dense_overlapping` at **0.983 / 0.843**, 11 misses of 70 where particles
  overlap; and `afm_sparse_low_snr` at **recall 0.000**, six particles and none found — **B-062**,
  filed not fixed, because it moves numbers. **`tem_dark_particles` 22 of 22 at 0.36 px** turns
  ADR-0023's "0 → 22 blobs" from a count into a measurement. Radius error is negative on every AFM
  phantom and positive on both image phantoms — a calibration offset, not scatter, which is why
  the signed error is reported next to the absolute one. 21 tests. **A phantom is not a sample**,
  written into the ADR before the numbers existed so they could not quietly widen it.

- **M3-T14** ✅ (2026-08-07, **ADR-0031**) — **D-16 and D-17 fixed: one measurement schema, and a
  `bbox` that means something.** Four producers had four column sets; the fix is a **core** —
  `particle_id x_px y_px area_px method` — plus blocks that are present in full or absent in full,
  with `method` naming the producer so a consumer knows which to expect. **Not** one superset with
  NaN: that says SEM/TEM has heights and they are all missing, and this milestone has spent six
  ADRs on absent versus substituted. Reading the four producers found three faults where the audit
  named one. One quantity under two names — `score`/`sam_score` (the audit caught it),
  `mask_area_px`/`area_px` (it did not). Columns that varied **per row**, because both SAM2
  producers assembled records with `if k in res`. And **two quantities under one name**:
  `radius_nm` was the detector's blob radius in the baseline table and the measured mask's radius
  in the SEM/TEM one, so concatenating them silently averaged two different measurements — now
  `detector_radius_nm` (where we looked) and `radius_nm` (what we found). `bbox` became
  `tuple[int, int, int, int] | None`, the sixth substitute value deleted this milestone, and the
  `type: ignore` M2-T02 wrote to expire itself expired on schedule. Detect mode returns the
  modality's empty table — the case ADR-0027 named and left open. Delta: **62 differences, all of
  them names, dtypes or added columns; 35 column digests unchanged and 0 changed**, with the
  renamed column's digest byte-identical to its predecessor and `peak_nm == height_nm +
  baseline_nm` on every phantom. **The harness had the same bug the code did** — `list(det.bbox)`
  assumed the tuple was always there, which is D-16 living inside the tool built to detect it. 31
  tests over five tables; the two SAM2 producers run against a **stub predictor** returning three
  candidate masks and their scores, because there are no weights here or in CI — so their golden
  delta is zero *by construction*, stated rather than implied. **mypy unchanged at 12**: two new
  errors appeared and were fixed, one of them a comparison that could only be False, in code
  written minutes earlier.

- **M3-T13** ✅ (2026-08-07, **ADR-0030**) — **D-15 fixed: the library has one way of saying no.**
  The audit's table was five inputs and five behaviours; the harness's own matrix was worse —
  eleven degenerate inputs against five entry points produced `ValueError`, `TypeError`,
  `IndexError`, `LinAlgError` and `RuntimeError`, and `detect_particles` answered a 1-D array, a
  3-D array, a NaN map and an infinite map with **a clean empty result**, so an unusable input and
  an empty sample were the same answer. `core/errors.py` now holds seven classes, **each also
  inheriting the builtin it replaced at its site** — the `json.JSONDecodeError` pattern, which is
  what makes this one commit rather than a migration of every `except` clause in the notebooks —
  and `core/validation.py` holds the one `ensure_height_map` that fourteen entry points call. A
  height map is 2-D, non-empty, integer-or-real and **finite**; the last is the decision, and it
  **supersedes ADR-0018 on non-finite input only**, a flat or negative map still being valid data
  with nothing in it. It also supersedes M3-T08 on boolean input one commit later: a mask is
  refused rather than levelled, so D-13's boolean pathology is unreachable instead of corrected.
  Delta: **129 differences, no measured value among them** — 32 exception types, 28 messages that
  became ours, 15 `raised_in`, 13 cells that used to answer, 11 results that stopped being
  returned. **Foreign messages in the golden: 15 → 0**, which empties the `_unchecked` category
  ADR-0022 created; the mechanism stays, because the next library upgrade can refill it. The
  twelve phantom-level differences are exception types on two probes that were already failing —
  **no phantom value moved**, which had to hold, since every phantom is a valid image. 109 tests,
  centred on 7 bad inputs × 10 entry points — **70 combinations, one error type** — and the same
  sweep proving a valid map passes all ten, because validation that rejects real data would be
  worse than the defect it fixes. Two findings filed rather than fixed: **B-060** (levelling that
  fits around a dropped scan line) and **B-061** (a rough opening radius of 0, which is reachable,
  makes the opening the identity, and is what ADR-0025 recorded — so refusing it would move a
  number). **mypy unchanged at 12.**

- **M3-T08** ✅ (2026-08-07, **ADR-0029**) — **D-13 fixed: levelling returns the residuals it
  computed.** `flatten_lines` pre-allocated with `np.empty_like(z)`, and the residual of a row's
  own fit is fractional by construction, so an output array narrower than float64 rounded every
  value away. The allocation promotes with `np.promote_types(z.dtype, np.float64)` — the rule
  `flatten_plane` has always followed by letting NumPy promote `z - plane`; a hardcoded float64
  would have matched it by coincidence and diverged on the one dtype `flatten_plane` keeps wide.
  Delta: **13 differences — 8 dtypes `float32 -> float64`, 4 sums, 1 added group; no phantom
  moves**, since every recorded chain is float64 before `flatten_lines` sees it — which is why
  the golden never caught this and why the audit's R9 asked for an integer case. **It understated
  the defect twice.** An integer output does not truncate a negative residual, it **wraps** it: on
  the newly recorded 8-bit phantom 100 % of pixels are wrong, by up to **257**, and every pit is
  rendered as a peak — a reader would have seen features that are not there rather than a
  degraded map. And boolean input, unmeasured, came back as a *mask* of where the residual was
  non-zero. The exposed caller is **`load_microscopy_image`**, `uint8` from `cv2.imread` and the
  only file entry point SEM/TEM has. The four moved sums are the fix as a physical property: a
  least-squares residual sums to zero, and float32 storage was leaving it at 1e-6 instead of
  1e-13. Every rejection deferred to **M3-T13**, deliberately — the three degenerate inputs that
  raise still raise exactly what they raised before. 17 tests; restoring `np.empty_like(z)` turns
  **14** red, the three survivors being the float64 cases. **mypy unchanged at 12.**

- **M3-T18** ✅ (2026-08-06, **no ADR — a side effect of M3-T05**) — `YoloDetector._last_result`
  was initialised to `None` and therefore *typed* `None`, so every attribute read off it was a
  mypy error. M3-T05 needed a second array from it and would have added a third; the field is now
  annotated `Any`, as its own comment already described it, and all three errors went. **No
  runtime guard was added and none is needed** — every access is two lines below the assignment
  in the same method, and the public `last_result` property returning `None` before the first
  `detect()` is the documented meaning of the field. Recorded as done rather than as its own
  commit, because that is what happened.

- **M3-T05** ✅ (2026-08-06, **ADR-0028**) — **D-09 fixed: a detection carries its own score, or
  none.** The model scores every box, `cfg.yolo_conf` filters on those scores, and the conversion
  dropped them, so every YOLO detection reported 1.0. Both backends now pass theirs
  (`boxes.conf`; `CombineDetections.filtered_confidences`, post-NMS); a length mismatch raises,
  because a shifted score reads as a measurement of the wrong box; `0.0` survives, because it is
  falsy and an `or` fallback would erase the least confident detection. `confidence` became
  `float | None` defaulting to `None` — **LoG had been claiming 1.0 as well**, and its response
  is not a probability. Delta: **29 keys added, 0 values changed**; the finding is that
  `default_detection_confidence` had **never been recorded**, so the golden could not have caught
  D-09. 7 tests; restoring the drop turns 6 red. **mypy 14 → 12.**

- **M3-T12** ✅ (2026-08-06, **ADR-0027**) — **D-08 fixed: an empty measurement table keeps its
  columns.** Two ordinary outcomes drop a row — a mask past the image edge, a non-positive height
  — and when they took the last one, "no particles" and "no such column" became the same object.
  `BASELINE_COLUMNS` declares twelve columns and their dtypes; `empty_baseline_table()` returns
  them with zero rows. Dtypes are part of the promise, and the drift guard is a test on the
  **populated** path, because the golden's empty case has no columns to compare. Delta: **78
  differences, 0 values moved** — and one of the six blocks is not the synthetic probe:
  **`afm_sparse_low_snr` detects 0 blobs on its ordinary path**, so the defect was live on a real
  phantom's normal run since the baseline. Left for M3-T14 with reasons: the two SAM2 producers
  vary their columns *per row*, and detect mode's schema is modality-dependent. **B-059 filed**:
  `nan <= 0` is `False`, so a NaN height reaches the table. 7 tests; restoring
  `pd.DataFrame(results)` turns 3 red.

- **M3-T17** ✅ (2026-08-06, **ADR-0026**) — **D-07's third face, and the last.** The SPM
  fallback for a header with no `Scan Size` set `scan_size_nm = None` and divided by `samps` on
  the next line. Now `(None, None, z)`: absent metadata, intact height map. `Samps/line: 0` and a
  stated `Scan Size: 0` raise instead, each naming its field — ADR-0025's absent-versus-wrong
  distinction, applied to the second loader so the two agree. **0 golden differences, and none
  was possible**: `afm_io` has no phantom, and that is recorded rather than left blank.
  **mypy 15 → 14** — the annotation `-> np.ndarray` on a function returning a three-tuple was
  this defect's static shadow, in the baseline since M1-T04. 3 tests; restoring the division
  turns 3 red.

- **M3-T20** ✅ (2026-08-06, **ADR-0025**) — **the other half of D-07.** `load_afm(fmt="npy")`
  fabricated `pixel_size_nm=1.0` and `scan_size_nm=float(z.shape[0])`, so every `_nm` downstream
  was a pixel count wearing nanometre units and no consumer could tell — and `or` swallowed an
  explicit `0.0` on the way. The loader now passes through what it was given, `AFMRawData` and
  `PreprocessingResult` carry `float | None`, and a scale that *is* given must be positive.
  `build_substrate_map` takes `None`: `radii_nm` and `typical_radius_nm` are absent, and the
  `min_size_nm` filter is skipped **with a warning** — silently would be D-04 again. Delta:
  **5 keys added, 0 values changed**; the new `build_substrate_map_no_scale` records that an
  unscaled run equals a scaled one with `min_size_nm=0` — bit-identical substrate on four
  phantoms, and on `afm_sparse_low_snr` **17 objects → 3351**, typical radius **2.99 → 0.80 px**,
  opening radius **8 → 5**. The ADR's draft claimed the pixel-space result was unaffected; the
  golden disagreed on one phantom in five and the ADR now says what was measured. **M3-T17
  inherits a contract instead of a question.** 10 tests; restoring the fabrication turns 6 red.

- **M3-T02** ✅ (2026-08-06, **ADR-0024**, decision **B2**) — **D-04 fixed: the minimum particle
  size is a physical size.** `min_size_nm` was converted to pixels with `int()` at three sites,
  compared against `radii_px`, and then — three lines later, twice, identically — `radii_px` was
  converted back to nanometres for the result. The comparison now happens where the parameter was
  always stated: `radii_nm >= min_size_nm`. B2 answered **filter in nanometres**; a "floor of at
  least 1 px", the other candidate, was rejected because at 29.3 nm/px it would discard
  everything under 29.3 nm. Delta: **47 differences — 27 changed, 15 added, 5 removed**;
  `afm_sparse_low_snr` **75 → 17** objects, the other four AFM phantoms **byte-identical**, and
  **no height moves anywhere** (the final opening radius is 8 before and after, so `substrate`
  and `z_above` are unchanged). **The phantom built for D-04 does not move**, which is the
  finding: at 9.77 nm/px a single pixel is already 5.5 nm, so the broken filter and the correct
  one agree. Re-read of all **628** scan headers: 90 % floored to zero (568), of which **365
  (58 %) had nothing to remove**, **203 (32 %)** lost a working noise filter, and the finest
  **60 (10 %)** were hurt by truncation rather than the floor. **mypy unchanged at 15** — a unit
  error is invisible to a type checker. 5 tests; restoring the `int()` turns 3 red. The
  duplicated `radii_nm` assignment went with it: the change forced the line to move.

- **M3-T10** ✅ (2026-08-05, **ADR-0023**, decision **B3**) — **D-12 fixed: TEM finds 22 of 22
  where it found 0.** Both detectors kept the bright side unconditionally — the LoG one by
  thresholding and by `blob_log` itself, the YOLO one by inverting every image because the weights
  expect dark particles — and TEM images by absorption, so both were working on the background.
  B3 answered **configured, with a per-modality default**: an auto-detector's failure mode is
  D-12's own (zero particles, no error), and the operator could not tell a bad guess from an empty
  sample. `Polarity`, written in M2-T02 for this task and adopted by nothing since, is now a
  `PipelineConfig` field whose `None` resolves to the modality's convention in `run_pipeline`.
  **One inversion at the entrance**, `max - z`: its own inverse, and positive-maximum-safe per
  ADR-0018. Both detectors in one commit, because it is one defect mirrored. Delta: **19 values
  changed, 12 keys added** — `tem_dark_particles` 0 → 22 blobs, its prepared YOLO input 43.3 →
  211.7 mean grey, `config_fields` 12 → 13; **SEM and all five AFM phantoms byte-identical**.
  **Not claimed:** better YOLO detections — inference is outside the gate, so what is shown is
  that the input is right. 14 tests.

- **B-058** ✅ (2026-08-05, **ADR-0022**) — the golden compared exception messages exactly, and
  most of them are CPython's or a library's. 3.14 reworded `too many values to unpack` and the
  first real CI run called it characterization drift (M1-T08). Now the **type** and the **raising
  function** are always compared and the **message** only when we wrote it — the frame must be
  inside `nanoscope` *and* the raising line must be an explicit `raise`, because either alone
  misclassifies: `h, w = z.shape` in our file is CPython's wording, and skimage raises explicitly
  too. **15 keys renamed to `error_message_unchecked`** (skipped by `compare`, still recorded),
  **7 remain compared, all `estimate_radius_otsu`'s** — the ones PROJECT_RULES §3 governs. 0
  values changed. **A Python upgrade no longer reads as drift**, which `STATE.md` listed as the
  precondition for touching the interpreter. 6 tests.

- **M3-T21** ✅ (2026-08-05, **ADR-0021**, decision **B7**) — the tiled YOLO backend **has never
  tiled**: `_prepare_image` emits one 640 px square and the crop shape is 640, so `get_crops_xy`
  computes one step per axis. It ran the direct backend's work through an extra library, more
  slowly, and small particles were never seen at native resolution — the only reason tiling
  exists. The overlap cannot rescue it (`int((side−shape)/step)+1` is 1 for any step when
  `side == shape`, asserted at 0/25/50/75 %); only input size can, and real tiling needs 1120 px.
  B7 answered **keep it, stop defaulting to it**: `use_tiling=False` in both `YoloDetector` and
  `PipelineConfig`, and the degenerate case now says so in the log. Delta: **zero golden
  difference** — inference is outside the gate — but the backends are **not bit-identical** even
  at one crop (`CombineDetections` adds a second NMS), so real detections may move and **no claim
  is made that either is better**; M3-T15 owns that question, and until it exists nobody can
  choose between "upsample to 1120" and "smaller crops". 9 tests.

- **M3-T09** ✅ (2026-08-05, **ADR-0020**, decision **B4**) — **D-10 fixed**. `disk(8.5)` is an
  18x18 element with no centre pixel, so the opening was biased by half a pixel; three sites fed
  it a float and each did something different. The operator answered B4 **round up**, and the
  `ceil` lives in `get_substrate_map` — the funnel all three pass through — so one line fixes
  them all. Up rather than down because a radius smaller than a particle recovers a "substrate"
  containing the particle's own top, while an over-large disk only over-smooths, which the method
  already tolerates. `build_substrate_map` reports the integer it used, keeping ADR-0014's
  principle. Delta: **696 golden values move, 0 keys added** — radius +1 or +2 on all five AFM
  phantoms, **no particle count changes**, largest height move **0.049 nm (0.37 %)** on
  `afm_dense_overlapping`. The 696 are propagation, not magnitude. **mypy 18 → 15**, and the
  three that went were this defect's static shadow, in the baseline since M1-T04. 11 tests;
  restoring the floor turns 4 red.

- **M3-T11** ✅ (2026-08-05, **ADR-0019**) — **D-07 fixed**. `MicroscopyData.nm_per_pixel` is
  `float | None`, `run_pipeline` passes it to the detector unread, and both detectors multiplied
  by it: `TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'`. An SEM or TEM
  image without scale metadata had exactly one outcome, and it was an exception. Now `None`
  propagates and the physical value becomes **absent**, which is the invariant D-07 states and
  the one `measure_geometry_from_mask` has kept since M2-T06. `Detection.radius_nm` is
  `float | None`; the blob array's nm column is `NaN`, because one ndarray column is one dtype,
  and `_blobs_to_detections` converts it at the entity boundary. **That NaN is not the NaN
  ADR-0018 removed one commit earlier** — this one is a marker in a reporting column, read by one
  line and never computed with; that one came out of arithmetic and reached a threshold
  comparison. Delta: **168 golden keys added, 0 changed** — every phantom has a scale, so nothing
  recorded moves. **mypy 19 → 18, and the error that went was the defect**: `pipeline.py:62` had
  been reporting D-07 at the assignment rather than at the crash since M1-T04. 8 tests;
  substituting `pixel_size_nm or 1.0` — the tempting wrong fix, and what the npy loader does
  today — turns 4 red.

- **M3-T07** ✅ (2026-08-05, **ADR-0018**) — **D-11 fixed**. The LoG path normalises with
  `z_above / z_above.max()` in two places and checked the divisor in neither. `max() == 0` gives
  a wholly `nan` image, `blob_log` finds nothing, and the operator is told to *lower the
  threshold* — a knob that cannot help. `max() < 0` **inverts the topography**, so the substrate
  outshines the peaks; measured on caps at −10 nm the adaptive threshold came out **2.4997**,
  compared against responses that live in `[0, 1]`. The guard is `not z_max > 0`, **not**
  `z_max <= 0`, because the two differ exactly on `nan` and `nan` is the case that spreads.
  **Zero particles is the answer, not an error** — the opposite call from ADR-0017 one commit
  earlier, because there the *caller* asked the impossible and here the *data* is simply flat,
  which is a legitimate scan region. Delta: **65 golden keys added, 0 changed** — every phantom
  goes through `build_substrate_map`, which guarantees `z_above >= 0`, so nothing on the working
  path moves. The negative case is reachable only via `LogDetector.detect` on raw SEM/TEM, which
  is **D-12**, still on B3. **The harness could not see this defect at all** and two changes in
  the same commit fixed that: `negative_with_structure` (the old `all_negative` is *constant*, so
  the flip has nothing to flip) and recording scalars instead of the string `"non-array"` — that
  line is why 2.4997 sat unrecorded since Phase 0. 11 tests; restoring the raw division turns 3
  red, and the two that stay green do so by construction: a `nan` image also yields no blobs,
  which is precisely how the defect survived.

- **M3-T06** ✅ (2026-08-05, **ADR-0017**) — **D-05 and D-06 fixed**, one commit because they
  are the same eight lines. The size filter could remove every object, and then `np.median([])`
  returned `nan` with a warning; the `nan` reached the LoG sigma range and failed two calls
  later as `zero-size array to reduction operation minimum`. It now raises where it happens,
  naming the parameter, its value **and the largest object measured** — without that third
  number, "no particles here" and "your minimum is 100× too large" read identically. And
  `n_objects` counts survivors instead of the pre-filter population. Delta: **8 golden
  differences** — `n_objects_reported` **1023 → 75** on `afm_sparse_low_snr` (13.6× over-count
  of single-pixel noise), the `extreme_aspect` degenerate input now fails as
  `estimate_radius_otsu` instead of `cannot convert float NaN to integer` one call downstream,
  plus 5 added keys recording D-05's own reproduction. **Only one phantom moved, and why is the
  point:** D-04 floors `min_size_pixel` to 0 on coarse scans, so the filter usually removes
  nothing — this fix starts mattering on real data the day **B2** is answered. 4 tests;
  restoring the old behaviour turns 3 red. **It also turned an M3-T01 test red**, one that had
  been passing because the sizing silently returned `nan` into a field it never read.

- **M3-T04** ✅ (2026-08-05, **ADR-0016**) — **D-21 fixed**: `_prepare_image` squashed every
  scan into a 640 × 640 square and `_scale_boxes` stretched the boxes back per axis. The two
  agreed, so boxes landed correctly — the defect is that **on a 2:1 scan the model saw
  ellipses**, and `radius_px = min(w, h) / 2` reported the smaller half-axis as a radius. Now
  one isotropic scale, a border of 255 (what the lowest point looks like after the inversion,
  so it reads as substrate rather than as an edge), applied *after* the normalisation, and one
  helper shared by the forward and inverse maps so they cannot drift. Delta: **0 golden
  differences, 7 keys added** — square scans are byte-identical and every phantom is square,
  so the harness gained `non_square_half_height` to characterize the path at all. 5 geometry
  tests; restoring the squash turns 4 red. **Found while reading, filed not fixed: M3-T21.**

- **M3-T03** ✅ (2026-08-05, **ADR-0015**) — **D-03 fixed**: `_prepare_image` cast the float
  height map to `uint8` *before* normalising, keeping only whichever integers 0…255 fell
  inside the map's range and wrapping the rest, then stretching the survivors so the result
  looked correctly exposed. Delta: **67 golden differences, all under
  `yolo_input_preparation`** across all 7 phantoms; unique grey levels **8–208 → 239–256**,
  retention **3.1%–81.2% → 100%**. The spread is the finding — **the cleaner the scan, the
  worse the loss** — and on the quiet 5 nm phantom the old image is **anti-correlated**
  (−0.499) with the correct one, so this was never merely a resolution defect. The cast
  truncates rather than rounds, matching the harness's own reference, which drops
  `mean_abs_diff_vs_normalize_first` to **0.0** and turns the field Phase 0 added to size
  the defect into a permanent guard. 6 tests; restoring the order turns 5 red.
  **Not claimed: better detections.** The weights were trained on images the old path
  produced, and inference is outside the gate (§6) — M3-T15 and M7 own that question.

- **M3-T01** ✅ (2026-08-04, **ADR-0014**) — **D-01 fixed**: `build_substrate_map`'s
  manual-radius branch raised `UnboundLocalError` on **100% of calls** since it was
  written, because the shared `return` read a variable only the other branch bound. The fix
  is one line — `opening_radius = manual_radius_px` — and it deliberately applies **no
  rounding and no floor**: both would pre-empt open decision B4 (M3-T09) or silently
  override an explicit request. Delta: **50 golden differences, every one under
  `build_substrate_map_manual`**; the automatic path is untouched. The harness now records
  the branch's returned arrays instead of only its failure — otherwise the fix would have
  left it less characterized than while broken. 6 tests; restoring the bug turns 5 red.

### M2 — Domain extraction ✅ (closed 2026-08-04)

- **M2-T01** ✅ (2026-08-04) — `nanoscope/` exists: the six layers from ADR-0011
  (`app` `core` `application` `infrastructure` `gui` `resources`) plus `py.typed`, each
  `__init__` stating that layer's half of the dependency rule. Distribution renamed
  `afm-analysis` → `nanoscope`. The regenerated `uv.lock` was **diffed package by package
  before committing** — 119 shared packages, **0 version changes** — because CI runs
  `uv sync --locked` and a quiet re-resolution of numpy or scipy would move the golden for
  a reason unrelated to the task. mypy now checks 20 files instead of 13, and the strict
  `nanoscope.*` override that M1-T04 wrote before the package existed **binds for the
  first time**: 0 errors, strict from line one. **Zero code moved**; no sub-package below
  the layer level, since each arrives with its content in M2-T02…T08.

- **M2-T02** ✅ (2026-08-04) — the first scientific code to move, in **three commits** so
  that drift would be attributable without bisecting. The six dataclasses left
  `src/types.py` for `nanoscope/core/entities/`; `src/types.py` is now a shim that defines
  nothing, verified by loading the pre-move module beside the new one — identical fields,
  order, defaults and factories, and `src.types.X is nanoscope.core.entities.X` for all
  six. **One `Detection` class in the process, not two.** The strict `nanoscope.*` override
  then caught three things verbatim legacy code does not satisfy: two bare generics
  tightened to `dict[str, Any]`, and `Detection.bbox` given a scoped `type: ignore` —
  mypy complaining there *is* **D-16**, and fixing it moves a number the golden records, so
  M3 owns it; `warn_unused_ignores` makes the ignore expire itself. **nanoscope: 0 mypy
  errors.** Finally `Modality`, `Polarity`, `PixelScale`, `DeviceKind` with 8
  mutation-validated tests — **defined, adopted by nothing**, because adoption changes what
  `asdict` produces. Golden: **zero drift**.

- **M2-T03** ✅ (2026-08-04) — preprocessing moved to
  `nanoscope/core/science/preprocessing/` (`flatten.py` + `substrate.py`); `preprocess.py`
  is a shim. The first move of real behaviour — plane fitting, line detrending,
  morphological opening, Otsu. **Proved before the gate ran:** all six functions
  AST-identical, docstrings differing only in trailing whitespace, and the 5 mypy errors
  travelled with the code (21 before, 21 after). Golden: **zero drift**. What the task
  actually settled is how legacy enters a strict package: **declared once in configuration**
  — mypy at default strictness for `nanoscope.core.science.*`, ruff still blocking there
  but ignoring six named rules — instead of a `type: ignore` on every audited defect across
  fifteen more moves. Every entry names the task that deletes it (M2-T11, M2-T12, M3).

- **M2-T04…T06** ✅ (2026-08-04) — three tasks on one branch (they share shims), **16
  definitions moved, golden zero drift**. I/O split along parsing-versus-the-world:
  SPM decoding to `core/science/io/`, the path-opening functions to
  `infrastructure/storage/`. The LoG detector and its ABC to `core/science/detection/` —
  all 7 definitions AST-identical, and `detect_particles` is recorded for all 8 phantoms.
  Measurement split AFM height from mask geometry, which is the point of M2-T06: the
  modality-neutral code was trapped in an AFM module, so the SEM/TEM path depended on
  `src.measure` by accident. Four more `src/` modules are shims. **The `ImageLoader` port
  was deliberately not written** — M2-T08 defines the ports wholesale. mypy 21 → 21.
  Three of ruff's safe fixes landed in `loaders.py` and are named, not rounded up to
  "verbatim". **`RUF046` was wrong about the science**: `round(np.float64)` is not an int,
  so obeying it would have changed the dtype of every measurement DataFrame's `x_px`
  column — it is now ignored with that reason attached.

- **M2-T07 / M2-T08** ✅ (2026-08-04) — the model-backed code left the domain:
  `YoloDetector` → `infrastructure/models/`, the SAM2 runners beside it, and
  `afm_to_rgb`/`overlay_masks` → `infrastructure/imaging/` (neither ever belonged to
  SAM2). **Nothing under `core` imports torch, ultralytics, sam2 or patched_yolo_infer any
  more** — the dependency rule is now a fact, and a test asserts it against `sys.modules`.
  `F821` caught two dangling references the split created, before any test ran. mypy 21 →
  21, after both moved modules joined `core.science` at default strictness. **M2-T08 was
  narrowed on purpose: one port, not seven.** `Detector` is satisfied today by
  `LogDetector` and `YoloDetector` from opposite layers; the other six have no
  implementation and no caller, so they ship with their first adapter, and
  `core/ports/__init__.py` carries the table naming the task for each.

- **M2-T09 / M2-T10** ✅ (2026-08-04) — the layout became enforceable and the rules became
  executable. **All five import cycles (D-18) had one cause**: `src/__init__.py` re-exported
  the pipeline, and Python runs a package `__init__` first, so importing the "dependency
  root" loaded SAM2 and matplotlib. Nothing ever used `from src import X` — emptying one
  file fixed all five. `import src.types` **1198 → 187 modules, 0.77 s → 0.07 s**;
  `nanoscope.core.entities` **626 → 185**, pandas moved behind `TYPE_CHECKING`.
  `test_import_graph.py` checks direction statically over the AST and weight dynamically in
  a subprocess; both proven to fail. **The M2 exit criterion "< 100 modules" was
  unachievable** — numpy alone is 141 — and is corrected in `Roadmap.md` to a named
  heavy-import assertion plus a 250 bound. M2-T10 put the capability matrix in
  `application/capabilities.py` and **fixed D-14**: validation now runs before any detector
  is constructed, with byte-identical messages. 12 tests carry it, because the golden never
  calls `run_pipeline`.

- **M2-T11…T14** ✅ (2026-08-04) — the library stopped printing, started speaking English,
  shed four dead functions and became installable. **Zero numbers moved**, and for the first
  time that took a *declared* golden re-baseline: 6 changed lines, none of them numeric —
  4 translated exception messages plus `stdout_lines` 8→0 and 4→0, because the golden
  records how much a function prints. **It also caught a bug in M2-T11 before any human
  did** (`"1%%"` is only an escape when `logging` formats, which it does not without args).
  **No `LogSink` port — ADR-0013**: it would only wrap `logging`, whose `Handler` is already
  the extension point. That is the second of seven planned ports to dissolve on contact with
  reality. **M2-T13 deleted 4 of the audit's 10 "unreachable" functions and kept 6** —
  `estimate_log_threshold` is recorded by the golden, `load_microscopy_image` is the only
  SEM/TEM entry point, three are used by the notebooks. `nanoscope` is now a real wheel
  (`py.typed` in, `src/` out) and the `pythonpath` hack is half deleted. Ruff findings inside
  `nanoscope/` with ignores off: **64 → 13**.

- **M2-T15 / M2-T16** ✅ (2026-08-04) — **`src/` deleted entirely**, and the milestone with
  it. The title understated the task: three modules had never had a shim and had to move
  first (`pipeline` and `preprocessing_pipeline` → `application/use_cases/`, `visualization`
  → `infrastructure/imaging/`). `pythonpath` deleted outright; mypy points at one package.
  A test caught a naming trap a review would not have: a module and a function of the same
  name shadow each other through `__init__`. M2-T16 rewrote `PROJECT_CONTEXT.md`, which had
  drifted to describing `src/`, the deleted frontend and a `pytest.ini` removed in M1-T05.

### M1 — Repository hygiene ✅ (closed 2026-08-04)

- **M1-T01** ✅ (2026-08-03) — tracked files 2 877 → **77**; `frontend/node_modules`
  (2 800 files) untracked; `yolov8s-world.pt` (26 MB) removed from the index before it
  entered history; `.gitignore` rewritten; `.claude/settings.json` now shared; junk
  deleted (`.zip`, four `__pycache__/`, tool caches, empty `output/` and `notebooks/`,
  stray root `package-lock.json`); `plan.md` archived to `docs/archive/`.
  Characterization: **zero drift**.
- **M1-T11** ✅ — absorbed into M1-T01.
- **M1-T02** ✅ (2026-08-03) — pytest 9.1.1, pytest-cov 7.1.0, ruff 0.16.1, mypy 2.3.0
  declared and installed; no runtime version moved, golden still stable. Baseline
  measured: **196 ruff findings** (109 in `src/`), **30 mypy errors**, **1 test, failing**.
  Nothing fixed — that is M1-T03/T04 and M2.
- **M1-T03** ✅ (2026-08-03) — ruff configuration repaired: `fix = true` removed (it made
  `ruff check` rewrite sources), `select`/`ignore` moved under `[tool.ruff.lint]`,
  py311 → py312, template `known-first-party` fixed, dead `S101` dropped, notebooks
  excluded from lint. `src/` findings unchanged at 109 — a repair, not a rule change.
  Total 196 → 128. Characterization: **zero drift**.
- **M1-T04** ✅ (2026-08-04) — mypy configured: strict for `nanoscope.*` from its first
  line; `src/` checked but **not** silenced (22 errors after per-module stub handling).
  All 30 default errors classified before writing config: 13 statically confirm audit
  defects **D-01, D-02, D-07, D-10, D-16**, and **3 new defects** were found and filed
  (**M3-T17…T19**), including a crash in the SPM parser's no-`Scan Size` fallback.
- **M1-T05** ✅ (2026-08-04) — the characterization golden now runs under `pytest`, via a
  single new seam in `capture.py` (`diff_against_golden()`); the CLI is unchanged. Marked
  `slow` (**192 s measured**, not the ~100 s the docs claimed); `pytest -m "not slow"`
  skips it in 1.4 s. `pytest.ini` folded into `pyproject.toml` and deleted — while it
  existed, pytest ignored `[tool.pytest.ini_options]` silently. The negative case was
  proven, not assumed: a perturbed golden produced a red run naming the moved quantity.
  **The M2 safety net is now mechanical.**
- **M1-T06** ✅ (2026-08-04) — `tests/test_io.py` (no assertions, wrong exception, absent
  fixture path) deleted; replaced by `tests/unit/test_afm_io.py`: **22 tests** over a
  synthetic Nanoscope byte stream derived from a real local header — round trip,
  calibration, unit conversion, 8 failure modes, npy and SEM/TEM. No binary fixture, no
  `data/`. **`pytest` is green for the first time (23 passed, 200 s).** The suite was
  validated by mutation: 4 edits to the parser, 3 killed immediately, and the 4th exposed a
  test that could not fail — now fixed. One new defect found → **M3-T20**.
- **M1-T07** ✅ (2026-08-04) — pre-commit: **9 hooks, each demonstrated failing** on a
  deliberately bad staged file. ruff runs as a `repo: local` hook on the project's own
  version, so no second version is ever declared. Rewriting hooks (format, whitespace) skip
  `src/` **and `preprocess_batch.py`** — the `--all-files` sweep caught them editing the
  scientific core, which the original `^src/` exclusion missed; refusing hooks apply
  everywhere. pytest and mypy stay off the commit path by design. `src/` files modified:
  **0**; golden: zero drift.
- **M1-T08** ✅ (2026-08-04) — CI written and verified locally: format → lint → tests+golden,
  `src/` reported not blocking. CI installs a `ci` group with **no torch, ultralytics, sam2
  or patched-yolo-infer** — every heavy import in `src/` turned out to be function-local —
  and a step fails the job if one appears. Two traps caught by running it: `uv run` re-syncs
  and would have reinstalled the full runtime (`UV_NO_SYNC` set), and `ruff format` rewrites
  Python inside Markdown docs (`*.md` excluded). The legacy exclusion moved into
  `pyproject.toml`, declared once for hooks and CI. Both rejection cases confirmed red.
  **Then it was pushed, and three runs found what local verification could not:** no
  readable failure reason (job logs need admin → diagnostics added), a non-existent
  `setup-uv@v9` tag (my error; both actions now pinned exactly), and — the real one — a
  single golden difference that was an exception *message*, not a number. CI resolved
  **Python 3.14**, which reworded `too many values to unpack`; 3.12 is now pinned and
  asserted. **Run 4 is green.** The underlying fragility — the golden stores CPython
  exception text — is filed as **B-058** and needs an ADR, not a quiet edit.
- **M1-T09** ✅ (2026-08-04) — notebook outputs stripped with the configured hook:
  **8.3 MB → 32 KB**, every one of the 45 code cells intact, and the outputs remain in git
  history. Both notebooks moved to `notebooks/` with a README stating they are experiments,
  that nothing may import them, and how to recover the outputs. `main.ipynb` — a tracked
  **0-byte file that was not valid JSON** (audit §330) — deleted. Tracked working tree
  17 MB → **7.8 MB**. **`pre-commit run --all-files` is green for the first time**; the
  last red was a missing final newline in one archived document.
- **M1-T10** ✅ (2026-08-04) — one gate, one description: a 53-line `Makefile`
  (`check` `format` `lint` `test` `fast` `golden` `types` `lint-legacy`; bare `make` lists
  them), and **CI rewritten to call the targets** — the workflow no longer holds a copy of
  any command, which was the point. Proven to fail closed: a misformatted file stopped
  `check` at step 1 in **0.04 s**, exit 2, never reaching the 190 s test step; a failing
  test failed its target. `types`/`lint-legacy` stay outside `check` because the legacy
  baseline is non-zero by design — a gate that cannot pass is a gate people skip. Writing
  it exposed that the three existing descriptions had already drifted: `PROJECT_RULES` §6
  listed `mypy nanoscope` (no such package yet) and a golden command M1-T05 had folded
  into `pytest`. **CI run 14 green on the first try, 216 s**, environment assertion intact.
  **M1 closes here.**

### Decisions executed (2026-08-04)

- **B1 → `nanoscope`** — ADR-0011 Accepted. Unblocks every M2 task.
- **B5 → delete** — **ADR-0012** (supersedes ADR-0007): `frontend/` and
  `preprocess_batch.py` removed. Tracked files **78 → 63**, and the blocking lint/format
  carve-out shrank from two paths to one, `src/`, which M2 then dissolves. Ruff findings in
  the legacy core **117 → 109**, all now in `src/`. Both files remain in git history.

### M0 — Engineering foundation (2026-08-03)

- Repository analysed: 12 source modules / 2 021 LOC, plus a React client, notebooks and
  an existing Phase 0 audit
- Strengths and weaknesses recorded with evidence → `docs/Architecture.md` §2
- Target Clean Architecture defined (`core` / `application` / `infrastructure` / `gui` / `app`)
- Project constitution written → `docs/PROJECT_RULES.md`
- 10 milestones, 110 tasks → `docs/Roadmap.md`, `docs/TASKS.md`
- 11 ADRs written → `docs/ADR/`
- Session/state protocol established → this file, `docs/Progress.md`, `docs/CURRENT_TASK.md`
- First task selected → `M1-T01`

### Inherited from earlier work (pre-M0, already in the repository)

- Working scientific pipeline: SPM I/O, flattening, substrate estimation, LoG and YOLO
  detection, SAM2 segmentation, height measurement
- Phase 0 audit with 24 reproduced defects → `docs/audit/2026-07-28-baseline-audit.md`
- Characterization golden baseline with 8 seeded phantoms →
  `docs/audit/characterization-baseline.md`, `tests/characterization/`

---

## In progress

**M2 is closed and M3 is well under way.** Sixteen tasks done — M3-T01 to T12, T17, T18, T20,
T21 — plus B-058, across five sessions. **Every `critical` and every `high` defect the audit
reproduced is closed**, which is the first of M3's five exit criteria, and **D-13 went with
them**. What is left is `medium` and below: T13, T14, plus T15 (the evaluation harness) and T16.

**Levelling now agrees with itself about dtype** (M3-T08). Both halves promote the way NumPy
does, so an 8-bit image — which is what the SEM/TEM loader returns — levels to its residuals
rather than to a wrapped integer map.

**And the library now has one way of saying no** (M3-T13). Fourteen entry points, one contract,
seven exception classes that each also *are* the builtin they replaced. The thing that made D-15
scientific rather than cosmetic is closed with it: `detect_particles` can no longer answer a NaN
map, a 1-D array or a 3-D array with "no particles found".

**And one way of reporting what it measured** (M3-T14). Four producers, one schema, one name per
quantity — and, more importantly, one quantity per name: `radius_nm` used to mean the detector's
radius in one table and the measured mask's in another.

**And, at last, a way to ask whether any of it is any good** (M3-T15). The golden now records
precision, recall and localisation against ground truth on all seven phantoms. Four of them are
perfect; one finds nothing at all, which is **B-062**.

**And no `NaN` reaches a measurement table** (M3-T22) — the third site of ADR-0018's `not x > 0`,
and the last defect carried over from an earlier task in this milestone.

**And the rough opening is always an opening** (M3-T23), and it no longer rounds itself down
first (M3-T24). A substrate identical to the image is unreachable through the automatic path, the
cost ADR-0025 attributed entirely to the missing size filter turns out to have been four fifths
the collapsed radius, and `_integer_radius` is finally the only rounding a radius gets — which is
what ADR-0020 said in the first place.

**The YOLO input path is now correct in three respects** — the data survives preparation, the
sample keeps its shape, and the polarity matches the modality — and none of those claims extends
to detection quality, which nothing in the gate can measure; **M3-T15 is the task that would
change that, and three ADRs have now had to write "not claimed" for want of it.** The LoG path no
longer constructs a `nan` image, and its adaptive threshold always lands in the interval it is
compared against. **The noise filter runs for the first time** on the scans where it was floored
away. **An unknown scale is a state on all three of its routes**, and nothing fabricates a
number to stand in for one — not a pixel size, not a minimum particle size, not a confidence, not
an empty table's columns.

**Repository state:** `main` is at `aceb5c7` and carries all of M0, M1, M2 and M3-T01. All
of M3's work lives on **one branch, `sci/m3-numerical-correctness`**, 24 commits ahead of `main`. **The 32 task branches were consolidated into it on 2026-08-06** at the operator's
instruction: the stack was strictly linear, so every one of them was an ancestor of the tip and
no commit was lost — that was verified branch by branch before anything was deleted, locally and
on `origin`.

**This is a declared deviation from PROJECT_RULES §7, "one task per branch."** Eleven tasks now
share a branch. The rule's purpose — that a task's change be attributable on its own — is still
served, because it never rested on the branch: each task is one commit, with its own ADR, its own
golden update and its own quantified delta in `Progress.md`, which is what ADR-0010 actually
requires. The branch was only ever a label. If the rule is meant to bind the branch too, it needs
an amendment saying so; it is recorded here rather than left as a silent violation.

All fourteen task branches were green on CI before they were deleted (#44–#50, #52, #54–#56), and
the surviving branch was the same commit CI ran on as **#56**. **M3-T08 is green as #61** (407 s), **M3-T13 as #63**, **M3-T14 as #64**, **M3-T15 as #65**, **M3-T22 as #67**, **M3-T23 as #69**, **M3-T24 as #71**, **M3-T25 as #73** and **M3-T26 as #74** — every task commit of this session verified on the machine without torch.

**There is no `src/`.** One package, `nanoscope`, 41 modules across four layers, installed
rather than path-hacked.

**Legacy in transit is declared, not hidden.** `nanoscope.core.science.*` runs at mypy's
default strictness and carries six named ruff ignores; every entry names the task that
deletes it (M2-T11, M2-T12, M3). The rest of `nanoscope` stays strict and 0.

Locally, `make check` is green end to end: format, lint, then the full suite including the
golden, exit 0.

---

## Blocked / needs decision

Decisions only the operator can make. Each blocks a specific task.

| # | Question | Blocks | Why it needs the operator |
|---|---|---|---|
| B6 | **Real sample data in git.** `data/` holds 628 SPM scans and is ignored. Should one small representative scan be committed as a test fixture? | M3-T16 | Data ownership and repository size |

**Answered 2026-08-05 by the operator, and all now executed:**

- **B4 → round up.** M3-T09, ADR-0020.
- **B7 → keep the tiled backend, stop defaulting to it.** M3-T21, ADR-0021.
- **B3 → polarity is configured, with a per-modality default.** M3-T10, ADR-0023.
- **B2 → filter in nanometres, delete the `int()`.** M3-T02, ADR-0024. The floor-of-1-px
  alternative was rejected in the ADR: at 29.3 nm/px it discards everything under 29.3 nm.

**Closed 2026-08-04 by the operator:**

- **B1 — package name → `nanoscope`.** ADR-0011 moves from Proposed to **Accepted**. This
  was the last thing blocking **M2**; M2-T01 can start as soon as M1 closes.
- **B5 — fate of the parked work → delete.** `frontend/` (21 tracked files) and
  `preprocess_batch.py` removed under **ADR-0012**, which supersedes ADR-0007. The third
  part of B5, the notebooks, was answered differently in M1-T09: kept, stripped, moved.

None of the remaining questions blocks M1 or M2.

---

## Next

1. **`M5-T04` — the project explorer**, which is where ADR-0044's obligation lands: an image can
   now be removed from a panel, and the confirmation has to count the annotations it would take
   with it. **`M5-T07`** still owes ADR-0043's thread hop for job listeners
2. **`make types` joins `make check` as blocking.** M4 is over, so the M1 exit criterion deferred
   "while the legacy core is `src/`" is now only about the **6** inherited errors: four in
   `use_cases/pipeline.py` — three of which are the `Detector` port's absence, which M5-T01's
   composition root is the natural place to remove — and two third-party overloads
3. **The four open M3 findings and M3-T16** (blocked on **B6**) are unchanged; **B-068** joins
   them as a question only the operator can answer, and **B-040** goes last of all because it
   rewrites every SHA above it
4. **B-058 is done (ADR-0022)** — a Python upgrade no longer reads as drift, so the 3.12 pin in
   CI is now a choice rather than a constraint
5. **B-054** (two README figures over 1 MB) is the one M1 exit criterion left open;
   it belongs to the README rewrite in M9-T01

---

## Health indicators

| Indicator | Value | Target | Source |
|---|---|---|---|
| Tracked files | **127** (was 2 854) | see note | `git ls-files \| wc -l` |
| Tracked working tree | **7.6 MB** ✅ (was 17 MB) | — | `git ls-files -z \| xargs -0 du -ch` |
| Tracked model weights | **0** ✅ (was 1) | 0 | `git ls-files '*.pt'` |
| `.git` size | 81 MB | — | `du -sh .git` — history unchanged, see B-040 |
| Library LOC | 2 021 | — | `wc -l nanoscope/**/*.py` |
| Meaningful tests | **888, all passing** ✅ (was 1, failing) | ≥ 80% of core | `pytest -q` |
| Golden enforced automatically | **yes** ✅ (was: by discipline) | yes | `pytest` |
| `src/` modules moved into `nanoscope/` | **12 of 12** ✅ — `src/` deleted | 12 | `git ls-files` |
| ruff findings, declared-and-owned | **14** in `nanoscope/` (was 109 in `src/`) | 0 | `make lint-legacy` |
| ruff findings, blocking | **0** ✅ | 0 | `make lint` |
| mypy errors | **6**, all inherited with moved code, none silenced; new code strict | 0 | `make types` |
| Characterization phantoms | 8 (7 carry `yolo_input_preparation`) | 8 | `tests/characterization/` |
| Open defects | **17** (was 28) — B-068 filed by M4-T15 — every reproduced numerical defect is closed; what is left is documentation (D-24 → M9) and the new M3-T19 | 0 critical | audit §2, M3-T17…T21 |
| Import cycles | **0** ✅ (was 5), and a test refuses new ones | 0 | `tests/unit/test_import_graph.py` |
| `print` calls in library code | **0** ✅ (was 13), asserted per module | 0 | `tests/unit/test_logging.py` |
| Non-English lines in library code | **0** ✅ (was 197) | 0 | `grep -rn "[а-яА-ЯёЁ]"` |
| Lint/type/test gate | **green end to end** ✅ — hooks on commit, CI on push | stays green | GitHub Actions |
| The gate has one definition | **yes** ✅ — `make check`, and CI calls the same targets | one | `Makefile` |
| Tracked files over 1 MB | **2** ❌ — two README figures, B-054 | 0 | `git ls-files` + `ls -l` |

> **The `< 100` target has done its job and expired — the count passed it at M2-T07.** It was M1's measure of
> *junk* — 2 800 `node_modules` files. M2 adds real source: each move leaves a shim and
> creates two or three modules. Passing 100 means the extraction is working, not that
> hygiene regressed. The
> meaningful successor is the row above it: **tracked files over 1 MB**, which must stay
> at zero once B-054 closes.
| Commit-time gate | **9 hooks, all proven to fire** ✅ (was: none) | enforced | `pre-commit run` |
