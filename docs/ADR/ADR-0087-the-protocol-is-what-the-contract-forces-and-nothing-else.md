# ADR-0087 — The remote protocol is what the contract forces, and nothing else

- **Status:** Accepted
- **Date:** 2026-09-03
- **Deciders:** operator + agent (M8-T07)
- **Affects:** `infrastructure/training/remote.py`, `infrastructure/training/wire.py`,
  `tests/contract/` · M8

## Context

This is **M8's fourth and last exit criterion** — *`RemoteTrainingProvider` satisfies the same port
and is covered by contract tests* — and the moment ADR-0080 §1 was written for:

> *"Let the port be wrong now, cheaply, instead of in M8-T07 when a second implementation discovers
> it."*

Fifteen assertions have been waiting since M8-T01 for an implementation that is **not in this
process**. The contract suite's own docstring says why it polls `status()` instead of waiting on a
listener: *"that is what a caller on the other side of a network has to do anyway."*

**ADR-0006 named this task's failure mode in M0**, in its own Negative section: *"Two
implementations of a port whose second implementation has no user yet; the remote protocol risks
being designed for an imagined deployment."* That is a real risk and it is not a reason to skip the
task — it is the reason to let the **contract** decide the protocol's size.

Three things were measured before anything was written.

**1. The obvious serialisation is silently wrong.** `dataclasses.asdict` + `json.dumps` on a
`TrainingRun`:

```
asdict + json.dumps: ok, length 501
reconstructed: dict str
round-trips equal: False
```

It does not raise. It produces 501 valid characters, and `TrainingRun(**json.loads(text))` hands
back a run whose `dataset` is a `dict` and whose `config` and `metrics` are dicts too. Worse than a
loud failure: `TrainingStatus` is a `StrEnum`, so `is_finished` **keeps working** on the string —
which is exactly what would let such a run travel a long way before `run.metrics[0].epoch` raised an
`AttributeError` somewhere unrelated.

**2. Compressing the upload makes it bigger.** A 40-scan dataset built by M8-T02 is **5.2 MB** on
disk; `tar.gz` of it is **5.3 MB** in 0.16 s. It is PNGs and short text, so gzip spends CPU to add
bytes.

**3. What the contract forces across a process boundary.** Three assertions cannot pass unless bytes
move, and they are the protocol's whole content:

| Assertion | What it forces |
|---|---|
| `(project_root / run.weights_path).is_file()` | the weights come **back** |
| a worker that cannot read the dataset | it goes **out** |
| `run.device is not None` | the worker says what **it** ran on |

## Decision

### 1. Four endpoints, each one forced by an assertion above

```
POST /runs              a tar of the dataset, plus the spec and config as JSON
GET  /runs/<id>         the snapshot
POST /runs/<id>/cancel  accepted, always
GET  /runs/<id>/weights the bytes a succeeded run produced
```

Nothing else. No authentication, no listing, no resumption, no log streaming, no versioning: every
one of them would be designed for a deployment nobody has, which is the failure ADR-0006 predicted
for this task by name. When there is a worker, the envelope grows a version and this paragraph is
what records that its absence was a decision.

`tar`, not `tar.gz`: measurement 2.

### 2. The wire codec is written, not derived

`infrastructure/training/wire.py`, one module used by both ends, because measurement 1 says the
derived one is wrong in a way that does not announce itself — and a codec written twice is two
codecs that agree until they do not.

**Decoding reconstructs the entities**, so their constructors run: an `EpochMetrics` naming a metric
this application does not know is refused at the boundary rather than becoming a chart. ADR-0080 §4
declared that vocabulary once and its guard now reaches the network without being written a second
time.

### 3. Relative paths are the interoperability, and they already existed

The dataset is uploaded under the **same relative root** the client knows it by, and the weights come
back to the **same relative path** the worker reports. Neither side translates.

ADR-0003 made every path in a project relative to the project so the project would survive being
moved. That decision, made for a filesystem, is what lets one string be true under two different
roots on two different machines — and it is the reason this protocol needs no path mapping at all.

### 4. The weights are downloaded before `SUCCEEDED` is published

The contract asserts that a succeeded run's weights are a file. A listener that heard `SUCCEEDED`
first would find them missing for as long as the transfer takes — the disagreement ADR-0080 §5
removed by refusing `collect_artifacts()`, arriving from the other direction. So the **last status
transition is the client's**, not the worker's, and M8-T04's `start_training` can still register a
model the moment it hears the news.

### 5. Polling is a plain thread, and `status()` does not touch the network

The one place this milestone departs from ADR-0043. `LocalTrainingProvider` drives its run with
`JobRunner` because there the run *is* the work; here the work is on another machine and what runs
locally is a watcher that sleeps. `max_workers` is 2, and a sleeping loop holding one of them for
six hours would leave this application with one worker to import, analyse and export with — a worse
version of the cost M8-T05 already had to state.

`status()` answers from the last snapshot the watcher fetched. The contract polls it every 10 ms;
a provider that turned each call into a request would be one nobody could poll, and a snapshot is
precisely the thing that can be answered from memory (ADR-0080 §3).

### 6. A worker that stops answering makes the run `FAILED`, with the reason

Not `RUNNING` for ever, and not a silent stall. A watcher that has lost its subject knows one thing
— that it cannot observe the run — and saying so is the only honest terminal state it has.

**ADR-0084 §8 is not contradicted.** That rule is about a *stored* run nobody is watching, whose
status is what was true when a process died. This is a live watcher reporting **its own** failure,
which is an observation rather than a substitution.

### 7. Nothing is wired, and there is no setting for a worker's address

ADR-0041 for the eighth time, and here it is load-bearing rather than procedural: a
`training.remote_url` preference would be exactly the imagined deployment ADR-0006 warned about,
made permanent in a settings file. The provider takes a base URL as a constructor argument; the
caller that supplies one arrives with a worker.

### 8. The worker in the tests is the fake behind `http.server`, on a different root

A **fixture, not a product**: this task ships a client, and no worker ships with this project.
Stdlib, so no dependency, and the training behind it is `FakeTrainingProvider` — the thing the
contract has been checked against since M8-T01 — so the only new subject is the client.

**Its root is a different directory from the client's project**, which is the whole design of the
test. With one directory the upload and the download are no-ops, the contract still passes, and it
proves nothing.

## Consequences

**Positive**

- **M8's fourth exit criterion is met, and the milestone's central claim is settled**: fifteen
  assertions written against a fake, satisfied by a real trainer in M8-T03 and now by a client that
  can see none of the run it describes — **not one of them edited**. The port survived a process
  boundary, which is what M8-T01 spent a task proving in advance.
- ADR-0003's relative paths turned out to be the interoperability layer for a wire they were not
  written for. That is the second time a decision made for one reason paid somewhere else in this
  milestone.
- The metric vocabulary guards the network for free. A worker that reports something `METRIC_BLOCKS`
  does not name fails at the epoch it arrives in.
- A codec exists for `TrainingRun`, which M8-T04 needed in SQL and wrote by hand; the two are still
  separate, and that is now a visible duplication rather than an invisible one.

**Negative**

- **The protocol is provisional and no worker implements it.** It is four endpoints chosen by a test
  suite, and the first real worker may want different ones. Stated here rather than discovered
  later, and kept small so that changing it is cheap.
- **Two representations of a `TrainingRun` now exist** — this codec and M8-T04's SQL columns. They
  cannot disagree today because the contract compares field for field on one side and a round-trip
  test on the other, but a third would be one too many.
- **`status()` can be stale by up to `poll_seconds`.** That is what makes it cheap, and a caller
  that needs the newest answer is a caller that should be listening.
- **A watcher thread per run**, outside `JobRunner` and outside anything the job-status widget shows.
  It sleeps, and it is a departure from the one thread policy ADR-0043 settled.
- **The upload is one request with the whole dataset in memory.** 5.2 MB measured for forty scans;
  a project of four hundred would want streaming, and this does not do it.

**Neutral**

- `requests` was already a dependency, and `http.server` is stdlib: nothing was added.
- The remote contract suite is **6 s and in the gate**, needing neither torch nor a network.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Decline the task, as M4 declined three | The criterion is M8's, the contract was written in M8-T01 *for* this, and there is a real discovery here — the port had never crossed a process |
| Design a full deployment protocol (auth, listing, resumption) | The imagined deployment ADR-0006 warned about, at four times the size |
| `dataclasses.asdict` for the wire | Measured: 501 valid characters and a run that compares unequal, with `is_finished` still working — the quietest possible wrong |
| Put the codec in `core` | `core` would then know it is on a wire; the encoding is infrastructure's, and both ends of this protocol are infrastructure |
| Reuse M8-T04's SQL encoding | It is storage's, shaped by columns, and `infrastructure/training/` may not import `infrastructure/storage/` |
| Absolute paths in the protocol | Two machines, two roots; ADR-0003 already solved this and the solution is free here |
| Publish `SUCCEEDED` and download the weights after | A run that claims weights it does not have — ADR-0080 §5's disagreement, on a schedule |
| `status()` fetches from the worker | The contract polls it every 10 ms; a network call there is a provider nobody may poll |
| Drive the watcher on `JobRunner` | A sleeping loop holding one of two workers for six hours |
| Leave a lost worker `RUNNING` | A run that never ends and a UI that never stops waiting |
| Ship a worker as well | Twice the surface, and the second half would be the guess: a worker is a deployment, and there is none |
| Test with one directory for both sides | The transfers become no-ops and the suite proves nothing |

## Compliance

- `tests/contract/test_remote_training_provider.py` — **the fifteen contract assertions, unchanged**,
  in two classes (with a held-out set and without), against a stub worker on a real socket with its
  own root; plus ten assertions the contract cannot make: the dataset lands on the other root, the
  weights come back and match byte for byte across two roots, a run survives the wire field for
  field, a worker that vanishes ends the run with a reason, a missing dataset is refused before
  anything is uploaded, `status()` answers after the worker is gone, an id from another client is
  refused, a cancel arriving before `start` returns is remembered, and cancelling into a void does
  not raise.
- `tests/unit/test_training_wire.py` — the round trip, the measured failure of the derived version
  kept as an assertion, and five refusals at the edge: an unknown metric, an unknown status, a
  missing field, a payload that is not an object, and an epoch numbered from zero.
- `tests/unit/test_import_graph.py` — the two new modules are covered by the parametrised layer and
  Qt guards without being added to a list.
- The golden is byte-identical. Nothing in this task is on a numerical path.

## References

- ADR-0006 — two implementations, chosen in M0, and the Negative clause this task was measured against
- ADR-0080 — the port, the snapshot, the vocabulary, and *let the port be wrong now*
- ADR-0003 — every path relative to the project; §3 is where that pays on a wire
- ADR-0043 — the job runner and the one thread policy §5 departs from, with the reason
- ADR-0082 / ADR-0084 / ADR-0085 — the local provider, the record, and the window, all of which this
  provider must be indistinguishable from
