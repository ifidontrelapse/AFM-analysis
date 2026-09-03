# CURRENT TASK

**ID:** `M8-T07`
**Title:** `RemoteTrainingProvider`: protocol, client, contract tests
**Milestone:** M8 — Training module, seventh task
**Defect:** — · **ADR:** **ADR-0087** (to be written)
**Branch:** `feat/m8-training`
**Status:** **planning 2026-09-03.** Not started.

---

## Why this task is seventh

It is **M8's fourth and last exit criterion**:

> *`RemoteTrainingProvider` satisfies the same port and is covered by contract tests.*

And it is the task the whole milestone was arranged around. ADR-0080 justified writing a port
before its adapter in exactly these words: *"let the port be wrong now, cheaply, instead of in
M8-T07 when a second implementation discovers it."* Fifteen assertions have been waiting since
M8-T01 for an implementation that is **not in this process**, and the contract suite's own docstring
says why it polls `status()` rather than waiting on a listener: *"that is what a caller on the other
side of a network has to do anyway."*

**ADR-0006 named this task's risk in M0**, in its own Negative section: *"Two implementations of a
port whose second implementation has no user yet; the remote protocol risks being designed for an
imagined deployment."* That risk is real and it is not a reason to skip the task — it is the reason
to keep the protocol to **exactly what the contract forces**, and to say plainly that no worker
ships with it.

---

## What was measured before planning

**1. The obvious serialisation is silently wrong.** `dataclasses.asdict` + `json.dumps` on a
`TrainingRun`:

```
asdict + json.dumps: ok, length 501
reconstructed: dict str
round-trips equal: False
```

It does not raise. It produces 501 valid characters, and `TrainingRun(**json.loads(text))` hands
back a run whose `dataset` is a `dict` and whose `status` is a `str` — a snapshot that compares
unequal to the one that was sent, and that no `is_finished` check reads correctly. **A wire format
this project can be wrong about quietly** is the thing to write on purpose rather than reach for.

The contract already guards it: `run.dataset == dataset` and `run.config == config` are asserted on
the run `start` returns.

**2. Compressing the upload makes it bigger.** A 40-scan dataset built by M8-T02:

```
40 scans of 512x512 -> dataset on disk: 5.2 MB
tar.gz:                                 5.3 MB in 0.16 s
```

The dataset is PNGs and short text files, so gzip spends CPU to add 0.1 MB. **`tar`, not
`tar.gz`.**

**3. What the contract forces across a process boundary.** Three assertions cannot pass unless bytes
actually move, and they are the protocol's whole content:

| Assertion | What it forces |
|---|---|
| `(project_root / run.weights_path).is_file()` | the weights come **back** |
| the fake's own `(self._root / dataset.root).is_dir()` guard | the dataset goes **out** |
| `run.device is not None` | the worker says what **it** ran on |

And `not Path(run.weights_path).is_absolute()` is what makes both sides able to use the same string:
a relative path means the same thing under two different roots, which is ADR-0003's rule paying for
itself on a wire it was not written for.

---

## The decisions

**1. The protocol is four endpoints, and each one is forced by an assertion above.**

```
POST /runs              tar of the dataset + the spec and config as JSON  -> the run
GET  /runs/<id>         -> the run
POST /runs/<id>/cancel  -> accepted, always
GET  /runs/<id>/weights -> the bytes
```

Nothing else. No authentication, no listing, no resumption, no log streaming: each would be designed
for a deployment nobody has, which is the failure ADR-0006 predicted for this task by name.

**2. The wire codec is written, not derived.** `infrastructure/training/wire.py`, one module used by
both ends, because measurement 1 says the derived one is wrong in a way that does not announce
itself. Enums by value, `Device` and `EpochMetrics` and the two specs by field, and a decoder that
**reconstructs the entities** so `EpochMetrics.__post_init__` still refuses a metric this
application cannot name — which is ADR-0080 §4's guard reaching the network for free.

**3. Relative paths are the interoperability, and they already exist.** The dataset is uploaded
under the *same relative root* the client knows it by, and the weights come back to the same
relative path the worker reports. Neither side translates, because ADR-0003 already made every path
in a project relative to it, and that is what makes one string true under two roots.

**4. The weights are downloaded before `SUCCEEDED` is published.**

The contract asserts that a succeeded run's weights are a file, and a listener that hears
`SUCCEEDED` first would find them missing for as long as the transfer takes — which is exactly the
disagreement ADR-0080 §5 removed by refusing `collect_artifacts()`, arriving from the other
direction. The last status transition is the client's, not the worker's.

**5. Polling is a plain thread, and it is the one place this task departs from ADR-0043.**

`LocalTrainingProvider` drives its run with `JobRunner` because the run *is* the work. Here the work
is on another machine and what runs locally is a watcher that sleeps. `max_workers` is 2, and a
sleeping poll loop holding one of them for six hours would leave this application with one worker to
import, analyse and export with — worse than what M8-T05 already had to state. So a daemon thread,
and `status()` answers from the **last polled snapshot** rather than the network, which is what the
contract's own `POLL_S = 0.01` requires of anything that is not to be hammered.

**6. A worker that stops answering makes the run `FAILED`, with the reason.**

Not `RUNNING` for ever, and not a silent stall. It is the one status this provider decides for
itself, and it is honest: what is known locally is that the run cannot be observed. ADR-0084 §8's
rule is not contradicted — that one is about a *stored* run nobody is watching, and this one is
about a live watcher that has lost its subject.

**7. Nothing is wired into the composition root, and there is no setting for a worker's address.**

ADR-0041's rule for the eighth time, and here it is load-bearing rather than procedural: a
`training.remote_url` preference would be the imagined deployment ADR-0006 warned about, made
permanent in a settings file. The provider takes a base URL as a constructor argument; the caller
that supplies one arrives with a worker.

**8. The worker in the tests is `FakeTrainingProvider` behind `http.server`.**

Stdlib, no new dependency, and it is a **fixture rather than a product**: this task ships a client,
not a server. The test's worker root is a **different directory** from the client's project root —
which is the whole point, because with one directory the upload and the download are no-ops and the
suite would prove nothing.

---

## Scope

**In scope**

1. `infrastructure/training/wire.py` — the JSON codec both ends use
2. `infrastructure/training/remote.py` — `RemoteTrainingProvider`, on `requests` (already a
   dependency)
3. `tests/contract/test_remote_training_provider.py` — the **fifteen existing assertions,
   unchanged**, against a stub worker on a real socket with a separate root
4. Tests for what only a client can get wrong: a lost worker, a refused start, a cancel that races
5. **ADR-0087** + the ADR index

**Out of scope**

- **A worker implementation** — this task ships a client and says so; the stub is a test fixture
- **Wiring, and a stored worker address** — decision 7
- **Authentication, TLS, multi-tenancy, resumption, log streaming** — every one of them a guess
  about a deployment nobody has
- **Uploading the project** — a worker trains on a dataset, which is all the port gives it

---

## Definition of done

- [ ] `RemoteTrainingProvider` passes `TrainingProviderContract` **unchanged** — M8's fourth criterion
- [ ] The dataset reaches the worker and the weights come back, across two separate roots
- [ ] A `TrainingRun` survives the wire intact, `dataset` and `config` compared field for field
- [ ] A worker that stops answering ends the run rather than leaving it running
- [ ] ADR-0087 + the ADR index
- [ ] `make check` green, golden byte-identical
- [ ] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`, `Roadmap.md`
- [ ] Commit: `M8-T07: the same port, on the other side of a socket`
