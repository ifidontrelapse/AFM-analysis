# Notebooks

Experiments. **Not interfaces** (PROJECT_RULES §7).

| Notebook | What it is |
|---|---|
| `afm_gold_nanoparticles.ipynb` | End-to-end pipeline exploration on real gold-nanoparticle scans |
| `preprocessing.ipynb` | Flattening and substrate-estimation exploration |

Three rules, all of them enforced rather than requested:

1. **No production code path may import or depend on a notebook.** If something here is
   worth keeping, it belongs in the library with a test.
2. **Outputs are stripped on commit.** The `nbstripout` hook does it (M1-T07); together the
   two notebooks were 8.3 MB of embedded PNGs and are now 32 KB of code. Install the hooks
   in a fresh clone with `uv run pre-commit install`, or you will re-commit the outputs.
3. **They are not part of the quality gate.** Ruff excludes `*.ipynb`, and nothing runs them
   in CI. They may be broken against the current `src/` — see below.

## State

Neither notebook was executed as part of M1-T09, so whether they still run against today's
`src/` is unverified. They predate the `AFMRawData` return-convention change (`e8caf25`,
defect **D-02**), which silently broke `preprocess_batch.py` in exactly that way, so treat a
failure here as expected rather than surprising.

They also read from `data/`, which is git-ignored — 628 local scans that are not in the
repository. A fresh clone cannot run them at all.

## Recovering the stripped outputs

They are not lost; every output is still in git history:

```bash
git show 09fd5f4:afm_gold_nanoparticles.ipynb > /tmp/with_outputs.ipynb
```

(Any commit before M1-T09 works — the files were at the repository root back then.)
