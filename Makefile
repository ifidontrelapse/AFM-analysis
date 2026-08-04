# The quality gate, in one place (M1-T10, PROJECT_RULES §6).
#
# CI calls these targets instead of re-listing the commands, so the workflow and
# a local run cannot drift apart — which is the whole point of the file. Why each
# command is what it is: docs/Development.md §4. This is only the list.
#
# Recipes are not silenced with `@`: `make check` should teach the commands, not
# hide them.

.PHONY: help check fast format lint test golden types lint-legacy
.DEFAULT_GOAL := help

# `make -j check` would run the gate's steps concurrently and interleave their
# output; the order is the point.
.NOTPARALLEL:

help:
	@echo 'check        the full gate, in the order CI runs it (~3 min)'
	@echo '  format     ruff format --check'
	@echo '  lint       ruff check, excluding src/'
	@echo '  test       pytest, golden included (~200 s)'
	@echo 'fast         pytest without the golden (~1 s) — the inner loop, not a merge gate'
	@echo 'golden       the characterization golden alone'
	@echo 'types        mypy — reports on src/, non-zero today, not part of check'
	@echo 'lint-legacy  ruff findings inside src/ — report only'

check: format lint test

format:
	uv run ruff format --check .

lint:
	uv run ruff check . --no-fix

test:
	uv run pytest -q

fast:
	uv run pytest -q -m "not slow"

golden:
	uv run pytest -q tests/characterization/test_golden.py

# Reported, never silenced, never blocking (M1-T04, M1-T07): src/ carries 109 ruff
# findings and 22 mypy errors, and blocking on them would freeze all sixteen M2
# tasks. Both exit non-zero, which is honest — that is why neither is in `check`.
# `--no-pretty` so one wording reads the same in a terminal and in a CI job
# summary; the pretty snippets only make it longer.
types:
	uv run mypy --no-pretty

lint-legacy:
	uv run ruff check src --no-fix --no-force-exclude --statistics
