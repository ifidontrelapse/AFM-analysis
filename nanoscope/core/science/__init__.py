"""The preserved scientific core — today's `src/`, moved rather than rewritten.

Everything here arrives verbatim under the characterization golden (M2-T03…T07):
same algorithms, same constants, same order of operations, whitespace excepted.
The defects the audit found travel with the code and are fixed in M3, each with a
declared numerical delta — never quietly, during a move.

That legacy status is declared once in `pyproject.toml` rather than repeated as
`type: ignore` and `noqa` down every module: mypy runs at default strictness here
instead of the strict `nanoscope.*` settings, and ruff's per-file ignores name the
three things M2-T11, M2-T12 and M3 remove. Both shrink to nothing as those tasks
land; neither hides anything, and the counts stay in CI's run summary.
"""
