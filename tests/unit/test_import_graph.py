"""The dependency rule, enforced by a machine instead of by review (M2-T09).

Two things are checked, and they fail for different reasons:

1. **Direction.** `core` must not import `application`, `infrastructure` or `gui`.
   This is a static check over the source, so it catches an edge even when the
   module that would prove it is never executed.
2. **Weight.** Importing the domain must not drag in matplotlib, torch or Qt. This
   is dynamic — it looks at `sys.modules` — because the cost is real only when the
   import actually happens, and a function-local `import torch` is fine while a
   module-level one is not.

Both were true by accident before this file existed. Now they are true on purpose.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CORE = ROOT / "nanoscope" / "core"

# Anything the domain must never reach for. `application` is included: `core`
# defines the ports, `application` consumes them, and the arrow points inward.
FORBIDDEN_FOR_CORE = ("nanoscope.application", "nanoscope.infrastructure", "nanoscope.gui", "src")

# Packages whose presence in `sys.modules` means the domain got expensive.
# `pandas` is on the list because M2-T09 pushed it behind `TYPE_CHECKING`; if it
# comes back at run time, that regression should be loud rather than gradual.
HEAVY = (
    "torch",
    "ultralytics",
    "sam2",
    "patched_yolo_infer",
    "matplotlib",
    "PySide6",
    "cv2",
    "pandas",
)


def _imported_modules(source: pathlib.Path) -> set[str]:
    """Every module named by an `import`/`from` statement, at any nesting depth."""
    tree = ast.parse(source.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


CORE_MODULES = sorted(CORE.rglob("*.py"))


def test_the_core_package_is_not_empty() -> None:
    """Guards the two tests below: a glob that matches nothing passes vacuously."""
    assert len(CORE_MODULES) >= 10, CORE_MODULES


@pytest.mark.parametrize("path", CORE_MODULES, ids=lambda p: str(p.relative_to(CORE)))
def test_core_imports_nothing_from_an_outer_layer(path: pathlib.Path) -> None:
    offenders = {
        name
        for name in _imported_modules(path)
        for bad in FORBIDDEN_FOR_CORE
        if name == bad or name.startswith(bad + ".")
    }
    assert not offenders, (
        f"{path.relative_to(ROOT)} imports {sorted(offenders)}. The dependency rule points "
        "inward: core declares ports, outer layers implement them (ADR-0001)."
    )


@pytest.mark.parametrize(
    "module", ["nanoscope.core.entities", "nanoscope.core.ports", "nanoscope.core.science"]
)
def test_importing_the_domain_stays_cheap(module: str) -> None:
    """Run in a subprocess: `sys.modules` in this process is already polluted by pytest."""
    code = f"import sys; import {module}; print(len(sys.modules)); print(' '.join(sorted(sys.modules)))"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT
    )
    count_line, names_line = out.stdout.splitlines()
    loaded = set(names_line.split())

    heavy = sorted(h for h in HEAVY if h in loaded)
    assert not heavy, f"importing {module} loaded {heavy}"

    # ~35 interpreter builtins + ~141 for numpy, which the domain is allowed to
    # use (Architecture §3: "pure Python + NumPy"). The bound is deliberately not
    # the "< 100" in the M2 exit criteria — that number predates measuring numpy,
    # and is unreachable for any module holding an ndarray annotation. What it was
    # protecting is the `heavy` assertion above; this one catches a new dependency
    # sneaking in under the numpy noise floor.
    assert int(count_line) < 250, f"importing {module} loaded {count_line} modules"


def test_the_weight_check_can_fail() -> None:
    """A guard that cannot fail is decoration — prove this one detects matplotlib."""
    code = "import sys; import matplotlib; print(' '.join(sorted(sys.modules)))"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT
    )
    assert "matplotlib" in set(out.stdout.split())
