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
PACKAGE = ROOT / "nanoscope"
GUI = PACKAGE / "gui"

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


# ── Qt stays in the GUI (M4-T15) ──────────────────────────────────────────────
#
# M4's sixth exit criterion says "no Qt imported anywhere", which is literally
# true today because `gui/` is empty — and will stop being true in M5, when
# `gui/` imports PySide6 because that is its job. The durable form of the
# criterion is the one written here: **nothing outside `gui/` imports it.**
#
# Added while the check is trivially true on purpose: a guard added after the
# first violation is a guard that has already failed once.

QT = ("PySide6", "PyQt5", "PyQt6", "shiboken6")

NON_GUI_MODULES = sorted(path for path in PACKAGE.rglob("*.py") if not path.is_relative_to(GUI))


def test_the_package_is_not_empty() -> None:
    """A glob that matches nothing passes vacuously."""
    assert len(NON_GUI_MODULES) >= 30, NON_GUI_MODULES


@pytest.mark.parametrize("path", NON_GUI_MODULES, ids=lambda p: str(p.relative_to(PACKAGE)))
def test_qt_stays_inside_the_gui(path: pathlib.Path) -> None:
    offenders = {
        name
        for name in _imported_modules(path)
        for bad in QT
        if name == bad or name.startswith(bad + ".")
    }
    assert not offenders, (
        f"{path.relative_to(ROOT)} imports {sorted(offenders)}. Qt belongs to `gui/`: the "
        "application layer must be runnable headless, which is M4's exit criterion and M5's "
        "test strategy (Architecture §3.2)."
    )


# ── The GUI reaches inward, and only so far (M5-T06) ─────────────────────────
#
# Two rules, both from Architecture §3.2's table, both previously true by habit:
#
# 1. **`gui` may not import `core.science` or `infrastructure`.** M5's fifth exit
#    criterion asks for a check that proves it; this is that check, in the same
#    form as every other rule this project enforces — a test, not a review note.
# 2. **A panel may not import the composition root.** M5-T06 gave the panels a
#    viewmodel; without this, the next one quietly takes the container again and
#    the layer becomes decoration.

FORBIDDEN_FOR_GUI = ("nanoscope.core.science", "nanoscope.infrastructure", "torch")

GUI_MODULES = sorted(GUI.rglob("*.py"))
PANEL_MODULES = sorted((GUI / "panels").rglob("*.py"))


def test_the_gui_package_is_not_empty() -> None:
    """A glob that matches nothing passes vacuously."""
    assert len(GUI_MODULES) >= 5 and len(PANEL_MODULES) >= 3, (GUI_MODULES, PANEL_MODULES)


@pytest.mark.parametrize("path", GUI_MODULES, ids=lambda p: str(p.relative_to(GUI)))
def test_the_gui_does_not_reach_past_the_application_layer(path: pathlib.Path) -> None:
    offenders = {
        name
        for name in _imported_modules(path)
        for bad in FORBIDDEN_FOR_GUI
        if name == bad or name.startswith(bad + ".")
    }
    assert not offenders, (
        f"{path.relative_to(ROOT)} imports {sorted(offenders)}. A widget may format and lay out; "
        "deciding what to compute, or how a value becomes a colour, belongs to `application` "
        "(Architecture §3.2, ADR-0056)."
    )


@pytest.mark.parametrize("path", PANEL_MODULES, ids=lambda p: str(p.relative_to(GUI)))
def test_no_panel_holds_the_composition_root(path: pathlib.Path) -> None:
    #: `== "nanoscope.app"` and not `startswith`, which also matches
    #: `nanoscope.application` — the layer the panels are *supposed* to use.
    offenders = {
        name
        for name in _imported_modules(path)
        if name == "nanoscope.app" or name.startswith("nanoscope.app.")
    }
    assert not offenders, (
        f"{path.relative_to(ROOT)} imports {sorted(offenders)}. Panels take the session "
        "viewmodel; the container is constructed in one place and held by one object "
        "(ADR-0057)."
    )


@pytest.mark.parametrize(
    "module",
    ["nanoscope.application.use_cases", "nanoscope.application.jobs", "nanoscope.app.logging"],
)
def test_the_application_layer_runs_without_qt(module: str) -> None:
    """Statically clean is not enough: a transitive import would still pull Qt
    into a headless process. Run in a subprocess, like the weight check above."""
    code = f"import sys; import {module}; print(' '.join(sorted(sys.modules)))"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT
    )
    loaded = set(out.stdout.split())

    assert not [q for q in QT if q in loaded], f"importing {module} loaded Qt"
