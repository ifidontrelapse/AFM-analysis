"""`nanoscope` — the entry point, before there is a window (M5-T01, ADR-0052).

The console script works **headless today**. PySide6 is not yet a dependency of
this project, and an entry point that only works once a window exists cannot be
run in CI and cannot be run by an operator debugging a project that will not
open. Opening a project and saying what is in it needs no display; showing it
does, and that is the next task's branch to fill.

**Our errors are messages; anything else keeps its traceback.** ADR-0030 built
that distinction so a surface like this one could use it: a `NanoscopeError` is
the library saying no about the operator's data, and printing a stack trace at
them for a directory that is not a project is an application blaming its user
for its own diagnostics.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from nanoscope import __version__
from nanoscope.app.container import Nanoscope
from nanoscope.app.logging import configure_logging
from nanoscope.core.entities.project import IntegrityReport, OpenedProject
from nanoscope.core.errors import NanoscopeError

#: Exit codes. `2` is argparse's for a usage error, so ours start above it.
OK = 0
REFUSED = 1
UNAVAILABLE = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nanoscope",
        description="Desktop analysis of nanoparticle microscopy images (AFM, SEM, TEM).",
    )
    parser.add_argument("--version", action="version", version=f"nanoscope {__version__}")
    parser.add_argument(
        "--project",
        metavar="PATH",
        help="a project directory to open. Printed and closed again, until M5-T02's window exists",
    )
    parser.add_argument(
        "--devices",
        action="store_true",
        help="list the devices inference could run on, and which one would be chosen",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="launch the graphical interface (M5-T02; PySide6 is not installed yet)",
    )
    parser.add_argument("--debug", action="store_true", help="log at DEBUG, and echo to stderr")
    parser.add_argument(
        "--log-file", metavar="PATH", help="where to write the application log (default: XDG state)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the application. Returns the process exit code.

    Returns rather than exits, so a test can call it — and so the `console_scripts`
    wrapper is the only thing that ever calls `sys.exit`.
    """
    args = build_parser().parse_args(argv)

    level = logging.DEBUG if args.debug else logging.INFO
    log_file = configure_logging(level=level, path=args.log_file, console=args.debug)

    with Nanoscope() as app:
        try:
            if args.devices:
                _print_devices(app)
            if args.project:
                _print_project(app.open(args.project))
            if args.gui:
                return _launch_gui()
        except NanoscopeError as refusal:
            #: A message, not a traceback: this is the library saying no about
            #: the operator's data, which is exactly the distinction ADR-0030
            #: was built to make available here.
            print(f"nanoscope: {refusal}", file=sys.stderr)
            logging.getLogger(__name__).error("refused: %s", refusal)
            return REFUSED

        if not (args.devices or args.project or args.gui):
            print(f"nanoscope {__version__}. Nothing asked for; try --help.")
            print(f"Logging to {log_file}")

    return OK


def _print_devices(app: Nanoscope) -> None:
    """What hardware exists, and what would be used — the question a support
    request starts with, and one an operator can now answer themselves."""
    for device in app.devices.available():
        print(f"  {device}")
    print(f"selected: {app.select_device()}")


def _print_project(opened: OpenedProject) -> None:
    """What is in the project — and what is wrong with it.

    The integrity report is *shown*, which is the obligation ADR-0040 ended on:
    it reports and changes nothing, and a report nobody reads is a report that
    did nothing. This is the first caller in the project that can read it.
    """
    print(f"{opened.name}: {len(opened.images)} image(s)")
    for image in opened.images:
        scale = f"{image.pixel_size_nm} nm/px" if image.pixel_size_nm else "scale unknown"
        print(f"  {image.display_name} ({image.modality}, {scale})")
    _print_integrity(opened.integrity)


def _print_integrity(report: IntegrityReport) -> None:
    if report.is_clean:
        print("the index and the files agree")
        return

    for image in report.missing_files:
        print(f"  missing: {image.relative_path} (its row is kept — the file may be elsewhere)")
    for path in report.untracked_files:
        print(f"  untracked: {path} (no row claims it; nothing was imported)")


def _launch_gui() -> int:
    """The branch M5-T02 fills.

    A sentence rather than an `ImportError` traceback: asking for a window this
    version cannot open is a well-formed request with no implementation, which
    is the one thing this application is careful to distinguish (ADR-0030).
    """
    print(
        "nanoscope: the graphical interface arrives in M5-T02; "
        "PySide6 is not a dependency of this version yet.",
        file=sys.stderr,
    )
    return UNAVAILABLE


if __name__ == "__main__":  # pragma: no cover — exercised by `python -m nanoscope.app.main`
    sys.exit(main())
