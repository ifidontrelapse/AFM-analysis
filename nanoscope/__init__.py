"""nanoscope — desktop analysis of nanoparticle microscopy (AFM, SEM, TEM).

Clean Architecture, four rings, one composition root; the dependency rule points
inward and each subpackage's docstring states its half of it (ADR-0001, ADR-0011,
`docs/Architecture.md` §3).

Empty for now by design: M2-T01 created the layers, and M2-T02…T08 fill them by
moving `src/` in, one module per commit, each proving zero drift against the
characterization golden. M3 corrected its numerics, M4 built the application
layer around it, and M5 is putting a window on top.
"""

__all__ = ["__version__"]


def __getattr__(name: str) -> str:
    """`nanoscope.__version__`, read from the installed distribution — **lazily**.

    Read at import time it cost eleven modules (`importlib.metadata` pulls in
    `zipfile` and `email`), which the import-weight guard caught immediately:
    importing the domain must stay cheap, and M2-T09 wrote a test that says so
    in numbers. PEP 562 module `__getattr__` moves the cost to whoever asks —
    which is the entry point, and it does not care.

    The version itself lives in `pyproject.toml` and nowhere else.
    """
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("nanoscope")
        except PackageNotFoundError:  # pragma: no cover — a checkout with no install
            return "0.0.0+unknown"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
