"""nanoscope — desktop analysis of nanoparticle microscopy (AFM, SEM, TEM).

Clean Architecture, four rings, one composition root; the dependency rule points
inward and each subpackage's docstring states its half of it (ADR-0001, ADR-0011,
`docs/Architecture.md` §3).

Empty for now by design: M2-T01 created the layers, and M2-T02…T08 fill them by
moving `src/` in, one module per commit, each proving zero drift against the
characterization golden.
"""
