"""Use cases, DTOs, the capability matrix, job orchestration.

Depends on `core` and on `core.ports` — never on a concrete adapter, and never on
`gui`.

`capabilities` is the first thing here (M2-T10): the one executable copy of which
(modality, detector, mode) combinations exist, consulted before any inference runs.
"""
