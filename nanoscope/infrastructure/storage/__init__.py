"""Reading images off a disk, and the shape of the directory they live in.

The first adapter in the project: everything here takes a path and opens it.
M2-T08 puts an `ImageLoader` port in front of the loaders; until then they are
called directly, exactly as they were from `src/afm_io.py`.

`project_format` joined them in M4-T01 — the executable half of the project
directory contract (`docs/ProjectFormat.md`, ADR-0038) — and `database` in
M4-T02, which owns the one file in that directory that is not a document, and
`project_repository` in M4-T03 — the first thing here that reads and writes
rows rather than files.
"""

from nanoscope.infrastructure.storage.database import (
    MIGRATIONS,
    SCHEMA_VERSION,
    connect,
    migrate,
    open_database,
    schema_version,
)
from nanoscope.infrastructure.storage.loaders import (
    load_afm,
    load_microscopy_image,
)
from nanoscope.infrastructure.storage.project_format import (
    DATABASE_NAME,
    DIRECTORIES,
    FORMAT_VERSION,
    MANIFEST_NAME,
    ProjectManifest,
    check_compatible,
    new_manifest,
    open_manifest,
    read_manifest,
    write_manifest,
)
from nanoscope.infrastructure.storage.project_repository import (
    SqliteProjectRepository,
    sha256_of,
)

__all__ = [
    "DATABASE_NAME",
    "DIRECTORIES",
    "FORMAT_VERSION",
    "MANIFEST_NAME",
    "MIGRATIONS",
    "SCHEMA_VERSION",
    "ProjectManifest",
    "SqliteProjectRepository",
    "check_compatible",
    "connect",
    "load_afm",
    "load_microscopy_image",
    "migrate",
    "new_manifest",
    "open_database",
    "open_manifest",
    "read_manifest",
    "schema_version",
    "sha256_of",
    "write_manifest",
]
