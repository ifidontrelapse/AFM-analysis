"""The project directory format, as the half of it a program can execute (M4-T01).

The contract itself is `docs/ProjectFormat.md`; the decisions behind it are
ADR-0038, on top of ADR-0003's layout. This module is deliberately thin — the
names of the directories, the manifest, and the version check — because a
specification nothing executes drifts from the code within two tasks, and the
operator's data is on the other side of this one.

What is *not* here: creating a project (M4-T04, which needs the repository under
it), the SQLite schema and its migrations (M4-T02), and the integrity check that
reconciles the index against the filesystem (M4-T03). Each needs a database this
task does not have.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from nanoscope.core.errors import ProjectFormatError

#: The layout this application writes and can read. One integer, bumped when a
#: reader that does not know about the change would misread a project — not for
#: additions a reader can ignore (ADR-0038).
FORMAT_VERSION = 1

#: The file that makes a directory a project. Read before anything else, and
#: readable when `database.sqlite` is corrupt — which is the point of it.
MANIFEST_NAME = "project.json"

DATABASE_NAME = "database.sqlite"

#: Created on `CreateProject` (M4-T04) and expected by every reader. `cache/` is
#: the only one that may be deleted behind the application's back.
DIRECTORIES = (
    "images",
    "annotations",
    "results",
    "exports",
    "models",
    "logs",
    "cache",
)

CACHE_DIRECTORY = "cache"


@dataclass(frozen=True)
class ProjectManifest:
    """What `project.json` states about the directory it sits in.

    `format_version` describes the *directory*. The database carries its own
    `schema_version` (M4-T02) because the two change for different reasons, and
    because this one has to be readable without opening the database.
    """

    name: str
    format_version: int
    created_utc: str

    #: Fields this version does not know about, carried through unchanged.
    #: Without this, an older application rewriting a newer project's manifest
    #: would silently delete what it did not recognise — and an additive change
    #: is only additive if it survives the round trip (`ProjectFormat.md` §2).
    extra: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                **self.extra,
                "name": self.name,
                "format_version": self.format_version,
                "created_utc": self.created_utc,
            },
            indent=2,
        )


def new_manifest(name: str) -> ProjectManifest:
    """A manifest for a project being created now, at the current version."""
    return ProjectManifest(
        name=name,
        format_version=FORMAT_VERSION,
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def manifest_path(project_dir: Path | str) -> Path:
    return Path(project_dir) / MANIFEST_NAME


def write_manifest(project_dir: Path | str, manifest: ProjectManifest) -> Path:
    path = manifest_path(project_dir)
    path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    return path


def read_manifest(project_dir: Path | str) -> ProjectManifest:
    """Read `project.json`, or say why this directory is not a project.

    Raises:
        ProjectFormatError: the manifest is absent, is not JSON, is not an
            object, or is missing a required field. Every message names the
            path, because the operator's next move is to look at it.
    """
    path = manifest_path(project_dir)

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProjectFormatError(
            f"not a project directory: no {MANIFEST_NAME} in {Path(project_dir)}"
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProjectFormatError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ProjectFormatError(f"{path} must contain a JSON object, not {type(data).__name__}")

    missing = [key for key in ("name", "format_version", "created_utc") if key not in data]
    if missing:
        raise ProjectFormatError(f"{path} is missing required field(s): {', '.join(missing)}")

    version = data["format_version"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise ProjectFormatError(
            f"{path}: format_version must be an integer, got {version!r}",
        )

    known = {"name", "format_version", "created_utc"}
    return ProjectManifest(
        name=str(data["name"]),
        format_version=version,
        created_utc=str(data["created_utc"]),
        extra={k: v for k, v in data.items() if k not in known},
    )


def check_compatible(manifest: ProjectManifest) -> None:
    """Refuse a project this version cannot honestly open.

    Newer is refused, older is accepted: a forward migration cannot be written
    by the past, and opening a project written by a later version would mean
    guessing what its fields mean (ADR-0003's compliance rule, ADR-0038's
    matrix). Migrating an older project is M4-T02's job; the rule that it *may*
    be opened is this one's.
    """
    if manifest.format_version > FORMAT_VERSION:
        raise ProjectFormatError(
            f"project format version {manifest.format_version} is newer than this "
            f"application understands (version {FORMAT_VERSION}); upgrade nanoscope to open it"
        )


def open_manifest(project_dir: Path | str) -> ProjectManifest:
    """`read_manifest` + `check_compatible` — what every caller actually wants."""
    manifest = read_manifest(project_dir)
    check_compatible(manifest)
    return manifest
