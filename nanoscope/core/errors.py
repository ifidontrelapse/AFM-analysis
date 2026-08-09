"""Every error this library raises on purpose (D-15, ADR-0030).

Before this module the answer to "what does `nanoscope` do with input it cannot
use" was a table with five exception types in it — `ValueError`, `TypeError`,
`IndexError`, `LinAlgError`, `RuntimeError` — and, for four inputs, no error at
all. None of them was ours, so a caller could not distinguish "this library
rejected your argument" from "a library we call fell over".

**Every class here also inherits the builtin it replaces at the site it is
raised from.** `InvalidImageError` is a `ValueError` because that is what
`flatten_plane` raised before; `MissingFileError` is a `FileNotFoundError`
because that is what `load_microscopy_image` raised. Existing `except` clauses —
in the notebooks, which are the only callers this library has — keep catching
exactly what they caught. It is the `json.JSONDecodeError` pattern, and it is
what makes the taxonomy adoptable in one commit instead of a migration.

What each class means is the whole point of having more than one, so:

- `InvalidInputError` — *the caller* passed something no analysis can run on.
  The fix is on the caller's side.
- `AnalysisFailedError` — the input was valid and the analysis has no answer to
  give. The fix is a different parameter, a different image, or accepting that
  there is nothing there.
- `UnsupportedRequestError` — the combination requested does not exist in this
  version. The fix is a different combination.
- `DataFormatError` / `MissingFileError` — the file is unreadable or absent. The
  fix is a different file.

`AnalysisFailedError` is deliberately *not* raised for an empty result: zero
particles on a valid image is an answer, and ADR-0018 settled that. It is raised
when the caller asked for something the data cannot supply — ADR-0017's case.
"""

from __future__ import annotations


class NanoscopeError(Exception):
    """Base class for every error this library raises deliberately.

    `except NanoscopeError` is the way to catch "the library said no" without
    also catching the bugs, which stay as their own builtin types.
    """


class InvalidInputError(NanoscopeError, ValueError):
    """An argument this library cannot work with. The caller's to fix."""


class InvalidImageError(InvalidInputError):
    """An image argument that is not a usable map: wrong rank, empty, a
    non-numeric dtype, or containing values that are not finite."""


class InvalidParameterError(InvalidInputError):
    """A scalar argument outside its domain — a non-positive pixel scale, a
    negative minimum size, a polynomial order the image cannot support."""


class UnsupportedRequestError(NanoscopeError, ValueError):
    """A (modality, detector, mode) combination with no implementation.

    Distinct from `InvalidInputError` because nothing about the arguments is
    malformed: the request is well-formed and this version cannot serve it.
    """


class DataFormatError(NanoscopeError, ValueError):
    """A file whose contents this library cannot read — a missing header field,
    a header that states an impossible value, an unsupported format name."""


class MissingFileError(NanoscopeError, FileNotFoundError):
    """A file that is not there, or that the image decoder returned nothing for."""


class ProjectFormatError(NanoscopeError, ValueError):
    """A directory that does not describe a project this version can open.

    Four cases, one error, because they are one statement to the operator —
    *this is not something I can open* — and each message names which
    (M4-T01, ADR-0038): no manifest, an unparseable one, one missing a required
    field, or one declaring a format version newer than this application knows.

    A `ValueError` and not a `FileNotFoundError` even when the manifest is
    absent: the claim is about the *directory*, which exists. "There is no
    `project.json` here" is how a directory says it is not a project, not an
    accident of a path.
    """


class AnalysisFailedError(NanoscopeError, ValueError):
    """The input was valid and the analysis could not produce a result.

    Raised where the answer would otherwise be a fabricated number: Otsu finding
    no objects at all, or the size filter removing every one of them (ADR-0017).
    """
