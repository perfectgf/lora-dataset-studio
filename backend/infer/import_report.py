"""Say which of the two import failures actually happened.

Every worker in this folder opens with the same block: import the ML stack,
and on failure print one JSON line the parent shows to the user. That line said
``ML deps missing`` whatever went wrong, and the two things it collapsed have
opposite repairs:

* **Absent** — the interpreter really does not have the package. The repair is
  an install, and Setup has a button for it.
* **Present but broken** — the package is there and raises while importing
  itself or one of its transitive dependencies: a version that predates this
  Python, a torch/torchvision mismatch, a native DLL that will not load, a
  stray copy in a user site-packages directory shadowing the good one.

Told "ML deps missing", a user reinstalls what is already installed, gets
"Requirement already satisfied" for every package the installer knows about,
and learns nothing. That is not hypothetical: an ``eventlet`` left in a user
site-packages by an unrelated project — imported transitively from
``open_clip``, and too old for Python 3.12 — reported ``ML deps missing:
AttributeError: module 'ssl' has no attribute 'wrap_socket'`` 855 times while
nothing was missing at all.

So the sentence names the CASE first and then carries the child's own words
verbatim, because the exception text is the only part that says which package
and which line. `exc.name` is what makes the distinction reliable rather than a
guess at the message: CPython fills it in on ``ModuleNotFoundError`` and leaves
it empty on the failures raised from INSIDE a module that did import.
"""
from __future__ import annotations


def import_failure(exc: BaseException) -> str:
    """One sentence for a failed dependency import, in the child's own words."""
    name = getattr(exc, 'name', None)
    if isinstance(exc, ModuleNotFoundError) and name:
        return (f'ML dependency not installed: no module named {name!r} in this '
                f'interpreter — install it from Setup')
    return (f'ML dependency installed but broken: importing it raised '
            f'{type(exc).__name__}: {exc}')
