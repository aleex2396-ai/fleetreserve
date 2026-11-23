"""Opens of made-up sealed files counted by identity on disk, whatever path names them.

`watch(*folders)` records, while it is active, every `open`, `os.listdir` and `os.scandir` event
(Python's audit events, `sys.addaudithook`) whose target is a file or folder at or under
`folders`, matched by device and inode (`os.stat`, links followed), so a spelling of the path in
other letter case or Unicode form, a symbolic link and a hard link are all caught. The identities
are taken when the watch starts. An audit hook cannot be removed, so one hook is added once per
process; it records nothing while no watch is active.

A name opened from a folder's descriptor (`os.open(name, flags, dir_fd=...)`, as
`fleetreserve.plain_read` opens every name) raises an `open` event that names no folder, so the
hook cannot find what it names. While a watch is active `os.open` is therefore wrapped: an open
with `dir_fd` of a relative name is matched by the identity of the descriptor it gives
(`os.fstat`), recorded as ("open", "<name> (from a folder descriptor)") unless the hook already
matched the same identity from the current folder. `descriptor_opens(listener)` gives the same
wrapper to another counter (`tests/wp12/opened_from.py`). The wrapper is put in place when the
first listener starts, calling the `os.open` in place then, and taken out when the last one ends,
if it is still `os.open` then; with no listener it only calls that `os.open`.

A wrapper can stay in place after its watch: a function put in place inside the watch (as
`opens_of` in tests/wp14 does) is `os.open` at the watch's end, and the patch that put it there
puts the wrapper back afterwards. A later function that calls the `os.open` it took (that
wrapper), wrapped in turn by a new watch, makes each open go wrapper, function, wrapper: the
wrapper entered a second time within one call calls the `os.open` taken at import, so the chain
ends there instead of calling the function again without end (`RecursionError`).
"""

import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

EVENTS = ("open", "os.listdir", "os.scandir")
_ACTIVE: list = []
_ADDED: list = []
_LISTENERS: list = []
_REAL_OPEN: list = []
_IMPORT_OPEN = os.open
_CALLING = threading.local()


def identity(path) -> tuple[int, int] | None:
    try:
        found = os.stat(path)
    except (OSError, TypeError, ValueError):
        return None
    return found.st_dev, found.st_ino


def _hook(event, args) -> None:
    if not _ACTIVE or event not in EVENTS or not args:
        return
    target = args[0]
    if target is None or isinstance(target, int):
        return
    try:
        name = os.fspath(target)
    except TypeError:
        return
    found = identity(name)
    if found is None:
        return
    for watched, seen in _ACTIVE:
        if found in watched:
            seen.append((event, os.fsdecode(name)))


def _open_watched(path, flags, mode=0o777, *, dir_fd=None):
    """`os.open`, telling each listener of an open from a folder's descriptor."""
    if getattr(_CALLING, "inside", False):
        # entered again within one call, through a function this wrapper calls
        return _IMPORT_OPEN(path, flags, mode, dir_fd=dir_fd)
    _CALLING.inside = True
    try:
        descriptor = _REAL_OPEN[0](path, flags, mode, dir_fd=dir_fd)
    finally:
        _CALLING.inside = False
    if not _LISTENERS or dir_fd is None:
        return descriptor
    try:
        name = os.fsdecode(os.fspath(path))
        found = os.fstat(descriptor)
    except (OSError, TypeError, ValueError):
        return descriptor
    if os.path.isabs(name):
        return descriptor  # `dir_fd` is not used; the hook matched the path itself
    mark = (found.st_dev, found.st_ino)
    if identity(name) == mark:
        return descriptor  # the hook matched the same name from the current folder
    for listener in list(_LISTENERS):
        listener[0](name, mark)
    return descriptor


@contextmanager
def descriptor_opens(listener):
    """While active, `listener(name, mark)` is called for each open of a relative `name` from a
    folder's descriptor whose file the current folder does not name, `mark` its (device, inode)."""
    if os.open is not _open_watched:
        _REAL_OPEN[:] = [os.open]
        os.open = _open_watched
    entry = [listener]
    _LISTENERS.append(entry)
    try:
        yield
    finally:
        _LISTENERS[:] = [e for e in _LISTENERS if e is not entry]
        if not _LISTENERS and os.open is _open_watched:
            os.open = _REAL_OPEN[0]


def identities(*folders) -> set:
    found = set()
    for folder in folders:
        folder = Path(folder)
        for path in [folder, *folder.rglob("*")]:
            mark = identity(path)
            if mark is not None:
                found.add(mark)
    return found


@contextmanager
def watch(*folders):
    """The list of (event, path) of every open, listing or scan of a watched file or folder."""
    if not _ADDED:
        sys.addaudithook(_hook)
        _ADDED.append(True)
    entry = (identities(*folders), [])

    def from_descriptor(name: str, mark: tuple) -> None:
        if mark in entry[0]:
            entry[1].append(("open", f"{name} (from a folder descriptor)"))

    with descriptor_opens(from_descriptor):
        _ACTIVE.append(entry)
        try:
            yield entry[1]
        finally:
            _ACTIVE[:] = [e for e in _ACTIVE if e is not entry]
