"""Opens of chosen files, each marked by whether a chosen module's code was running when it was
made.

`opened(files, module)` records, while it is active, every `open` audit event (Python's audit
events, `sys.addaudithook`; `os.open` raises one too) whose target is one of `files`, matched by
device and inode (`os.stat`, links followed), as (path, inside): `inside` is true when a frame of
the module's file is on the call stack of the open. The identities are taken when the watch
starts. A name opened from a folder's descriptor (`dir_fd`, as `plain_read` opens every name)
shows no folder in the event; it is matched by the identity of the file it opened, through the
`os.open` wrapper of `sealed_watch` (`descriptor_opens`), so a same-named file in another folder
is never counted. An audit hook cannot be removed, so one hook is added once per process; it
records nothing while no watch is active.
"""

import os
import sys
from contextlib import contextmanager

from sealed_watch import descriptor_opens

_ACTIVE: list = []
_ADDED: list = []


def _identity(path) -> tuple[int, int] | None:
    try:
        found = os.stat(path)
    except (OSError, TypeError, ValueError):
        return None
    return found.st_dev, found.st_ino


def _inside(module_file: str) -> bool:
    frame = sys._getframe(2)
    while frame is not None:
        if os.path.realpath(frame.f_code.co_filename) == module_file:
            return True
        frame = frame.f_back
    return False


def _hook(event, args) -> None:
    if not _ACTIVE or event != "open" or not args:
        return
    target = args[0]
    if target is None or isinstance(target, int):
        return
    try:
        name = os.fspath(target)
    except TypeError:
        return
    found = _identity(name)
    for files, module_file, seen in _ACTIVE:
        if found in files:
            seen.append((os.fsdecode(name), _inside(module_file)))


@contextmanager
def opened(files, module):
    """The list of (path, inside) of every open of one of `files` (module text)."""
    if not _ADDED:
        sys.addaudithook(_hook)
        _ADDED.append(True)
    marks = {mark for mark in (_identity(path) for path in files) if mark is not None}
    entry = (marks, os.path.realpath(module.__file__), [])

    def from_descriptor(name: str, mark: tuple) -> None:
        if mark in marks:
            entry[2].append((name, _inside(entry[1])))

    with descriptor_opens(from_descriptor):
        _ACTIVE.append(entry)
        try:
            yield entry[2]
        finally:
            _ACTIVE[:] = [e for e in _ACTIVE if e is not entry]
