"""One reader of files and folders by their own paths below a top folder, for the checks that
must never open what a link or a ".." points at (the WP14 check, `tools/sealed_run_check.py`, and
the WP12 check, `tools/rehearsal_check.py`).

`read(top, path)` gives a file's bytes and `listing(top, folder)` a folder's names. Both open the
path through `opened`: its parts below `top` are taken as written, before any normalising, and a
"..", a "." or an empty part (an absolute start among them), or a path not beginning with `top`,
raises `NotPlain` with nothing opened; then each name is opened from `top` down, from the
descriptor of the folder above it, never following a symbolic link (`O_NOFOLLOW`, after a look
at the name), so a link at the file or at any folder between `top` and it raises `LinkOnPath`
and is never followed. Both are `NeverRead`, an `OSError`; any other `OSError` of a look or an
open is raised as it is.
"""

import os
import stat
from pathlib import Path


class NeverRead(OSError):
    """A path a checker never opens: a link on it (`LinkOnPath`) or a name that is not plain
    (`NotPlain`): the check that needs it could not run."""


class NotPlain(NeverRead):
    """A path whose parts under `top`, as written before any normalising, hold "..", "." or an
    empty part (an absolute start among them), or which does not begin with `top`."""

    def __init__(self, path: str):
        super().__init__(
            f"{path} is not a plain path under the top folder (a '..', '.' or empty part, or an "
            "absolute start): never opened"
        )
        self.path = path


class LinkOnPath(NeverRead):
    """A symbolic link at a file being read, or at a folder between `top` and it: never
    followed. `file` and `link` are their paths under `top`."""

    def __init__(self, file: str, link: str):
        super().__init__(
            f"{file} is a symbolic link"
            if link == file
            else f"{file} lies under the symbolic link {link}"
        )
        self.file, self.link = file, link


def no_link(name, flags: int, folder: int | None = None) -> int:
    """The opener of every name a checker opens by its own path, from the descriptor `folder`:
    a link raises `OSError`."""
    return os.open(name, flags | os.O_NOFOLLOW, dir_fd=folder)


def plain_parts(root, path) -> tuple[str, ...]:
    """The names of `path` below `root`, as written, with no normalising: `NotPlain` when the
    text does not begin with `root` (as given, or made absolute) and "/", or when a name below it
    is "..", "." or empty (an absolute start or a doubled "/")."""
    text, top = os.fspath(path), os.fspath(root)
    for start in (top, os.path.abspath(top)):
        if text.startswith(start.rstrip("/") + "/"):
            rest = text[len(start.rstrip("/")) + 1 :]
            break
    else:
        if top != os.curdir or os.path.isabs(text):
            raise NotPlain(text)
        rest = text  # `Path(".") / name` drops the "./"
    names = tuple(rest.split("/"))
    if any(name in ("", os.curdir, os.pardir) for name in names):
        raise NotPlain(text)
    return names


def opened(root: Path, path: Path, *, listing: bool = False) -> int:
    """A descriptor of the file `path` (of the folder, with `listing`), opened one name at a time
    from `root` down, each from the descriptor of the folder above it by `no_link`, after a look
    (`lstat`) at the name: a symbolic link at any name raises `LinkOnPath` and is never followed
    (one made between the look and the open is refused by the opener); a path not under `root`,
    and any other `OSError` of a look or an open, are raised. The path's parts are taken as written,
    before any normalising (`plain_parts`): a "..", a "." or an absolute start raises `NotPlain`,
    and nothing is opened."""
    names = plain_parts(root, path)
    top = os.path.abspath(root)
    file = "/".join(names)
    folder = os.open(top, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for at, name in enumerate(names):
            here = "/".join(names[: at + 1])
            last = at == len(names) - 1
            flags = os.O_RDONLY | (os.O_DIRECTORY if listing or not last else 0)
            if stat.S_ISLNK(os.lstat(name, dir_fd=folder).st_mode):
                raise LinkOnPath(file, here)
            try:
                inner = no_link(name, flags, folder)
            except OSError:
                if stat.S_ISLNK(os.lstat(name, dir_fd=folder).st_mode):
                    raise LinkOnPath(file, here) from None
                raise
            os.close(folder)
            folder = inner
    except BaseException:
        os.close(folder)
        raise
    return folder


def read(root: Path, path: Path) -> bytes:
    """The bytes of the file `path`, opened by `opened` from `root`; raises its `OSError`s."""
    descriptor = opened(root, path)
    try:
        chunks = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def listing(root: Path, folder: Path) -> list[str]:
    """The names in the folder `folder`, opened by `opened` from `root`; raises its `OSError`s."""
    descriptor = opened(root, folder, listing=True)
    try:
        return os.listdir(descriptor)
    finally:
        os.close(descriptor)
