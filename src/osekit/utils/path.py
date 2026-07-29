from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike


def move_tree(
    source: Path,
    destination: Path,
    excluded_paths: set[Path] | None = None,
) -> None:
    """Move all content from a source folder to a destination folder.

    Paths given in ``excluded_files`` will not be affected.

    Parameters
    ----------
    source : Path
        The folder from which the content will be moved.
    destination: Path
        The destination folder in which the content will be moved.
    excluded_paths: set[Path]
        Paths that won't be affected by the moving.
        These paths refer to files/folders directly within the source folder.
        If a path point to a folder, all of its content will be left untouched.
        If a nested file like ``source/foo/bar`` is included without
        including ``foo`` (which is directly within the ``source`` folder),
        all the content of ``foo`` (including ``bar``) will be moved regardless.

    """
    if excluded_paths is None:
        excluded_paths = set()
    destination.mkdir(parents=True, exist_ok=True)
    for file in source.glob("*"):
        if file in excluded_paths or file == destination or file in destination.parents:
            continue
        file_destination = destination / file.parent.relative_to(source)
        file_destination.mkdir(parents=True, exist_ok=True)
        if not (file_destination / file.name).exists():
            shutil.move(file, file_destination)
    if not any(destination.iterdir()):
        destination.rmdir()


def is_absolute(path: PathLike | str) -> bool:
    """Check if a path is an absolute path in any OS format."""
    for formatted_path in (PureWindowsPath(path), PurePosixPath(path), Path(path)):
        if formatted_path.is_absolute():
            return True
    return False


def ensure_within_base(path: Path, base: Path) -> Path:
    """Ensure that a path resolves to a location contained within a base folder.

    This guards against path traversal / path injection: if ``path`` is built
    (even partly) from untrusted data (e.g. a value read from a JSON project
    file, or from another user's dataset), a crafted value such as
    ``../../../home/other_user/.ssh`` could otherwise make OSEkit read,
    write, move or delete files outside the folder it is supposed to
    operate in.

    Parameters
    ----------
    path: Path
        The path to validate. Doesn't need to exist yet (this also covers
        paths that are about to be created, e.g. before a ``mkdir()``).
    base: Path
        The folder that ``path`` is expected to stay within.
        This is typically the ``Project`` root folder, i.e. the boundary of
        what the running process is allowed to touch.

    Returns
    -------
    Path:
        The resolved (absolute, symlink-free) version of ``path``.

    Raises
    ------
    ValueError:
        If the resolved ``path`` is not ``base`` itself and not located
        inside ``base``.

    Examples
    --------
    >>> ensure_within_base(Path("/data/project/log"), Path("/data/project"))
    PosixPath('/data/project/log')
    >>> ensure_within_base(Path("/data/project/../../etc"), Path("/data/project"))
    Traceback (most recent call last):
        ...
    ValueError: Path '/data/project/../../etc' escapes the allowed directory '/data/project'.

    """  # noqa: E501
    resolved_path = path.resolve()
    resolved_base = base.resolve()
    if resolved_path != resolved_base and resolved_base not in resolved_path.parents:
        msg = f"Path '{path}' escapes the allowed directory '{base}'."
        raise ValueError(msg)
    return resolved_path
