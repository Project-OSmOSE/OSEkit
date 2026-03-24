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
        shutil.move(file, file_destination / file.name)
    if not any(destination.iterdir()):
        destination.rmdir()


def is_absolute(path: PathLike | str) -> bool:
    """Check if a path is an absolute path in any OS format."""
    for formatted_path in (PureWindowsPath(path), PurePosixPath(path), Path(path)):
        if formatted_path.is_absolute():
            return True
    return False
