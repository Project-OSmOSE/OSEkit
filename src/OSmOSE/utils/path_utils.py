from __future__ import annotations

import shutil
from pathlib import Path

from OSmOSE.config import DPDEFAULT


def make_path(path: Path, *, mode=DPDEFAULT) -> Path:
    """Create a path folder by folder with correct permissions.

    If a folder in the path already exists, it will not be modified (even if the specified mode differs).

    Parameters
    ----------
        path: `Path`
            The complete path to create.
        mode: `int`, optional, keyword-only
            The permission code of the complete path. The default is 0o755, meaning the owner has all permissions over the files of the path,
            and the owner group and others only have read and execute rights, but not writing.

    """
    for parent in path.parents[::-1]:
        parent.mkdir(mode=mode, exist_ok=True)

    if not path.is_file():
        path.mkdir(mode=mode, exist_ok=True)

    return path


def str2Path(path: str) -> Path:
    """From string to Path

    Parameters
    ----------
    path : str
        The complete path as a string.

    Returns
    -------
    Path
        The complete path as a Path.

    """
    return Path(path)


def move_tree(
    source: Path,
    destination: Path,
    excluded_paths: set[Path] | None = None,
) -> None:
    """Move all content from a source folder to a destination folder.

    Paths given in excluded_files will not be affected.

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
        If a nested file like source/foo/bar is included without including foo (which
        is directly within the source folder), all the content of foo (including bar)
        will be moved regardless.

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
