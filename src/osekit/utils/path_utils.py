from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


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


def absolute_path(target_path: Path, root_path: Path | None = None) -> Path:
    """Return the absolute path of a file from a relative root path.

    If the file is already absolute or if the root path is None, will
    return the absolute path.

    Parameters
    ----------
    target_path: Path
        Path to return as absolute.
        Should already be absolute if no root_path is provided.
    root_path: Path | None
        Path from which the target_path is relatively expressed.
        If None, target_path should be absolute.

    Returns
    -------
    Path:
        The absolute path to target_path.

    """
    if target_path.is_absolute():
        return target_path
    if root_path is None:
        msg = (
            rf"No root path is specified."
            rf"Target path {target_path!s} should be absolute."
        )
        raise ValueError(msg)
    return root_path / target_path
