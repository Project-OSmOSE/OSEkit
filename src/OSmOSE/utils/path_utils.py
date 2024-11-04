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
