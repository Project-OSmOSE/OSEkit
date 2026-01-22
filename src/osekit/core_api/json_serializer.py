"""Functions used to serialize the API objects to json files."""

import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath  # noqa: F401
from typing import Literal

from osekit.utils.path_utils import is_absolute


def absolute_to_relative(
    target_path: os.PathLike | str,
    root_path: os.PathLike | str,
) -> Path:
    """Convert an absolute path in a relative path.

    Parameters
    ----------
    target_path: os.PathLike | str
        Absolute path to the target.
    root_path: os.PathLike | str
        Path relative to which the target should be expressed.

    Returns
    -------
    Path:
        Target path expressed relative to the root path.

    """
    target_path, root_path = map(Path, (target_path, root_path))
    return Path(os.path.relpath(path=target_path, start=root_path))


def relative_to_absolute(
    target_path: os.PathLike | str,
    root_path: os.PathLike | str,
) -> Path:
    r"""Convert a relative path in an absolute path.

    Parameters
    ----------
    target_path: os.PathLike | str
        Relative path to the target.
        If the target is absolute, the first folder in the root parents that appear
        in the target parent will be considered as common.
    root_path: os.PathLike | str
        Path from which the target is expressed.

    Returns
    -------
    Path:
        Absolute path to the target.

    >>> str(PureWindowsPath(relative_to_absolute(target_path=r'relative\target', root_path=r'C:\absolute\root')))
    'C:\\absolute\\root\\relative\\target'
    >>> str(PureWindowsPath(relative_to_absolute(target_path=r'D:\absolute\path\root\target', root_path=r'C:\absolute\root')))
    'C:\\absolute\\path\\root\\target'
    >>> str(PurePosixPath(relative_to_absolute(target_path=r'C:/user/cool/data/audio/fun.wav', root_path=r'/home/dataset/cool/processed/stuff')))
    '/home/dataset/cool/data/audio/fun.wav'

    """
    target_path, root_path = map(PureWindowsPath, (target_path, root_path))
    if is_absolute(target_path):
        first_common_folder = next(
            folder
            for folder in root_path.parts
            if (folder in target_path.parts and folder != root_path.anchor)
        )
        return Path(
            PureWindowsPath(
                str(root_path).split(first_common_folder)[0]
                + first_common_folder
                + str(target_path).split(first_common_folder, maxsplit=1)[1],
            ),
        )
    return Path(root_path / target_path).resolve()


def set_path_reference(
    serialized_dict: dict,
    root_path: Path,
    reference: Literal["absolute", "relative"],
) -> None:
    """Recursively set all paths either relative to ``root_path`` or absolute from ``root_path``.

    Parameters
    ----------
    serialized_dict: dict
        Dictionary in which the path values must be modified.
    root_path: Path
        Root path:
         - From which the relative files are expressed.
         - From which the absolute files should be expressed as relative
    reference: Literal["absolute","relative"]
        How to express the file names:
        ``"absolute"``: Files names are converted from
        relative to ``root_path`` to absolute.
        ``"relative"``: Files names are converted from
        absolute to relative to ``root_path``.

    """
    for key, value in serialized_dict.items():
        if type(value) is dict:
            set_path_reference(value, root_path, reference)
            continue
        if key in ("path", "folder", "json"):
            serialized_dict[key] = (
                str(relative_to_absolute(target_path=value, root_path=root_path))
                if reference == "absolute"
                else str(absolute_to_relative(target_path=value, root_path=root_path))
            )


def serialize_json(path: Path, serialized_dict: dict) -> None:
    """Serialize a dictionary in a JSON file.

    Parameters
    ----------
    path: Path
        Path of the JSON file to be written.
    serialized_dict: dict
        Dictionary to be serialized.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    set_path_reference(
        serialized_dict=serialized_dict,
        root_path=path.parent,
        reference="relative",
    )
    with path.open("w") as file:
        file.write(json.dumps(serialized_dict, sort_keys=True, indent=4))


def deserialize_json(path: Path) -> dict:
    """Deserialize a JSON file into a dictionary.

    Parameters
    ----------
    path: Path
        Path of the JSON file to be deserialized.

    Returns
    -------
    dict:
        Deserialized dictionary.

    """
    with path.open("r") as f:
        dictionary = json.loads(f.read())
    set_path_reference(
        serialized_dict=dictionary,
        root_path=path.parent,
        reference="absolute",
    )
    return dictionary
