"""Functions used to serialize the API objects to json files."""

import json
import os
from pathlib import Path
from typing import Literal


def set_path_reference(
    serialized_dict: dict,
    root_path: Path,
    reference: Literal["absolute", "relative"],
) -> None:
    """Recursively set all paths either relative to root_path or absolute from root_path.

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
        "absolute": Files names are converted from relative to root_path to absolute.
        "relative": Files names are converted from absolute to relative to root_path.

    """
    for key, value in serialized_dict.items():
        if type(value) is dict:
            set_path_reference(value, root_path, reference)
            continue
        if key == "path":
            serialized_dict[key] = (
                (root_path / value).resolve()
                if reference == "absolute"
                else os.path.relpath(value, root_path)
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
