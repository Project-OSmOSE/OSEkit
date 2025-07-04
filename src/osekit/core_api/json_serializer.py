"""Functions used to serialize the API objects to json files."""

import json
from pathlib import Path


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
        return json.loads(f.read())
