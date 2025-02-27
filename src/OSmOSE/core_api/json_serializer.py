import json
from pathlib import Path


def serialize_json(path: Path, serialized_dict: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        file.write(json.dumps(serialized_dict))

def deserialize_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.loads(f.read())
