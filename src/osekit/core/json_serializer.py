"""Functions used to serialize the API objects to json files."""

import json
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path, PurePosixPath, PureWindowsPath  # noqa: F401
from typing import Any

from osekit.utils.path import is_absolute

_anchor: ContextVar[Path | None] = ContextVar("anchor_path", default=None)

_PATH_KEYS = {"path", "folder", "json"}


@contextmanager
def path_anchor(anchor: Path) -> Generator[None, None, None]:
    """Set the anchor for serializing paths.

    All paths that are within ``anchor`` will be serialized as relative paths.
    All paths that are outside ``anchor`` will be serialized as absolute paths.

    Parameters
    ----------
    anchor: Path
        Path of the anchor.

    """
    token = _anchor.set(anchor)
    try:
        yield
    finally:
        _anchor.reset(token)


def _serialize_path(value: Path, root_path: Path) -> str:
    """Serialize a path into a string.

    If the path is within the ``_anchor`` path, it will be serialized as relative
    to ``root_path`` (provided this is possible).
    Else, it will be serialized with its absolute path.

    Parameters
    ----------
    value: Path
        Path to serialize.
    root_path: Path
        Root path according to which paths are serialized as relative.

    Returns
    -------
    str:
        The serialized path.

    """
    anchor = _anchor.get()
    if anchor is not None and not value.is_relative_to(anchor):
        # Path is outside the project: we store it as an absolute path.
        return value.as_posix()
    try:
        return Path(os.path.relpath(path=value, start=root_path)).as_posix()
    except ValueError:
        # Value is not expressible as relative to root
        return value.as_posix()


def _deserialize_path(value: str, root_path: Path | None = None) -> Path:
    """Deserialize a string into a path.

    Is the path is not absolute, it is considered to be relative to ``root_path``.

    Parameters
    ----------
    value: str
        The string representation of the path to deserialize.
    root_path: Path|None
        The path ``value`` is relative to, if ``value`` is not absolute.

    Returns
    -------
    Path
        The deserialized path corresponding to ``value``.

    """
    if is_absolute(path=value):
        return Path(value)
    if root_path is None:
        msg = (
            f"{value} is not an absolute serialized path.\n"
            f"Please provide a Path as root_path."
        )
        raise ValueError(msg)
    return (root_path / value).resolve()


def serialize_paths(
    data: dict[str, Any],
    root_path: Path,
) -> None:
    """Serialize all paths in ``data``.

    ``data`` is edited in-place.

    Parameters
    ----------
    data: dict[str,Any]
        Input dictionary.
        Will be edited in-place.
    root_path: Path
        Path that serves as root for relative paths.

    """
    _transform_paths(
        data=data,
        root_path=root_path,
        transform=_serialize_path,
    )


def deserialize_paths(
    data: dict[str, Any],
    root_path: Path,
) -> None:
    """Deserialize all paths in ``data`` to ``Path`` instances.

    ``data`` is edited in-place.

    Parameters
    ----------
    data: dict[str,Any]
        Input dictionary.
        Will be edited in-place.
    root_path: Path
        Path that serves as root for relative paths.

    """
    _transform_paths(
        data=data,
        root_path=root_path,
        transform=_deserialize_path,
    )


def _transform_paths(
    data: dict[str, Any],
    root_path: Path,
    transform: Callable[..., Path | str],
) -> None:
    """In-place transform of the paths in ``data``.

    Each path (identified thanks to ``json_serializer._PATH_KEYS`` key names) will be transformed
    according to the input ``transform``.

    Parameters
    ----------
    data: dict[str, Any]
        Input dictionary.
        Will be edited in-place.
    root_path: Path
        Path that serves as root for relative paths.
    transform: Callable[...,Path|str]
        Transform to apply to the path values.

    """
    for key, value in data.items():
        if type(value) is dict:
            _transform_paths(data=value, root_path=root_path, transform=transform)
            continue
        if key in _PATH_KEYS:
            data[key] = transform(value=value, root_path=root_path)


def serialize_json(path: Path, data: dict) -> None:
    """Serialize a dictionary in a JSON file.

    Parameters
    ----------
    path: Path
        Path of the JSON file to be written.
    data: dict
        Dictionary to be serialized.

    """
    if not (parent_folder := path.parent).exists():
        parent_folder.mkdir(parents=True)

    serialize_paths(
        data=data,
        root_path=path.parent,
    )
    with path.open("w") as file:
        file.write(json.dumps(data, sort_keys=True, indent=4))


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

    deserialize_paths(
        data=dictionary,
        root_path=path.parent,
    )
    return dictionary
