from __future__ import annotations

import json
import os
import shutil
import time
from bisect import bisect
from importlib.resources import as_file
from importlib.util import find_spec
from pathlib import Path
from typing import NamedTuple

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


from osekit.config import global_logging_context as glc
from osekit.config import print_logger

_is_grp_supported = bool(find_spec("grp"))


@glc.set_logger(print_logger)
def display_folder_storage_info(dir_path: str) -> None:
    """Print a detail of the current disk usage."""
    usage = shutil.disk_usage(dir_path)

    def str_usage(key: str, value: int) -> str:
        return f"{f'{key} storage space:':<30}{f'{round(value / (1024**4), 1)} TB':>10}"

    total = str_usage("Total", usage.total)
    used = str_usage("Used", usage.used)
    free = str_usage("Available", usage.free)
    glc.logger.info("%s\n%s\n%s\n%s", total, used, f"{'-' * 30:^40}", free)


def read_config(raw_config: str | dict | Path) -> NamedTuple:
    """Read the given configuration file or dict and converts it to a namedtuple. Only TOML and JSON formats are accepted for now.

    Parameter
    ---------
    raw_config : `str` or `Path` or `dict`
        The path of the configuration file, or the dict object containing the configuration.

    Returns
    -------
    config : `namedtuple`
        The configuration as a `namedtuple` object.

    Raises
    ------
    FileNotFoundError
        Raised if the raw_config is a string that does not correspond to a valid path, or the raw_config file is not in TOML, JSON or YAML formats.
    TypeError
        Raised if the raw_config is anything else than a string, a PurePath or a dict.
    NotImplementedError
        Raised if the raw_config file is in YAML format

    """
    match raw_config:
        case Path():
            with as_file(raw_config) as input_config:
                raw_config = input_config

        case str():
            if not Path(raw_config).is_file:
                raise FileNotFoundError(
                    f"The configuration file {raw_config} does not exist.",
                )

        case dict():
            pass
        case _:
            raise TypeError(
                "The raw_config must be either of type str, dict or Traversable.",
            )

    if not isinstance(raw_config, dict):
        with open(raw_config, "rb") as input_config:
            match Path(raw_config).suffix:
                case ".toml":
                    raw_config = tomllib.load(input_config)
                case ".json":
                    raw_config = json.load(input_config)
                case ".yaml":
                    raise NotImplementedError(
                        "YAML support will eventually get there (unfortunately)",
                    )
                case _:
                    raise FileNotFoundError(
                        f"The provided configuration file extension ({Path(raw_config).suffix} is not a valid extension. Please use .toml or .json files.",
                    )

    return raw_config


def chmod_if_needed(path: Path, mode: int) -> None:
    """Change the permission of a path if user doesn't have read and write access.

    Parameters
    ----------
    path: Path
        Path of the file or folder in which permission should be changed.
    mode: int
        Permissions as used by os.chmod()

    """
    if not _is_grp_supported:
        return
    if all(os.access(path, p) for p in (os.R_OK, os.W_OK)):
        return

    try:
        path.chmod(mode)
    except PermissionError as e:
        message = (
            f"You do not have the permission to write to {path}, "
            "nor to change its permissions."
        )
        glc.logger.error(message)
        raise PermissionError(message) from e


def change_owner_group(path: Path, owner_group: str) -> None:
    """Change the owner group of the given path.

    Parameters
    ----------
    path:
        Path of which the owner group should be changed.
    owner_group:
        The new owner group.
        A warning is logged if the grp module is supported (Unix os) but
        no owner_group is passed.

    """
    if not _is_grp_supported:
        return
    if owner_group is None:
        glc.logger.warning("You did not set the group owner of the dataset.")
        return
    if path.group() == owner_group:
        return
    glc.logger.debug("Setting osekit permission to the dataset..")

    try:
        import grp

        gid = grp.getgrnam(owner_group).gr_gid
    except KeyError as e:
        message = f"Group {owner_group} does not exist."
        glc.logger.error(message)
        raise KeyError(message) from e

    try:
        os.chown(path, -1, gid)
    except PermissionError as e:
        message = (
            f"You do not have the permission to change the owner of {path}."
            f"The group owner has not been changed "
            f"from {path.group()} to {owner_group}."
        )
        glc.logger.error(message)
        raise PermissionError(message) from e


def get_umask() -> int:
    """Return the current umask."""
    umask = os.umask(0)
    os.umask(umask)
    return umask


def file_indexes_per_batch(
    total_nb_files: int,
    nb_batches: int,
) -> list[tuple[int, int]]:
    """Compute the start and stop file indexes for each batch.

    The number of files is equitably distributed among batches.
    Example: 10 files distributed among 4 batches will lead to
    batches indexes [(0,3), (3,6), (6,8), (8,10)].

    Parameters
    ----------
    total_nb_files: int
        Number of files processed by ball batches
    nb_batches: int
        Number of batches in the analysis

    Returns
    -------
    list[tuple[int,int]]:
    A list of tuples representing the start and stop index of files processed by each
    batch in the analysis.

    Examples
    --------
    >>> file_indexes_per_batch(10,4)
    [(0, 3), (3, 6), (6, 8), (8, 10)]
    >>> file_indexes_per_batch(1448,10)
    [(0, 145), (145, 290), (290, 435), (435, 580), (580, 725), (725, 870), (870, 1015), (1015, 1160), (1160, 1304), (1304, 1448)]

    """  # noqa: E501
    batch_lengths = [
        length
        for length in nb_files_per_batch(total_nb_files, nb_batches)
        if length > 0
    ]
    return [
        (sum(batch_lengths[:b]), sum(batch_lengths[:b]) + batch_lengths[b])
        for b in range(len(batch_lengths))
    ]


def nb_files_per_batch(total_nb_files: int, nb_batches: int) -> list[int]:
    """Compute the number of files processed by each batch in the analysis.

    The number of files is equitably distributed among batches.
    Example: 10 files distributed among 4 batches will lead to
    batches containing [3,3,2,2] files.

    Parameters
    ----------
    total_nb_files: int
        Number of files processed by ball batches
    nb_batches: int
        Number of batches in the analysis

    Returns
    -------
    list(int):
    A list representing the number of files processed by each batch in the analysis.

    Examples
    --------
    >>> nb_files_per_batch(10,4)
    [3, 3, 2, 2]
    >>> nb_files_per_batch(1448,10)
    [145, 145, 145, 145, 145, 145, 145, 145, 144, 144]

    """
    return [
        total_nb_files // nb_batches + (1 if i < total_nb_files % nb_batches else 0)
        for i in range(nb_batches)
    ]


def locked(lock_file: Path) -> callable:
    """Use a lock file for managing priorities between processes.

    If the specified lock file already exists, the decorated function execution will be
    suspended until the lock file is removed.

    The lock_file will then be created before the execution of the decorated function,
    and removed once the function has been executed.

    Parameters
    ----------
    lock_file: Path
        Path to the lock file.

    """

    def inner(func: callable) -> callable:
        def wrapper(*args: any, **kwargs: any) -> any:
            # Wait for the lock to be released
            while lock_file.exists():
                time.sleep(1)

            # Create lock file
            lock_file.touch()

            r = func(*args, **kwargs)

            # Release lock file
            lock_file.unlink()

            return r

        return wrapper

    return inner


def get_closest_value_index(target: float, values: list[float]) -> int:
    """Get the index of the closest value in a list from a target value.

    Parameters
    ----------
    target: float
        Target value from which the closest value is to be found.
    values: list[float]
        List of values in which the closest value from target is
        to be found.

    Returns
    -------
    int
        Index of the closest value from target in values.

    """
    closest_upper_index = min(bisect(values, target), len(values) - 1)
    closest_lower_index = max(0, closest_upper_index - 1)
    return min(
        (closest_lower_index, closest_upper_index),
        key=lambda i: abs(values[i] - target),
    )
