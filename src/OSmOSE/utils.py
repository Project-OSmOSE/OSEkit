from pathlib import PurePath
from importlib.resources import as_file
import os
import shutil
import struct
from collections import namedtuple
from distutils.errors import UnknownFileError
from typing import Union, NamedTuple, Tuple

import json

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def display_folder_storage_infos(dir_path: str) -> None:
    usage = shutil.disk_usage(dir_path)
    print("Total storage space (TB):", round(usage.total / (1024**4), 1))
    print("Used storage space (TB):", round(usage.used / (1024**4), 1))
    print("-----------------------")
    print("Available storage space (TB):", round(usage.free / (1024**4), 1))


def list_not_built_datasets(datasets_folder_path: str) -> None:
    """Prints the available datasets that have not been built by the `Dataset.build()` function.

    Parameter
    ---------
    dataset_folder_path: str
        The path to the directory containing the datasets"""

    dataset_list = [
        directory
        for directory in sorted(os.listdir(datasets_folder_path))
        if os.path.isdir(os.path.join(datasets_folder_path, directory))
    ]
    list_not_built_datasets = []

    for dataset_directory in dataset_list:
        if os.path.exists(
            os.path.join(datasets_folder_path, dataset_directory, "raw/audio/original/")
        ):
            list_not_built_datasets.append(dataset_directory)

    print("List of the datasets not built yet:")

    for dataset in list_not_built_datasets:
        print("  - {}".format(dataset))


def read_config(raw_config: Union[str, dict, PurePath]) -> NamedTuple:
    """Read the given configuration file or dict and converts it to a namedtuple. Only TOML and JSON formats are accepted for now.

    Parameter
    ---------
    raw_config : `str` or `PurePath` or `dict`
        The path of the configuration file, or the dict object containing the configuration.

    Returns
    -------
    config : `namedtuple`
        The configuration as a `namedtuple` object.

    Raises
    ------
    FileNotFoundError
        Raised if the raw_config is a string that does not correspond to a valid path.
    TypeError
        Raised if the raw_config is anything else than a string, a PurePath or a dict.
    NotImplementedError
        Raised if the raw_config file is in YAML format
    UnknownFileError
        Raised if the raw_config file is not in TOML, JSON or YAML formats."""

    match raw_config:
        case PurePath():
            with as_file(raw_config) as input_config:
                raw_config = input_config

        case str():
            if not os.path.isfile(raw_config):
                raise FileNotFoundError(
                    f"The configuration file {raw_config} does not exist."
                )

        case dict():
            pass
        case _:
            raise TypeError(
                "The raw_config must be either of type str, dict or Traversable."
            )

    if not isinstance(raw_config, dict):
        with open(raw_config, "rb") as input_config:
            match os.path.splitext(raw_config)[1]:
                case ".toml":
                    raw_config = tomllib.load(input_config)
                case ".json":
                    raw_config = json.load(input_config)
                case ".yaml":
                    raise NotImplementedError(
                        "YAML support will eventually get there (unfortunately)"
                    )
                case _:
                    raise UnknownFileError(
                        f"The provided configuration file extension ({os.path.splitext(raw_config)[1]} is not a valid extension. Please use .toml or .json files."
                    )

    return namedtuple("GenericDict", raw_config.keys())(**raw_config)


def read_header(file: str) -> Tuple[int, float, int, int]:
    """Read the first bytes of a wav file and extract its characteristics.

    At the very least, only the first 44 bytes are read. If the `data` chunk is not right after the header chunk,
    the subsequent chunks will be read until the `data` chunk is found. If there is no `data` chunk, all the file will be read.

    Parameter
    ---------
    file: str
        The absolute path of the wav file whose header will be read.

    Returns
    -------
    samplerate : `int`
        The number of samples in one frame.
    frames : `float`
        The number of frames, corresponding to the file duration in seconds.
    channels : `int`
        The number of audio channels.
    sampwidth : `int`
        The sample width.

    Note
    ----
    When there is no `data` chunk, the `frames` value will fall back on the size written in the header. This can be incorrect,
    if the file has been corrupted or the writing process has been interrupted before completion.
    """
    with open(file, "rb") as fh:
        _, size, _ = struct.unpack("<4sI4s", fh.read(12))
        chunk_header = fh.read(8)
        subchunkid, _ = struct.unpack("<4sI", chunk_header)

        if subchunkid == b"fmt ":
            _, channels, samplerate, _, _, sampwidth = struct.unpack(
                "HHIIHH", fh.read(16)
            )

        chunkOffset = fh.tell()
        found_data = False
        while chunkOffset < size and not found_data:
            fh.seek(chunkOffset)
            subchunk2id, subchunk2size = struct.unpack("<4sI", fh.read(8))
            if subchunk2id == b"data":
                found_data = True

            chunkOffset = chunkOffset + subchunk2size + 8

        if not found_data:
            print(
                "No data chunk found while reading the header. Will fallback on the header size."
            )
            subchunk2size = size - 36

        sampwidth = (sampwidth + 7) // 8
        framesize = channels * sampwidth
        frames = subchunk2size / framesize

        if (size - 36) != subchunk2size:
            print(
                f"Warning : the size indicated in the header is not the same as the actual file size. This might mean that the file is truncated or otherwise corrupt.\
                \nSupposed size: {size} bytes \nActual size: {subchunk2size} bytes."
            )

        return samplerate, frames, channels, sampwidth
