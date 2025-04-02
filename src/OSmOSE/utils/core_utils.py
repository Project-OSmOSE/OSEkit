from __future__ import annotations

import glob
import json
import math
import os
import random
import shutil
import struct
from importlib.resources import as_file
from importlib.util import find_spec
from pathlib import Path
from typing import List, NamedTuple, Tuple, Union

import pandas as pd
import pytz

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import datetime as dt
import re

import numpy as np
import soundfile as sf

from OSmOSE.config import OSMOSE_PATH, print_logger
from OSmOSE.config import global_logging_context as glc

_is_grp_supported = bool(find_spec("grp"))


@glc.set_logger(print_logger)
def display_folder_storage_info(dir_path: str) -> None:
    usage = shutil.disk_usage(dir_path)

    def str_usage(key: str, value: int) -> str:
        return f"{f'{key} storage space:':<30}{f'{round(value/ (1024**4), 1)} TB':>10}"

    total = str_usage("Total", usage.total)
    used = str_usage("Used", usage.used)
    free = str_usage("Available", usage.free)
    glc.logger.info("%s\n%s\n%s\n%s", total, used, f"{'-'*30:^40}", free)


@glc.set_logger(print_logger)
def list_not_built_dataset(path_osmose: str, project: str = None) -> None:
    """Prints the available datasets that have not been built by the `Dataset.build()` function.

    Parameter
    ---------
    dataset_folder_path: str
        The path to the directory containing the project/datasets.
    project: str
        Name of the project folder containing the datasets.
    """
    ds_folder = Path(path_osmose, project)

    dataset_list = [
        directory
        for directory in ds_folder.iterdir()
        if ds_folder.joinpath(directory).is_dir()
    ]

    dataset_list = sorted(
        dataset_list,
        key=lambda path: str(path).lower(),
    )  # case insensitive alphabetical sorting of datasets

    list_not_built_dataset = []
    list_unknown_dataset = []

    for dataset_directory in dataset_list:
        dataset_directory = ds_folder.joinpath(dataset_directory)
        if os.access(dataset_directory, os.R_OK):
            metadata_path = next(
                dataset_directory.joinpath(OSMOSE_PATH.raw_audio).rglob("metadata.csv"),
                None,
            )
            timestamp_path = next(
                dataset_directory.joinpath(OSMOSE_PATH.raw_audio).rglob(
                    "timestamp.csv",
                ),
                None,
            )

            if not (
                metadata_path
                and metadata_path.exists()
                and timestamp_path
                and timestamp_path.exists()
                and not dataset_directory.joinpath(
                    OSMOSE_PATH.raw_audio,
                    "original",
                ).exists()
            ):
                list_not_built_dataset.append(dataset_directory)
        else:
            list_unknown_dataset.append(dataset_directory)

    not_built_formatted = "\n".join(
        [f"  - {dataset.name}" for dataset in list_not_built_dataset],
    )
    glc.logger.info(
        f"""List of the datasets that are not built yet:\n\n{not_built_formatted}""",
    )

    if list_unknown_dataset:
        unreachable_formatted = "\n".join(
            [f"  - {dataset.name}" for dataset in list_unknown_dataset],
        )
        glc.logger.info(
            f"""List of unreachable datasets (probably due to insufficient permissions):\n\n{unreachable_formatted}""",
        )


@glc.set_logger(print_logger)
def list_dataset(path_osmose: str, project: str = None) -> None:
    """Prints the available datasets that have been built by the `Dataset.build()` function.

    Parameter
    ---------
    dataset_folder_path: str
        The path to the directory containing the project/datasets.
    project: str
        Name of the project folder containing the datasets.
    """
    ds_folder = Path(path_osmose, project)

    dataset_list = [
        directory
        for directory in ds_folder.iterdir()
        if ds_folder.joinpath(directory).is_dir()
    ]

    dataset_list = sorted(
        dataset_list,
        key=lambda path: str(path).lower(),
    )  # case insensitive alphabetical sorting of datasets

    list_built_dataset = []
    list_unknown_dataset = []

    for dataset_directory in dataset_list:
        dataset_directory = ds_folder.joinpath(dataset_directory)
        if os.access(dataset_directory, os.R_OK):
            metadata_path = next(
                dataset_directory.joinpath(OSMOSE_PATH.raw_audio).rglob("metadata.csv"),
                None,
            )
            timestamp_path = next(
                dataset_directory.joinpath(OSMOSE_PATH.raw_audio).rglob(
                    "timestamp.csv",
                ),
                None,
            )

            if (
                metadata_path
                and metadata_path.exists()
                and timestamp_path
                and timestamp_path.exists()
                and not dataset_directory.joinpath(
                    OSMOSE_PATH.raw_audio,
                    "original",
                ).exists()
            ):
                list_built_dataset.append(dataset_directory)

        else:
            list_unknown_dataset.append(dataset_directory)

    if list_built_dataset:
        built_formatted = "\n".join(
            [f"  - {dataset.name}" for dataset in list_built_dataset],
        )
        glc.logger.info(
            f"""List of the built datasets under {ds_folder}:\n\n{built_formatted}""",
        )
    else:
        glc.logger.info(f"No dataset found under '{ds_folder}'")

    if list_unknown_dataset:
        unreachable_formatted = "\n".join(
            [f"  - {dataset.name}" for dataset in list_unknown_dataset],
        )
        glc.logger.info(
            f"""List of unreachable datasets (probably due to insufficient permissions):\n\n{unreachable_formatted}""",
        )


@glc.set_logger(print_logger)
def list_aplose(path_osmose: str, project: str = ""):
    """Prints the available APLOSE datasets containing result files (result and task_status).

    Parameter
    ---------
    dataset_folder_path: str
        The path to the directory containing the project / datasets.
    project: str
        Name of the project folder containing the datasets.
    """
    ds_folder = Path(path_osmose, project)

    dataset_list = [
        directory
        for directory in ds_folder.iterdir()
        if ds_folder.joinpath(directory).is_dir()
    ]

    dataset_list = sorted(
        dataset_list,
        key=lambda path: str(path).lower(),
    )  # case insensitive alphabetical sorting of datasets

    list_built_dataset = []
    list_unknown_dataset = []
    list_aplose_result = []
    list_aplose_task_status = []

    for dataset_directory in dataset_list:
        aplose_path = Path(dataset_directory, OSMOSE_PATH.aplose)

        result_path = next(
            iter(glob.glob(os.path.join(aplose_path, "**_results**.csv"))),
            None,
        )

        task_status_path = (
            result_path.replace("results", "task_status") if result_path else None
        )

        if os.access(dataset_directory, os.R_OK):
            if (
                result_path
                and Path(result_path).exists()
                and task_status_path
                and Path(task_status_path).exists()
            ):
                list_built_dataset.append(dataset_directory)
                list_aplose_result.append(Path(result_path))
                list_aplose_task_status.append(Path(task_status_path))
        else:
            list_unknown_dataset.append(dataset_directory)

    if list_built_dataset:
        aplose_formatted = "\n".join(
            [
                f"  - {dataset.name}\n\tresult file: {r.name}\n\ttask status file: {ts.name}"
                for dataset, r, ts in zip(
                    list_built_dataset,
                    list_aplose_result,
                    list_aplose_task_status,
                )
            ],
        )
        glc.logger.info(
            f"""List of the datasets with APLOSE result files under {ds_folder}:\n\n{aplose_formatted}""",
        )
    else:
        glc.logger.info(
            f"No dataset with APLOSE result files found under '{ds_folder}'",
        )

    if list_unknown_dataset:
        unreachable_formatted = "\n".join(
            [f"  - {dataset.name}" for dataset in list_unknown_dataset],
        )
        glc.logger.info(
            f"""List of unreachable datasets (probably due to insufficient permissions):\n\n{unreachable_formatted}""",
        )


def read_config(raw_config: Union[str, dict, Path]) -> NamedTuple:
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


def read_header(file: str) -> Tuple[int, float, int, int, int]:
    """Read the first bytes of a wav/flac file and extract its characteristics.
    At the very least, only the first 44 bytes are read. If the `data` chunk is not right after the header chunk,
    the subsequent chunks will be read until the `data` chunk is found. If there is no `data` chunk, all the file will be read.
    Parameter
    ---------
    file: str
        The absolute path of the audio file whose header will be read.

    Returns:
    -------
    samplerate : `int`
        The number of samples in one frame.
    frames : `float`
        The number of frames, corresponding to the file duration in seconds.
    channels : `int`
        The number of audio channels.
    sampwidth : `int`
        The sample width.

    Note:
    ----
    When there is no `data` chunk, the `frames` value will fall back on the size written in the header. This can be incorrect,
    if the file has been corrupted or the writing process has been interrupted before completion.

    """
    with open(file, "rb") as fh:
        header = fh.read(4)

        if header == b"RIFF":
            # WAV file processing
            _, size, wave_format = struct.unpack("<4sI4s", header + fh.read(8))

            while True:
                chunk_header = fh.read(8)
                if len(chunk_header) < 8:
                    break  # Reached the end of the file or a corrupted file

                subchunkid, subchunk_size = struct.unpack("<4sI", chunk_header)

                if subchunkid == b"fmt ":
                    # Process the fmt chunk
                    fmt_chunk_data = fh.read(subchunk_size)
                    _, channels, samplerate, _, _, sampwidth = struct.unpack(
                        "<HHIIHH",
                        fmt_chunk_data[:16],
                    )
                    break
                fh.seek(subchunk_size, 1)

            chunkOffset = fh.tell()
            found_data = False
            while chunkOffset < size and not found_data:
                fh.seek(chunkOffset)
                subchunk2id, subchunk2size = struct.unpack("<4sI", fh.read(8))
                if subchunk2id == b"data":
                    found_data = True

                chunkOffset = chunkOffset + subchunk2size + 8

            if not found_data:
                glc.logger.warning(
                    "No data chunk found while reading the header. Will fallback on the header size.",
                )
                subchunk2size = size - 36

            sampwidth = (sampwidth + 7) // 8
            framesize = channels * sampwidth
            frames = subchunk2size / framesize

            return samplerate, frames, sampwidth, channels, size

        if header == b"fLaC":
            # FLAC file processing
            is_last = False
            while not is_last:
                block_header = fh.read(4)
                block_type = block_header[0] & 0x7F
                is_last = (block_header[0] & 0x80) != 0
                block_size = struct.unpack(">I", b"\x00" + block_header[1:])[0]

                if block_type == 0:  # STREAMINFO block
                    block_data = fh.read(block_size)
                    samplerate = (
                        struct.unpack(">I", b"\x00" + block_data[10:13])[0] >> 4
                    )
                    channels = ((block_data[12] & 0x0E) >> 1) + 1
                    sampwidth = (
                        ((block_data[12] & 0x01) << 4) | ((block_data[13] & 0xF0) >> 4)
                    ) + 1
                    size = struct.unpack(">Q", b"\x00" * 4 + block_data[14:18])[0]
                    frames = size / samplerate

                    return samplerate, frames, sampwidth, channels, size
                    break
                fh.seek(block_size, 1)  # Skip this block
        else:
            raise ValueError("Unsupported file format")


def safe_read(
    file_path: str,
    *,
    nan: float = 0.0,
    posinf: any = None,
    neginf: any = None,
) -> Tuple[np.ndarray, int]:
    """Open a file using Soundfile and clean up the data to be used safely
    Currently, only checks for `NaN`, `inf` and `-inf` presence. The default behavior is the same as `np.nan_to_num`:
    `NaNs` are transformed into 0.0, `inf` and `-inf` are transformed into the maximum and minimum values of their dtype.

    Parameters
    ----------
        file_path: `str`
            The path to the audio file to safely read.
        nan: `float`, optional, keyword_only
            The value that will replace `NaNs`. Default is 0.0
        posinf: `any`, optional, keyword_only
            The value that will replace `inf`. Default behavior is the maximum value of the data type.
        neginf: `any`, optional, keyword_only
            The value that will replace `-inf`. Default behavior is the minimum value of the data type.

    Returns
    -------
        audio_data: `NDArray`
            The cleaned audio data as a numpy array.
        sample_rate: `int`
            The sample rate of the data.

    """
    audio_data, sample_rate = sf.read(file_path)

    nan_nb = sum(np.isnan(audio_data))
    if hasattr(nan_nb, "__iter__"):
        nan_nb = sum(nan_nb)  # Double sum to account for multiple channels

    if nan_nb > 0:
        glc.logger.warning(
            f"{nan_nb} NaN detected in file {Path(file_path).name}. They will be replaced with {nan}.",
        )

    np.nan_to_num(audio_data, copy=False, nan=nan, posinf=posinf, neginf=neginf)

    return audio_data, sample_rate


def check_n_files(
    file_list: list,
    n: int,
    *,
    output_path: str = None,
    auto_normalization: bool = False,
) -> bool:
    """Check n files at random for anomalies and may normalize them.
    Currently, check if the data for wav in PCM float format are between -1.0 and 1.0. If the number of files that
    fail the test is higher than the threshold (which is 10% of n by default, with an absolute minimum of 1), all the
    dataset will be normalized and written in another file.

    Parameters
    ----------
        file_list: `list`
            The list of files to be evaluated. It must be equal or longer than n.
        n: `int`
            The number of files to evaluate. To lower resource consumption, it is advised to check only a subset of the dataset.
            10 files taken at random should provide an acceptable idea of the whole dataset.
        output_path: `str`, optional, keyword-only
            The path to the folder where the normalized files will be written. If auto_normalization is set to True, then
            it must have a value.
        auto_normalization: `bool`, optional, keyword_only
            Whether the normalization should proceed automatically or not if the threshold is reached. As a safeguard, the default is False.

    Returns
    -------
        normalized: `bool`
            Indicates whether or not the dataset has been normalized.

    """
    n = min(n, len(file_list))

    # if "float" in str(sf.info(file_list[0])): # to understand
    bad_files = []
    glc.logger.debug(
        "Testing whether samples are within [-1,1] for the following audio files:",
    )
    for audio_file in random.sample(file_list, n):
        data, sr = safe_read(audio_file)
        if not (np.max(data) <= 1.0 and np.min(data) >= -1.0):
            bad_files.append(audio_file)
            glc.logger.warning(f"- {audio_file.name} -> FAILED")
        else:
            glc.logger.debug(f"- {audio_file.name} -> PASSED")

    return len(bad_files)


def get_files(path, extensions):
    all_files = []
    for ext in extensions:
        all_files.extend(Path(path).glob(ext))
    return all_files


# TO DO : function not optimized in case you use it in a for loop , because it will reload .csv for each audiofile , should
# be able to take as input the already loaded timestamps
def get_timestamp_of_audio_file(path_timestamp_file: Path, audio_file_name: str) -> str:
    timestamps = pd.read_csv(path_timestamp_file)
    # get timestamp of the audio file
    return str(
        timestamps["timestamp"][timestamps["filename"] == audio_file_name].values[0],
    )


def t_rounder(t: pd.Timestamp, res: int) -> pd.Timestamp:
    """Rounds a Timestamp according to the user specified resolution : 10s / 1min / 10 min / 1h / 24h

    Parameters
    ----------
        t: pd.Timestamp
            Timestamp to round
        res: 'int'
            integer corresponding to the new resolution in seconds

    Returns
    -------
        t: pd.Timestamp
            rounded Timestamp

    """
    if res == 600:  # 10min
        minute = t.minute
        minute = math.floor(minute / 10) * 10
        t = t.replace(minute=minute, second=0, microsecond=0)
    elif res == 10:  # 10s
        seconde = t.second
        seconde = math.floor(seconde / 10) * 10
        t = t.replace(second=seconde, microsecond=0)
    elif res == 60:  # 1min
        t = t.replace(second=0, microsecond=0)
    elif res == 3600:  # 1h
        t = t.replace(minute=0, second=0, microsecond=0)
    elif res == 86400:  # 24h
        t = t.replace(hour=0, minute=0, second=0, microsecond=0)
    elif res == 3:
        t = t.replace(microsecond=0)
    else:
        raise ValueError(f"res={res}s: Resolution not available")
    return t


@glc.set_logger(print_logger)
def check_available_file_resolution(path_osmose: str, project_ID: str, dataset_ID: str):
    """Lists the file resolution for a given dataset

    Parameters
    ----------
    path_osmose: 'str'
        usually '/home/datawork-osmose/dataset/'

    project_ID: 'str'
        Name of the project, can be '' if the dataset is directly stored under path_osmose

    dataset_ID: 'str'
        Name of the dataset

    Returns
    -------
    The list of the dataset resolutions is being printed and returned as a list of str

    """
    base_path = os.path.join(path_osmose, project_ID, dataset_ID, "data", "audio")
    resolution = os.listdir(base_path)
    resolution_str = "\n".join(resolution)

    message = (
        f"Dataset : {project_ID}/{dataset_ID}\n"
        "Available Resolution (LengthFile_samplerate) :\n"
        f"{resolution_str}"
    )
    glc.logger.info(message)

    return resolution


def extract_config(
    path_osmose: str,
    list_project_ID: List[str],
    list_dataset_ID: List[str],
    out_dir: str,
):
    """Extracts the configuration file of a list of datasets that have been built
    and whose spectrograms have been computed.
    A directory per audio resolution is being created and a directory
    for the spectrogram configuration as well.

    Parameters
    ----------
    path_osmose: 'str'
        usually '/home/datawork-osmose/dataset/'

    list_project_ID: List[str]
        List of the projects, can be '' if datasets are directly stored under path_osmose

    list_dataset_ID: 'str'
        List of the datasets

    out_dir: 'str'
        Directory where the files are exported

    Returns
    -------

    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for project_ID, dataset_ID in zip(list_project_ID, list_dataset_ID):
        dataset_resolution = check_available_file_resolution(
            path_osmose,
            project_ID,
            dataset_ID,
        )

        for dr in dataset_resolution:
            ### audio config files
            path1 = os.path.join(
                path_osmose,
                project_ID,
                dataset_ID,
                "data",
                "audio",
                dr,
            )
            files1 = glob.glob(os.path.join(path1, "**.csv"))

            full_path1 = os.path.join(out_dir, "export_" + dataset_ID, dr)
            if not os.path.exists(full_path1):
                os.makedirs(full_path1)
            [shutil.copy(file, full_path1) for file in files1]

        ### spectro config files
        path2 = os.path.join(
            path_osmose,
            project_ID,
            dataset_ID,
            "processed",
            "spectrogram",
        )
        files2 = []
        for root, dirs, files in os.walk(path2):
            files2.extend(
                [
                    os.path.join(root, file)
                    for file in files
                    if file.lower().endswith(".csv")
                ],
            )

        full_path2 = os.path.join(out_dir, "export_" + dataset_ID, "spectro")
        if not os.path.exists(full_path2):
            os.makedirs(full_path2)
        [shutil.copy(file, full_path2) for file in files2]

    glc.logger.info(f"\nFiles exported to {out_dir}")


def extract_datetime(
    var: str,
    tz: pytz._FixedOffset = None,
    formats=None,
) -> Union[pd.Timestamp, str]:
    """Extracts datetime from filename based on the date format

    Parameters
    ----------
        var: 'str'
            name of the wav file
        tz: pytz._FixedOffset
            timezone info
        formats: 'str'
            The date template in strftime format.
            For example, `2017/02/24` has the template `%Y/%m/%d`
            For more information on strftime template, see https://strftime.org/

    Returns
    -------
        date_obj: pd.Timestamp
            datetime corresponding to the datetime found in var

    """
    if formats is None:
        # add more format if necessary
        formats = [
            r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}",
            r"\d{2}\d{2}\d{2}\d{2}\d{2}\d{2}",
            r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}",
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
            r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}",
            r"\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2}",
            r"\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2}",
        ]
    match = None
    for f in formats:
        match = re.search(f, var)
        if match:
            break
    if match:
        dt_string = match.group()
        if f == r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}":
            dt_format = "%Y-%m-%dT%H-%M-%S"
        elif f == r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}":
            dt_format = "%Y-%m-%d_%H-%M-%S"
        elif f == r"\d{2}\d{2}\d{2}\d{2}\d{2}\d{2}":
            dt_format = "%y%m%d%H%M%S"
        elif f == r"\d{2}\d{2}\d{2}_\d{2}\d{2}\d{2}":
            dt_format = "%y%m%d_%H%M%S"
        elif f == r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}":
            dt_format = "%Y-%m-%d %H:%M:%S"
        elif f == r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}":
            dt_format = "%Y-%m-%dT%H:%M:%S"
        elif f == r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}":
            dt_format = "%Y_%m_%d_%H_%M_%S"
        elif f == r"\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2}":
            dt_format = "%Y_%m_%dT%H_%M_%S"
        elif f == r"\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2}":
            dt_format = "%Y%m%dT%H%M%S"

        date_obj = pd.to_datetime(dt_string, format=dt_format)

        if tz is None:
            return date_obj
        if type(tz) is dt.timezone:
            offset_minutes = tz.utcoffset(None).total_seconds() / 60
            pytz_fixed_offset = pytz.FixedOffset(int(offset_minutes))
            date_obj = pytz_fixed_offset.localize(date_obj)
        else:
            date_obj = tz.localize(date_obj)

        return date_obj
    raise ValueError(f"{var}: No datetime found")


def add_entry_for_APLOSE(path: str, file: str, info: pd.DataFrame):
    """Add entry for APLOSE dataset csv file

    Parameters
    ----------
        path: 'str'
            path to the file
        file: 'str'
            csv file
        info: 'DataFrame'
            info of the entry
            'path' / 'dataset' / 'spectro_duration' / 'dataset_sr' / 'files_type'

    Returns
    -------

    """
    dataset_csv = Path(path, file)

    if dataset_csv.exists():
        meta = pd.read_csv(dataset_csv)
        meta = pd.concat([meta, info], ignore_index=True).sort_values(
            by=["path", "dataset"],
            ascending=True,
        )
        meta.to_csv(dataset_csv, index=False)


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
    glc.logger.debug("Setting OSmOSE permission to the dataset..")

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
    A list of tuples representing the start and stop index of files processed by each batch in the analysis.

    Examples
    --------
    >>> file_indexes_per_batch(10,4)
    [(0, 3), (3, 6), (6, 8), (8, 10)]
    >>> file_indexes_per_batch(1448,10)
    [(0, 145), (145, 290), (290, 435), (435, 580), (580, 725), (725, 870), (870, 1015), (1015, 1160), (1160, 1304), (1304, 1448)]

    """
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
