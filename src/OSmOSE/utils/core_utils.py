import os
from warnings import warn
from pathlib import Path
from importlib.resources import as_file
import random
import shutil
import struct
from collections import namedtuple
import sys
from typing import Union, NamedTuple, Tuple, List
import pytz

import pandas as pd

import json

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import soundfile as sf
import numpy as np
from OSmOSE.config import *


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
        The path to the directory containing the datasets."""

    ds_folder = Path(datasets_folder_path)

    dataset_list = [
        directory
        for directory in ds_folder.iterdir()
        if ds_folder.joinpath(directory).is_dir()
    ]

    dataset_list = sorted(
        dataset_list, key=lambda path: str(path).lower()
    )  # case insensitive alphabetical sorting of datasets

    list_not_built_datasets = []
    list_unknown_datasets = []

    for dataset_directory in dataset_list:
        dataset_directory = ds_folder.joinpath(dataset_directory)
        if os.access(dataset_directory, os.R_OK):
            metadata_path = next(
                dataset_directory.joinpath(OSMOSE_PATH.raw_audio).rglob("metadata.csv"),
                None,
            )
            timestamp_path = next(
                dataset_directory.joinpath(OSMOSE_PATH.raw_audio).rglob(
                    "timestamp.csv"
                ),
                None,
            )

            if not (
                metadata_path
                and metadata_path.exists()
                and timestamp_path
                and timestamp_path.exists()
                and not dataset_directory.joinpath(
                    OSMOSE_PATH.raw_audio, "original"
                ).exists()
            ):
                list_not_built_datasets.append(dataset_directory)
        else:
            list_unknown_datasets.append(dataset_directory)

    not_built_formatted = "\n".join(
        [f"  - {dataset.name}" for dataset in list_not_built_datasets]
    )
    print(f"""List of the datasets that aren't built yet:\n{not_built_formatted}""")

    unreachable_formatted = "\n".join(
        [f"  - {dataset.name}" for dataset in list_unknown_datasets]
    )
    print(
        f"""List of unreachable datasets (probably due to insufficient permissions) :\n{unreachable_formatted}"""
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
        Raised if the raw_config file is in YAML format"""

    match raw_config:
        case Path():
            with as_file(raw_config) as input_config:
                raw_config = input_config

        case str():
            if not Path(raw_config).is_file:
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
            match Path(raw_config).suffix:
                case ".toml":
                    raw_config = tomllib.load(input_config)
                case ".json":
                    raw_config = json.load(input_config)
                case ".yaml":
                    raise NotImplementedError(
                        "YAML support will eventually get there (unfortunately)"
                    )
                case _:
                    raise FileNotFoundError(
                        f"The provided configuration file extension ({Path(raw_config).suffix} is not a valid extension. Please use .toml or .json files."
                    )

    return raw_config


# def convert(template: NamedTuple, dictionary: dict) -> NamedTuple:
#     """Convert a dictionary in a Named Tuple"""
#     for key, value in dictionary.items():
#         if isinstance(value, dict):
#             dictionary[key] = convert(template, value)
#     return template(**dictionary)


def read_header(file: str) -> Tuple[int, float, int, int, int]:
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

        # if (size - 72) > subchunk2size:
        #     print(
        #         f"Warning : the size indicated in the header is not the same as the actual file size. This might mean that the file is truncated or otherwise corrupt.\
        #         \nSupposed size: {size} bytes \nActual size: {subchunk2size} bytes."
        #     )

        return samplerate, frames, sampwidth, channels, size


def safe_read(
    file_path: str, *, nan: float = 0.0, posinf: any = None, neginf: any = None
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
            The sample rate of the data."""
    audio_data, sample_rate = sf.read(file_path)

    nan_nb = sum(np.isnan(audio_data))
    if hasattr(nan_nb, "__iter__"):
        nan_nb = sum(nan_nb)  # Double sum to account for multiple channels

    if nan_nb > 0:
        warn(
            f"{nan_nb} NaN detected in file {Path(file_path).name}. They will be replaced with {nan}."
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

    if n > len(file_list):
        n = len(file_list)

    # if "float" in str(sf.info(file_list[0])): # to understand
    bad_files = []
    print(f"Testing whether samples are within [-1,1] for the following audio files:")
    for audio_file in random.sample(file_list, n):
        data, sr = safe_read(audio_file)
        if not (np.max(data) <= 1.0 and np.min(data) >= -1.0):
            bad_files.append(audio_file)
            print(f"- {audio_file.name} -> FAILED")
        else:
            print(f"- {audio_file.name} -> PASSED")
    print(f"\n")

    return len(bad_files)


def set_umask():
    os.umask(0o002)


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
        timestamps["timestamp"][timestamps["filename"] == audio_file_name].values[0]
    )


def t_rounder(t: pd.datetime, res: int) -> pd.datetime:
    """Rounds a Timestamp according to the user specified resolution : 10s / 1min / 10 min / 1h / 24h

    Parameters
    -------
        t: pd.datetime
            Timestamp to round
        res: 'int'
            integer corresponding to the new resolution in seconds

    Returns
    -------
        t: pd.datetime
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


def list_dataset(path_osmose: str, campaign_folder: str = None):
    """Lists all the datasets available, i.e. built datasets, under given path.
    A dataset is defined as built if it contains the following folders : 'data', 'log', 'processed', 'other'.
    The function check in the immediate directories of the given path and one level deeper
    in case a campaign folder is present, i.e. a folder that contains several datasets.
    If user only wants to print the datasets under a specific campaign only, then 'campaign_folder' argument
    should be provided and the function will only check for dataset structure under this folder.

    Parameter
    ---------
    path_osmose: 'str'
        usually '/home/datawork-osmose/dataset/'

    path_osmose: 'str'
        campaign name to check for dataset strucure, if provided

    Returns
    -------
    The list of the datasets and theirs associated campaign is being printed
    In case of denied read permissions, the list of datasets with no permission will be printed with its owner
    """

    dataset, denied_dataset, campaign, owner = [], [], [], []
    if campaign_folder is not None:
        path_osmose = os.path.join(path_osmose, campaign_folder)

    # Iterate over immediate subdirectories of the root directory
    for entry in os.scandir(path_osmose):
        if entry.is_dir():
            try:
                subdirectories = set(os.listdir(entry.path))
                if all(
                    subdir in subdirectories
                    for subdir in ["data", "log", "processed", "other"]
                ):
                    dataset.append(os.path.basename(entry.path))
                    if campaign_folder is not None:
                        campaign.append(campaign_folder)
                    else:
                        campaign.append("/")

                if campaign_folder is None:
                    # If the immediate subdirectory doesn't contain the required subdirectories,
                    # check one level deeper (campaigns directories)
                    for sub_entry in os.scandir(entry.path):
                        if sub_entry.is_dir():
                            try:
                                sub_subdirectories = set(os.listdir(sub_entry.path))
                                if all(
                                    subdir in sub_subdirectories
                                    for subdir in ["data", "log", "processed", "other"]
                                ):
                                    dataset.append(os.path.basename(sub_entry.path))
                                    campaign.append(Path(sub_entry.path).parts[-2])
                            except PermissionError:
                                denied_dataset.append(sub_entry.path)
                                owner_id = os.stat(sub_entry.path).st_uid
                                owner_name = pwd.getpwuid(owner_id).pw_name
                                owner.append(owner_name)

            except PermissionError:
                denied_dataset.append(entry.path)
                owner_id = os.stat(entry.path).st_uid
                owner_name = pwd.getpwuid(owner_id).pw_name
                owner.append(owner_name)
                continue

    if dataset != []:
        print("Built datasets:")
        combined = list(zip(dataset, campaign))
        combined.sort()
        dataset, campaign = zip(*combined)
        for cp, ds in zip(campaign, dataset):
            print(f"  - campaign: {cp} -- dataset: {ds}")
    else:
        raise ValueError(f"No dataset available under {path_osmose}")

    if denied_dataset != []:
        print("\nNo permission to read:")
        combined = list(zip(denied_dataset, owner))
        combined.sort()
        denied_dataset, owner = zip(*combined)
        for ds, ow in zip(denied_dataset, owner):
            print(f"  - {ds} -- Owner ID: {ow}")


def list_aplose(path_osmose: str):
    """Checks whether an APLOSE annotation file is stored in a dataset folder.
    The search for an aPLOSE file is performed under one of the following directory structures:
        - campaign_name/dataset_name/processed/aplose/csv_result_file
        - dataset_name/processed/aplose/csv_result_file
    If either structure is found, it will be printed.

    Parameter
    ---------
    path_osmose: 'str'
        usually '/home/datawork-osmose/dataset/'

    Returns
    -------
    The list of the datasets and theirs associated campaign and APLOSE annotation file is being printed
    In case of denied read permissions, the list of datasets with no permission will be printed with its owner
    """

    campaign_path, aplose_path, aplose_file, no_permission, owner = [], [], [], [], []

    # Iterate over directories directly under path_osmose_dataset
    for dir_name in os.listdir(path_osmose):
        dir_path = os.path.join(path_osmose, dir_name)

        # Check if the item is a directory
        if os.path.isdir(dir_path):
            try:
                processed_dir_path = os.path.join(dir_path, "processed")

                # Check if 'processed' directory exists
                if os.path.isdir(processed_dir_path):
                    aplose_dir_path = os.path.join(processed_dir_path, "aplose")
                    if os.path.exists(aplose_dir_path):
                        csv_path = glob.glob(
                            os.path.join(aplose_dir_path, "**_results**.csv")
                        )
                        for csv in csv_path:
                            campaign_path.append("/")
                            aplose_path.append(dir_name)
                            aplose_file.append(os.path.basename(csv))
                else:
                    # If 'processed' directory doesn't exist, look one level deeper
                    for subdir_name in os.listdir(dir_path):
                        subdir_path = os.path.join(dir_path, subdir_name)
                        processed_subdir_path = os.path.join(subdir_path, "processed")

                        # Check if the item is a directory
                        if os.path.isdir(processed_subdir_path):
                            try:
                                # Check if 'processed' directory exists
                                if os.path.isdir(processed_subdir_path):
                                    aplose_subdir_path = os.path.join(
                                        processed_subdir_path, "aplose"
                                    )
                                    if os.path.exists(aplose_subdir_path):
                                        csv_path = glob.glob(
                                            os.path.join(
                                                aplose_dir_path, "**_results**.csv"
                                            )
                                        )
                                        for csv in csv_path:
                                            campaign_path.append(
                                                Path(aplose_subdir_path).parts[-4]
                                            )
                                            aplose_path.append(subdir_name)
                                            aplose_file.append(os.path.basename(csv))

                            except PermissionError:
                                owner_id = os.stat(subdir_path).st_uid
                                owner_name = pwd.getpwuid(owner_id).pw_name
                                owner.append(owner_name)
                                no_permission.append(subdir_path)

            except PermissionError:
                owner_id = os.stat(dir_path).st_uid
                owner_name = pwd.getpwuid(owner_id).pw_name
                owner.append(owner_name)
                no_permission.append(dir_path)

    # Print datasets with no read permission
    print("\nNo permission to read:")
    for ds, ow in zip(no_permission, owner):
        print(f"  - {ds} -- Owner ID: {ow}")

    # Print campaigns -- datasets -- APLOSE files
    print("\nAvailable APLOSE annotation files:")
    for campaign, ds, f in zip(campaign_path, aplose_path, aplose_file):
        print(f"  - campaign: {campaign} -- dataset: {ds} -- file: {f}")


def check_available_file_resolution(
    path_osmose: str, campaign_ID: str, dataset_ID: str
):
    """Lists the file resolution for a given dataset

    Parameters
    ---------
    path_osmose: 'str'
        usually '/home/datawork-osmose/dataset/'

    campaign_ID: 'str'
        Name of the campaign, can be '' if the dataset is directly stored under path_osmose

    dataset_ID: 'str'
        Name of the dataset

    Returns
    -------
    The list of the dataset resolutions is being printed and returned as a list of str
    """

    base_path = os.path.join(path_osmose, campaign_ID, dataset_ID, "data", "audio")
    resolution = os.listdir(base_path)

    print(f"\nDataset : {campaign_ID}/{dataset_ID}")
    print("Available Resolution (LengthFile_samplerate) :", end="\n")

    [print(f" {r}") for r in resolution]

    return resolution


def extract_config(
    path_osmose: str,
    list_campaign_ID: List[str],
    list_dataset_ID: List[str],
    out_dir: str,
):
    """Extracts the configuration file of a list of datasets that have been built
    and whose spectrograms have been computed.
    A directory per audio resolution is being created and a directory
    for the spectrogram configuration as well.

    Parameters
    ---------
    path_osmose: 'str'
        usually '/home/datawork-osmose/dataset/'

    list_campaign_ID: List[str]
        List of the campaigns, can be '' if datasets are directly stored under path_osmose

    list_dataset_ID: 'str'
        List of the datasets

    out_dir: 'str'
        Directory where the files are exported

    Returns
    -------
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for campaign_ID, dataset_ID in zip(list_campaign_ID, list_dataset_ID):

        dataset_resolution = check_available_file_resolution(
            path_osmose, campaign_ID, dataset_ID
        )

        for dr in dataset_resolution:
            ### audio config files
            path1 = os.path.join(
                path_osmose, campaign_ID, dataset_ID, "data", "audio", dr
            )
            files1 = glob.glob(os.path.join(path1, "**.csv"))

            full_path1 = os.path.join(out_dir, "export_" + dataset_ID, dr)
            if not os.path.exists(full_path1):
                os.makedirs(full_path1)
            [shutil.copy(file, full_path1) for file in files1]

        ### spectro config files
        path2 = os.path.join(
            path_osmose, campaign_ID, dataset_ID, "processed", "spectrogram"
        )
        files2 = []
        for root, dirs, files in os.walk(path2):
            files2.extend(
                [
                    os.path.join(root, file)
                    for file in files
                    if file.lower().endswith(".csv")
                ]
            )

        full_path2 = os.path.join(out_dir, "export_" + dataset_ID, "spectro")
        if not os.path.exists(full_path2):
            os.makedirs(full_path2)
        [shutil.copy(file, full_path2) for file in files2]

    print(f"\nFiles exported to {out_dir}")


def extract_datetime(
    var: str, tz: pytz._FixedOffset = None, formats=None
) -> Union[pd.Timestamp, str]:
    """Extracts datetime from filename based on the date format

    Parameters
    -------
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
        date_obj: pd.datetime
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
        elif type(tz) is dt.timezone:
            offset_minutes = tz.utcoffset(None).total_seconds() / 60
            pytz_fixed_offset = pytz.FixedOffset(int(offset_minutes))
            date_obj = pytz_fixed_offset.localize(date_obj)
        else:
            date_obj = tz.localize(date_obj)

        return date_obj
    else:
        raise ValueError(f"{var}: No datetime found")
