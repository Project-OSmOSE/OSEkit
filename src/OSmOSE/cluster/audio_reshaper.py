import os
import re
import pandas as pd
from datetime import datetime
from typing import List, Union, Literal
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import soundfile as sf
from librosa import resample

from OSmOSE.utils.core_utils import set_umask, select_audio_file
from OSmOSE.utils.path_utils import make_path
from OSmOSE.config import DPDEFAULT, FPDEFAULT


def reshape(
    input_files: Union[str, list],
    chunk_size: int,
    *,
    file_metadata_path: Union[str, Path] = None,
    timestamp_path: Union[str, Path] = None,
    output_dir_path: Union[str, Path] = None,
    datetime_begin: str = None,
    datetime_end: str = None,
    new_sr: int = -1,
    batch_ind_min: int = 0,
    batch_ind_max: int = -1,
    last_file_behavior: Literal["truncate", "pad", "discard"] = "pad",
    verbose: bool = False,
    overwrite: bool = True,
) -> List[str]:
    """Reshape all audio files in the folder to be of the specified duration from datetime_begin to datetime_end. If chunk_size is superior to the base duration of the files, they will be fused according to their order in the timestamp.csv file in the same folder.

    Parameters:
    -----------
        input_files: `str` or `list(str)`
            Either the directory containing the audio files and the timestamp.csv file, in which case all audio files will be considered,
            OR a list of audio files all located in the same directory alongside a timestamp.csv, in which case only they will be used.

        chunk_size: `int`
            The target duration for all the reshaped files, in seconds.

        file_metadata_path: `str` or `Path`
            The path to the file_metadata.csv file

        timestamp_path: `str` or `Path`
            The path to the timestamp.csv file

        datetime_begin: `str`
            A datetime formatted string corresponding to the beginning of the reshaped audio files

        datetime_end: `str`
            A datetime formatted string corresponding to the end of the reshaped audio files

        output_dir_path: `str`, optional, keyword-only
            The directory where the newly created audio files will be created. If none is provided,
            it will be the same as the input directory. This is not recommended.

        batch_ind_min: `int`, optional, keyword-only
            The first file of the list to be processed. Default is 0.

        batch_ind_max: `int`, optional, keyword-only
            The last file of the list to be processed. Default is -1, meaning the entire list is processed.

        last_file_behavior: `{"truncate","pad","discard"}, optional, keyword-only
            Tells the reshaper what to do with if the last data of the last file is too small to fill a whole file.
            This parameter is only active if `batch_ind_max` is `-1`
            - `truncate` creates a truncated file with the remaining data, which will have a different duration than the others.
            - `pad` creates a file of the same duration than the others, where the missing data is filled with 0.
            - `discard` ignores the remaining data. The last seconds/minutes/hours of audio will be lost in the reshaping.
        The default method is `pad`.

        verbose: `bool`, optional, keyword-only
            Whether to display informative messages or not.

        overwrite: `bool`, optional, keyword-only
            Deletes the content of `output_dir_path` before writing the results. If it is implicitly the `input_files` directory,
            nothing happens. WARNING: If `output_dir_path` is explicitly set to be the same as `input_files`, then it will be overwritten!

    Returns:
    --------
        The list of the path of newly created audio files.
    """
    set_umask()

    # datetimes check
    if not datetime_begin or not datetime_end:
        raise ValueError(
            "The begin and end datetimes must be a valid date and time string format. Please consider using the following format: 'YYYY-MM-DDTHH:MM:SS±HHMM'"
        )

    regex = r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[+-]\d{4}$"

    if not re.match(regex, datetime_begin):
        raise ValueError(
            f"The datetime string '{datetime_begin}' is not in the valid format 'YYYY-MM-DDTHH:MM:SS±HHMM'."
        )

    try:
        pd.Timestamp(datetime_begin)
    except ValueError as e:
        raise ValueError(
            f"Failed to parse datetime string '{datetime_begin}': {e}. Please provide a valid format."
        )

    if not re.match(regex, datetime_end):
        raise ValueError(
            f"The datetime string '{datetime_end}' is not in the valid format 'YYYY-MM-DDTHH:MM:SS±HHMM'."
        )

    try:
        pd.Timestamp(datetime_end)
    except ValueError as e:
        raise ValueError(
            f"Failed to parse datetime string '{datetime_end}': {e}. Please provide a valid format."
        )

    # last_file_behavior check
    if last_file_behavior not in ["truncate", "pad", "discard"]:
        raise ValueError(
            f"Bad value {last_file_behavior} for last_file_behavior parameter. Must be one of truncate, pad or discard."
        )

    # input_files / input_dir_path checks
    if isinstance(input_files, list):
        input_dir_path = Path(input_files[0]).parent
        files = [Path(file) for file in input_files]
        if verbose:
            print(f"Input directory detected as {input_dir_path}")
    else:
        input_dir_path = Path(input_files)

    if not input_dir_path.is_dir():
        raise ValueError(
            f"The input files must either be a valid folder path or a list of file path, not {str(input_dir_path)}."
        )

    if not input_dir_path.joinpath("timestamp.csv").exists() and (
        not timestamp_path or not timestamp_path.exists()
    ):
        raise FileNotFoundError(
            f"The timestamp.csv file must be present in the directory {input_dir_path} and correspond to the audio files in the same location, or be specified in the argument."
        )

    # output_dir_path check
    if not output_dir_path:
        print("No output directory provided. Will use the input directory instead.")
        output_dir_path = input_dir_path
        if overwrite:
            print(
                "Cannot overwrite input directory when the output directory is implicit! Choose a different output directory instead."
            )
    output_dir_path = Path(output_dir_path)
    make_path(output_dir_path, mode=DPDEFAULT)

    # read timestamp.csv
    input_timestamp = pd.read_csv(
        timestamp_path
        if timestamp_path and timestamp_path.exists()
        else input_dir_path.joinpath("timestamp.csv")
    )

    # read file_metadata.csv
    if file_metadata_path and file_metadata_path.exists():
        file_metadata = pd.read_csv(file_metadata_path, parse_dates=["timestamp"])
    else:
        file_metadata = pd.read_csv(
            input_dir_path.joinpath("file_metadata.csv"), parse_dates=["timestamp"]
        )

    print(type(file_metadata["timestamp"].iloc[0]))

    # DataFrame creation with the new audio files to be created and the original audio files needed to create each of those new audio
    df_file = select_audio_file(
        file_metadata=file_metadata,
        dt_begin=pd.Timestamp(datetime_begin),
        dt_end=pd.Timestamp(datetime_end),
        duration=chunk_size,
        last_file_behavior=last_file_behavior,
    )

    # selection of a subset (or batch) of the DataFrame according to provided indexes
    df_file_batch = df_file[
        batch_ind_min : (batch_ind_max + 1 if batch_ind_max > 0 else len(df_file))
    ].reset_index(drop=True)

    result = []
    timestamp_list = []
    list_seg_name = []
    list_seg_timestamp = []

    for i in range(len(df_file_batch)):

        if verbose:
            print(f"New file: {df_file_batch['filename'][i]}")
            print(f"Selected original files: {df_file_batch['selection'][i]}")

        if any(df_file_batch["selection"][i]):

            files = df_file_batch["selection"][i]

            # first we resample if necessary then we concatenate data necessary to construct the new audio file
            audio_data = np.empty(shape=[0])
            for f in files:
                input_file = input_dir_path.joinpath(f)
                fmt = input_file.suffixes[-1].replace(".", "")

                # getting file information and data
                with sf.SoundFile(input_file) as audio_file:
                    sample_rate = audio_file.samplerate
                    subtype = audio_file.subtype
                    audio_data_slice = audio_file.read()

                if new_sr == -1:
                    new_sr = sample_rate

                # if the file sample rate is different from the target sample rate, we resample the audio data
                elif new_sr != sample_rate:
                    audio_data_slice = resample(
                        audio_data_slice, orig_sr=sample_rate, target_sr=new_sr
                    )
                    sample_rate = new_sr

                audio_data = np.concatenate((audio_data, audio_data_slice))

            # now we check if the data begins before / after the begin datetime of the new audio file
            # case 1 : it begins after, then we need to pad the begining of new file with zeros
            if df_file_batch["dt_start"][i] < min(
                df_file_batch["selection_datetime_begin"][i]
            ):
                offset_beginning = (
                    min(df_file_batch["selection_datetime_begin"][i])
                    - df_file_batch["dt_start"][i]
                ).total_seconds()
                fill = np.zeros(int(offset_beginning * sample_rate))
                audio_data = np.concatenate((fill, audio_data))[
                    : chunk_size * sample_rate
                ]
            # case 2  : it begins before, then we need to cut out the useless data anterior to begining of new file
            # note : the case where it begins right on the same time is taken care of here
            elif df_file_batch["dt_start"][i] >= min(
                df_file_batch["selection_datetime_begin"][i]
            ):
                offset_beginning = (
                    df_file_batch["dt_start"][i]
                    - min(df_file_batch["selection_datetime_begin"][i])
                ).total_seconds()
                audio_data = audio_data[int(offset_beginning * sample_rate) :]

                # case where the audio file ends right on the same datetime than the begin datetime, then audio_data is empty
                if len(audio_data) == 0:
                    continue

            # if audio_data is longer than desired duration we cut out the end to get the desired duration
            if len(audio_data) > chunk_size * sample_rate:
                audio_data = audio_data[: chunk_size * sample_rate]

            # if it is the last batch, the last original audio might be too short, then we pad with zeros if "pad" is selected
            # OR we shorten it because the duration if the last new audio file is < than the duration
            if (
                len(audio_data) < chunk_size * sample_rate
                or (
                    df_file_batch["dt_end"][i] - df_file_batch["dt_start"][i]
                ).total_seconds()
                < chunk_size
            ):

                dur = (
                    df_file_batch["dt_end"][i] - df_file_batch["dt_start"][i]
                ).total_seconds()
                audio_data = audio_data[: int(dur * sample_rate)]

                if last_file_behavior == "pad":
                    offset_end = (chunk_size * sample_rate) - len(audio_data)
                    fill = np.zeros(int(offset_end))
                    audio_data = np.concatenate((audio_data, fill))
                elif last_file_behavior == "discard":
                    # todo: check if this works
                    print(f"Last file is discarded as it is shorter than {chunk_size}s")
                    return
                elif last_file_behavior == "truncate":
                    # todo: check if this works
                    print("truncate l214")
                    continue

            # at this point audio_data should have the desired size, then we proceed and write the new wav file
            outfilename = output_dir_path.joinpath(
                f"{df_file_batch['filename'][i].replace('-','_').replace(':','_').replace('.','_').replace('+','_')}.{fmt}"
            )
            result.append(outfilename.name)

            list_seg_name.append(outfilename.name)
            list_seg_timestamp.append(
                datetime.strftime(
                    df_file_batch["dt_start"][i], "%Y-%m-%dT%H:%M:%S.%f%z"
                )
            )

            timestamp_list.append(
                datetime.strftime(
                    df_file_batch["dt_start"][i], "%Y-%m-%dT%H:%M:%S.%f%z"
                )
            )

            sf.write(outfilename, audio_data, sample_rate, format=fmt, subtype=subtype)
            os.chmod(outfilename, mode=FPDEFAULT)

            if verbose:
                print(
                    f"{outfilename} written! File is {(len(audio_data)/sample_rate)} seconds long."
                )

    # writing infos to timestamp_*.csv
    input_timestamp = pd.DataFrame({"filename": result, "timestamp": timestamp_list})
    input_timestamp.sort_values(by=["timestamp"], inplace=True)
    input_timestamp.drop_duplicates().to_csv(
        output_dir_path.joinpath(f"timestamp_{batch_ind_min}.csv"),
        index=False,
        na_rep="NaN",
    )
    os.chmod(output_dir_path.joinpath(f"timestamp_{batch_ind_min}.csv"), mode=FPDEFAULT)


if __name__ == "__main__":
    parser = ArgumentParser()
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--input-files",
        "-i",
        help="The files to be reshaped, as either the path to a directory containing audio files and a timestamp.csv or a list of filenames all in the same directory alongside a timestamp.csv.",
    )
    required.add_argument(
        "--file-metadata-path",
        type=str,
        help="Path to file metadata",
    )
    required.add_argument(
        "--datetime-begin",
        type=str,
        help="Datetime string to begin the reshape at",
    )
    required.add_argument(
        "--datetime-end",
        type=str,
        help="Datetime string to end the reshape at",
    )
    required.add_argument(
        "--chunk-size",
        "-s",
        type=int,
        help="The time in seconds of the reshaped files.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="The path to the directory to write reshaped files. Default is same as --input-files directory.",
    )
    parser.add_argument(
        "--batch-ind-min",
        "-min",
        type=int,
        default=0,
        help="The first file of the list to be processed. Default is 0.",
    )
    parser.add_argument(
        "--batch-ind-max",
        "-max",
        type=int,
        default=-1,
        help="The last file of the list to be processed. Default is -1, meaning the entire list is processed.",
    )
    parser.add_argument(
        "--max-delta-interval",
        type=int,
        default=5,
        help="The maximum number of second allowed for a delta between two timestamp_list to still be considered the same. Default is 5s up and down.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Whether the script prints informative messages. Default is true.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="If set, deletes all content in --output-dir before writing the output. Default false, deactivated if the --output-dir is the same as --input-file dir.",
    )
    parser.add_argument(
        "--last-file-behavior",
        default="pad",
        help="Tells the program what to do with the remaining data that are shorter than the chunk size. Possible arguments are pad (the default), which pads with silence until the last file has the same length as the others; truncate to create a shorter file with only the leftover data; discard to not do anything with the last data and throw it away.",
    )
    parser.add_argument(
        "--timestamp-path", default=None, help="Path to the original timestamp file."
    )
    parser.add_argument(
        "--new-sr",
        type=int,
        default=0,
        help="Sampling rate",
    )

    args = parser.parse_args()

    input_files = (
        args.input_files.split(" ")
        if not Path(args.input_files).is_dir()
        else args.input_files
    )

    print("Parameters :", args)

    files = reshape(
        chunk_size=args.chunk_size,
        file_metadata_path=args.file_metadata_path,
        input_files=input_files,
        output_dir_path=args.output_dir,
        new_sr=args.new_sr,
        datetime_begin=args.datetime_begin,
        datetime_end=args.datetime_end,
        batch_ind_min=args.batch_ind_min,
        batch_ind_max=args.batch_ind_max,
        timestamp_path=Path(args.timestamp_path),
        last_file_behavior=args.last_file_behavior,
        verbose=args.verbose,
        overwrite=args.overwrite,
    )
