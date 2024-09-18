import os
import re
import pandas as pd
from typing import List, Union, Literal
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import soundfile as sf
from librosa import resample

from OSmOSE.utils.core_utils import set_umask
from OSmOSE.utils.path_utils import make_path
from OSmOSE.config import DPDEFAULT, FPDEFAULT
from OSmOSE.utils import get_audio_file


def reshape(
    input_files: Union[str, list],
    segment_size: int,
    *,
    file_metadata_path: Union[str, Path] = None,
    timestamp_path: Union[str, Path] = None,
    output_dir_path: Union[str, Path] = None,
    datetime_begin: str = None,
    datetime_end: str = None,
    new_sr: int = -1,
    batch: int = 0,
    batch_num: int = 1,
    batch_ind_min: int = 0,
    batch_ind_max: int = -1,
    last_file_behavior: Literal["truncate", "pad", "discard"] = "pad",
    concat: bool = True,
    verbose: bool = False,
    overwrite: bool = True,
    trigger: int = 5,
) -> List[str]:
    """
    Reshape all audio files in the folder to be of the specified duration and/or sampling rate.
    The begin and end datetime can be specified as well by the user,
    if not, the begin datetime of the first original audio file and the end datetime of the last audio file will be used
    """

    """
    Reshape all audio files in the folder to be of the specified duration and/or sampling rate.

    Parameters:
    -----------
    input_files : Union[str, list]
        Path to a directory containing audio files.

    segment_size : int
        The desired duration of each audio segment in seconds.

    file_metadata_path : Union[str, Path], optional
        Path to file_metadata.csv file.

    timestamp_path : Union[str, Path], optional
        Path to timestamps.csv file.

    output_dir_path : Union[str, Path], optional
        Directory where the processed audio files will be saved.

    datetime_begin : str, optional
        Start date and time in ISO 8601 format (e.g., 'YYYY-MM-DDTHH:MM:SS'). This defines the beginning of
        the time window for processing. If not specified, the start time of the first original audio file will be used.

    datetime_end : str, optional
        End date and time in ISO 8601 format (e.g., 'YYYY-MM-DDTHH:MM:SS'). This defines the end of the time
        window for processing. If not specified, the end time of the last original audio file will be used.

    new_sr : int, optional
        Desired sampling rate for the output audio files. If set to -1 (default), the original sampling rate
        of the audio files will be retained.

    batch : int, optional
        The current batch number to process when dealing with large datasets. Default is 0, which processes
        all files at once.

    batch_num : int, optional
        The total number of batches to divide the processing into. Default is 1.

    batch_ind_min : int, optional
        The minimum index of the files to include in the current batch. Default is 0.

    batch_ind_max : int, optional
        The maximum index of the files to include in the current batch. Default is -1, which processes all
        files up to the end.

    last_file_behavior : Literal["truncate", "pad", "discard"], optional
        Specifies how to handle the last audio segment if it is shorter than the desired segment size:
        - "truncate": Truncate the segment to the shorter length.
        - "pad": Pad the segment with zeros to match the segment size (default).
        - "discard": Discard the segment entirely.

    concat : bool, optional
        If True, concatenate all original audio data. If False,
        each segment will be saved as a separate file. Default is True.

    verbose : bool, optional
        If True, print detailed information about the processing steps. Default is False.

    overwrite : bool, optional
        If True, overwrite existing files in the output directory with the same name. Default is True.

    trigger : int, optional
        Integer from 0 to 100 to filter out segments with a number of sample inferior to (trigger * spectrogram duration * segment_sample_rate)
    """

    set_umask()
    segment_duration = pd.Timedelta(seconds=segment_size)

    # Validation for trigger
    if not (0 <= trigger <= 100):
        raise ValueError(
            "The 'trigger' parameter must be an integer between 0 and 100."
        )

    # Validate datetimes format
    regex = r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[+-]\d{4}$"
    regex2 = r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$"
    datetime_format_error = (
        "Please use the following format: 'YYYY-MM-DDTHH:MM:SS+/-HHMM'."
    )

    if datetime_begin:
        if not re.match(regex, datetime_begin) and not re.match(regex2, datetime_begin):
            raise ValueError(
                f"Invalid format for datetime_begin. {datetime_format_error}"
            )
        datetime_begin = pd.Timestamp(datetime_begin)

    if datetime_end:
        if not re.match(regex, datetime_end) and not re.match(regex2, datetime_end):
            raise ValueError(
                f"Invalid format for datetime_end. {datetime_format_error}"
            )
        datetime_end = pd.Timestamp(datetime_end)

    # Validate last_file_behavior
    if last_file_behavior not in ["truncate", "pad", "discard"]:
        raise ValueError(
            f"Invalid last_file_behavior: '{last_file_behavior}'. Must be one of 'truncate', 'pad', or 'discard'."
        )

    # Prepare file paths
    if isinstance(input_files, list):
        input_dir_path = Path(input_files[0]).parent
        files = [Path(file) for file in input_files]
    else:
        input_dir_path = Path(input_files)
        files = get_audio_file(file_path=input_dir_path)

    if not input_dir_path.is_dir():
        raise ValueError(
            f"The input files must either be a valid folder path or a list of file path, not {input_dir_path}."
        )

    if not (input_dir_path / "timestamp.csv").exists() and (
        not timestamp_path or not timestamp_path.exists()
    ):
        raise FileNotFoundError(
            f"The timestamp.csv file must be present in the directory {Path(input_dir_path)} and correspond to the audio files in the same location, or be specified in the argument."
        )

    # Output directory
    if not output_dir_path:
        output_dir_path = input_dir_path
        if overwrite:
            raise ValueError(
                "Cannot overwrite input directory when the output directory is implicit!"
            )
    output_dir_path = Path(output_dir_path)
    make_path(output_dir_path, mode=DPDEFAULT)

    # Load timestamp and metadata
    input_timestamp = pd.read_csv(
        timestamp_path
        if timestamp_path and timestamp_path.exists()
        else input_dir_path / "timestamp.csv"
    )

    if file_metadata_path and file_metadata_path.exists():
        file_metadata = pd.read_csv(file_metadata_path, parse_dates=["timestamp"])
    else:
        file_metadata = pd.read_csv(
            input_dir_path / "file_metadata.csv", parse_dates=["timestamp"]
        )

    filename = [file.name for file in files]
    file_metadata = file_metadata[file_metadata["filename"].isin(filename)].reset_index(
        drop=True
    )

    if not datetime_begin:
        datetime_begin = file_metadata["timestamp"].iloc[0]

    if not datetime_end:
        datetime_end = file_metadata["timestamp"].iloc[-1] + pd.Timedelta(
            file_metadata["duration"].iloc[-1], unit="s"
        )

    # In the case of a single batch where batch_ind_max = -1, assign its true value otherwise the main for loop does not work
    if batch == 0 and batch_ind_min == 0 and batch_ind_max == -1:
        if concat:
            batch_ind_max = (
                len(
                    list(
                        pd.date_range(
                            start=pd.Timestamp(datetime_begin),
                            end=pd.Timestamp(datetime_end)
                            - pd.Timedelta(value=int(f"{segment_size}"), unit="s"),
                            freq=f"{segment_size}s",
                        )
                    )
                )
                - 1
            )
        else:
            batch_ind_max = len(file_metadata) - 1

    # segment timestamps
    if concat:
        new_file = pd.date_range(
            start=datetime_begin,
            end=datetime_end,
            freq=f"{segment_size}s",
        ).to_list()
    else:
        origin_timestamp = file_metadata[
            (file_metadata["timestamp"] >= datetime_begin)
            & (file_metadata["timestamp"] <= datetime_end)
        ]
        new_file = []
        for i, ts in enumerate(origin_timestamp["timestamp"]):
            current_ts = ts
            original_timedelta = pd.Timedelta(
                seconds=origin_timestamp["duration"].iloc[i]
            )

            while (
                current_ts <= ts + original_timedelta
                and (ts + original_timedelta - current_ts).total_seconds() > 5
            ):
                new_file.append(current_ts)
                current_ts += segment_duration
        new_file.append(current_ts + segment_duration)

    # sample rates
    orig_sr = file_metadata["origin_sr"][0]
    segment_sample_rate = orig_sr if new_sr == -1 else new_sr

    # origin audio time vector list
    file_time_vec_list = [
        file_metadata["timestamp"][i].timestamp()
        + (
            np.arange(0, file_metadata["duration"][i] * segment_sample_rate)
            / segment_sample_rate
        )
        for i in range(len(file_metadata))
    ]

    result = []
    timestamp_list = []
    for i in range((batch_ind_max - batch_ind_min) + 1):

        audio_data = np.zeros(shape=segment_size * segment_sample_rate)

        if concat:
            segment_datetime_begin = datetime_begin + (
                (i + batch_ind_min) * segment_duration
            )
            segment_datetime_end = min(
                segment_datetime_begin + segment_duration, datetime_end
            )
        else:
            segment_datetime_begin = new_file[i + batch_ind_min]
            segment_datetime_end = segment_datetime_begin + segment_duration

        # segment time vector
        time_vec = (
            segment_datetime_begin.timestamp()
            + np.arange(0, segment_size * segment_sample_rate) / segment_sample_rate
        )

        len_sig = 0
        audio_data_sum = audio_data.sum()
        for index, row in file_metadata.iterrows():

            file_datetime_begin = row["timestamp"]
            file_datetime_end = row["timestamp"] + pd.Timedelta(seconds=row["duration"])

            # check if original audio file overlaps with the current segment
            if (
                file_datetime_end > segment_datetime_begin
                and file_datetime_begin < segment_datetime_end
            ):

                start_offset = (
                    0
                    if file_datetime_begin >= segment_datetime_begin
                    else int(
                        (segment_datetime_begin - file_datetime_begin).total_seconds()
                        * orig_sr
                    )
                )

                end_offset = (
                    round(orig_sr * row["duration"])
                    if file_datetime_end <= segment_datetime_end
                    else int(
                        (segment_datetime_end - file_datetime_begin).total_seconds()
                        * orig_sr
                    )
                )

                # read the appropriate section of the origin audio file
                with sf.SoundFile(input_dir_path / row["filename"]) as audio_file:
                    audio_file.seek(start_offset)
                    sig = audio_file.read(frames=end_offset - start_offset)

                # resample
                if orig_sr != segment_sample_rate:
                    sig = resample(sig, orig_sr=orig_sr, target_sr=segment_sample_rate)

                # origin audio time vector
                file_time_vec = file_time_vec_list[index]

                audio_data[
                    (time_vec < file_time_vec[0])
                    .sum() : (time_vec < file_time_vec[0])
                    .sum()
                    + len(sig)
                ] = sig

                len_sig += len(sig)

                if not concat:
                    break

        if np.sum(audio_data) == 0:
            print(
                f"No data available for file {segment_datetime_begin.strftime('%Y_%m_%d_%H_%M_%S')}.wav. Skipping...\n"
            )
            continue

        if len_sig < 0.01 * trigger * segment_size * segment_sample_rate:
            print(
                f"Not enough data available for file {segment_datetime_begin.strftime('%Y_%m_%d_%H_%M_%S')}.wav ({len_sig / segment_sample_rate:.2f}s < {trigger}% of the spectrogram duration of {segment_size}s). Skipping...\n"
            )
            continue

        # Define the output file name and save the audio segment
        outfilename = (
            output_dir_path
            / f"{segment_datetime_begin.strftime('%Y_%m_%d_%H_%M_%S')}.wav"
        )

        if not overwrite and outfilename.exists():
            if verbose:
                print(
                    f"File {outfilename} already exists and overwrite is set to False. Skipping...\n"
                )
            continue

        result.append(outfilename.name)
        timestamp_list.append(segment_datetime_begin.strftime("%Y-%m-%dT%H:%M:%S.%f%z"))

        sf.write(
            outfilename,
            audio_data,
            segment_sample_rate,
        )
        os.chmod(outfilename, mode=FPDEFAULT)

        if verbose:
            print(
                f"Saved file from {segment_datetime_begin} to {segment_datetime_end} as {outfilename}\n"
            )
    # %%
    # writing infos to timestamp_*.csv
    input_timestamp = pd.DataFrame({"filename": result, "timestamp": timestamp_list})
    input_timestamp.sort_values(by=["timestamp"], inplace=True)
    input_timestamp.drop_duplicates().to_csv(
        output_dir_path / f"timestamp_{batch}.csv",
        index=False,
        na_rep="NaN",
    )
    os.chmod((output_dir_path / f"timestamp_{batch}.csv"), mode=FPDEFAULT)
    print(f"Saved timestamp csv file as timestamp_{batch}.csv\n")

    print(f"Reshape for batch_{batch} completed")

    return


if __name__ == "__main__":
    parser = ArgumentParser()
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--input-files",
        "-i",
        help="The files to be reshaped, as either the path to a directory containing audio files and a timestamp.csv or a list of filenames all in the same directory alongside a timestamp.csv.",
    )
    required.add_argument(
        "--segment-size",
        "-s",
        type=int,
        help="The time in seconds of the reshaped files.",
    )
    parser.add_argument(
        "--datetime-begin",
        type=str,
        help="Datetime string to begin the reshape at",
    )
    parser.add_argument(
        "--datetime-end",
        type=str,
        help="Datetime string to end the reshape at",
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
        "--batch",
        type=int,
        help="Batch index",
    )
    parser.add_argument(
        "--batch-num",
        type=int,
        help="Batch number",
    )
    parser.add_argument(
        "--batch-ind-max",
        "-max",
        type=int,
        default=-1,
        help="The last file of the list to be processed. Default is -1, meaning the entire list is processed.",
    )
    parser.add_argument(
        "--trigger",
        "-trig",
        type=int,
        default=5,
        help="Integer to filter out segment with not enough samples",
    )
    parser.add_argument(
        "--concat",
        default=True,
        type=str,
        help="Whether the script concatenate audio segments or not. If not, the segments are 0-padded if necessary to fit the defined duration.",
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
        help="Tells the program what to do with the remaining data that are shorter than the segment size. Possible arguments are pad (the default), which pads with silence until the last file has the same length as the others; truncate to create a shorter file with only the leftover data; discard to not do anything with the last data and throw it away.",
    )
    parser.add_argument(
        "--timestamp-path", default=None, help="Path to the original timestamp file."
    )
    parser.add_argument(
        "--file-metadata-path",
        help="Path to file metadata",
        default=None,
    )
    parser.add_argument(
        "--new-sr",
        type=int,
        default=-1,
        help="Sampling rate",
    )

    args = parser.parse_args()

    input_files = (
        args.input_files.split(" ")
        if not Path(args.input_files).is_dir()
        else args.input_files
    )

    print(f"### Parameters: {args}\n")

    files = reshape(
        segment_size=args.segment_size,
        file_metadata_path=Path(args.file_metadata_path),
        timestamp_path=Path(args.timestamp_path),
        input_files=input_files,
        output_dir_path=args.output_dir,
        new_sr=args.new_sr,
        datetime_begin=args.datetime_begin,
        datetime_end=args.datetime_end,
        batch=args.batch,
        batch_num=args.batch_num,
        batch_ind_min=args.batch_ind_min,
        batch_ind_max=args.batch_ind_max,
        concat=args.concat.lower() == "true",
        last_file_behavior=args.last_file_behavior,
        verbose=args.verbose,
        overwrite=args.overwrite,
        trigger=args.trigger,
    )
