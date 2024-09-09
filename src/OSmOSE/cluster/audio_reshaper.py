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
    """

    set_umask()

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

    segment_duration = pd.Timedelta(seconds=segment_size)
    result = []
    timestamp_list = []
    previous_segment = np.empty(shape=[0])
    initial_padding = np.empty(shape=[0])
    f = 0  # written file

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

    # Iterate over each segment
    for i in range((batch_ind_max - batch_ind_min) + 1):

        audio_data = np.empty(shape=[0])

        if concat:
            segment_datetime_begin = datetime_begin + (
                (i + batch_ind_min) * segment_duration
            )
            segment_datetime_end = min(
                segment_datetime_begin + segment_duration, datetime_end
            )
        else:
            segment_datetime_begin = file_metadata["timestamp"].iloc[i + batch_ind_min]
            segment_datetime_end = segment_datetime_begin + segment_duration

        for index, row in file_metadata.iterrows():

            file_datetime_begin = row["timestamp"]
            file_datetime_end = row["timestamp"] + pd.Timedelta(seconds=row["duration"])

            # Check if original audio file overlaps with the current segment
            if (
                file_datetime_end > segment_datetime_begin
                and file_datetime_begin < segment_datetime_end
            ):
                with sf.SoundFile(input_dir_path / row["filename"]) as audio_file:
                    file_sample_rate = audio_file.samplerate

                    # Determine the part of the file that fits into the segment
                    if file_datetime_begin < segment_datetime_begin:
                        # File starts before the segment, so we skip the initial part of the file
                        start_offset = int(
                            (
                                segment_datetime_begin - file_datetime_begin
                            ).total_seconds()
                            * file_sample_rate
                        )
                    elif file_datetime_begin >= segment_datetime_begin:
                        # File does not start before the segment, no offset
                        start_offset = 0

                        # First segment begins after first original file : zero padding
                        if f == 0 and previous_segment.shape == np.zeros(shape=0).shape:
                            initial_padding = np.zeros(
                                shape=int(
                                    (
                                        file_datetime_begin - segment_datetime_begin
                                    ).total_seconds()
                                    * file_sample_rate
                                )
                            )

                    if file_datetime_end > segment_datetime_end:
                        # File ends after the segment, so we cut off the end part of the file
                        end_offset = int(
                            (segment_datetime_end - file_datetime_begin).total_seconds()
                            * file_sample_rate
                        )
                    else:
                        # File ends before the segment, we read the rest of the audio file
                        end_offset = len(audio_file)

                    # Read the appropriate section of the file
                    audio_file.seek(start_offset)
                    audio_data = audio_file.read(frames=end_offset - start_offset)

                    # Initial padding
                    if initial_padding.shape != np.zeros(shape=0).shape:
                        audio_data = np.concatenate((initial_padding, audio_data))
                        initial_padding = np.zeros(shape=0)

                    # Resampling
                    segment_sample_rate = file_sample_rate if new_sr == -1 else -1

                    if file_sample_rate != segment_sample_rate:
                        audio_data = resample(
                            audio_data, orig_sr=file_sample_rate, target_sr=new_sr
                        )
                        segment_sample_rate = new_sr

                    audio_data = np.concatenate((previous_segment, audio_data))

                    expected_frames = int(segment_size * segment_sample_rate)

                    # The segment is shorter than expected
                    if len(audio_data) < expected_frames:
                        if concat:
                            # Not last batch
                            if batch + 1 < batch_num:
                                # Not last segment
                                if i <= (batch_ind_max - batch_ind_min):
                                    previous_segment = audio_data
                                    continue

                            # Last batch
                            elif batch + 1 == batch_num:

                                # Last batch but not last segment
                                if i <= (batch_ind_max - batch_ind_min):

                                    # Not last original audio file
                                    if index + 1 < len(file_metadata):
                                        previous_segment = audio_data
                                        continue

                                    # Last original audio file
                                    elif index + 1 == len(file_metadata):
                                        padding = np.zeros(
                                            expected_frames - len(audio_data),
                                            dtype=np.float32,
                                        )
                                        audio_data = np.concatenate(
                                            (audio_data, padding)
                                        )
                                        previous_segment = np.empty(shape=[0])

                                # Last batch and last segment
                                else:
                                    if last_file_behavior == "pad":
                                        padding = np.zeros(
                                            expected_frames - len(audio_data),
                                            dtype=np.float32,
                                        )
                                        audio_data = np.concatenate(
                                            (audio_data, padding)
                                        )
                                    elif last_file_behavior == "truncate":
                                        continue
                                    elif last_file_behavior == "discard":
                                        break
                        else:
                            padding = np.zeros(
                                expected_frames - len(audio_data),
                                dtype=np.float32,
                            )
                            audio_data = np.concatenate((audio_data, padding))
                            previous_segment = np.empty(shape=[0])
                            break

                    # The segment is the expected length
                    elif len(audio_data) == expected_frames:
                        previous_segment = np.empty(shape=[0])
                        continue

                    # The segment is longer than the expected length
                    else:
                        previous_segment = audio_data[expected_frames:]
                        audio_data = audio_data[:expected_frames]

            elif file_datetime_begin > segment_datetime_end:
                if index + 1 == len(file_metadata):
                    if audio_data.shape == np.empty(shape=[0]).shape:
                        print(
                            f"No data available for file {segment_datetime_begin.strftime('%Y_%m_%d_%H_%M_%S')}.wav. Skipping...\n"
                        )
            else:
                if index + 1 == len(file_metadata):
                    if audio_data.shape == np.empty(shape=[0]).shape:
                        print(
                            f"No data available for file {segment_datetime_begin.strftime('%Y_%m_%d_%H_%M_%S')}.wav. Skipping...\n"
                        )

        # Define the output file name and save the audio segment
        outfilename = (
            output_dir_path
            / f"{segment_datetime_begin.strftime('%Y_%m_%d_%H_%M_%S')}.wav"
        )
        result.append(outfilename.name)
        timestamp_list.append(segment_datetime_begin.strftime("%Y-%m-%dT%H:%M:%S.%f%z"))

        if not overwrite and os.path.exists(outfilename):
            if verbose:
                print(
                    f"File {outfilename} already exists and overwrite is set to False. Skipping...\n"
                )
            continue

        if audio_data.shape != np.empty(shape=[0]).shape:
            sf.write(
                outfilename,
                audio_data,
                segment_sample_rate,
            )
            os.chmod(outfilename, mode=FPDEFAULT)

            # Increment the number of written file
            f += 1

            if verbose:
                print(
                    f"Saved file from {segment_datetime_begin} to {segment_datetime_end} as {outfilename}\n"
                )

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
    )
