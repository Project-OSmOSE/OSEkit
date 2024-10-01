import os
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import soundfile as sf
from librosa import resample

from OSmOSE.utils import (
    DPDEFAULT,
    FPDEFAULT,
    to_timestamp,
    get_all_audio_files,
    set_umask,
    make_path,
)


def reshape(
    input_files: str | list,
    segment_size: int,
    *,
    file_metadata_path: str | Path = None,
    timestamp_path: str | Path = None,
    output_dir_path: str | Path = None,
    datetime_begin: str = None,
    datetime_end: str = None,
    new_sr: int = -1,
    batch: int = 0,
    batch_num: int = 1,
    batch_ind_min: int = 0,
    batch_ind_max: int = -1,
    concat: bool = True,
    verbose: bool = False,
    overwrite: bool = True,
    threshold: int = 5,
):
    """
    Reshape all audio files in the folder to be of the specified duration and/or sampling rate.

    Parameters
    ----------
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

    concat : bool, optional
        If True, concatenate all original audio data. If False,
        each segment will be saved as a separate file. Default is True.

    verbose : bool, optional
        If True, print detailed information about the processing steps. Default is False.

    overwrite : bool, optional
        If True, overwrite existing files in the output directory with the same name. Default is True.

    threshold : int, optional
        Integer from 0 to 100 to filter out segments with a number of sample inferior to (threshold * spectrogram duration * new_sr)
    """
    set_umask()
    segment_duration = pd.Timedelta(seconds=segment_size)
    msg_log = ""

    # validation for threshold
    if not (0 <= threshold <= 100):
        raise ValueError(
            "The 'threshold' parameter must be an integer between 0 and 100."
        )

    # validation datetimes
    if datetime_begin:
        datetime_begin = to_timestamp(datetime_begin)

    if datetime_end:
        datetime_end = to_timestamp(datetime_end)

    # prepare file paths
    if file_metadata_path and isinstance(file_metadata_path, str):
        file_metadata_path = Path(file_metadata_path)
    if timestamp_path and isinstance(timestamp_path, str):
        timestamp_path = Path(timestamp_path)
    if output_dir_path and isinstance(output_dir_path, str):
        output_dir_path = Path(output_dir_path)
    if input_files and isinstance(input_files, str):
        input_files = Path(input_files)
    if isinstance(input_files, list):
        input_dir_path = Path(input_files[0]).parent
        files = [Path(file) for file in input_files]
    else:
        input_dir_path = Path(input_files)
        files = get_all_audio_files(directory=input_dir_path)

    if not input_dir_path.exists():
        raise ValueError(
            f"The input files must be a valid folder path, not '{input_dir_path}'."
        )

    if not (input_dir_path / "timestamp.csv").exists() and (
        not timestamp_path or not timestamp_path.exists()
    ):
        raise FileNotFoundError(
            f"The timestamp.csv file must be present in the directory {Path(input_dir_path)} and correspond to the audio files in the same location, or be specified in the argument."
        )

    # output directory
    if not output_dir_path:
        output_dir_path = input_dir_path
        if overwrite:
            raise ValueError(
                "Cannot overwrite input directory when the output directory is implicit!"
            )
    if not output_dir_path.exists():
        make_path(output_dir_path, mode=DPDEFAULT)

    # load timestamp and metadata
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

    filenames = [file.name for file in files]
    file_metadata = file_metadata[
        file_metadata["filename"].isin(filenames)
    ].reset_index(drop=True)

    # datetimes
    if not datetime_begin:
        datetime_begin = file_metadata["timestamp"].iloc[0]

    if not datetime_end:
        datetime_end = file_metadata["timestamp"].iloc[-1] + pd.Timedelta(
            file_metadata["duration"].iloc[-1], unit="s"
        )

    # segment timestamps
    if concat:
        new_file = pd.date_range(
            start=datetime_begin,
            end=datetime_end,
            freq=f"{segment_size}s",
        ).to_list()[:-1]
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

            while current_ts <= ts + original_timedelta:
                new_file.append(current_ts)
                current_ts += segment_duration

    if batch_ind_max == -1:
        batch_ind_max = len(new_file) - 1

    # sample rates
    orig_sr = file_metadata["origin_sr"][0]
    new_sr = orig_sr if new_sr == -1 else new_sr

    result = []
    timestamp_list = []
    for i in range(batch_ind_max - batch_ind_min + 1):

        audio_data = np.zeros(shape=segment_size * new_sr)

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
            + np.arange(0, segment_size * new_sr) / new_sr
        )

        len_sig = 0
        for index, row in file_metadata.iterrows():

            file_datetime_begin = row["timestamp"]
            file_datetime_end = row["timestamp"] + pd.Timedelta(seconds=row["duration"])

            # check if original audio file overlaps with the current segment
            if (
                file_datetime_end > segment_datetime_begin
                and file_datetime_begin < segment_datetime_end
            ):

                start_offset = int(
                    max(
                        0.0,
                        (segment_datetime_begin - file_datetime_begin).total_seconds(),
                    )
                    * orig_sr
                )

                end_offset = int(
                    min(
                        (segment_datetime_end - file_datetime_begin),
                        (file_datetime_end - file_datetime_begin),
                    ).total_seconds()
                    * orig_sr
                )

                # read the appropriate section of the origin audio file
                with sf.SoundFile(input_dir_path / row["filename"]) as audio_file:
                    audio_file.seek(start_offset)
                    sig = audio_file.read(frames=end_offset - start_offset)
                    orig_len = len(sig)

                # resample
                # During the conversion, because the ratio of original and new sampling rates​ is not an integer,
                # the resample algorithm may introduce a slight rounding error when computing the exact number of output samples.
                # This rounding error can sometimes result in one extra sample.
                if orig_sr != new_sr:
                    sig = resample(sig, orig_sr=orig_sr, target_sr=new_sr)
                    expected_len = int((new_sr * orig_len) / orig_sr)
                    sig = sig[:expected_len]

                # origin audio time vector
                file_time_vec = file_metadata["timestamp"][index].timestamp() + (
                    np.arange(0, file_metadata["duration"][index] * new_sr) / new_sr
                )

                audio_data[
                    (time_vec < file_time_vec[0])
                    .sum() : (time_vec < file_time_vec[0])
                    .sum()
                    + len(sig)
                ] = sig

                len_sig += len(sig)

                if not concat:
                    break

        # Define the output file name and save the audio segment
        out_filename = (
            output_dir_path
            / f"{segment_datetime_begin.strftime('%Y_%m_%d_%H_%M_%S')}.wav"
        )

        # if no data available
        if np.sum(audio_data) == 0:
            msg_log += f"No data available for file {out_filename.name}. Skipping...\n"
            continue

        # if not enough data available
        if len_sig < 0.01 * threshold * segment_size * new_sr:
            msg_log += f"Not enough data available for file {out_filename.name} ({len_sig / new_sr:.2f}s < {threshold}% of the spectrogram duration of {segment_size}s). Skipping...\n"
            continue

        # if audio file already exists and overwrite parameter is set to False
        if not overwrite and out_filename.exists():
            msg_log += f"File {out_filename} already exists and overwrite is set to False. Skipping...\n"
            continue

        result.append(out_filename.name)
        timestamp_list.append(segment_datetime_begin.strftime("%Y-%m-%dT%H:%M:%S.%f%z"))

        sf.write(
            out_filename,
            audio_data,
            new_sr,
        )
        os.chmod(out_filename, mode=FPDEFAULT)
        msg_log += f"Saved file from {segment_datetime_begin} to {segment_datetime_end} as {out_filename}\n"

    # writing infos to timestamp_*.csv
    input_timestamp = pd.DataFrame({"filename": result, "timestamp": timestamp_list})
    input_timestamp.sort_values(by=["timestamp"], inplace=True)
    input_timestamp.drop_duplicates().to_csv(
        output_dir_path / f"timestamp_{batch}.csv",
        index=False,
        na_rep="NaN",
    )
    os.chmod((output_dir_path / f"timestamp_{batch}.csv"), mode=FPDEFAULT)
    msg_log += f"Saved timestamp csv file as timestamp_{batch}.csv\n"
    msg_log += f"Reshape for batch_{batch} completed\n"

    if verbose:
        print(msg_log)

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
        "--threshold",
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
        "--timestamp-path",
        default=None,
        help="Path to the original timestamp file.",
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
        verbose=args.verbose,
        overwrite=args.overwrite,
        threshold=args.threshold,
    )
