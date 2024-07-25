import os
import pandas as pd
from datetime import datetime
from typing import List, Union, Literal
from argparse import ArgumentParser
from pathlib import Path
import numpy as np

import soundfile as sf
from librosa import resample

from OSmOSE.utils.core_utils import set_umask
from OSmOSE.utils.path_utils import make_path
from OSmOSE.config import DPDEFAULT, FPDEFAULT, OSMOSE_PATH


def reshape_MD(
    input_files: Union[str, list],
    chunk_size: int,
    *,
    new_sr: int = -1,
    datetime_begin: pd.Timestamp = None,
    df_file: pd.DataFrame = None,
    output_dir_path: str = None,
    batch_ind_min: int = 0,
    batch_ind_max: int = -1,
    max_delta_interval: int = 5,
    last_file_behavior: Literal["truncate", "pad", "discard"] = "pad",
    offset_beginning: int = 0,
    offset_end: int = 0,
    timestamp_path: Path = None,
    verbose: bool = False,
    overwrite: bool = True,
    force_reshape: bool = False,
    merge_files: bool = False,
    audio_file_overlap: int = 0,
) -> List[str]:
    """Reshape all audio files in the folder to be of the specified duration and to begin at the user-defined datetime.
    If chunk_size is superior to the base duration of the files, they will be fused according to their order in the timestamp.csv file
    in the same folder.

    Parameters:
    -----------
        input_files: `str` or `list(str)`
            Either the directory containing the audio files and the timestamp.csv file, in which case all audio files will be considered,
            OR a list of audio files all located in the same directory alongside a timestamp.csv, in which case only they will be used.

        chunk_size: `int`
            The target duration for all the reshaped files, in seconds.

        output_dir_path: `str`, optional, keyword-only
            The directory where the newly created audio files will be created. If none is provided,
            it will be the same as the input directory. This is not recommended.

        batch_ind_min: `int`, optional, keyword-only
            The first file of the list to be processed. Default is 0.

        batch_ind_max: `int`, optional, keyword-only
            The last file of the list to be processed. Default is -1, meaning the entire list is processed.

        max_delta_interval: `int`, optional, keyword-only
            The maximum number of second allowed for a delta between two timestamp_list to still be considered the same.
            Default is 5s up and down.

        last_file_behavior: `{"truncate","pad","discard"}, optional, keyword-only
            Tells the reshaper what to do with if the last data of the last file is too small to fill a whole file.
            This parameter is only active if `batch_ind_max` is `-1`
            - `truncate` creates a truncated file with the remaining data, which will have a different duration than the others.
            - `pad` creates a file of the same duration than the others, where the missing data is filled with 0.
            - `discard` ignores the remaining data. The last seconds/minutes/hours of audio will be lost in the reshaping.
        The default method is `pad`.

        offset_beginning: `int`, optional, keyword-only
            The number of seconds that should be skipped in the first input file. When parallelising the reshaping,
            it would mean that the beginning of the file is being processed by another job. Default is 0.

        offset_end: `int`, optional, keyword-only
            The number of seconds that should be ignored in the last input file. When parallelising the reshaping, it would mean that the end of this file is processed by another job.
            Default is 0, meaning that nothing is ignored.

        verbose: `bool`, optional, keyword-only
            Whether to display informative messages or not.

        overwrite: `bool`, optional, keyword-only
            Deletes the content of `output_dir_path` before writing the results. If it is implicitly the `input_files` directory,
            nothing happens. WARNING: If `output_dir_path` is explicitly set to be the same as `input_files`, then it will be overwritten!

        force_reshape: `bool`, optional, keyword-only
            Ignore all warnings and non-fatal errors while reshaping.

        merge_files: `bool`, optional, keyword-only
            Whether to merge files when reshaping them. If set to False, then the chunk_size can only be smaller than the original duration, and the remaining
            data will follow the last_file_behavior (default: pad). The default is True.
    Returns:
    --------
        The list of the path of newly created audio files.
    """
    set_umask()
    save_meta_res = False

    if isinstance(input_files, list):
        input_dir_path = Path(input_files[0]).parent
        files = [Path(file) for file in input_files]
        if verbose:
            print(f"Input directory detected as {input_dir_path}")
    else:
        input_dir_path = Path(input_files)

    # Validation
    if last_file_behavior not in ["truncate", "pad", "discard"]:
        raise ValueError(
            f"Bad value {last_file_behavior} for last_file_behavior parameters. Must be one of truncate, pad or discard."
        )

    implicit_output = False
    if not output_dir_path:
        print("No output directory provided. Will use the input directory instead.")
        implicit_output = True
        output_dir_path = input_dir_path
        if overwrite:
            print(
                "Cannot overwrite input directory when the output directory is implicit! Choose a different output directory instead."
            )

    output_dir_path = Path(output_dir_path)

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

    make_path(output_dir_path, mode=DPDEFAULT)

    input_timestamp = pd.read_csv(
        timestamp_path
        if timestamp_path and timestamp_path.exists()
        else input_dir_path.joinpath("timestamp.csv")
    )

    df_file_batch = df_file[batch_ind_min:(batch_ind_max if batch_ind_max > 0 else input_timestamp.size)]

    # we select here the original audio to reshape based on which new audio file are to be created in this batch
    flattened_unique_list = []
    for sublist in df_file_batch['selection']:
        for item in sublist:
            if item not in flattened_unique_list:
                flattened_unique_list.append(item)

    result = []
    timestamp_list = []
    list_seg_name = []
    list_seg_timestamp = []
    sample_rate = 0
    i = 0

    while i < len(df_file_batch):

        if verbose:
            print(f"New file: {df_file_batch['filename'][i]}")
            print(f"Selected original files: {df_file_batch['selection'][i]}")

        files = df_file_batch['selection'][i]

        # first we resample if necessary then we concatenate data necessary to construct the new audio file
        audio_data = np.empty(shape=[0])
        for f in files:
            input_file = input_dir_path.joinpath(f)
            fmt = input_file.suffixes[-1].replace(".", "")

            # Getting file information and data
            with sf.SoundFile(input_file) as audio_file:
                sample_rate = audio_file.samplerate
                subtype = audio_file.subtype
                audio_data_slice = audio_file.read()

            if new_sr == -1:
                new_sr = sample_rate

            # If the file sample rate is different from the target sample rate, we resample the audio data
            elif new_sr != sample_rate:
                audio_data_slice = resample(audio_data_slice, orig_sr=sample_rate, target_sr=new_sr)
                sample_rate = new_sr

            audio_data = np.concatenate((audio_data, audio_data_slice))

        # now we check if the data begins before / after the begin datetime of the new audio file
        # case 1 : it begins after, then we need to pad the begining of new file with zeros
        if df_file_batch['dt_start'][i] < min(df_file_batch['selection_datetime_begin'][i]):
            offset_beginning = (min(df_file_batch['selection_datetime_begin'][i]) - df_file_batch['dt_start'][i]).total_seconds()
            fill = np.zeros(int(offset_beginning * sample_rate))
            audio_data = np.concatenate((fill, audio_data))[: chunk_size * sample_rate]
        # case 2  : it begins before, then we need to cut out the useless data anterior to begining of new file
        # note : the case where it begins right on the same time is taken care of here
        elif df_file_batch['dt_start'][i] >= min(df_file_batch['selection_datetime_begin'][i]):
            offset_beginning = (df_file_batch['dt_start'][i] - min(df_file_batch['selection_datetime_begin'][i])).total_seconds()
            audio_data = audio_data[int(offset_beginning * sample_rate):]

        # if audio_data is longer than desired duration we cut out the end to get the desired duration
        if len(audio_data) > chunk_size * sample_rate:
            audio_data = audio_data[: chunk_size * sample_rate]

        # if it is the last batch, the last original audio might be too short, then we pad with zeros if "pad" is selected
        if len(audio_data) < chunk_size * sample_rate:
            if last_file_behavior == "pad":
                offset_end = (chunk_size * sample_rate) - len(audio_data)
                fill = np.zeros(int(offset_end * sample_rate))
                audio_data = np.concatenate((audio_data, fill))
            elif last_file_behavior == "discard":
                # todo: check if this works
                print(f'Last file is discarded as it is shorter than {chunk_size}s')
                # return
            elif last_file_behavior == "truncate":
                # todo: check if this works
                print('hihi')
                # continue

        # at this point audio_data should have the desired size, then we proceed and write the new wav file
        outfilename = output_dir_path.joinpath(
            f"{df_file_batch['filename'][i].replace('-','_').replace(':','_').replace('.','_').replace('+','_')}.{fmt}"
        )
        result.append(outfilename.name)

        list_seg_name.append(outfilename.name)
        list_seg_timestamp.append(
            datetime.strftime(df_file_batch['dt_start'][i], "%Y-%m-%dT%H:%M:%S.%f%z")
        )

        timestamp_list.append(
            datetime.strftime(df_file_batch['dt_start'][i], "%Y-%m-%dT%H:%M:%S.%f%z")
        )

        sf.write(
            outfilename, audio_data, sample_rate, format=fmt, subtype=subtype
        )
        os.chmod(outfilename, mode=FPDEFAULT)

        if verbose:
            print(
                f"{outfilename} written! File is {(len(audio_data)/sample_rate)} seconds long."
            )

        i += 1
        continue
        ##########################################################################################################################

        if (
            overwrite
            and not implicit_output
            and output_dir_path == input_dir_path
            and output_dir_path == input_dir_path
            and i < len(files) - 1
        ):
            print(f"Deleting {files[i]}")
            input_dir_path.joinpath(files[i]).unlink()

        if save_meta_res:
            df = pd.DataFrame(
                {
                    "list_seg_name": list_seg_name,
                    "list_seg_timestamp": list_seg_timestamp,
                }
            )
            df.to_csv(f"{self.path.joinpath(OSMOSE_PATH.log, files[i])}.csv")
        i += 1
        continue


if __name__ == "__main__":
    parser = ArgumentParser()
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--input-files",
        "-i",
        help="The files to be reshaped, as either the path to a directory containing audio files and a timestamp.csv or a list of filenames all in the same directory alongside a timestamp.csv.",
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
        "--offset-beginning",
        type=int,
        default=0,
        help="number of seconds that should be skipped in the first input file. When parallelising the reshaping, it would mean that the beginning of the file is being processed by another job. Default is 0.",
    )
    parser.add_argument(
        "--offset-end",
        type=int,
        default=0,
        help="The number of seconds that should be ignored in the last input file. When parallelising the reshaping, it would mean that the end of this file is processed by another job. Default is 0, meaning that nothing is ignored.",
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
        "--force",
        "-f",
        action="store_true",
        help="Ignore all warnings and non-fatal errors while reshaping.",
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
        "--no-merge",
        action="store_false",
        help="Don't try to merge the reshaped files.",
    )  # When absent = we merge file; when present = we don't merge -> merge_file is False
    parser.add_argument(
        "--audio-file-overlap",
        type=int,
        default=0,
        help="Overlap between audio files after segmentation. Default is 0, meaning no overlap.",
    )
    parser.add_argument(
        "--new-sr",
        type=int,
        default=0,
        help="Sampling rate",
    )
    parser.add_argument(
        "--datetime-begin",
        type=pd.datetime,
        default=None,
        help="Datetime at which you want to begin the segmentation",
    )
    args = parser.parse_args()

    input_files = (
        args.input_files.split(" ")
        if not Path(args.input_files).is_dir()
        else args.input_files
    )

    print("Parameters :", args)

    files = reshape_MD(
        chunk_size=args.chunk_size,
        input_files=input_files,
        output_dir_path=args.output_dir,
        new_sr=args.new_sr,
        datetime_begin=args.datetime_begin,
        batch_ind_min=args.batch_ind_min,
        batch_ind_max=args.batch_ind_max,
        offset_beginning=args.offset_beginning,
        offset_end=args.offset_end,
        timestamp_path=Path(args.timestamp_path),
        max_delta_interval=args.max_delta_interval,
        last_file_behavior=args.last_file_behavior,
        verbose=args.verbose,
        overwrite=args.overwrite,
        force_reshape=args.force,
        merge_files=args.no_merge,
        audio_file_overlap=args.audio_file_overlap,
    )
