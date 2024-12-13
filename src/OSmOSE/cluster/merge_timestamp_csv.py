import itertools
import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def merge_timestamp_csv(input_files: str, umask: int):
    os.umask(umask)
    input_dir_path = Path(input_files)

    list_conca_timestamps = []
    list_conca_filename = []
    for ll in list(input_dir_path.glob("timestamp_*")):
        print(f"read and remove file {ll}")
        list_conca_timestamps.append(list(pd.read_csv(ll)["timestamp"].values))
        list_conca_filename.append(list(pd.read_csv(ll)["filename"].values))
        os.remove(ll)

    print(f"save file {input_dir_path.joinpath('timestamp.csv')!s}")
    df = pd.DataFrame(
        {
            "filename": list(itertools.chain(*list_conca_filename)),
            "timestamp": list(itertools.chain(*list_conca_timestamps)),
        },
    )
    df.sort_values(by=["timestamp"], inplace=True)
    df.to_csv(input_dir_path.joinpath("timestamp.csv"), index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--input-files",
        "-i",
        help="The files to be reshaped, as either the path to a directory containing audio files and a timestamp.csv or a list of filenames all in the same directory alongside a timestamp.csv.",
    )
    parser.add_argument(
        "--umask",
        type=int,
        default=0o002,
        help="Umask to apply on the created files permissions.",
    )

    args = parser.parse_args()

    input_files = (
        args.input_files.split(" ")
        if not Path(args.input_files).is_dir()
        else args.input_files
    )

    files = merge_timestamp_csv(input_files=input_files, umask=args.umask)
