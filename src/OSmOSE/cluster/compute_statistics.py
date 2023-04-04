from pathlib import Path
import sys
import csv
import argparse

import soundfile as sf
import numpy as np
from scipy import signal

from OSmOSE.utils import set_umask

def Write_zscore_norma_params(
    *,
    input_dir: str,
    output_file: str,
    HPfilter_min_freq: int,
    batch_ind_min: int = 0,
    batch_ind_max: int = -1
):
    """Computes the normalization parameters for the Zscore normalisation of the dataset and writes it to a csv.

    This function can also be called from the command line. Type `get_zscore_params.py -h` to see the list of arguments.
    Will write the mean and standard deviation for each file in a .csv. To get the parameters, you need to read the csv and
    compute the means of each columns.

    Parameters
    ----------
    input_dir: `str`
        The absolute path to the input directory. All audio files within this directory will be used.

    output_file: `str`
        The absolute path of the output csv file.

    HPfilter_min_freq: `int`
        The minimum audio frequency under which data will be filtrated. If set to 0, then `sys.float_info.epsilon` will be used.

    batch_ind_min: `int`
        The first file of the list to be processed. Default is 0.

    batch_ind_max: `int`
        The last file of the list to be processed. Default is -1, meaning the entire list is processed.
    """
    set_umask()
    all_files = sorted(Path(input_dir).glob("*wav"))
    # If batch_ind_max is -1, we go to the end of the list.
    wav_list = all_files[
        batch_ind_min : batch_ind_max if batch_ind_max != -1 else len(all_files)
    ]

    list_summaryStats = []

    for wav in wav_list:
        data, sample_rate = sf.read(wav)

        bpcoef = signal.butter(
            20,
            np.array(
                [max(HPfilter_min_freq, sys.float_info.epsilon), sample_rate / 2 - 1]
            ),
            fs=sample_rate,
            output="sos",
            btype="bandpass",
        )
        data = signal.sosfilt(bpcoef, data)

        list_summaryStats.append([wav, np.mean(data), np.std(data)])

    with open(output_file, "w", newline="") as f:
        write = csv.writer(f)
        write.writerow(["filename", "mean", "std"])
        write.writerows(list_summaryStats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Python script to get the mean and standard deviation of audio files, used in zscore parameter calculation."
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--input-dir",
        "-i",
        required=True,
        help="The input folder where the files to read are.",
    )
    required.add_argument(
        "--output-file",
        "-o",
        required=True,
        help="The csv file where the results will be written.",
    )
    required.add_argument(
        "--hpfilter-min-freq",
        "-fmin",
        required=True,
        type=int,
        help="The High Pass Filter.",
    )
    parser.add_argument(
        "--ind-min",
        "-min",
        type=int,
        default=0,
        help="The first file to consider. Default is 0.",
    )
    parser.add_argument(
        "--ind-max",
        "-max",
        type=int,
        default=-1,
        help="The last file to consider. -1 means consider all files from ind-min. Default is -1",
    )

    args = parser.parse_args()

    Write_zscore_norma_params(
        input_dir=args.input_dir,
        output_file=args.output_file,
        HPfilter_min_freq=args.HPfilter_min_freq,
        batch_ind_min=args.batch_ind_min,
        batch_ind_max=args.batch_ind_max,
    )
