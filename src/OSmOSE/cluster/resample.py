import argparse
import platform
import subprocess
from pathlib import Path

from OSmOSE.utils.audio_utils import get_all_audio_files


def resample(
    *,
    input_dir: Path,
    output_dir: Path,
    target_sr: int,
    batch_ind_min: int = 0,
    batch_ind_max: int = -1,
):
    """Resample all audio files in a given directory to a target sample rate and write them to a new directory.

    Parameters
    ----------
        input_dir: `Path`, keyword-only
            The input directory containing the input files.
        output_dir: `Path`, keyword-only
            The output directory to write the resampled files.
        target_sr: `int`, keyword-only
            The target sample rate.
        batch_ind_min: `int`, keyword-only
            The index of the first file of the batch. The default is 0.
        batch_ind_max: `int`, keyword-only
            The index of the last file of the batch. The default is -1, meaning the last file of the input directory.

    """
    if platform.system() == "Windows":
        print("Sox is unavailable on Windows")
        return
    all_files = get_all_audio_files(input_dir)

    # If batch_ind_max is -1, we go to the end of the list.
    audio_files_list = all_files[
        batch_ind_min : batch_ind_max if batch_ind_max != -1 else len(all_files)
    ]

    # tfm = sox.Transformer()
    # tfm.set_output_format(rate=target_sr)

    for audio_file in audio_files_list:
        subprocess.run(
            f"sox {audio_file!s} -r {target_sr!s} -t wavpcm {Path(output_dir, audio_file.name)!s}",
            shell=True,
            check=False,
        )

        print(f"{audio_file.name} resampled to {target_sr}!")
    #     tfm.build_file(
    #         input_filepath=str(audio_file),
    #         output_filepath=str(Path(output_dir, audio_file.name)),
    #     )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Python script to resample a list of audio files using the sox tool. Does not change the file duration.",
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--input-dir",
        "-i",
        required=True,
        help="The input folder where the files to resample are.",
    )
    required.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="The output folder of the resampled files.",
    )
    required.add_argument(
        "--target-sr",
        "-sr",
        required=True,
        type=int,
        help="The target samplerate.",
    )
    parser.add_argument(
        "--batch-ind-min",
        "-min",
        type=int,
        default=0,
        help="The first file to consider. Default is 0.",
    )
    parser.add_argument(
        "--batch-ind-max",
        "-max",
        type=int,
        default=-1,
        help="The last file to consider. -1 means consider all files from ind-min. Default is -1",
    )

    args = parser.parse_args()

    print("Parameters :", args)

    resample(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        target_sr=args.target_sr,
        batch_ind_min=args.batch_ind_min,
        batch_ind_max=args.batch_ind_max,
    )
