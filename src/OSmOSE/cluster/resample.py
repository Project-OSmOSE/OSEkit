import argparse
from pathlib import Path
import sox


def resample(
    *,
    input_dir: str,
    output_dir: str,
    target_sr: int,
    batch_ind_min: int = 0,
    batch_ind_max: int = -1,
):
    return
    all_files = sorted(Path(input_dir).glob("*wav"))

    # If batch_ind_max is -1, we go to the end of the list.
    audio_files_list = all_files[
        batch_ind_min : batch_ind_max if batch_ind_max != -1 else len(all_files)
    ]

    tfm = sox.Transformer()
    tfm.set_output_format(rate=target_sr)

    for audio_file in audio_files_list:
        tfm.build_file(
            input_filepath=audio_file,
            output_filepath=Path(output_dir, audio_file.name),
        )

        print(f"{audio_file.name} resampled to {target_sr}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Python script to resample a list of audio files using the sox tool. Does not change the file duration."
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
        "--target-sr", "-sr", required=True, type=int, help="The target samplerate."
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

    resample(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_sr=args.targets_sr,
        batch_ind_min=args.batch_ind_min,
        batch_ind_max=args.batch_ind_max,
    )
