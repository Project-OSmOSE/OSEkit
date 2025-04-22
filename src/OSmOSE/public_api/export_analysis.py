from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING

from OSmOSE.config import resample_quality_settings
from OSmOSE.public_api import Analysis
from OSmOSE.public_api.dataset import Dataset

if TYPE_CHECKING:
    from OSmOSE.core_api.audio_dataset import AudioDataset
    from OSmOSE.core_api.spectro_dataset import SpectroDataset


def write_analysis(
    analysis: Analysis,
    ads: AudioDataset,
    sds: SpectroDataset,
    subtype: str,
    matrix_folder_name: str,
    spectrogram_folder_name: str,
    link: bool = True,
    first: int = 0,
    last: int | None = None,
) -> None:
    """Write SpectroDataset output files to disk.

    Parameters
    ----------
    analysis: Analysis
        Flags that should be use to specify the type of analysis to run.
        See Dataset.Analysis docstring for more info.
    subtype: str | None
        Subtype of the written audio files as provided by the soundfile module.
        Defaulted as the default 16-bit PCM for WAV audio files.
        This parameter has no effect if Analysis.AUDIO is not in analysis.
    ads: AudioDataset
        The AudioDataset of which the data should be written.
    sds: SpectroDataset
        The SpectroDataset of which the data should be written.
    matrix_folder_name: Path
        The folder in which the matrix npz files should be written.
    spectrogram_folder_name: Path
        The folder in which the spectrogram png files should be written.
    link: bool
        If set to True, the ads data will be linked to the exported files.
    first: int
        Index of the first data object to write.
    last: int|None
        Index after the last data object to write.

    """
    if Analysis.AUDIO in analysis:
        ads.write(
            folder=ads.folder,
            subtype=subtype,
            link=link,
            first=first,
            last=last,
        )
        ads.write_json(ads.folder)

    if Analysis.MATRIX not in analysis and Analysis.SPECTROGRAM not in analysis:
        return

    # Avoid re-computing the reshaped audio
    if Analysis.AUDIO in analysis:
        sds.link_audio_dataset(ads, first=first, last=last)

    if Analysis.MATRIX in analysis and Analysis.SPECTROGRAM in analysis:
        sds.save_all(
            matrix_folder=sds.folder / matrix_folder_name,
            spectrogram_folder=sds.folder / spectrogram_folder_name,
            link=link,
            first=first,
            last=last,
        )
    elif Analysis.SPECTROGRAM in analysis:
        sds.save_spectrogram(
            folder=sds.folder / spectrogram_folder_name,
            first=first,
            last=last,
        )
    elif Analysis.MATRIX in analysis:
        sds.write(
            folder=sds.folder / matrix_folder_name,
            link=link,
            first=first,
            last=last,
        )

    # Update the sds from the JSON in case it has already been modified in another job
    sds.update_json_audio_data(first=first, last=last)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--dataset-json-path",
        "-p",
        required=True,
        help="The path to the Dataset JSON file.",
        type=str,
    )
    required.add_argument(
        "--analysis",
        "-a",
        required=True,
        help="Flags representing which files to export during this analysis.",
        type=int,
    )
    required.add_argument(
        "--ads-name",
        "-ads",
        required=True,
        help="Name of the AudioDataset to export during this analysis.",
        type=str,
    )
    required.add_argument(
        "--sds-name",
        "-sds",
        required=True,
        help="Name of the SpectroDataset to export during this analysis.",
        type=str,
    )
    parser.add_argument(
        "--subtype",
        "-sbtp",
        required=False,
        help="The subtype format of the audio files to export.",
        type=str,
        default=None,
    )
    required.add_argument(
        "--matrix-folder-name",
        "-mfn",
        required=True,
        help="The name of the folder in which the npz matrix files are written.",
        type=str,
    )
    required.add_argument(
        "--spectrogram-folder-name",
        "-sfn",
        required=True,
        help="The name of the folder in which the png spectrogram files are written.",
        type=str,
    )
    required.add_argument(
        "--first",
        "-f",
        required=True,
        help="The index of the first file to export.",
        type=int,
        default=0,
    )
    required.add_argument(
        "--last",
        "-l",
        required=True,
        help="The index after the last file to export.",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--downsampling-quality",
        "-dq",
        required=False,
        help="The downsampling quality preset as specified in the soxr library.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--upsampling-quality",
        "-uq",
        required=False,
        help="The upsampling quality preset as specified in the soxr library.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--umask",
        type=int,
        default=0o002,
        help="The umask to apply on the created file permissions.",
    )

    args = parser.parse_args()

    os.umask(args.umask)

    if args.downsampling_quality is not None:
        resample_quality_settings["downsample"] = args.downsampling_quality
    if args.upsampling_quality is not None:
        resample_quality_settings["upsample"] = args.upsampling_quality

    dataset = Dataset.from_json(file=Path(args.dataset_json_path))

    ads, sds = (
        dataset.get_dataset(ds_name) if ds_name else None
        for ds_name in (args.ads_name, args.sds_name)
    )
    subtype = None if args.subtype.lower() == "none" else args.subtype

    analysis = Analysis(args.analysis)

    write_analysis(
        analysis=analysis,
        ads=ads,
        sds=sds,
        subtype=subtype,
        matrix_folder_name=args.matrix_folder_name,
        spectrogram_folder_name=args.spectrogram_folder_name,
        first=args.first,
        last=args.last,
        link=True,
    )
