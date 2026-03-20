"""Module that provides scripts for running public API transforms."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from osekit import config, setup_logging
from osekit.config import global_logging_context as glc
from osekit.core.audio_dataset import AudioDataset
from osekit.public.project import Project
from osekit.public.transform import OutputType
from osekit.utils.deserialization import deserialize_spectro_or_ltas_dataset

if TYPE_CHECKING:
    from osekit.core.spectro_dataset import SpectroDataset


def write_transform_output(
    output_type: OutputType,
    ads: AudioDataset | None,
    sds: SpectroDataset | None,
    subtype: str | None = None,
    matrix_folder_path: Path | None = None,
    spectrogram_folder_path: Path | None = None,
    welch_folder_path: Path | None = None,
    first: int = 0,
    last: int | None = None,
    logger: logging.Logger | None = None,
    *,
    link: bool = True,
) -> None:
    """Write ``SpectroDataset`` output files to disk.

    Parameters
    ----------
    output_type: OutputType
        Flags that should be used to specify the type of transform to run.
        See ``Transform.OutputType`` docstring for more info.
    subtype: str | None
        Subtype of the written audio files as provided by the soundfile module.
        Defaulted as the default ``16-bit PCM`` for ``wav`` audio files.
        This parameter has no effect if ``Transform.AUDIO`` is not in transform.
    ads: AudioDataset
        The ``AudioDataset`` of which the data should be written.
    sds: SpectroDataset
        The ``SpectroDataset`` of which the data should be written.
    matrix_folder_path: Path
        The folder in which the matrix ``npz`` files should be written.
    spectrogram_folder_path: Path
        The folder in which the spectrogram ``png`` files should be written.
    welch_folder_path: Path
        The folder in which the welch ``npz`` files should be written.
    link: bool
        If ``True``, the ads data will be linked to the exported files.
    first: int
        Index of the first data object to write.
    last: int|None
        Index after the last data object to write.
    logger: logging.Logger | None
        Logger to use to log the transform steps.

    """
    logger = glc.logger if logger is None else logger

    logger.info("Running transform...")

    if OutputType.AUDIO in output_type:
        logger.info("Writing audio files...")
        ads.write(
            folder=ads.folder,
            subtype=subtype,
            link=link,
            first=first,
            last=last,
        )
        ads.write_json(ads.folder)

    if (
        OutputType.SPECTRUM not in output_type
        and OutputType.SPECTROGRAM not in output_type
        and OutputType.WELCH not in output_type
    ):
        return

    # Avoid re-computing the reshaped audio
    if OutputType.AUDIO in output_type:
        sds.link_audio_dataset(ads, first=first, last=last)

    if OutputType.SPECTRUM in output_type and OutputType.SPECTROGRAM in output_type:
        logger.info("Computing and writing spectrum matrices and spectrograms...")
        sds.save_all(
            matrix_folder=matrix_folder_path,
            spectrogram_folder=spectrogram_folder_path,
            link=link,
            first=first,
            last=last,
        )
    elif OutputType.SPECTROGRAM in output_type:
        logger.info("Computing and writing spectrograms...")
        sds.save_spectrogram(
            folder=spectrogram_folder_path,
            first=first,
            last=last,
        )
    elif OutputType.SPECTRUM in output_type:
        logger.info("Computing and writing spectrum matrices...")
        sds.write(
            folder=matrix_folder_path,
            link=link,
            first=first,
            last=last,
        )
    if OutputType.WELCH in output_type:
        logger.info("Computing and writing welches...")
        sds.write_welch(
            folder=welch_folder_path,
            first=first,
            last=last,
        )

    # Update the sds from the JSON in case it has already been modified in another job
    sds.update_json_audio_data(first=first, last=last)
    logger.info("Transform done!")


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Export audio/spectro datasets.",
    )

    parser.add_argument(
        "--transform",
        "-a",
        required=True,
        help="Flags representing which files to export. See OutputType doc for more info.",
        type=int,
    )

    parser.add_argument(
        "--ads-json",
        "-ads",
        required=False,
        help="Path to the JSON of the AudioDataset to export.",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--sds-json",
        "-sds",
        required=False,
        help="Path to the JSON of the SpectroDataset to export.",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--subtype",
        "-sbtp",
        required=False,
        help="The subtype format of the audio files to export.",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--matrix-folder-path",
        "-mf",
        required=False,
        help="The path of the folder in which the npz matrix files are written.",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--spectrogram-folder-path",
        "-sf",
        required=False,
        help="The path of the folder in which the png spectrogram files are written.",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--welch-folder-path",
        "-wf",
        required=False,
        help="The path of the folder in which the npz welch files are written.",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--first",
        "-f",
        required=False,
        help="The index of the first file to export.",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--last",
        "-l",
        required=False,
        help="The index after the last file to export.",
        type=int,
        default=None,
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
        required=False,
        type=int,
        default=0o002,
        help="The umask to apply on the created file permissions.",
    )

    parser.add_argument(
        "--tqdm-disable",
        required=False,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable TQDM progress bars.",
    )

    parser.add_argument(
        "--multiprocessing",
        required=False,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Turn multiprocessing on or off.",
    )

    parser.add_argument(
        "--use-logging-setup",
        required=False,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Call osekit.setup_logging() before running the transform.",
    )

    parser.add_argument(
        "--nb-processes",
        required=False,
        type=str,
        default=None,
        help="Set the number of processes to use.",
    )

    parser.add_argument(
        "--dataset-json-path",
        "-p",
        required=False,
        help="The path to the Project JSON file of which to use the logger.",
        type=str,
        default=None,
    )

    return parser


def main() -> None:
    """Export a transform."""
    args = create_parser().parse_args()

    os.environ["DISABLE_TQDM"] = str(args.tqdm_disable)

    if args.use_logging_setup:
        setup_logging()

    config.multiprocessing["is_active"] = args.multiprocessing
    if (nb_processes := args.nb_processes) is not None:
        config.multiprocessing["nb_processes"] = (
            None if nb_processes.lower() == "none" else int(nb_processes)
        )

    os.umask(args.umask)

    if args.downsampling_quality is not None:
        config.resample_quality_settings["downsample"] = args.downsampling_quality
    if args.upsampling_quality is not None:
        config.resample_quality_settings["upsample"] = args.upsampling_quality

    logger = (
        logging.getLogger()
        if (args.dataset_json_path is None or args.dataset_json_path.lower() == "none")
        else Project.from_json(Path(args.dataset_json_path)).logger
    )

    ads = (
        AudioDataset.from_json(Path(args.ads_json))
        if args.ads_json.lower() != "none"
        else None
    )
    sds = (
        deserialize_spectro_or_ltas_dataset(path=Path(args.sds_json))
        if args.sds_json.lower() != "none"
        else None
    )

    subtype = None if args.subtype.lower() == "none" else args.subtype

    output_type = OutputType(args.transform)

    write_transform_output(
        output_type=output_type,
        ads=ads,
        sds=sds,
        subtype=subtype,
        matrix_folder_path=Path(args.matrix_folder_path),
        spectrogram_folder_path=Path(args.spectrogram_folder_path),
        welch_folder_path=Path(args.welch_folder_path),
        first=args.first,
        last=args.last,
        link=True,
        logger=logger,
    )


if __name__ == "__main__":
    main()
