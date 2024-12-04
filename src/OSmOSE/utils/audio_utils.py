from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from OSmOSE.config import (
    AUDIO_METADATA,
    BUILD_DURATION_DELTA_THRESHOLD,
    SUPPORTED_AUDIO_FORMAT,
)


def is_supported_audio_format(filename: Path) -> bool:
    """Check if a given file is a supported audio file based on its extension.

    Parameters
    ----------
    filename : Path
        The path to the file to be checked.

    Returns
    -------
    bool
        True if the file has an extension that matches a supported audio format,
        False otherwise.

    """
    return filename.suffix.lower() in SUPPORTED_AUDIO_FORMAT


def get_all_audio_files(directory: Path) -> list[Path]:
    """Retrieve all supported audio files from a given directory.

    Parameters
    ----------
    file_path : Path
        The path to the directory to search for audio files

    Returns
    -------
    list[Path]
        A list of `Path` objects corresponding to the supported audio files
        found in the directory.

    """
    return sorted(
        file
        for extension in SUPPORTED_AUDIO_FORMAT
        for file in directory.glob(f"*{extension}")
    )


def get_audio_metadata(audio_file: Path) -> dict:
    """Read metadata from the audio file.

    Parameters
    ----------
    audio_file: pathlib.Path
        The path to the audio file.

    Returns
    -------
    dict:
        A dictionary containing the metadata of the audio file.
        The metadata list is grabbed from OSmOSE.config.AUDIO_METADATA.

    """
    return {key: f(audio_file) for key, f in AUDIO_METADATA.items()}


def check_audio(
    audio_metadata: pd.DataFrame,
    timestamps: pd.DataFrame,
) -> None:
    """Raise errors if the audio files present anomalies.

    Parameters
    ----------
    audio_metadata: pandas.DataFrame
        metadata of the audio files
    timestamps: pandas.DataFrame
        timestamps of the audio files

    Raises
    ------
    FileNotFoundError:
        If the number of audio files doesn't match the number of timestamps.
    ValueError:
        If the audio files present one of the following anomalies:
        - Different sampling rates among audio files
        - Large duration differences (> 5% of the mean duration) among audio files

    """
    if any(
        (unlisted_file := file) not in timestamps["filename"].unique()
        for file in audio_metadata["filename"]
    ):
        message = f"{unlisted_file} has not been found in timestamp.csv"
        raise FileNotFoundError(message)

    if any(
        (missing_file := filename) not in audio_metadata["filename"].unique()
        for filename in timestamps["filename"]
    ):
        message = f"{missing_file} is listed in timestamp.csv but hasn't be found."
        raise FileNotFoundError(message)

    if len(audio_metadata["origin_sr"].unique()) > 1:
        message = (
            "Your files do not have all the same sampling rate. "
            f"Found sampling rates: {', '.join(str(sr) + ' Hz' for sr in audio_metadata['origin_sr'].unique())}."
        )
        raise ValueError(message)

    mean_duration = audio_metadata["duration"].mean()
    if any(
        abs(mean_duration - d) > BUILD_DURATION_DELTA_THRESHOLD * mean_duration
        for d in audio_metadata["duration"].unique()
    ):
        message = "Your audio files have large duration discrepancies."
        raise ValueError(message)

def generate_sample_audio(
    nb_files: int,
    nb_samples: int,
    series_type: Literal["repeat", "increase"] = "repeat",
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> list[np.ndarray]:
    """Generate sample audio data.

    Parameters
    ----------
    nb_files: int
        Number of audio data to generate.
    nb_samples: int
        Number of samples per audio data.
    series_type: Literal["repeat", "increase"] (Optional)
        "repeat": audio data contain the same linear values from min to max.
        "increase": audio data contain increasing values from min to max.
        Defaults to "repeat".
    min_value: float
        Minimum value of the audio data.
    max_value: float
        Maximum value of the audio data.

    Returns
    -------
    list[numpy.ndarray]:
        The generated audio data.

    """
    if series_type == "repeat":
        return np.split(np.tile(np.linspace(min_value, max_value, nb_samples), nb_files), nb_files)
    if series_type == "increase":
        return np.split(np.linspace(min_value, max_value, nb_samples * nb_files), nb_files)
    return np.split(np.empty(nb_samples * nb_files), nb_files)
