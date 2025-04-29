from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import soundfile as sf
import soxr
from pandas import Timedelta

from OSmOSE.config import (
    AUDIO_METADATA,
    BUILD_DURATION_DELTA_THRESHOLD,
    SUPPORTED_AUDIO_FORMAT,
    resample_quality_settings,
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
    directory : Path
        The path to the directory to search for audio files

    Returns
    -------
    list[Path]
        A list of `Path` objects corresponding to the supported audio files
        found in the directory.

    """
    return sorted(
        file
        for file in directory.iterdir()
        if any(
            file.name.lower().endswith(extension)
            for extension in SUPPORTED_AUDIO_FORMAT
        )
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
    with sf.SoundFile(audio_file, "r") as af:
        return {key: f(af) for key, f in AUDIO_METADATA.items()}


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
    audio_files = set(audio_metadata["filename"].unique())
    timestamps = set(timestamps["filename"].unique())

    unlisted_files = sorted(audio_files - timestamps)
    missing_files = sorted(timestamps - audio_files)

    if unlisted_files:
        message = (
            "The following files have not been found in timestamp.csv:\n\t"
            + "\n\t".join(unlisted_files)
        )
        raise FileNotFoundError(message)

    if missing_files:
        message = (
            "The following files are listed in timestamp.csv but hasn't be found:\n\t"
            + "\n\t".join(missing_files)
        )
        raise FileNotFoundError(message)

    if len(sample_rates := audio_metadata["origin_sr"].unique()) > 1:
        message = (
            "Your files do not have all the same sampling rate.\n"
            f"Found sampling rates: {', '.join(str(sr) + ' Hz' for sr in sample_rates)}."
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
    series_type: Literal["repeat", "increase", "sine"] = "repeat",
    sine_frequency: float = 1000.0,
    min_value: float = 0.0,
    max_value: float = 1.0,
    duration: float = 1.0,
    dtype: np.dtype = np.float64,
) -> list[np.ndarray]:
    """Generate sample audio data.

    Parameters
    ----------
    nb_files: int
        Number of audio data to generate.
    nb_samples: int
        Number of samples per audio data.
    series_type: Literal["repeat", "increase", "sine"] (Optional)
        "repeat": audio data contain the same linear values from min to max.
        "increase": audio data contain increasing values from min to max.
        "sine": audio data contain sine waves with a peak value of max_value.
        Defaults to "repeat".
    sine_frequency: float (Optional)
        Frequency of the sine waves.
        Has no effect if series_type is not "sine".
    min_value: float
        Minimum value of the audio data.
    max_value: float
        Maximum value of the audio data.
    duration: float
        Duration of the audio data in seconds.
        Used to compute the frequency of sine waves.
    dtype: np.dtype
        The type of the output array.

    Returns
    -------
    list[numpy.ndarray]:
        The generated audio data.

    """
    if duration is None:
        duration = Timedelta(seconds=1)
    if series_type == "repeat":
        return np.split(
            np.tile(
                np.linspace(min_value, max_value, nb_samples, dtype=dtype),
                nb_files,
            ),
            nb_files,
        )
    if series_type == "increase":
        return np.split(
            np.linspace(min_value, max_value, nb_samples * nb_files, dtype=dtype),
            nb_files,
        )
    if series_type == "sine":
        t = np.linspace(0, duration, nb_samples)
        return np.split(
            np.tile(
                np.sin(2 * np.pi * sine_frequency * t, dtype=dtype) * max_value,
                nb_files,
            ),
            nb_files,
        )
    return np.split(np.empty(nb_samples * nb_files, dtype=dtype), nb_files)


def resample(data: np.ndarray, origin_sr: float, target_sr: float) -> np.ndarray:
    """Resample the audio data using soxr.

    Parameters
    ----------
    data: np.ndarray
        The audio data to resample.
    origin_sr:
        The sampling rate of the audio data.
    target_sr:
        The sampling rate of the resampled audio data.

    Returns
    -------
    np.ndarray
        The resampled audio data.

    """
    quality = (
        resample_quality_settings["upsample"]
        if target_sr > origin_sr
        else resample_quality_settings["downsample"]
    )
    return soxr.resample(data, origin_sr, target_sr, quality=quality)
