from __future__ import annotations

from typing import Literal

import numpy as np
import soxr
from pandas import Timedelta

from osekit.config import (
    resample_quality_settings,
)


def generate_sample_audio(
    nb_files: int,
    nb_samples: int,
    series_type: Literal["repeat", "increase", "sine", "noise"] = "repeat",
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
        "noise": audio data contains white gaussian noise (mean=0., std=1.)
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
    if series_type == "noise":
        generator = np.random.default_rng(seed=1)
        sig = generator.normal(0.0, 1.0, size=nb_samples)
        return np.split(
            sig,
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
