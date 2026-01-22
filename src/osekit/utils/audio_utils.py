from __future__ import annotations

import enum
from typing import Literal, Self

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
        ``"repeat"``: audio data contain the same linear values from ``min`` to ``max``.
        ``"increase"``: audio data contain increasing values from ``min`` to ``max``.
        ``"sine"``: audio data contain sine waves with a peak value of ``max_value``.
        ``"noise"``: audio data contains white gaussian noise (``mean=0.``, ``std=1.``)
        Defaults to ``"repeat"``.
    sine_frequency: float (Optional)
        Frequency of the sine waves.
        Has no effect if ``series_type != "sine"``.
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
    """Resample the audio data using ``soxr``.

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


def normalize_raw(values: np.ndarray) -> np.ndarray:
    """No normalization of the audio data."""
    return values


def normalize_dc_reject(
    values: np.ndarray,
    dc_component: float | None = None,
) -> np.ndarray:
    """Reject the DC component of the audio data."""
    return values - (values.mean() if dc_component is None else dc_component)


def normalize_peak(values: np.ndarray, peak: float | None = None) -> np.ndarray:
    """Return values normalized so that the peak value is ``1.0``."""
    return values / (max(abs(values)) if peak is None else peak)


def normalize_zscore(
    values: np.ndarray,
    mean: float | None = None,
    std: float | None = None,
) -> np.ndarray:
    """Return normalized zscore from the audio data."""
    mean = values.mean() if mean is None else mean
    std = values.std() if std is None else std
    return (values - mean) / std


class NormalizationValider(enum.EnumMeta):
    """Metaclass used for validating the normalization flag.

    This is used because only ``REJECT_DC`` can be combined with (exactly)
    one other normalization.

    """

    def __call__(cls, *args, **kwargs) -> Self:  # noqa: ANN002, ANN003
        """Overwrite the call dunder."""
        instance = super().__call__(*args, **kwargs)

        mask = instance.value & ~Normalization.DC_REJECT.value
        if mask & (mask - 1):
            message = (
                "Combined normalizations can only be DC_REJECT combined "
                "with exactly one other normalization type."
            )
            raise ValueError(message)

        return instance


class Normalization(enum.Flag, metaclass=NormalizationValider):
    """Normalization to apply to the audio data.

    ``RAW``: No normalization is done.

    ``DC_REJECT``: Reject the DC component of the audio data.

    ``PEAK``: Divide the data by the absolute peak so that the peak value is ``1.0``.

    ``ZSCORE``: Normalize the data to a z-score with a mean of ``0.0`` and a
    std of ``1.0``.

    """

    RAW = enum.auto()
    DC_REJECT = enum.auto()
    PEAK = enum.auto()
    ZSCORE = enum.auto()


def normalize(
    values: np.ndarray,
    normalization: Normalization,
    mean: float | None = None,
    peak: float | None = None,
    std: float | None = None,
) -> np.ndarray:
    """Normalize the audio data."""
    if Normalization.DC_REJECT in normalization:
        values = normalize_dc_reject(values=values, dc_component=mean)
    if Normalization.PEAK in normalization:
        values = normalize_peak(values=values, peak=peak)
    if Normalization.ZSCORE in normalization:
        values = normalize_zscore(values=values, mean=mean, std=std)
    return values
