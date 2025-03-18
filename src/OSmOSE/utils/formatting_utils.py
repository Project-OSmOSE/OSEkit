from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from OSmOSE.config import FORBIDDEN_FILENAME_CHARACTERS
from OSmOSE.config import global_logging_context as glc

if TYPE_CHECKING:
    from pathlib import Path


def aplose2raven(
    df: pd.DataFrame,
    audio_datetimes: list[pd.Timestamp],
    audio_durations: list[float],
) -> pd.DataFrame:
    r"""Format an APLOSE result DataFrame to a Raven result DataFrame.

    The list of audio files and durations considered for the Raven campaign should be
    provided to account for the deviations between the advertised and actual
    file durations.

    Parameters
    ----------
    df: Dataframe,
        APLOSE formatted result DataFrame.

    audio_datetimes: list[pd.Timestamp]
        list of tz-aware timestamps from considered audio files.

    audio_durations: list[float]
        list of all considered audio file durations.

    Returns
    -------
    Raven formatted DataFrame.

    Example of use
    --------------
    aplose_file = Path("path/to/aplose/result/file")
    timestamp_list = list(filenames)
    duration_list = list(durations)

    df = (
        pd.read_csv(aplose_file, parse_dates=["start_datetime", "end_datetime"])
        .sort_values("start_datetime")
        .reset_index(drop=True)
    )
    df_raven = aplose2raven(df, filename_list, duration_list)

    # export to Raven format
    df2raven.to_csv('path/to/result/file.txt', sep='\t', index=False)
    # Raven export tab-separated files with a txt extension

    """
    # index of the corresponding wav file for each detection
    index_detection = (
        np.searchsorted(audio_datetimes, df["start_datetime"], side="right") - 1
    )

    # time differences between consecutive datetimes and add wav_duration
    filename_diff = [td.total_seconds() for td in np.diff(audio_datetimes).tolist()]
    adjust = [0]
    adjust.extend([t1 - t2 for (t1, t2) in zip(audio_durations[:-1], filename_diff)])
    cumsum_adjust = list(np.cumsum(adjust))

    # adjusted datetimes to match Raven annoying functioning
    begin_datetime_adjusted = [
        det + pd.Timedelta(seconds=cumsum_adjust[ind])
        for (det, ind) in zip(df["start_datetime"], index_detection)
    ]
    end_datetime_adjusted = [
        det + pd.Timedelta(seconds=cumsum_adjust[ind])
        for (det, ind) in zip(df["end_datetime"], index_detection)
    ]
    begin_time_adjusted = [
        (d - audio_datetimes[0]).total_seconds() for d in begin_datetime_adjusted
    ]
    end_time_adjusted = [
        (d - audio_datetimes[0]).total_seconds() for d in end_datetime_adjusted
    ]

    df2raven = pd.DataFrame()
    df2raven["Selection"] = list(range(1, len(df) + 1))
    df2raven["View"], df2raven["Channel"] = [1] * len(df), [1] * len(df)
    df2raven["Begin Time (s)"] = begin_time_adjusted
    df2raven["End Time (s)"] = end_time_adjusted
    df2raven["Low Freq (Hz)"] = df["start_frequency"]
    df2raven["High Freq (Hz)"] = df["end_frequency"]

    return df2raven


def clean_filenames(filenames: list[Path]) -> tuple[list[Path], dict[Path, Path]]:
    """Clean filenames to replace forbidden characters in the OSmOSE format.

    Parameters
    ----------
    filenames: list[Path]
        Iterable of paths to audio files.

    Returns
    -------
    tuple[list[Path],dict[Path,Path]]
        list[Path]: filenames where the incorrect filenames have been replaced.
        dict[Path,Path]: Dictionary with incorrect filenames
        as keys and corrected filenames as values.

    """
    corrected_files = {}
    for idx, file in enumerate(filenames):
        if not has_forbidden_characters(file.name) and file.suffix.islower():
            continue
        corrected = file.stem
        corrected = clean_forbidden_characters(corrected)
        corrected = file.parent / f"{corrected}{file.suffix.lower()}"
        corrected_files[file] = corrected
        filenames[idx] = corrected

    if corrected_files:
        glc.logger.warning(
            "Audio file names have been cleaned.",
        )
    return filenames, corrected_files


def has_forbidden_characters(filename: str) -> bool:
    """Return true if the filename contains forbidden characters."""
    return any(c in FORBIDDEN_FILENAME_CHARACTERS for c in filename)


def clean_forbidden_characters(text: str) -> str:
    """Replace all forbidden characters in a given string with its replacement."""
    for forbidden_character, replacement in FORBIDDEN_FILENAME_CHARACTERS.items():
        text = text.replace(forbidden_character, replacement)
    return text
