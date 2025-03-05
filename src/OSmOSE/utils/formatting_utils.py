from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from OSmOSE.config import FORBIDDEN_FILENAME_CHARACTERS
from OSmOSE.config import global_logging_context as glc

if TYPE_CHECKING:
    from pathlib import Path


def aplose2raven(df: pd.DataFrame) -> pd.DataFrame:
    """Export an APLOSE formatted result file to Raven formatted DataFrame

    Parameters
    ----------
    df: APLOSE formatted result DataFrame

    Returns
    -------
    df2raven: Raven formatted DataFrame

    Example of use
    --------------
    aplose_file = Path("path/to/aplose/result/file")
    df = (
        pd.read_csv(aplose_file, parse_dates=["start_datetime", "end_datetime"])
        .sort_values("start_datetime")
        .reset_index(drop=True)
    )

    df_raven = aplose2raven(df)

    # export to Raven format
    df2raven.to_csv('path/to/result/file.txt', sep='\t', index=False) # Raven export tab-separated files with a txt extension

    """
    start_time = [
        (st - df["start_datetime"][0]).total_seconds() for st in df["start_datetime"]
    ]
    end_time = [
        st + dur for st, dur in zip(start_time, df["end_time"] - df["start_time"])
    ]

    # annoying precision considerations
    precision = lambda v: len(str(v).split(".")[1])
    end_time_rounded = [
        et if type(et) is int else round(et, precision(st))
        for (st, et) in zip(start_time, end_time)
    ]

    df2raven = pd.DataFrame()
    df2raven["Selection"] = list(range(1, len(df) + 1))
    df2raven["View"], df2raven["Channel"] = [1] * len(df), [1] * len(df)
    df2raven["Begin Time (s)"] = start_time
    df2raven["End Time (s)"] = end_time_rounded
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
