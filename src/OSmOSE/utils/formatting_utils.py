from __future__ import annotations

import numpy as np
import pandas as pd


def aplose2raven(
    aplose_result: pd.DataFrame,
    audio_datetimes: list[pd.Timestamp],
    audio_durations: list[float],
) -> pd.DataFrame:
    r"""Format an APLOSE result DataFrame to a Raven result DataFrame.

    The list of audio files and durations considered for the Raven campaign should be
    provided to account for the deviations between the advertised and actual
    file durations.

    Parameters
    ----------
    aplose_result: Dataframe,
        APLOSE formatted result DataFrame.

    audio_datetimes: list[pd.Timestamp]
        list of tz-aware timestamps from considered audio files.

    audio_durations: list[float]
        list of all considered audio file durations in seconds.

    Returns
    -------
    Raven formatted DataFrame.

    Example of use
    --------------
    aplose_file = Path("path/to/aplose/result/file")
    timestamp_list = list(filenames)
    duration_list = list(durations)

    aplose_result = (
        pd.read_csv(aplose_file, parse_dates=["start_datetime", "end_datetime"])
        .sort_values("start_datetime")
        .reset_index(drop=True)
    )
    raven_result = aplose2raven(aplose_result, filename_list, duration_list)

    # export to Raven format: tab-separated files with a txt extension
    raven_result.to_csv('path/to/result/file.txt', sep='\t', index=False)

    """
    # index of the corresponding wav file for each detection
    index_detection = (
        np.searchsorted(audio_datetimes, aplose_result["start_datetime"], side="right")
        - 1
    )

    # time differences between consecutive datetimes and add wav_duration
    filename_diff = [td.total_seconds() for td in np.diff(audio_datetimes).tolist()]
    adjust = [0]
    adjust.extend([t1 - t2 for (t1, t2) in zip(audio_durations[:-1], filename_diff)])
    cumsum_adjust = list(np.cumsum(adjust))

    # adjusted datetimes to match Raven annoying functioning
    begin_datetime_adjusted = [
        det + pd.Timedelta(seconds=cumsum_adjust[ind])
        for (det, ind) in zip(aplose_result["start_datetime"], index_detection)
    ]
    end_datetime_adjusted = [
        det + pd.Timedelta(seconds=cumsum_adjust[ind])
        for (det, ind) in zip(aplose_result["end_datetime"], index_detection)
    ]
    begin_time_adjusted = [
        (d - audio_datetimes[0]).total_seconds() for d in begin_datetime_adjusted
    ]
    end_time_adjusted = [
        (d - audio_datetimes[0]).total_seconds() for d in end_datetime_adjusted
    ]

    raven_result = pd.DataFrame()
    raven_result["Selection"] = list(range(1, len(aplose_result) + 1))
    raven_result["View"] = [1] * len(aplose_result)
    raven_result["Channel"] = [1] * len(aplose_result)
    raven_result["Begin Time (s)"] = begin_time_adjusted
    raven_result["End Time (s)"] = end_time_adjusted
    raven_result["Low Freq (Hz)"] = aplose_result["start_frequency"]
    raven_result["High Freq (Hz)"] = aplose_result["end_frequency"]

    return raven_result
