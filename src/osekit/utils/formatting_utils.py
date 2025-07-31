from __future__ import annotations

import numpy as np
import pytz
from pandas import DataFrame, Timedelta

from osekit.core_api.audio_dataset import AudioFile


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

    # Add beg datetime of the wavfile
    aplose_result["wav_timestamp"] = [audio_datetimes[i] for i in index_detection]
    # Add theoretical start datetime so that the information stays written in the Raven .txt
    aplose_result["start_datetime_backup"] = aplose_result["start_datetime"]

    # time differences between consecutive datetimes and add wav_duration
    filename_diff = [td.total_seconds() for td in np.diff(audio_datetimes).tolist()]
    adjust = [0]
    adjust.extend([t1 - t2 for (t1, t2) in zip(audio_durations[:-1], filename_diff, strict=False)])
    cumsum_adjust = list(np.cumsum(adjust))

    # adjusted datetimes to match Raven annoying functioning
    begin_datetime_adjusted = []
    end_datetime_adjusted = []
    for (beg_det, end_det, beg_wav, ind) in (zip(aplose_result["start_datetime"], aplose_result["end_datetime"],
                          aplose_result["wav_timestamp"], index_detection, strict=False)):
        """
        For duty cycled data, if the aplose_result detections were reshaped (eg : to 60-second duration),
        the start of the detection might virtually be located in a OFF duty cycle phase.
        This would cause issue in Raven, because the OFF part are not represented,
        and the detection start will be located on the previous wav file.
        The following if condition apply the appropriate correction to make the Raven box start
         at the begining of the wav file
        """

        if (beg_wav + Timedelta(seconds=audio_durations[ind])) < beg_det < (beg_wav + Timedelta(seconds = filename_diff[ind])):
            corr_dur = (audio_datetimes[ind + 1] - beg_det).total_seconds()
            begin_datetime_adjusted.append(beg_det + Timedelta(seconds=cumsum_adjust[ind + 1]) + Timedelta(seconds=corr_dur))
            end_datetime_adjusted.append(end_det + Timedelta(seconds=cumsum_adjust[ind + 1]))
        else:
            # Else, apply normal raven time correction
            begin_datetime_adjusted.append(
                beg_det + Timedelta(seconds=cumsum_adjust[ind])
            )
            end_datetime_adjusted.append(
                end_det + Timedelta(seconds=cumsum_adjust[ind])
            )

    # Convert the datetimes to seconds from the start of first wav (raven format)
    begin_time_adjusted = [
        (d - audio_datetimes[0]).total_seconds() for d in begin_datetime_adjusted
    ]
    end_time_adjusted = [
        (d - audio_datetimes[0]).total_seconds() for d in end_datetime_adjusted
    ]
    # Build corrected Raven selection table
    raven_result = DataFrame()
    raven_result["Selection"] = list(range(1, len(aplose_result) + 1))
    raven_result["View"] = [1] * len(aplose_result)
    raven_result["Channel"] = [1] * len(aplose_result)
    raven_result["Begin Time (s)"] = begin_time_adjusted
    raven_result["End Time (s)"] = end_time_adjusted
    raven_result["Low Freq (Hz)"] = aplose_result["start_frequency"]
    raven_result["High Freq (Hz)"] = aplose_result["end_frequency"]
    raven_result["Begin Date Time Real"] = aplose_result["start_datetime_backup"]

    return raven_result
