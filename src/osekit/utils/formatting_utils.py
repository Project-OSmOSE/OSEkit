from __future__ import annotations

import numpy as np
from pandas import DataFrame, Timedelta, Timestamp



def aplose2raven(
    aplose_result: DataFrame,
    list_audio_begin_time: list[Timestamp],
    audio_durations: list[float],
) -> DataFrame:
    r"""Format an APLOSE result DataFrame to a Raven result DataFrame.

    The list of audio files and durations considered for the Raven campaign should be
    provided to account for the deviations between the advertised and actual
    file durations.

    Parameters
    ----------
    aplose_result: Dataframe,
        APLOSE formatted result DataFrame.

    list_audio_begin_time: list[pd.Timestamp]
        list of tz-aware timestamps from considered audio files begin time.

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
    # index of the corresponding audio file for each detection
    index_detection = (
        np.searchsorted(list_audio_begin_time, aplose_result["start_datetime"], side="right")
        - 1
    )

"""
The following time adjustment is necessary because Raven does not account for
the duty cycle, nor for any potential offset between the end of one file and the start
of the next. To ensure that detection timestamps in the Aplose format align perfectly
with the spectrograms displayed by Raven, we need to correct the number of
seconds, since the software only uses the elapsed time from the beginning of the
first file to generate the bounding boxes.
"""

    # Add the begin time of the audio file corresponding to each detection
    aplose_result["wav_timestamp"] = [list_audio_begin_time[i] for i in index_detection]

    # Compute the time gaps (in seconds) between consecutive audio file begin time
    audio_begin_timegap = [td.total_seconds() for td in np.diff(list_audio_begin_time).tolist()]
    # Adjustment values: difference between each file's duration and the gap until the next file
    # (needed to account for potential gaps/overlaps between files)
    adjustment_values = [0]
    adjustment_values.extend([t1 - t2 for (t1, t2) in zip(audio_durations[:-1], audio_begin_timegap, strict=False)])
    # Cumulative adjustment in seconds, to realign all detection timestamps consistently
    cumsum_adjust = list(np.cumsum(adjustment_values))

    detection_begin_datetime_adjusted = []
    detection_end_datetime_adjusted = []
    for (detection_begin_time, detection_end_time, audio_begin_time, ind) in (zip(aplose_result["start_datetime"], aplose_result["end_datetime"],
                          aplose_result["wav_timestamp"], index_detection, strict=False)):
        """
        For duty cycled data, if the aplose_result detections were reshaped (eg : to 60-second duration),
        the start or end of the detection might virtually be located in a OFF duty cycle phase.
        This would cause issue in Raven, because the OFF part are not represented,
        and the detection start will be located on the previous audio file.
        The 2 following 'if' conditions apply the appropriate correction to make the Raven box (1)starts or (2) ends
         at the appropriate timing in Raven (ie at the begining or end of an audio file).
        """

        audio_begin_time_adjusted = audio_begin_time + Timedelta(seconds=audio_durations[ind])
        next_audio_begin_time_adjusted = audio_begin_time + Timedelta(seconds = audio_begin_timegap[ind])
        if audio_begin_time_adjusted < detection_begin_time < next_audio_begin_time_adjusted:
            correction_duration = (list_audio_begin_time[ind + 1] - detection_begin_time).total_seconds()
            detection_begin_datetime_adjusted.append(detection_begin_time +
                                                     Timedelta(seconds=cumsum_adjust[ind + 1]) +
                                                     Timedelta(seconds=correction_duration))
            detection_end_datetime_adjusted.append(detection_end_time +
                                                   Timedelta(seconds=cumsum_adjust[ind + 1]))
        elif audio_begin_time_adjusted < detection_end_time < next_audio_begin_time_adjusted:
            detection_begin_datetime_adjusted.append(
                detection_begin_time + Timedelta(seconds=cumsum_adjust[ind])
            )
            correction_duration = ((detection_end_time-detection_begin_time).total_seconds() -
                                   ((audio_begin_time + Timedelta(seconds=audio_durations[ind]))
                                    -detection_begin_time).total_seconds())
            detection_end_datetime_adjusted.append(detection_end_time +
                                                   Timedelta(seconds=cumsum_adjust[ind]) -
                                                   Timedelta(seconds=correction_duration))

        else:
            # Else, apply normal raven time correction
            detection_begin_datetime_adjusted.append(
                detection_begin_time + Timedelta(seconds=cumsum_adjust[ind])
            )
            detection_end_datetime_adjusted.append(
                detection_end_time + Timedelta(seconds=cumsum_adjust[ind])
            )

    # Convert the datetimes to seconds from the start of first audio (raven format)
    begin_time_adjusted = [
        (d - list_audio_begin_time[0]).total_seconds() for d in detection_begin_datetime_adjusted
    ]
    end_time_adjusted = [
        (d - list_audio_begin_time[0]).total_seconds() for d in detection_end_datetime_adjusted
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
    raven_result["Begin Date Time Real"] = aplose_result["start_datetime"]

    return raven_result
