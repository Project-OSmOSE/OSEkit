from __future__ import annotations

import numpy as np
from pandas import DataFrame, Timedelta, Timestamp


def aplose2raven(
    aplose_result: DataFrame,
    list_audio_begin_time: list[Timestamp],
    audio_durations: list[Timedelta],
) -> DataFrame:
    r"""Format an APLOSE result ``DataFrame`` to a Raven result ``DataFrame``.

    The list of audio files and durations considered for the Raven campaign should be
    provided to account for the deviations between the advertised and actual
    file durations.

    Parameters
    ----------
    aplose_result: Dataframe,
        APLOSE formatted result ``DataFrame``.

    list_audio_begin_time: list[Timestamp]
        list of tz-aware timestamps from considered audio files begin time.

    audio_durations: list[Timedelta]
        list of all considered audio file durations.

    Returns
    -------
    Raven formatted ``DataFrame``.

    Example of use
    --------------
    >>> from pathlib import Path
    >>> from pandas import read_csv
    >>> from osekit.core_api.audio_dataset import AudioDataset
    >>> from osekit.utils.formatting_utils import aplose2raven

    >>> dataset_folder = Path(r"path\to\audio\folder")
    >>> dataset = AudioDataset.from_folder(dataset_folder,
    >>>                                    strptime_format="strptime_format",
    >>>                                    timezone='utc',
    >>>                                   )

    >>> begin_list = sorted([f.begin for f in list(dataset.files)])
    >>> duration_list = sorted([f.duration for f in list(dataset.files)])

    >>> csv = Path(r"path\to\result\csv")
    >>> df = read_csv(csv,
    >>>               parse_dates=["start_datetime", "end_datetime"]
    >>>               ).sort_values("start_datetime")
    >>>                .reset_index(drop=True)

    >>> df_raven = aplose2raven(df, begin_list, duration_list)
    >>> raven_result.to_csv('path/to/result/file.txt', sep='\t', index=False)

    """
    # index of the corresponding audio file for each detection
    index_detection = (
        np.searchsorted(
            list_audio_begin_time,
            aplose_result["start_datetime"],
            side="right",
        )
        - 1
    )

    """
    The following time adjustment is necessary because Raven does not account
    for the duty cycle, nor for any potential offset between the end of one
    file and the start of the next. To ensure that detection timestamps in
    APLOSE format align with the spectrograms displayed by Raven, a correction
    of the number of seconds is required, since the software only uses the
    elapsed time from the beginning of the first file to generate the bounding boxes.
    """

    # Add the begin time of the audio file corresponding to each detection
    aplose_result["wav_timestamp"] = [list_audio_begin_time[i] for i in index_detection]

    # Compute the time gaps between consecutive audio file begin time
    audio_begin_timegap = list(np.diff(list_audio_begin_time).tolist())

    # Adjustment values: difference between each file's duration
    # and the gap until the next file.
    # (Required to account for potential gaps/overlaps between files)
    adjustment_values = [Timedelta(0)]
    adjustment_values.extend(
        [
            t1 - t2
            for (t1, t2) in zip(audio_durations[:-1], audio_begin_timegap, strict=False)
        ],
    )

    # Cumulative adjustment in seconds, to realign all detection timestamps consistently
    cumsum_adjust = list(np.cumsum(adjustment_values))

    detection_begin_datetime_adjusted = []
    detection_end_datetime_adjusted = []
    for i in range(len(aplose_result)):
        detection_begin_time = aplose_result["start_datetime"].iloc[i]
        detection_end_time = aplose_result["end_datetime"].iloc[i]
        audio_begin_time = aplose_result["wav_timestamp"].iloc[i]
        ind = index_detection[i]
        """
        For duty cycled data, if aplose_result detections were reshaped (eg to 60s duration),
        the start or end of the detection might virtually be located in a OFF duty cycle phase.
        This would cause issue in Raven, because the OFF part are not represented,
        and the detection start will be located on the previous audio file.
        The 2 following 'if' conditions apply the appropriate correction
        to make the Raven box (1)starts or (2) ends.
        at the appropriate timing in Raven (ie at the begining or end of an audio file).
        """

        audio_begin_time_adjusted = audio_begin_time + audio_durations[ind]

        if ind < len(audio_begin_timegap):
            next_audio_begin_time_adjusted = audio_begin_time + audio_begin_timegap[ind]
        else:
            next_audio_begin_time_adjusted += audio_durations[ind]

        if (
            audio_begin_time_adjusted
            < detection_begin_time
            < next_audio_begin_time_adjusted
        ):
            correction_duration = list_audio_begin_time[ind + 1] - detection_begin_time
            detection_begin_datetime_adjusted.append(
                detection_begin_time + cumsum_adjust[ind + 1] + correction_duration,
            )
            detection_end_datetime_adjusted.append(
                detection_end_time + cumsum_adjust[ind + 1],
            )
        elif (
            audio_begin_time_adjusted
            < detection_end_time
            < next_audio_begin_time_adjusted
        ):
            detection_begin_datetime_adjusted.append(
                detection_begin_time + cumsum_adjust[ind],
            )
            correction_duration = (detection_end_time - detection_begin_time) - (
                (audio_begin_time + audio_durations[ind]) - detection_begin_time
            )
            detection_end_datetime_adjusted.append(
                detection_end_time + cumsum_adjust[ind] - correction_duration,
            )

        else:
            # Else, apply normal Raven time correction
            detection_begin_datetime_adjusted.append(
                detection_begin_time + cumsum_adjust[ind],
            )
            detection_end_datetime_adjusted.append(
                detection_end_time + cumsum_adjust[ind],
            )

    # Convert the datetimes to seconds from the start of first audio (raven format)
    begin_time_adjusted = [
        (d - list_audio_begin_time[0]).total_seconds()
        for d in detection_begin_datetime_adjusted
    ]
    end_time_adjusted = [
        (d - list_audio_begin_time[0]).total_seconds()
        for d in detection_end_datetime_adjusted
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
