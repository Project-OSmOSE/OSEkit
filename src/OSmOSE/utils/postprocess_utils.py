import numpy as np
import pandas as pd
import pytz
import csv
from typing import List
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter
import bisect
from OSmOSE.utils.core_utils import (
    t_rounder,
    extract_datetime,
)


def sorting_detections(
    file: List[str],
    tz: pytz._FixedOffset = None,
    date_begin: pd.Timestamp = None,
    date_end: pd.Timestamp = None,
    annotator: str = None,
    annotation: str = None,
    box: bool = False,
    timebin_new: int = None,
    user_sel: str = "all",
    fmin_filter: int = None,
    fmax_filter: int = None,
) -> (pd.DataFrame, pd.DataFrame):
    """Filters an Aplose formatted detection file according to user specified filters

    Parameters
    -------
        file: List[str], 'str'
            list of path(s) to the detection file(s), can be a str too

        tz: pytz._FixedOffset
            timezone info, to be specified if the user wants to change the TZ of the detections
            Default value: None

        date_begin: pd.Timestamp
            datetime to be specified if the user wants to select detections after date_begin
            Default value: None

        date_end: pd.Timestamp
            datetime to be specified if the user wants to select detections before date_end
            Default value: None

        annotator: 'str'
            string to be specified if the user wants to select the detection of a particular annotator
            Default value: None

        annotation: 'str'
            string to be specified if the user wants to select the detection of a particular label
            Default value: None

        box: 'bool'
            If set to True, keeps all the annotations (strong detection)
            If set to False keeps only the absence/presence box (weak detection)
            Default value: False

        timebin_new: 'int'
            int to be specified if the user already know the new time resolution to set the detection file to
            Default value: None

        user_sel: 'str'
            str to specify to filter detections of a file based on annotators
                -'union' : the common detections of all annotators and the unique detections of each annotators are selected
                -'intersection' : only the common detections of all annotators are selected
                -'all' : all the detections are selected
            Default value: 'all'

        fmax_filter: 'int'
            int, in the case where the user wants to filter out detections based on their max frequency
            Default value: None

        fmin_filter: 'int'
            int, in the case where the user wants to filter out detections based on their min frequency
            Default value: None

    Returns
    -------
        df: pd.DataFrame
            APLOSE formatted DataFrame corresponding to the filters applied and containing all the detections

        info: pd.DataFrame
            DataFrame containing infos such as max_time/max_freq/annotators/labels corresponding to each detection file
                - max_time: spectrogram temporal length
                - max_freq: sampling frequency * 0.5
                - annotators: list of annotators after filtering
                - labels: list of labels after filtering
    """

    # Find the proper delimiter for file
    with open(file, "r", newline="") as csv_file:
        try:
            temp_lines = csv_file.readline() + "\n" + csv_file.readline()
            dialect = csv.Sniffer().sniff(temp_lines, delimiters=",;")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file, sep=delimiter)
    list_annotators = list(df["annotator"].drop_duplicates())
    list_labels = list(df["annotation"].drop_duplicates())
    max_freq = int(max(df["end_frequency"]))
    max_time = int(max(df["end_time"]))

    # Convert datetime columns to proper format
    df["start_datetime"] = pd.to_datetime(
        df["start_datetime"], format="%Y-%m-%dT%H:%M:%S.%f%z"
    )
    df["end_datetime"] = pd.to_datetime(
        df["end_datetime"], format="%Y-%m-%dT%H:%M:%S.%f%z"
    )
    df = df.sort_values("start_datetime")

    # Apply timezone conversion if specified
    if tz is not None:
        df["start_datetime"] = [x.tz_convert(tz) for x in df["start_datetime"]]
        df["end_datetime"] = [x.tz_convert(tz) for x in df["end_datetime"]]

    tz_data = df["start_datetime"][0].tz

    # Filter detections based on date_begin/date_end if specified
    if date_begin is not None:
        df = df[df["start_datetime"] >= date_begin]

    if date_end is not None:
        df = df[df["end_datetime"] <= date_end]

    if date_begin is not None and date_end is not None:
        if date_begin >= date_end:
            raise ValueError("Error: date_begin > date_end")

    # Filter detections based on annotator if specified
    if annotator is not None:
        df = df.loc[(df["annotator"] == annotator)]
        list_annotators = [annotator]

    # Filter detections based on annotation if specified
    if annotation is not None:
        df = df.loc[(df["annotation"] == annotation)]
        list_labels = [annotation]

    # Filter detections based on fmin_filter/fmax_filter if specified
    if fmin_filter is not None:
        df = df[df["start_frequency"] >= fmin_filter]
        if len(df) == 0:
            raise Exception("No detection found after fmin filtering, upload aborted")

    if fmax_filter is not None:
        df = df[df["end_frequency"] <= fmax_filter]
        if len(df) == 0:
            raise Exception("No detection found after fmax filtering, upload aborted")

    # Filter detections based on timebin_new if specified
    df_nobox = df.loc[
        (df["start_time"] == 0)
        & (df["end_time"] == max_time)
        & (df["end_frequency"] == max_freq)
    ]
    if len(df_nobox) == 0:
        max_time = 0

    if box is False:
        if len(df_nobox) == 0:
            df = reshape_timebin(df=df.reset_index(drop=True), timebin_new=timebin_new)
            max_time = int(max(df["end_time"]))
        else:
            if timebin_new is not None:
                df = reshape_timebin(
                    df=df.reset_index(drop=True), timebin_new=timebin_new
                )
                max_time = int(max(df["end_time"]))
            else:
                df = df_nobox

    # Filter detections based on user_sel if specified
    if len(list_annotators) > 1:
        if user_sel == "union" or user_sel == "intersection":
            df_inter = pd.DataFrame()
            df_diff = pd.DataFrame()
            for label_sel in list_labels:
                df_label = df[df["annotation"] == label_sel]
                values = list(df_label["start_datetime"].drop_duplicates())
                common_values = []
                diff_values = []
                error_values = []
                for value in values:
                    if df_label["start_datetime"].to_list().count(value) == 2:
                        common_values.append(value)
                    elif df_label["start_datetime"].to_list().count(value) == 1:
                        diff_values.append(value)
                    else:
                        error_values.append(value)

                df_label_inter = df_label[
                    df_label["start_datetime"].isin(common_values)
                ].reset_index(drop=True)
                df_label_inter = df_label_inter.drop_duplicates(subset="start_datetime")
                df_inter = pd.concat([df_inter, df_label_inter]).reset_index(drop=True)

                df_label_diff = df_label[
                    df_label["start_datetime"].isin(diff_values)
                ].reset_index(drop=True)
                df_diff = pd.concat([df_diff, df_label_diff]).reset_index(drop=True)

            if user_sel == "intersection":
                df = df_inter
                list_annotators = [" ∩ ".join(list_annotators)]
            elif user_sel == "union":
                df = pd.concat([df_diff, df_inter]).reset_index(drop=True)
                df = df.sort_values("start_datetime")
                list_annotators = [" ∪ ".join(list_annotators)]

            df["annotator"] = list_annotators[0]

    columns = ["file", "max_time", "max_freq", "annotators", "labels", "tz_data"]
    info = pd.DataFrame(
        [[file, int(max_time), max_freq, list_annotators, list_labels, tz_data]],
        columns=columns,
    )

    return df.reset_index(drop=True), info


def reshape_timebin(df: pd.DataFrame, timebin_new: int = None) -> pd.DataFrame:
    """Changes the timebin (time resolution) of a detection dataframe
    ex :    -from a raw PAMGuard detection file to a detection file with 10s timebin
            -from an 10s detection file to a 1min / 1h / 24h detection file
    Parameters
    -------
        df: pd.DataFrame
            detection dataframe

        timebin_new: 'int'
            Time resolution to base the detections on, if not provided it is asked to the user
    Returns
    -------
        df_new: pd.DataFrame
            detection dataframe with the new timebin
    """
    if isinstance(df, pd.DataFrame) is False:
        raise ValueError("Not a dataframe passed, reshape aborted")

    annotators = list(df["annotator"].drop_duplicates())
    labels = list(df["annotation"].drop_duplicates())

    df_nobox = df.loc[
        (df["start_time"] == 0)
        & (df["end_time"] == max(df["end_time"]))
        & (df["end_frequency"] == max(df["end_frequency"]))
    ]
    max_time = 0 if len(df_nobox) == 0 else int(max(df["end_time"]))
    max_freq = int(max(df["end_frequency"]))

    tz_data = df["start_datetime"][0].tz

    if timebin_new == 10:
        pass
    elif timebin_new == 60:
        pass
    elif timebin_new == 600:
        pass
    elif timebin_new == 3600:
        pass
    elif timebin_new == 86400:
        pass
    else:
        raise ValueError(f"Time resolution {timebin_new}s not available")

    f = str(timebin_new) + "s"

    df_new = pd.DataFrame()
    if isinstance(annotators, str):
        annotators = [annotators]
    if isinstance(labels, str):
        labels = [labels]
    for annotator in annotators:
        for label in labels:
            df_detect_prov = df[
                (df["annotator"] == annotator) & (df["annotation"] == label)
            ]

            if len(df_detect_prov) == 0:
                continue

            t = t_rounder(t=df_detect_prov["start_datetime"].iloc[0], res=timebin_new)
            t2 = t_rounder(
                df_detect_prov["start_datetime"].iloc[-1], timebin_new
            ) + pd.timedelta(seconds=timebin_new)

            time_vector = [
                ts.timestamp() for ts in pd.date_range(start=t, end=t2, freq=f)
            ]

            # #here test to find for each time vector value which filename corresponds
            filenames = sorted(list(set(df_detect_prov["filename"])))
            if not all(isinstance(filename, str) for filename in filenames):
                if all(math.isnan(filename) for filename in filenames):
                    # FPOD case: the filenames of a FPOD csv file are NaN values
                    filenames = [
                        i.strftime("%Y-%m-%dT%H:%M:%S%z")
                        for i in df_detect_prov["start_datetime"]
                    ]

            ts_filenames = [
                extract_datetime(var=filename, tz=tz_data).timestamp()
                for filename in filenames
            ]

            filename_vector = []
            for ts in time_vector:
                index = bisect.bisect_left(ts_filenames, ts)
                if index == 0:
                    filename_vector.append(filenames[index])
                elif index == len(ts_filenames):
                    filename_vector.append(filenames[index - 1])
                else:
                    filename_vector.append(filenames[index - 1])

            times_detect_beg = [
                detect.timestamp() for detect in df_detect_prov["start_datetime"]
            ]
            times_detect_end = [
                detect.timestamp() for detect in df_detect_prov["end_datetime"]
            ]

            detect_vec, ranks, k = np.zeros(len(time_vector), dtype=int), [], 0
            for i in range(len(times_detect_beg)):
                for j in range(k, len(time_vector) - 1):
                    if int(times_detect_beg[i] * 1e7) in range(
                        int(time_vector[j] * 1e7), int(time_vector[j + 1] * 1e7)
                    ) or int(times_detect_end[i] * 1e7) in range(
                        int(time_vector[j] * 1e7), int(time_vector[j + 1] * 1e7)
                    ):
                        ranks.append(j)
                        k = j
                        break
                    else:
                        continue

            ranks = sorted(list(set(ranks)))
            detect_vec[ranks] = 1
            detect_vec = list(detect_vec)

            start_datetime_str, end_datetime_str, filename = [], [], []
            for i in range(len(time_vector)):
                if detect_vec[i] == 1:
                    start_datetime = pd.Timestamp(time_vector[i], unit="s", tz=tz_data)
                    start_datetime_str.append(
                        start_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f%z")[:-8]
                        + start_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f%z")[-5:-2]
                        + ":"
                        + start_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f%z")[-2:]
                    )
                    end_datetime = pd.Timestamp(
                        time_vector[i] + timebin_new, unit="s", tz=tz_data
                    )
                    end_datetime_str.append(
                        end_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f%z")[:-8]
                        + end_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f%z")[-5:-2]
                        + ":"
                        + end_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f%z")[-2:]
                    )
                    filename.append(filename_vector[i])

            df_new_prov = pd.DataFrame()
            dataset_str = list(set(df_detect_prov["dataset"]))

            df_new_prov["dataset"] = dataset_str * len(start_datetime_str)
            df_new_prov["filename"] = filename
            df_new_prov["start_time"] = [0] * len(start_datetime_str)
            df_new_prov["end_time"] = [timebin_new] * len(start_datetime_str)
            df_new_prov["start_frequency"] = [0] * len(start_datetime_str)
            df_new_prov["end_frequency"] = [max_freq] * len(start_datetime_str)
            df_new_prov["annotation"] = list(set(df_detect_prov["annotation"])) * len(
                start_datetime_str
            )
            df_new_prov["annotator"] = list(set(df_detect_prov["annotator"])) * len(
                start_datetime_str
            )
            df_new_prov["start_datetime"], df_new_prov["end_datetime"] = (
                start_datetime_str,
                end_datetime_str,
            )

            df_new = pd.concat([df_new, df_new_prov])

        df_new["start_datetime"] = [
            pd.to_datetime(d, format="%Y-%m-%dT%H:%M:%S.%f%z")
            for d in df_new["start_datetime"]
        ]
        df_new["end_datetime"] = [
            pd.to_datetime(d, format="%Y-%m-%dT%H:%M:%S.%f%z")
            for d in df_new["end_datetime"]
        ]
        df_new = df_new.sort_values(by=["start_datetime"])

    return df_new


def overview(data: pd.DataFrame):
    """Overview plots given an APLOSE formatted result file

    Parameter
    -------
        t: pd.DataFrame
            APLOSE formatted detection / annotation file

    Returns
    -------
    The list of annotation/detection per label and per annotator is being printed.
    Also plots of annotation per label and annotation per annotator is shown.
    """

    summary_label = (
        data.groupby("annotation")["annotator"].apply(Counter).unstack(fill_value=0)
    )
    summary_annotator = (
        data.groupby("annotator")["annotation"].apply(Counter).unstack(fill_value=0)
    )

    print("\nOverview of the detections :\n\n{0}".format(summary_label))
    print("---\n\n{0}".format(summary_annotator.to_string()))

    fig, (ax1, ax2) = plt.subplots(
        2, figsize=(12, 6), gridspec_kw={"height_ratios": [1, 1]}, facecolor="#36454F"
    )
    # ax1 = summary_label.plot(kind='bar', ax=ax1, color=['tab:blue', 'tab:orange'], edgecolor='black', linewidth=1)
    ax1 = summary_label.plot(kind="bar", ax=ax1, edgecolor="black", linewidth=1)
    ax2 = summary_annotator.plot(kind="bar", ax=ax2, edgecolor="black", linewidth=1)

    # facecolor
    ax1.set_facecolor("#36454F")
    ax2.set_facecolor("#36454F")

    # spacing between plots
    plt.subplots_adjust(hspace=0.4)

    # legend
    ax1.legend(loc="best", fontsize=10, frameon=1, framealpha=0.6)
    ax2.legend(loc="best", fontsize=10, frameon=1, framealpha=0.6)

    # ticks
    ax1.tick_params(axis="both", colors="w", rotation=0, labelsize=8)
    ax2.tick_params(axis="both", colors="w", rotation=0, labelsize=8)

    # labels
    ax1.set_ylabel(
        "Number of\nannotated calls", fontsize=15, color="w", multialignment="center"
    )
    ax1.set_xlabel(
        "Labels", fontsize=10, rotation=0, color="w", multialignment="center"
    )
    ax2.set_ylabel(
        "Number of\nannotated calls", fontsize=15, color="w", multialignment="center"
    )
    ax2.set_xlabel(
        "Annotator", fontsize=10, rotation=0, color="w", multialignment="center"
    )

    # spines
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_color("w")
    ax1.spines["left"].set_color("w")

    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_color("w")
    ax2.spines["left"].set_color("w")

    # y-grids
    ax1.yaxis.grid(color="gray", linestyle="--")
    ax2.yaxis.grid(color="gray", linestyle="--")
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)

    # titles
    title_font = {"fontsize": 15, "color": "w", "fontweight": "bold"}
    ax1.set_title(
        "Number of\nannotations per label",
        color="w",
        fontdict=title_font,
        pad=5,
        size=10,
        multialignment="center",
    )
    ax2.set_title(
        "Number of\nannotations per annotator",
        color="w",
        fontdict=title_font,
        pad=5,
        size=10,
        multialignment="center",
    )


def single_seasonality(
    data: pd.DataFrame,
    annot: str,
    label: str,
    date1: pd.Timestamp,
    date2: pd.Timestamp,
    timebin: int,
    reso_bin: int,
    res_min: int,
    time_locator: str,
    scale: str,
):
    """Single seasonality plot
    Given an APLOSE formatted DataFrame, annotator and label, compute the barplot of the detections

    Parameters
    -------
        data: pd.DataFrame
             APLOSE formatted detection / annotation file

        annot: 'str'
             Selected annotator

        label: 'str'
             Selected label

        date1: pd.Timestamp
             Begin datetime to filter out detection

        date2: pd.Timestamp
             End datetime to filter out detection

        reso_bin: 'str'
             The value can be either 'minute' / 'day' / 'week' / 'month'.

        res_min: 'int'
             ??????

        time_locator: 'str'
             str to select the x-tick resolution on the barplot
             The value can be either 'monthly' / 'beweekly' / 'daily' / 'hourly'.

        scale: 'str'
             Changes the y-axis to percentage of detection or to a raw number of detection
             The value can be either 'raw' or 'percentage'.

    Returns
    -------
    """

    tz_data = date1.tz

    if time_locator == "monthly":
        mdate1 = mdates.MonthLocator(interval=1)
        mdate2 = mdates.DateFormatter("%B", tz=tz_data)
    elif time_locator == "biweekly":
        mdate1 = mdates.DayLocator(interval=15, tz=tz_data)
        mdate2 = mdates.DateFormatter("%d-%B", tz=tz_data)
    elif time_locator == "daily":
        mdate1 = mdates.DayLocator(interval=1, tz=tz_data)
        mdate2 = mdates.DateFormatter("%d-%m", tz=tz_data)
    elif time_locator == "hourly":
        mdate1 = mdates.HourLocator(interval=1, tz=tz_data)
        mdate2 = mdates.DateFormatter("%H:%M", tz=tz_data)
    else:
        raise ValueError(f"Error time_locator: {time_locator}")

    if reso_bin == "minute":
        n_annot_max = (
            res_min * 60
        ) / timebin  # max nb of annoted time_bin max per res_min slice
        delta, start_vec, end_vec = (
            pd.Timedelta(seconds=60 * res_min),
            t_rounder(date1, res=600),
            t_rounder(date2 + pd.Timedelta(seconds=timebin), res=600),
        )
        time_vector = [
            start_vec + i * delta for i in range(int((end_vec - start_vec) / delta) + 1)
        ]
        y_label_txt = f"Number of detections\n({res_min} min)"
    elif reso_bin == "day":
        time_vector_ts = pd.date_range(date1, date2, freq="D", tz=tz_data)
        time_vector = [timestamp.date() for timestamp in time_vector_ts]
        n_annot_max = (24 * 60 * 60) / timebin
        y_label_txt = "Number of detections per day"
    elif reso_bin == "week":
        time_vector_ts = pd.date_range(date1, date2, freq="W-MON", tz=tz_data)
        time_vector = [timestamp.date() for timestamp in time_vector_ts]
        n_annot_max = (24 * 60 * 60 * 7) / timebin
        y_label_txt = "Number of detections per week (starting every monday)"
    elif reso_bin == "month":
        time_vector_ts = pd.date_range(date1, date2, freq="MS", tz=tz_data)
        time_vector = [timestamp.date() for timestamp in time_vector_ts]
        n_annot_max = (31 * 24 * 60 * 60) / timebin
        y_label_txt = "Number of detections per month"
    else:
        raise ValueError(f"Error reso_bin: {reso_bin}")

    df_1annot_1label = data[
        (data["annotator"] == annot) & (data["annotation"] == label)
    ]

    fig, ax = plt.subplots(figsize=(20, 9), facecolor="#36454F")
    [hist_y, hist_x, _] = ax.hist(
        df_1annot_1label["start_datetime"],
        bins=time_vector,
        color="crimson",
        edgecolor="black",
        linewidth=1,
    )

    # Compute df_hist for user to check the values contained in the histogram
    # hist_xt = [pd.to_datetime(x * 24 * 60 * 60, unit='s') for x in hist_x[:-1]]
    # df_hist = pd.DataFrame({'Date': hist_xt, 'Number of detection': hist_y.tolist()})

    # facecolor
    ax.set_facecolor("#36454F")

    # ticks
    ax.tick_params(axis="y", colors="w", rotation=0, labelsize=20)
    ax.tick_params(axis="x", colors="w", rotation=60, labelsize=15)

    ax.set_ylabel(y_label_txt, fontsize=20, color="w")

    # spines
    ax.spines["right"].set_color("w")
    ax.spines["top"].set_color("w")
    ax.spines["bottom"].set_color("w")
    ax.spines["left"].set_color("w")

    # titles
    fig.suptitle(
        "annotateur : " + annot + "\n" + "label : " + label,
        fontsize=24,
        y=0.98,
        color="w",
    )

    ax.xaxis.set_major_locator(mdate1)
    ax.xaxis.set_major_formatter(mdate2)
    plt.xlim(time_vector[0], time_vector[-1])
    ax.grid(color="w", linestyle="--", linewidth=0.2, axis="both")

    # change value 2 in bars = range(0, 110, 2) to change the space between two ticks
    # change value 0.08 in ax.set_ylim([0,n_annot_max * 0.08]) to change y max
    if scale == "percentage":
        bars = np.arange(0, 110, 10)
        y_pos = [n_annot_max * p / 100 for p in bars]
        ax.set_yticks(y_pos, bars)
        ax.set_ylim([0, n_annot_max])
        if reso_bin == "Minutes":
            ax.set_ylabel(
                "Detection rate % \n({0} min)".format(res_min), fontsize=20, color="w"
            )
        else:
            ax.set_ylabel("Detection rate % per month", fontsize=20, color="w")
    elif scale == "raw":
        pass
    else:
        raise ValueError(f"Error scale: {scale}")
