from collections.abc import Generator
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Iterable
import os

from OSmOSE.config import TIMESTAMP_FORMAT_AUDIO_FILE
from pandas import Timestamp
import re

_REGEX_BUILDER = {
    "%Y": "([12][0-9]{3})",
    "%y": "([0-9]{2})",
    "%m": "(0[1-9]|1[0-2])",
    "%d": "([0-2][0-9]|3[0-1])",
    "%H": "([0-1][0-9]|2[0-4])",
    "%I": "(0[1-9]|1[0-2])",
    "%p": "(AM|PM)",
    "%M": "([0-5][0-9])",
    "%S": "([0-5][0-9])",
    "%f": "([0-9]{6})",
    "%Z": "((?:\\w+)(?:[-/]\\w+)*(?:[\\+-]\\d+)?)",
    "%z": "([\\+-]\\d{4})",
}


def check_epoch(df):
    "Function that adds epoch column to dataframe"
    if "epoch" in df.columns:
        return df
    else:
        try:
            df["epoch"] = df.timestamp.apply(
                lambda x: datetime.strptime(x[:26], "%Y-%m-%dT%H:%M:%S.%f").timestamp()
            )
            return df
        except ValueError:
            print(
                "Please check that you have either a timestamp column (format ISO 8601 Micro s) or an epoch column"
            )
            return df


def substract_timestamps(
    input_timestamp: pd.DataFrame, files: List[str], index: int
) -> timedelta:
    """Substracts two timestamp_list from the "timestamp" column of a dataframe at the indexes of files[i] and files[i-1] and returns the time delta between them

    Parameters:
    -----------
        input_timestamp: the pandas DataFrame containing at least two columns: filename and timestamp

        files: the list of file names corresponding to the filename column of the dataframe

        index: the index of the file whose timestamp will be substracted

    Returns:
    --------
        The time between the two timestamp_list as a datetime.timedelta object"""

    if index == 0:
        return timedelta(seconds=0)

    cur_timestamp: str = input_timestamp[input_timestamp["filename"] == files[index]][
        "timestamp"
    ].values[0]
    cur_timestamp: datetime = to_timestamp(cur_timestamp)
    next_timestamp: str = input_timestamp[
        input_timestamp["filename"] == files[index + 1]
    ]["timestamp"].values[0]
    next_timestamp: datetime = to_timestamp(next_timestamp)

    return next_timestamp - cur_timestamp


def to_timestamp(string: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(string)
    except ValueError:
        raise ValueError(
            f"The timestamp '{string}' must match format %Y-%m-%dT%H:%M:%S%z."
        )


def strftime_osmose_format(date: pd.Timestamp) -> str:
    """
    Format a pandas Timestamp using strftime() and the OSmOSE time format %Y-%m-%dT%H:%M:%S.%f%z, with %f limited to a millisecond precision.
    If the input Timestamp is not localized, its localization will be defaulted as UTC.

    arameters
    ----------
    date: pandas.Timestamp
        The Timestamp to format

    Returns
    -------
    str:
        The formatted Timestamp

    Examples
    --------
    >>> strftime_osmose_format(Timestamp('2024-10-17 10:14:11.933634', tz="US/Eastern"))
    '2024-10-17T10:14:11.933-0400'
    """
    if date.tz is None:
        date = date.tz_localize("UTC")

    str_time = date.strftime(TIMESTAMP_FORMAT_AUDIO_FILE)
    str_time = (
        str_time[:-8] + str_time[-5:]
    )  # Changes microsecond precision to millisecond precision

    return str_time


def build_regex_from_datetime_template(datetime_template: str) -> str:
    """
    Builds the regular expression that is used to parse a Timestamp from a string following the given datetime strftime template

    Parameters
    ----------
    datetime_template: str
        A datetime template string using strftime codes

    Returns
    -------
    str
        A regex that can be used to parse a Timestamp from a string following the given datetime strftime template

    Examples
    --------
    >>> build_regex_from_datetime_template('year_%Y_hour_%H')
    'year_([12][0-9]{3})_hour_([0-1][0-9]|2[0-4])'
    """

    escaped_characters = "()"
    for escaped in escaped_characters:
        datetime_template = datetime_template.replace(escaped, f"\\{escaped}")
    for key, value in _REGEX_BUILDER.items():
        datetime_template = datetime_template.replace(key, value)
    return datetime_template


def is_datetime_template_valid(datetime_template: str) -> bool:
    """
    Checks the validity of a datetime template string. A datetame template string is used to extract a timestamp from a given string: 'year_%Y' is a valid datetame template for extracting '2016' from the 'year_2016' string.
    A datetime template string should use valid strftime codes (see https://strftime.org/).

    Parameters
    ----------
    datetime_template: str
        The datetime template

    Returns
    -------
    bool:
    True if datetime_template is valid (only uses supported strftime codes), False otherwise

    Examples
    --------
    >>> is_datetime_template_valid('year_%Y_hour_%H')
    True
    >>> is_datetime_template_valid('unsupported_code_%Z_hour_%H')
    False
    """
    strftime_identifiers = [key.lstrip("%") for key in _REGEX_BUILDER.keys()]
    percent_sign_indexes = (
        index for index, char in enumerate(datetime_template) if char == "%"
    )
    for index in percent_sign_indexes:
        if index == len(datetime_template) - 1:
            return False
        if datetime_template[index + 1] not in strftime_identifiers:
            return False
    return True


def extract_timestamp_from_filename(filename: str, datetime_template: str) -> Timestamp:
    """
    Extract a pandas.Timestamp from the filename string following the datetime_template specified.

    Parameters
    ----------
    filename: str
        The filename in which the timestamp should be extracted, ex '2016_06_13_14:12.txt'
    datetime_template: str
         The datetime template used in filename, using strftime codes (https://strftime.org/). Example: '%y%m%d_%H:%M:%S'

    Returns
    -------
    pandas.Timestamp:
        The timestamp extracted from filename according to datetime_template

    Examples
    --------
    >>> extract_timestamp_from_filename('2016_06_13_14:12.txt', '%Y_%m_%d_%H:%M')
    Timestamp('2016-06-13 14:12:00')
    >>> extract_timestamp_from_filename('date_12_03_21_hour_11:45:10_PM.wav', '%y_%m_%d_hour_%I:%M:%S_%p')
    Timestamp('2012-03-21 23:45:10')
    """

    if not is_datetime_template_valid(datetime_template):
        raise ValueError(f"{datetime_template} is not a supported strftime template")

    regex_pattern = build_regex_from_datetime_template(datetime_template)
    regex_result = re.findall(regex_pattern, filename)

    if not regex_result:
        raise ValueError(
            f"{filename} did not match the given {datetime_template} template"
        )

    date_string = "".join(regex_result[0])
    cleaned_date_template = "".join(
        c + datetime_template[i + 1]
        for i, c in enumerate(datetime_template)
        if c == "%"
    )  # MUST BE TESTED IN CASE OF "%i%" or "%%"
    return pd.to_datetime(date_string, format=cleaned_date_template)


def associate_timestamps(
    audio_files: Iterable[str], datetime_template: str
) -> pd.Series:
    """
    Returns a chronologically sorted pandas series containing the audio files as indexes and the extracted timestamp as values.

    Parameters
    ----------
    audio_files: Iterable[str]
        files from which the timestamps should be extracted. They must share a same datetime format.
    datetime_template: str
         The datetime template used in filename, using strftime codes (https://strftime.org/). Example: '%y%m%d_%H:%M:%S'

    Returns
    -------
    pandas.Series
        A series with the audio files names as index and the extracted timestamps as values.
    """
    files_with_timestamps = {
        file: extract_timestamp_from_filename(file, datetime_template)
        for file in audio_files
    }
    series = pd.Series(data=files_with_timestamps, name="timestamp")
    series.index.name = "filename"
    return series.sort_values().reset_index()


def get_timestamps(
    path_osmose_dataset: str, campaign_name: str, dataset_name: str, resolution: str
) -> pd.DataFrame:
    """Read infos from APLOSE timestamp csv file
    Parameters
    -------
        path_osmose_dataset: 'str'
            usually '/home/datawork-osmose/dataset/'

        campaign_name: 'str'
            Name of the campaign
        dataset_name: 'str'
            Name of the dataset
        resolution: 'str'
            Resolution of the dataset
    Returns
    -------
        df: pd.DataFrame
            The timestamp file is read and returned as a DataFrame
    """

    csv = os.path.join(
        path_osmose_dataset,
        campaign_name,
        dataset_name,
        "data",
        "audio",
        resolution,
        "timestamp.csv",
    )

    if os.path.exists(csv):
        df = pd.read_csv(csv, parse_dates=["timestamp"])
        return df
    else:
        raise ValueError(f"{csv} does not exist")
