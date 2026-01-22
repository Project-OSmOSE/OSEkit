"""Utils for working with timestamps."""

from __future__ import annotations  # Backwards compatibility with Python < 3.10

import math
import re

import pandas as pd
import pytz
from pandas import Timedelta, Timestamp

from osekit.config import TIMESTAMP_FORMAT_AUDIO_FILE
from osekit.config import global_logging_context as glc

_REGEX_BUILDER = {
    "%Y": r"([12]\d{3})",
    "%y": r"(\d{2})",
    "%m": r"(0[1-9]|1[0-2])",
    "%d": r"([0-2]\d|3[0-1])",
    "%H": r"([0-1]\d|2[0-4])",
    "%I": r"(0[1-9]|1[0-2])",
    "%p": r"(AM|PM)",
    "%M": r"([0-5]\d)",
    "%S": r"([0-5]\d)",
    "%f": r"(\d{1,6})",
    "%Z": r"((?:[a-zA-Z]+)(?:[-/]\w+)*(?:[\+-]\d+)?)",
    "%z": r"([\+-]\d{2}:?\d{2})",
}


def localize_timestamp(
    timestamp: Timestamp,
    timezone: str | pytz.timezone,
) -> Timestamp:
    """Localize a timestamp in the given timezone.

    Parameters
    ----------
    timestamp: pandas.Timestamp
        The timestamp to localize.
    timezone: str | pytz.timezone
        The timezone in which the timestamp is localized.

    Returns
    -------
    pandas.Timestamp
    The timestamp localized in the specified timezone.
    If the original timestamp was tz-aware, its tz will be converted
    to the new timezone.

    """
    if not timestamp.tz:
        return timestamp.tz_localize(timezone)

    if timestamp.utcoffset() != timestamp.tz_convert(timezone).utcoffset():
        glc.logger.warning(
            "The timestamps are tz-aware and you specified a different timezone.\n"
            f"Timestamps timezones {timestamp.tz} will be converted to {timezone}",
        )
    return timestamp.tz_convert(timezone)


def reformat_timestamp(
    old_timestamp_str: str,
    old_datetime_template: str,
) -> str:
    """Format a timestamp string from a given template to the osekit template.

    Parameters
    ----------
    old_timestamp_str: str
        The old timestamp string.
    old_datetime_template: str
        The datetime template of the old timestamp using strftime codes.

    Returns
    -------
    str:
        The formatted timestamp string in the osekit ``%Y-%m-%dT%H:%M:%S.%f%z`` format.

    """
    timestamp = strptime_from_text(
        text=old_timestamp_str,
        datetime_template=old_datetime_template,
    )
    return strftime_osmose_format(timestamp)


def strftime_osmose_format(date: pd.Timestamp) -> str:
    """Format a Timestamp to the osekit format.

    Parameters
    ----------
    date: pandas.Timestamp
        The Timestamp to format.
        If the Timestamp is timezone-naive, it will be localized to UTC.

    Returns
    -------
    str:
        The Timestamp in osekit's ``%Y-%m-%dT%H:%M:%S.%f%z`` format.
        ``%f`` is limited to a millisecond precision.

    Examples
    --------
    >>> strftime_osmose_format(Timestamp('2024-10-17 10:14:11.933634', tz="US/Eastern"))
    '2024-10-17T10:14:11.933-0400'

    """
    if date.tz is None:
        date = date.tz_localize("UTC")

    str_time = date.strftime(TIMESTAMP_FORMAT_AUDIO_FILE)
    return (
        str_time[:-8] + str_time[-5:]
    )  # Changes microsecond precision to millisecond precision


def build_regex_from_datetime_template(datetime_template: str) -> str:
    r"""Build the regular expression for parsing Timestamps based on a template string.

    Parameters
    ----------
    datetime_template: str
        A datetime template string using strftime codes.

    Returns
    -------
    str:
        A regex that can be used to parse a Timestamp from a string.
        The timestamp in the string must be written in the specified template

    Examples
    --------
    >>> build_regex_from_datetime_template('year_%Y_hour_%H')
    'year_([12]\\d{3})_hour_([0-1]\\d|2[0-4])'

    """
    escaped_characters = "()"
    for escaped in escaped_characters:
        datetime_template = datetime_template.replace(escaped, f"\\{escaped}")
    for key, value in _REGEX_BUILDER.items():
        datetime_template = datetime_template.replace(key, value)
    return datetime_template


def is_datetime_template_valid(datetime_template: str) -> bool:
    """Check the validity of a datetime template string.

    Parameters
    ----------
    datetime_template: str
        The datetime template following which timestamps are written.
        It should use valid strftime codes (see https://strftime.org/).

    Returns
    -------
    bool:
    ``True`` if ``datetime_template`` only uses supported strftime codes,
    ``False`` otherwise.

    Examples
    --------
    >>> is_datetime_template_valid('year_%Y_hour_%H')
    True
    >>> is_datetime_template_valid('unsupported_code_%u_hour_%H')
    False

    """
    strftime_identifiers = [key.lstrip("%") for key in _REGEX_BUILDER]
    percent_sign_indexes = (
        index for index, char in enumerate(datetime_template) if char == "%"
    )
    for index in percent_sign_indexes:
        if index == len(datetime_template) - 1:
            return False
        if datetime_template[index + 1] not in strftime_identifiers:
            return False
    return True


def strptime_from_text(text: str, datetime_template: str | list[str]) -> Timestamp:
    """Extract a Timestamp written in a string with a specified format.

    Parameters
    ----------
    text: str
        The text in which the timestamp should be extracted, ex ``'2016_06_13_14:12.txt'``.
    datetime_template: str | list[str]
         The datetime template used in the text.
         It should use valid strftime codes (https://strftime.org/).
         Example: ``'%y%m%d_%H:%M:%S'``.
         If ``datetime_template`` is a list of strings, the datetime will be parsed from
         the first template of the list that matches the input text.

    Returns
    -------
    pandas.Timestamp:
        The timestamp extracted from the text according to ``datetime_template``

    Examples
    --------
    >>> strptime_from_text('2016_06_13_14:12.txt', '%Y_%m_%d_%H:%M')
    Timestamp('2016-06-13 14:12:00')
    >>> strptime_from_text('D_12_03_21_hour_11:45:10_PM', '%y_%m_%d_hour_%I:%M:%S_%p')
    Timestamp('2012-03-21 23:45:10')
    >>> strptime_from_text('2016_06_13_14:12.txt', ['%Y_%m_%d_%H:%M%z','%Y_%m_%d_%H:%M'])
    Timestamp('2016-06-13 14:12:00')
    >>> strptime_from_text('2016_06_13_14:12+0500.txt', ['%Y_%m_%d_%H:%M%z','%Y_%m_%d_%H:%M'])
    Timestamp('2016-06-13 14:12:00+0500', tz='UTC+05:00')

    """  # noqa: E501
    if type(datetime_template) is str:
        datetime_template = [datetime_template]

    valid_datetime_template = ""
    regex_result = []
    msg = []

    for template in datetime_template:
        if not is_datetime_template_valid(template):
            msg.append(f"{template} is not a supported strftime template")
            continue

        regex_pattern = build_regex_from_datetime_template(template)
        regex_result = re.findall(regex_pattern, text)

        if not regex_result:
            msg.append(f"{text} did not match the given {template} template")
            continue

        valid_datetime_template = template
        break

    if not valid_datetime_template:
        raise ValueError("\n".join(msg))

    date_string = "_".join(regex_result[0])
    cleaned_date_template = "_".join(
        c + valid_datetime_template[i + 1]
        for i, c in enumerate(valid_datetime_template)
        if c == "%"
    )
    return pd.to_datetime(date_string, format=cleaned_date_template)


def last_window_end(
    begin: Timestamp,
    end: Timestamp,
    window_duration: Timedelta,
    window_hop: Timedelta,
) -> Timestamp:
    """Compute the end Timestamp of the last window for a sliding window starting from begin to end."""
    max_hops = math.ceil((end - begin).total_seconds() / window_hop.total_seconds()) - 1
    last_window_start = begin + window_hop * max_hops
    return last_window_start + window_duration
