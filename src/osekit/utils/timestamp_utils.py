"""Utils for working with timestamps."""

from __future__ import annotations  # Backwards compatibility with Python < 3.10

import math
import re
from typing import TYPE_CHECKING

from pandas import Timedelta, Timestamp, to_datetime

from osekit.config import TIMESTAMP_FORMAT_AUDIO_FILE
from osekit.config import global_logging_context as glc

if TYPE_CHECKING:
    import pytz


_REGEX_BUILDER = {
    "%Y": r"([12]\d{3})",
    "%y": r"(\d{2})",
    "%m": r"(0[1-9]|1[0-2])",
    "%-m": r"(1[0-2]|(?:(?<!\d)[1-9](?!\d)))",
    "%d": r"([0-2]\d|3[0-1])",
    "%-d": r"(3[01]|[12][0-9]|(?:(?<!\d)[1-9](?!\d)))",
    "%H": r"([0-1]\d|2[0-4])",
    "%-H": r"(2[0-3]|1[0-9]|(?:(?<!\d)[0-9](?!\d)))",
    "%I": r"(0[1-9]|1[0-2])",
    "%-I": r"(1[0-2]|(?:(?<!\d)[1-9](?!\d)))",
    "%p": r"(AM|PM)",
    "%M": r"([0-5]\d)",
    "%-M": r"([0-5][0-9]|(?:(?<!\d)[0-9](?!\d)))",
    "%S": r"([0-5]\d)",
    "%-S": r"([0-5][0-9]|(?:(?<!\d)[0-9](?!\d)))",
    "%f": r"(\d{1,6})",
    "%Z": r"((?:[a-zA-Z]+)(?:[-/]\w+)*(?:[\+-]\d+)?)",
    "%z": r"([\+-]\d{2}:?\d{2})",
}


def normalize_datetime(datetime: tuple[str], template: str) -> tuple[str, str]:
    """Convert a datetime and its template with non-zero padded parts.

    Parameters
    ----------
    datetime : tuple[str]
        A tuple of datetime component strings (e.g., ``('2024', '1', '15')``).
    template : str
        A datetime template string with format specifiers (e.g., ``'%Y_%-m_%d'``).
        Format specifiers starting with ``'%-'`` indicate non-zero-padded values
        that will be converted to zero-padded format.

    Returns
    -------
    tuple[str, str]
        A tuple containing:
        - A normalized template string with all format specifiers zero-padded
          (e.g., ``'%Y_%m_%d'``)
        - A normalized datetime string with all values zero-padded
          (e.g., ``'2024_01_15'``)

    Examples
    --------
    >>> normalize_datetime(('2024', '1', '15'), '%Y_%-m_%d')
    ('%Y_%m_%d', '2024_01_15')

    >>> normalize_datetime(('2024', '3', '5'), '%Y_%-m_%-d')
    ('%Y_%m_%d', '2024_03_05')

    """
    template_parts = re.findall(r"%-?[A-Za-z]", template)
    dt_dict = dict(zip(template_parts, datetime, strict=True))

    if sum(1 for _ in {k.lstrip("%-") for k in dt_dict}) < len(dt_dict):
        msg = "Format specifiers in template must be unique."
        raise ValueError(msg)

    clean_dt_dict = {}
    for key, value in dt_dict.items():
        if "-" in key:
            new_key = key.replace("-", "")
            new_value = f"{int(value):02}"
        else:
            new_key = key
            new_value = value

        clean_dt_dict[new_key] = new_value

    return "_".join(clean_dt_dict.keys()), "_".join(clean_dt_dict.values())


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


def strftime_osmose_format(date: Timestamp) -> str:
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
    strftime_identifier_lengths = {
        len(strftime_id) for strftime_id in strftime_identifiers
    }
    percent_sign_indexes = (
        index for index, char in enumerate(datetime_template) if char == "%"
    )
    for index in percent_sign_indexes:
        if index == len(datetime_template) - 1:
            return False
        if not any(
                datetime_template[index + 1: index + 1 + id_len] in strftime_identifiers
                for id_len in strftime_identifier_lengths
        ):
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
    if isinstance(datetime_template, str):
        datetime_template = [datetime_template]

    valid_datetime_template = None
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

    if valid_datetime_template is None:
        raise ValueError("\n".join(msg))

    cleaned_date_template, cleaned_date_string = normalize_datetime(
        datetime=regex_result[0],
        template=valid_datetime_template,
    )

    return to_datetime(cleaned_date_string, format=cleaned_date_template)


def last_window_end(
    begin: Timestamp,
    end: Timestamp,
    window_duration: Timedelta,
    window_hop: Timedelta,
) -> Timestamp:
    """Compute the end Timestamp of the last sliding window from begin to end."""
    max_hops = math.ceil((end - begin).total_seconds() / window_hop.total_seconds()) - 1
    last_window_start = begin + window_hop * max_hops
    return last_window_start + window_duration
