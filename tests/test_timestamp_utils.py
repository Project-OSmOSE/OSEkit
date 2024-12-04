from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd
import pytest
from pandas import DataFrame, Timestamp

from OSmOSE.utils.timestamp_utils import (
    adapt_timestamp_csv_to_osmose,
    associate_timestamps,
    build_regex_from_datetime_template,
    is_datetime_template_valid,
    is_overlapping,
    localize_timestamp,
    parse_timestamps_csv,
    reformat_timestamp,
    strftime_osmose_format,
    strptime_from_text,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("datetime_template", "expected"),
    [
        pytest.param("%y%m%d%H%M%S", True, id="simple_pattern"),
        pytest.param("%Y%y%m%d%H%M%S%I%p%f", True, id="all_strftime_codes"),
        pytest.param("%y %m %d %H %M %S", True, id="spaces_separated_codes"),
        pytest.param("%y:%m:%d%H.%M%S", True, id="special_chars_separated_codes"),
        pytest.param("%y%z%d%H%X%S", False, id="%X_is_wrong_strftime_code"),
        pytest.param("%y%z%d%H%%%S", False, id="%%_is_wrong_strftime_code"),
        pytest.param("%y%m%d_at_%H%M%S", True, id="alpha_letters_separated_codes"),
        pytest.param("%y%m%d%H%M%S%", False, id="trailing_%_is_wrong_strftime_code"),
        pytest.param("%y%m%d%H%M%S%z", True, id="utc_offset"),
        pytest.param("%y%m%d%H%M%S_%Z", True, id="timezone_name"),
    ],
)
def test_is_datetime_template_valid(datetime_template: str, expected: bool) -> None:
    assert is_datetime_template_valid(datetime_template) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("datetime_template", "expected"),
    [
        pytest.param(
            "%y%m%d%H%M%S",
            r"(\d{2})(0[1-9]|1[0-2])([0-2]\d|3[0-1])([0-1]\d|2[0-4])([0-5]\d)([0-5]\d)",
            id="simple_pattern",
        ),
        pytest.param(
            "%Y%y%m%d%H%M%S%I%p%f",
            r"([12]\d{3})(\d{2})(0[1-9]|1[0-2])([0-2]\d|3[0-1])([0-1]\d|2[0-4])"
            r"([0-5]\d)([0-5]\d)(0[1-9]|1[0-2])(AM|PM)(\d{1,6})",
            id="all_strftime_codes",
        ),
        pytest.param(
            "%y %m %d %H %M %S",
            r"(\d{2}) (0[1-9]|1[0-2]) ([0-2]\d|3[0-1]) "
            r"([0-1]\d|2[0-4]) ([0-5]\d) ([0-5]\d)",
            id="spaces_separated_codes",
        ),
        pytest.param(
            "%y:%m:%d%H.%M%S",
            r"(\d{2}):(0[1-9]|1[0-2]):([0-2]\d|3[0-1])([0-1]\d|2[0-4]).([0-5]\d)([0-5]\d)",
            id="special_chars_separated_codes",
        ),
        pytest.param(
            "%y%m%d_at_%H%M%S",
            r"(\d{2})(0[1-9]|1[0-2])([0-2]\d|3[0-1])_at_([0-1]\d|2[0-4])([0-5]\d)([0-5]\d)",
            id="alpha_letters_separated_codes",
        ),
        pytest.param(
            "{%y}%m%d(%H%M%S)",
            r"{(\d{2})}(0[1-9]|1[0-2])([0-2]\d|3[0-1])\(([0-1]\d|2[0-4])([0-5]\d)([0-5]\d)\)",
            id="parentheses_separated_codes",
        ),
        pytest.param(
            "%y%m%d%z",
            r"(\d{2})(0[1-9]|1[0-2])([0-2]\d|3[0-1])([\+-]\d{2}:?\d{2})",
            id="utc_offset",
        ),
        pytest.param(
            "%y%m%d%Z",
            r"(\d{2})(0[1-9]|1[0-2])([0-2]\d|3[0-1])((?:\w+)(?:[-/]\w+)*(?:[\+-]\d+)?)",
            id="timezone_name",
        ),
    ],
)
def test_build_regex_from_datetime_template(
    datetime_template: str,
    expected: str,
) -> None:
    assert build_regex_from_datetime_template(datetime_template) == expected


@pytest.mark.integ
@pytest.mark.parametrize(
    ("text", "datetime_template", "expected"),
    [
        pytest.param(
            "7189.230405144906.wav",
            "%y%m%d%H%M%S",
            Timestamp("2023-04-05 14:49:06"),
            id="plain_pattern",
        ),
        pytest.param(
            "7189.(230405)/(144906).wav",
            "(%y%m%d)/(%H%M%S)",
            Timestamp("2023-04-05 14:49:06"),
            id="escaped_parentheses",
        ),
        pytest.param(
            "7189.23_04_05_14:49:06.wav",
            "%y_%m_%d_%H:%M:%S",
            Timestamp("2023-04-05 14:49:06"),
            id="special_characters",
        ),
        pytest.param(
            "7189.230405_at_144906.wav",
            "%y%m%d_at_%H%M%S",
            Timestamp("2023-04-05 14:49:06"),
            id="alpha_letters",
        ),
        pytest.param(
            "7189.202323040514490602PM000010.wav",
            "%Y%y%m%d%H%M%S%I%p%f",
            Timestamp("2023-04-05 14:49:06.000010"),
            id="full_pattern",
        ),
        pytest.param(
            "7189.230405144906+0200.wav",
            "%y%m%d%H%M%S%z",
            Timestamp("2023-04-05 14:49:06+0200"),
            id="utc_positive_offset",
        ),
        pytest.param(
            "7189.230405144906-0200.wav",
            "%y%m%d%H%M%S%z",
            Timestamp("2023-04-05 14:49:06-0200"),
            id="utc_negative_offset",
        ),
        pytest.param(
            "7189.230405144906+020012.wav",
            "%y%m%d%H%M%S%z",
            Timestamp("2023-04-05 14:49:06+0200"),
            id="utc_offset_with_seconds",
        ),
        pytest.param(
            "7189.230405144906+020012.123456.wav",
            "%y%m%d%H%M%S%z",
            Timestamp("2023-04-05 14:49:06+0200"),
            id="utc_offset_with_microseconds",
        ),
        pytest.param(
            "7189.230405144906_Japan.wav",
            "%y%m%d%H%M%S_%Z",
            Timestamp("2023-04-05 14:49:06+0900", tz="UTC+09:00"),
            id="timezone_name",
        ),
        pytest.param(
            "7189.230405144906_Japan.wav",
            "%y%m%d%H%M%S",
            Timestamp("2023-04-05 14:49:06"),
            id="unspecified_timezone_name_doesnt_count",
        ),
        pytest.param(
            "7189.230405144906_Europe/Isle_of_Man.wav",
            "%y%m%d%H%M%S_%Z",
            Timestamp("2023-04-05 14:49:06+0100", tz="UTC+01:00"),
            id="timezone_name_with_special_chars",
        ),
        pytest.param(
            "7189.230405144906_America/North_Dakota/New_Salem.wav",
            "%y%m%d%H%M%S_%Z",
            Timestamp("2023-04-05 14:49:06-0500", tz="UTC-05:00"),
            id="timezone_name_with_double_slash",
        ),
        pytest.param(
            "7189.230405144906_Etc/GMT+2.wav",
            "%y%m%d%H%M%S_%Z",
            Timestamp("2023-04-05 14:49:06-0200", tz="UTC-02:00"),
            id="timezone_name_with_offset",
        ),
        pytest.param(
            "2023-04-05T14:49:06.123-02:00.wav",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            Timestamp("2023-04-05 14:49:06.123000-0200", tz="UTC-02:00"),
            id="osmose_format_timezone_string",
        ),
        pytest.param(
            "2023-04-05T14:49:06.123-0200.wav",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            Timestamp("2023-04-05 14:49:06.123000-0200", tz="UTC-02:00"),
            id="osmose_format_without_colon_timezone_string",
        ),
        pytest.param(
            "2023-04-05T14:49:06.123456-0200.wav",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            Timestamp("2023-04-05 14:49:06.123456-0200", tz="UTC-02:00"),
            id="microsecond_precision",
        ),
        pytest.param(
            "2023-04-05T14:49:06.1-0200.wav",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            Timestamp("2023-04-05 14:49:06.100000-0200", tz="UTC-02:00"),
            id="decisecond_precision",
        ),
        pytest.param(
            "14:49:06.1232023-04-05-0200.wav",
            "%H:%M:%S.%f%Y-%m-%d%z",
            Timestamp("2023-04-05 14:49:06.123000-0200", tz="UTC-02:00"),
            id="no_ambiguity_for_%f_precision",
        ),
    ],
)
def test_strptime_from_text(
    text: str,
    datetime_template: str,
    expected: Timestamp,
) -> None:
    assert strptime_from_text(text, datetime_template) == expected


@pytest.mark.integ
@pytest.mark.parametrize(
    ("text", "datetime_template", "expected"),
    [
        pytest.param(
            "7189.230405144906.wav",
            "%y%m%d%H%M%%S",
            pytest.raises(
                ValueError,
                match="%y%m%d%H%M%%S is not a supported strftime template",
            ),
            id="%%_is_wrong_strftime_code",
        ),
        pytest.param(
            "7189.230405144906.wav",
            "%y%m%d%H%M%P%S",
            pytest.raises(
                ValueError,
                match="%y%m%d%H%M%P%S is not a supported strftime template",
            ),
            id="%P_is_wrong_strftime_code",
        ),
        pytest.param(
            "7189.230405144906.wav",
            "%y%m%d%H%M%52%S",
            pytest.raises(
                ValueError,
                match="%y%m%d%H%M%52%S is not a supported strftime template",
            ),
            id="%5_is_wrong_strftime_code",
        ),
        pytest.param(
            "7189.230405_144906.wav",
            "%y%m%d%H%M%S",
            pytest.raises(
                ValueError,
                match="7189.230405_144906.wav did not match "
                "the given %y%m%d%H%M%S template",
            ),
            id="unmatching_pattern",
        ),
        pytest.param(
            "7189.230405144906.wav",
            "%Y%m%d%H%M%S",
            pytest.raises(
                ValueError,
                match="7189.230405144906.wav did not match "
                "the given %Y%m%d%H%M%S template",
            ),
            id="%Y_should_have_4_digits",
        ),
        pytest.param(
            "7189.230405146706.wav",
            "%y%m%d%H%M%S",
            pytest.raises(
                ValueError,
                match="7189.230405146706.wav did not match "
                "the given %y%m%d%H%M%S template",
            ),
            id="%M_should_be_lower_than_60",
        ),
        pytest.param(
            "7189.230405146706_0200.wav",
            "%y%m%d%H%M%S_%z",
            pytest.raises(
                ValueError,
                match="7189.230405146706_0200.wav did not match "
                "the given %y%m%d%H%M%S_%z template",
            ),
            id="incorrect_timezone_offset",
        ),
        pytest.param(
            "7189.230405146706_+2500.wav",
            "%y%m%d%H%M%S_%z",
            pytest.raises(
                ValueError,
                match=r"7189.230405146706_\+2500.wav did not match "
                "the given %y%m%d%H%M%S_%z template",
            ),
            id="out_of_range_timezone_offset",
        ),
        pytest.param(
            "7189.230405146706_Brest.wav",
            "%y%m%d%H%M%S_%Z",
            pytest.raises(
                ValueError,
                match="7189.230405146706_Brest.wav did not match "
                "the given %y%m%d%H%M%S_%Z template",
            ),
            id="incorrect_timezone_name",
        ),
        pytest.param(
            "2023-04-05T14:49:06.-0200.wav",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            pytest.raises(
                ValueError,
                match="2023-04-05T14:49:06.-0200.wav did not match "
                "the given %Y-%m-%dT%H:%M:%S.%f%z template",
            ),
            id="no_specified_%f",
        ),
    ],
)
def test_strptime_from_text_errors(
    text: str,
    datetime_template: str,
    expected: Timestamp,
) -> None:
    with expected as e:
        assert strptime_from_text(text, datetime_template) == e


@pytest.fixture
def correct_dataframe() -> DataFrame:
    return DataFrame(
        [
            [
                "something2345_2012_06_24__16:32:10.wav",
                Timestamp("2012-06-24 16:32:10"),
            ],
            [
                "something2345_2023_07_28__08:24:50.flac",
                Timestamp("2023-07-28 08:24:50"),
            ],
            [
                "something2345_2024_01_01__23:12:11.WAV",
                Timestamp("2024-01-01 23:12:11"),
            ],
            [
                "something2345_2024_02_02__02:02:02.FLAC",
                Timestamp("2024-02-02 02:02:02"),
            ],
        ],
        columns=["filename", "timestamp"],
    )


@pytest.mark.integ
def test_associate_timestamps(correct_dataframe: DataFrame) -> None:
    input_files = list(correct_dataframe["filename"])
    assert associate_timestamps((i for i in input_files), "%Y_%m_%d__%H:%M:%S").equals(
        correct_dataframe,
    )


@pytest.mark.integ
def test_associate_timestamps_error_with_incorrect_datetime_format(
    correct_dataframe: DataFrame,
) -> None:
    input_files = list(correct_dataframe["filename"])
    mismatching_datetime_format = "%Y%m%d__%H:%M:%S"
    incorrect_datetime_format = "%y%m%d%H%M%P%S"

    with pytest.raises(
        ValueError,
        match=f"{input_files[0]} did not match "
        f"the given {mismatching_datetime_format} template",
    ) as e:
        assert e == associate_timestamps(
            (i for i in input_files),
            mismatching_datetime_format,
        )

    with pytest.raises(
        ValueError,
        match=f"{incorrect_datetime_format} is not a supported strftime template",
    ):
        assert e == associate_timestamps(
            (i for i in input_files),
            incorrect_datetime_format,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        pytest.param(
            Timestamp("2024-10-17 10:14:11.933+0000"),
            "2024-10-17T10:14:11.933+0000",
            id="timestamp_with_timezone",
        ),
        pytest.param(
            Timestamp("2024-10-17 10:14:11+0000"),
            "2024-10-17T10:14:11.000+0000",
            id="increase_precision_to_millisecond",
        ),
        pytest.param(
            Timestamp("2024-10-17 10:14:11.933384+0000"),
            "2024-10-17T10:14:11.933+0000",
            id="reduce_precision_to_millisecond",
        ),
        pytest.param(
            Timestamp("2024-10-17 10:14:11.933293"),
            "2024-10-17T10:14:11.933+0000",
            id="no_timezone_defaults_to_utc",
        ),
        pytest.param(
            Timestamp("2024-10-17 10:14:11.933-0400"),
            "2024-10-17T10:14:11.933-0400",
            id="delta_timezone",
        ),
        pytest.param(
            Timestamp("2024-10-17 10:14:11.933", tz="US/Eastern"),
            "2024-10-17T10:14:11.933-0400",
            id="str_timezone",
        ),
    ],
)
def test_strftime_osmose_format(timestamp: Timestamp, expected: str) -> None:
    assert strftime_osmose_format(timestamp) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("timestamp_str", "old_template", "expected"),
    [
        pytest.param(
            "2024-10-17 10:14:11",
            "%Y-%m-%d %H:%M:%S",
            "2024-10-17T10:14:11.000+0000",
            id="space_separated_no_timezone",
        ),
        pytest.param(
            "20241017101411",
            "%Y%m%d%H%M%S",
            "2024-10-17T10:14:11.000+0000",
            id="no_separation_no_timezone",
        ),
        pytest.param(
            "2024-10-17 10:14:11 +03:00",
            "%Y-%m-%d %H:%M:%S %z",
            "2024-10-17T10:14:11.000+0300",
            id="UTC_offset_timezone",
        ),
        pytest.param(
            "2024-10-17 10:14:11 Canada/Pacific",
            "%Y-%m-%d %H:%M:%S %Z",
            "2024-10-17T10:14:11.000-0700",
            id="named_timezone_in_str",
        ),
        pytest.param(
            "2024-10-17 10:14:11 +0200",
            "%Y-%m-%d %H:%M:%S %z",
            "2024-10-17T10:14:11.000+0200",
            id="UTC_offset_timezone_without_colon",
        ),
        pytest.param(
            "2024-10-17 10:14:11 -0700",
            "%Y-%m-%d %H:%M:%S %z",
            "2024-10-17T10:14:11.000-0700",
            id="negative_UTC_offset_timezone",
        ),
        pytest.param(
            "2024-10-17 10:14:11 -0000",
            "%Y-%m-%d %H:%M:%S",
            "2024-10-17T10:14:11.000+0000",
            id="negative_zero_UTC_offset_timezone",
        ),
    ],
)
def test_reformat_timestamp(
    timestamp_str: str,
    old_template: str,
    expected: str,
) -> None:
    assert (
        reformat_timestamp(
            old_timestamp_str=timestamp_str,
            old_datetime_template=old_template,
        )
        == expected
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("timestamp", "timezone", "expected"),
    [
        pytest.param(
            Timestamp("2024-10-17 10:14:11"),
            "UTC",
            Timestamp("2024-10-17 10:14:11+0000"),
            id="UTC_timezone",
        ),
        pytest.param(
            Timestamp("2024-10-17 10:14:11"),
            "Pacific/Rarotonga",
            Timestamp("2024-10-17 10:14:11-1000"),
            id="Non-UTC_timezone",
        ),
        pytest.param(
            Timestamp("2024-10-17 10:14:11"),
            "+03:00",
            Timestamp("2024-10-17 10:14:11+0300"),
            id="UTC_offset_timezone",
        ),
        pytest.param(
            Timestamp("2024-10-17 10:14:11"),
            None,
            Timestamp("2024-10-17 10:14:11", tz="UTC"),
            id="No_timezone_defaults_to_UTC",
        ),
        pytest.param(
            Timestamp("2024-10-17 10:14:11+0200"),
            "+02:00",
            Timestamp("2024-10-17 10:14:11+0200"),
            id="tz-aware_timestamp",
        ),
        pytest.param(
            Timestamp("2024-10-17 10:14:11-0700"),
            "Canada/Pacific",
            Timestamp("2024-10-17 10:14:11-0700", tz="Canada/Pacific"),
            id="named_timezone_in_str",
        ),
        pytest.param(
            Timestamp("2024-10-17 10:14:11"),
            "+0200",
            Timestamp("2024-10-17T10:14:11.000+0200"),
            id="UTC_offset_timezone_without_colon",
        ),
        pytest.param(
            Timestamp("2024-10-17 10:14:11"),
            "-0700",
            Timestamp("2024-10-17T10:14:11.000-0700"),
            id="negative_UTC_offset_timezone",
        ),
        pytest.param(
            Timestamp("2024-10-17 10:14:11"),
            "-0000",
            "2024-10-17T10:14:11.000+0000",
            id="negative_zero_UTC_offset_timezone",
        ),
    ],
)
def test_localize_timestamp(
    timestamp: Timestamp,
    timezone: str,
    expected: Timestamp,
) -> None:
    pass


@pytest.mark.unit
@pytest.mark.parametrize(
    ("timestamp", "timezone", "expected"),
    [
        pytest.param(
            Timestamp("2024-10-17 10:14:11 +0200"),
            "+01:00",
            Timestamp("2024-10-17 09:14:11 +0100"),
            id="UTC_offset",
        ),
        pytest.param(
            Timestamp("2024-06-17 12:14:11+0200"),
            "America/Chihuahua",
            Timestamp("2024-06-17 04:14:11-0600"),
            id="named_timezone",
        ),
    ],
)
def test_localize_timestamp_warns_when_changing_timezone(
    timestamp: Timestamp,
    timezone: str,
    expected: Timestamp,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        result = localize_timestamp(timestamp=timestamp, timezone=timezone)

    assert (
        f"Timestamps timezones UTC+02:00 will be converted to {timezone}" in caplog.text
    )
    assert result == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("timestamp", "timezone", "expected"),
    [
        pytest.param(
            Timestamp("2024-06-17 10:14:11 +0200"),
            "Europe/Paris",
            Timestamp("2024-06-17 10:14:11 +0200", tz="Europe/Paris"),
            id="utc+2_to_paris_summer",
        ),
        pytest.param(
            Timestamp("2024-12-17 10:14:11 +0100"),
            "Europe/Paris",
            Timestamp("2024-12-17 10:14:11 +0100", tz="Europe/Paris"),
            id="utc+1_to_paris_winter",
        ),
        pytest.param(
            Timestamp("2024-12-17 10:14:11 +0100"),
            "+01:00",
            Timestamp("2024-12-17 10:14:11 +0100"),
            id="utc_offset",
        ),
        pytest.param(
            Timestamp("2024-06-17 10:14:11", tz="America/Chihuahua"),
            "-06:00",
            Timestamp("2024-06-17 10:14:11-0600", tz="America/Chihuahua"),
            id="utc_offset_with_named_tz",
        ),
    ],
)
def test_localize_timestamp_doesnt_warn_when_timezones_have_same_utcoffset(
    timestamp: Timestamp,
    timezone: str,
    expected: Timestamp,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        result = localize_timestamp(timestamp=timestamp, timezone=timezone)

    assert "will be converted" not in caplog.text
    assert result == expected


@pytest.mark.parametrize(
    ("filenames", "datetime_template", "timezone", "expected"),
    [
        pytest.param(
            [
                "audio_20240617101411.wav",
                "audio_20240617111411.wav",
                "audio_20240617121411.wav",
            ],
            "%Y%m%d%H%M%S",
            None,
            pd.DataFrame(
                data=[
                    [
                        "audio_20240617101411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 10:14:11+0000")),
                    ],
                    [
                        "audio_20240617111411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 11:14:11+0000")),
                    ],
                    [
                        "audio_20240617121411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 12:14:11+0000")),
                    ],
                ],
                columns=["filename", "timestamp"],
            ),
            id="no_timezone",
        ),
        pytest.param(
            [
                "audio_20240617101411.wav",
                "audio_20240617111411.wav",
                "audio_20240617121411.wav",
            ],
            "%Y%m%d%H%M%S",
            "Europe/Paris",
            pd.DataFrame(
                data=[
                    [
                        "audio_20240617101411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 10:14:11+0200")),
                    ],
                    [
                        "audio_20240617111411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 11:14:11+0200")),
                    ],
                    [
                        "audio_20240617121411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 12:14:11+0200")),
                    ],
                ],
                columns=["filename", "timestamp"],
            ),
            id="specified_timezone",
        ),
        pytest.param(
            [
                "audio_20240617101411.wav",
                "audio_20240617111411.wav",
                "audio_20240617121411.wav",
            ],
            "%Y%m%d%H%M%S",
            "-0200",
            pd.DataFrame(
                data=[
                    [
                        "audio_20240617101411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 10:14:11-0200")),
                    ],
                    [
                        "audio_20240617111411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 11:14:11-0200")),
                    ],
                    [
                        "audio_20240617121411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 12:14:11-0200")),
                    ],
                ],
                columns=["filename", "timestamp"],
            ),
            id="specified_utc_offset",
        ),
        pytest.param(
            [
                "audio_20240617101411+0200.wav",
                "audio_20240617111411+0200.wav",
                "audio_20240617121411+0200.wav",
            ],
            "%Y%m%d%H%M%S%z",
            "UTC",
            pd.DataFrame(
                data=[
                    [
                        "audio_20240617101411+0200.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 08:14:11+0000")),
                    ],
                    [
                        "audio_20240617111411+0200.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 09:14:11+0000")),
                    ],
                    [
                        "audio_20240617121411+0200.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 10:14:11+0000")),
                    ],
                ],
                columns=["filename", "timestamp"],
            ),
            id="conflict_in_timezone_results_in_conversion",
        ),
    ],
)
def test_parse_timestamps_csv(
    filenames: Iterable[str],
    datetime_template: str,
    timezone: str | None,
    expected: pd.DataFrame,
) -> None:
    assert parse_timestamps_csv(filenames, datetime_template, timezone).equals(expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("timestamps", "date_template", "timezone", "expected"),
    [
        pytest.param(
            pd.DataFrame(
                data=[
                    [
                        "audio_20240617101411.wav",
                        "2024-06-17 10:14:11-0200",
                    ],
                    [
                        "audio_20240617111411.wav",
                        "2024-06-17 11:14:11-0200",
                    ],
                    [
                        "audio_20240617121411.wav",
                        "2024-06-17 12:14:11-0200",
                    ],
                ],
                columns=["filename", "timestamp"],
            ),
            "%Y-%m-%d %H:%M:%S%z",
            None,
            pd.DataFrame(
                data=[
                    [
                        "audio_20240617101411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 10:14:11-0200")),
                    ],
                    [
                        "audio_20240617111411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 11:14:11-0200")),
                    ],
                    [
                        "audio_20240617121411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 12:14:11-0200")),
                    ],
                ],
                columns=["filename", "timestamp"],
            ),
            id="tz-aware_timestamps",
        ),
        pytest.param(
            pd.DataFrame(
                data=[
                    [
                        "audio_20240617101411.wav",
                        "2024-06-17 10:14:11",
                    ],
                    [
                        "audio_20240617111411.wav",
                        "2024-06-17 11:14:11",
                    ],
                    [
                        "audio_20240617121411.wav",
                        "2024-06-17 12:14:11",
                    ],
                ],
                columns=["filename", "timestamp"],
            ),
            "%Y-%m-%d %H:%M:%S",
            "-0200",
            pd.DataFrame(
                data=[
                    [
                        "audio_20240617101411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 10:14:11-0200")),
                    ],
                    [
                        "audio_20240617111411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 11:14:11-0200")),
                    ],
                    [
                        "audio_20240617121411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 12:14:11-0200")),
                    ],
                ],
                columns=["filename", "timestamp"],
            ),
            id="tz-naive_timestamps",
        ),
        pytest.param(
            pd.DataFrame(
                data=[
                    [
                        "audio_20240617101411.wav",
                        "2024-06-17 10:14:11-0200",
                    ],
                    [
                        "audio_20240617111411.wav",
                        "2024-06-17 11:14:11-0200",
                    ],
                    [
                        "audio_20240617121411.wav",
                        "2024-06-17 12:14:11-0200",
                    ],
                ],
                columns=["filename", "timestamp"],
            ),
            "%Y-%m-%d %H:%M:%S%z",
            "-0200",
            pd.DataFrame(
                data=[
                    [
                        "audio_20240617101411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 10:14:11-0200")),
                    ],
                    [
                        "audio_20240617111411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 11:14:11-0200")),
                    ],
                    [
                        "audio_20240617121411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 12:14:11-0200")),
                    ],
                ],
                columns=["filename", "timestamp"],
            ),
            id="non_conflicting_tz_timestamps",
        ),
        pytest.param(
            pd.DataFrame(
                data=[
                    [
                        "audio_20240617101411.wav",
                        "2024-06-17 10:14:11-0200",
                    ],
                    [
                        "audio_20240617111411.wav",
                        "2024-06-17 11:14:11-0200",
                    ],
                    [
                        "audio_20240617121411.wav",
                        "2024-06-17 12:14:11-0200",
                    ],
                ],
                columns=["filename", "timestamp"],
            ),
            "%Y-%m-%d %H:%M:%S%z",
            "GMT",
            pd.DataFrame(
                data=[
                    [
                        "audio_20240617101411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 12:14:11+0000")),
                    ],
                    [
                        "audio_20240617111411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 13:14:11+0000")),
                    ],
                    [
                        "audio_20240617121411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 14:14:11+0000")),
                    ],
                ],
                columns=["filename", "timestamp"],
            ),
            id="conflicting_tz_timestamps_lead_to_tz_conversion",
        ),
        pytest.param(
            pd.DataFrame(
                data=[
                    [
                        "audio_20240617101411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 10:14:11+0000")),
                    ],
                    [
                        "audio_20240617111411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 11:14:11+0000")),
                    ],
                    [
                        "audio_20240617121411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 12:14:11+0000")),
                    ],
                ],
                columns=["filename", "timestamp"],
            ),
            "%Y-%m-%d %H:%M:%S%z",
            None,
            pd.DataFrame(
                data=[
                    [
                        "audio_20240617101411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 10:14:11+0000")),
                    ],
                    [
                        "audio_20240617111411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 11:14:11+0000")),
                    ],
                    [
                        "audio_20240617121411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 12:14:11+0000")),
                    ],
                ],
                columns=["filename", "timestamp"],
            ),
            id="already_formatted_timestamp_should_not_raise_error",
        ),
        pytest.param(
            pd.DataFrame(
                data=[
                    [
                        "audio_20240617101411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 10:14:11+0000")),
                    ],
                    [
                        "audio_20240617111411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 11:14:11+0000")),
                    ],
                    [
                        "audio_20240617121411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 12:14:11+0000")),
                    ],
                ],
                columns=["filename", "timestamp"],
            ),
            "%Y-%m-%d %H:%M:%S%z",
            "+0200",
            pd.DataFrame(
                data=[
                    [
                        "audio_20240617101411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 12:14:11+0200")),
                    ],
                    [
                        "audio_20240617111411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 13:14:11+0200")),
                    ],
                    [
                        "audio_20240617121411.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 14:14:11+0200")),
                    ],
                ],
                columns=["filename", "timestamp"],
            ),
            id="already_formatted_timestamp_converts_tz",
        ),
        pytest.param(
            pd.DataFrame(
                data=[
                    [
                        "audio_2024-06-17 10:14:11.wav",
                        "2024-06-17 10:14:11-0200",
                    ],
                    [
                        "audio_2024-06-17 11:14:11.wav",
                        "2024-06-17 11:14:11-0200",
                    ],
                    [
                        "audio_2024-06-17 12:14:11.wav",
                        "2024-06-17 12:14:11-0200",
                    ],
                ],
                columns=["filename", "timestamp"],
            ),
            "%Y-%m-%d %H:%M:%S%z",
            None,
            pd.DataFrame(
                data=[
                    [
                        "audio_2024_06_17 10_14_11.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 10:14:11-0200")),
                    ],
                    [
                        "audio_2024_06_17 11_14_11.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 11:14:11-0200")),
                    ],
                    [
                        "audio_2024_06_17 12_14_11.wav",
                        strftime_osmose_format(Timestamp("2024-06-17 12:14:11-0200")),
                    ],
                ],
                columns=["filename", "timestamp"],
            ),
            id="cleaned_filenames",
        ),
    ],
)
def test_adapt_timestamp_csv_to_osmose(
    timestamps: pd.DataFrame,
    date_template: str,
    timezone: str | None,
    expected: pd.DataFrame,
) -> None:
    assert adapt_timestamp_csv_to_osmose(timestamps, date_template, timezone).equals(
        expected,
    )


@pytest.mark.parametrize(
    ("event1", "event2", "expected"),
    [
        pytest.param(
            (
                Timestamp("2024-01-01 00:00:00"),
                Timestamp("2024-01-02 00:00:00"),
            ),
            (
                Timestamp("2024-01-01 00:00:00"),
                Timestamp("2024-01-02 00:00:00"),
            ),
            True,
            id="same_event",
        ),
        pytest.param(
            (
                    Timestamp("2024-01-01 00:00:00"),
                    Timestamp("2024-01-02 00:00:00"),
            ),
            (
                    Timestamp("2024-01-01 12:00:00"),
                    Timestamp("2024-01-02 12:00:00"),
            ),
            True,
            id="overlapping_events",
        ),
        pytest.param(
            (
                    Timestamp("2024-01-01 12:00:00"),
                    Timestamp("2024-01-02 12:00:00"),
            ),
            (
                    Timestamp("2024-01-01 00:00:00"),
                    Timestamp("2024-01-02 00:00:00"),
            ),
            True,
            id="overlapping_events_reversed",
        ),
        pytest.param(
            (
                    Timestamp("2024-01-01 00:00:00"),
                    Timestamp("2024-01-02 00:00:00"),
            ),
            (
                    Timestamp("2024-01-01 12:00:00"),
                    Timestamp("2024-01-01 12:01:00"),
            ),
            True,
            id="embedded_events",
        ),
        pytest.param(
            (
                    Timestamp("2024-01-01 0:00:00"),
                    Timestamp("2024-01-01 12:00:00"),
            ),
            (
                    Timestamp("2024-01-02 00:00:00"),
                    Timestamp("2024-01-02 12:00:00"),
            ),
            False,
            id="non_overlapping_events",
        ),
        pytest.param(
            (
                    Timestamp("2024-01-02 0:00:00"),
                    Timestamp("2024-01-02 12:00:00"),
            ),
            (
                    Timestamp("2024-01-01 00:00:00"),
                    Timestamp("2024-01-01 12:00:00"),
            ),
            False,
            id="non_overlapping_events_reversed",
        ),
        pytest.param(
            (
                    Timestamp("2024-01-01 0:00:00"),
                    Timestamp("2024-01-01 12:00:00"),
            ),
            (
                    Timestamp("2024-01-01 12:00:00"),
                    Timestamp("2024-01-02 00:00:00"),
            ),
            False,
            id="border_sharing_isnt_overlapping",
        ),
    ],
)
def test_overlapping_events(
    event1: tuple[Timestamp, Timestamp],
    event2: tuple[Timestamp, Timestamp],
    expected: bool,
) -> None:
    assert is_overlapping(event1, event2) == expected
