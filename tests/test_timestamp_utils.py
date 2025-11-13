from __future__ import annotations

import logging

import pytest
from pandas import Timedelta, Timestamp

from osekit.utils.timestamp_utils import (
    build_regex_from_datetime_template,
    is_datetime_template_valid,
    last_window_end,
    localize_timestamp,
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
            r"(\d{2})(0[1-9]|1[0-2])([0-2]\d|3[0-1])((?:[a-zA-Z]+)(?:[-/]\w+)*(?:[\+-]\d+)?)",
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
        pytest.param(
            "14:49:06.1232023-04-05_UTC_soundfile.wav",
            "%H:%M:%S.%f%Y-%m-%d_%Z",
            Timestamp("2023-04-05 14:49:06.123000", tz="UTC"),
            id="underscore_after_%Z",
        ),
        pytest.param(
            "14:49:06.1232023-04-05_UTC_soundfile.wav",
            ["%H:%M:%S.%f%Y-%m-%d_%Z", "%y%m%d%H%M%S_%Z"],
            Timestamp("2023-04-05 14:49:06.123000", tz="UTC"),
            id="first_template_from_list_is_matching",
        ),
        pytest.param(
            "14:49:06.1232023-04-05_UTC_soundfile.wav",
            ["%y%m%d%H%M%S_%Z", "%H:%M:%S.%f%Y-%m-%d_%Z"],
            Timestamp("2023-04-05 14:49:06.123000", tz="UTC"),
            id="first_unmatching_but_second_matching",
        ),
        pytest.param(
            "14:49:06.1232023-04-05_UTC_soundfile.wav",
            ["%y%m%d%H%M%%S", "%H:%M:%S.%f%Y-%m-%d_%Z"],
            Timestamp("2023-04-05 14:49:06.123000", tz="UTC"),
            id="first_invalid_but_second_matching",
        ),
        pytest.param(
            "14:49:06.1232023-04-05_UTC_soundfile.wav",
            ["%y%m%d%H%M%%S", "%y%m%d%H%M%S_%Z", "%H:%M:%S.%f%Y-%m-%d_%Z"],
            Timestamp("2023-04-05 14:49:06.123000", tz="UTC"),
            id="invalid_then_unmatching_then_matching",
        ),
    ],
)
def test_strptime_from_text(
    text: str,
    datetime_template: str | list[str],
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
        pytest.param(
            "2023-04-05T14:49:06.-0200.wav",
            ["%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dU%H:%M:%S.%f"],
            pytest.raises(
                ValueError,
                match="2023-04-05T14:49:06.-0200.wav did not match "
                "the given %Y-%m-%dT%H:%M:%S.%f%z template\n"
                "2023-04-05T14:49:06.-0200.wav did not match "
                "the given %Y-%m-%dU%H:%M:%S.%f template",
            ),
            id="no_matching_template_in_list",
        ),
        pytest.param(
            "2023-04-05T14:49:06.-0200.wav",
            ["%Y-%m-%dT%H:%M:%S.%k", "%a-%m-%dT%H:%M:%S.%f"],
            pytest.raises(
                ValueError,
                match="%Y-%m-%dT%H:%M:%S.%k is not a supported strftime template\n"
                "%a-%m-%dT%H:%M:%S.%f is not a supported strftime template",
            ),
            id="no_valid_template_in_list",
        ),
        pytest.param(
            "2023-04-05T14:49:06.-0200.wav",
            ["%Y-%m-%dT%H:%M:%S.%f%z", "%a-%m-%dT%H:%M:%S.%f"],
            pytest.raises(
                ValueError,
                match="2023-04-05T14:49:06.-0200.wav did not match "
                "the given %Y-%m-%dT%H:%M:%S.%f%z template\n"
                "%a-%m-%dT%H:%M:%S.%f is not a supported strftime template",
            ),
            id="error_mix",
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
            Timestamp("2024-10-17 10:14:11+0000", tz="UTC"),
            id="UTC_timezone",
        ),
        pytest.param(
            Timestamp("2024-10-17 10:14:11"),
            "Pacific/Rarotonga",
            Timestamp("2024-10-17 10:14:11-1000", tz="Pacific/Rarotonga"),
            id="Non-UTC_timezone",
        ),
        pytest.param(
            Timestamp("2024-10-17 10:14:11"),
            "+03:00",
            Timestamp("2024-10-17 10:14:11+0300", tz="UTC+03:00"),
            id="UTC_offset_timezone",
        ),
        pytest.param(
            Timestamp("2024-10-17 10:14:11+0200"),
            "+02:00",
            Timestamp("2024-10-17 10:14:11+0200", tz="UTC+02:00"),
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
            Timestamp("2024-10-17T10:14:11.000+0200", tz="UTC+02:00"),
            id="UTC_offset_timezone_without_colon",
        ),
        pytest.param(
            Timestamp("2024-10-17 10:14:11"),
            "-0700",
            Timestamp("2024-10-17T10:14:11.000-0700", tz="UTC-07:00"),
            id="negative_UTC_offset_timezone",
        ),
        pytest.param(
            Timestamp("2024-10-17 10:14:11"),
            "-0000",
            Timestamp("2024-10-17T10:14:11.000+0000", tz="UTC"),
            id="negative_zero_UTC_offset_timezone",
        ),
    ],
)
def test_localize_timestamp(
    timestamp: Timestamp,
    timezone: str,
    expected: Timestamp,
) -> None:
    assert localize_timestamp(timestamp, timezone) == expected


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
    ("begin", "end", "window_duration", "window_hop", "expected"),
    [
        pytest.param(
            Timestamp("2000-01-01 00:00:00"),
            Timestamp("2000-01-01 00:01:00"),
            Timedelta(seconds=10),
            Timedelta(seconds=10),
            Timestamp("2000-01-01 00:01:00"),
            id="no_overlap_ends_at_end",
        ),
        pytest.param(
            Timestamp("2000-01-01 00:00:00"),
            Timestamp("2000-01-01 00:00:58"),
            Timedelta(seconds=10),
            Timedelta(seconds=10),
            Timestamp("2000-01-01 00:01:00"),
            id="no_overlap_ends_after_end",
        ),
        pytest.param(
            Timestamp("2000-01-01 00:00:00"),
            Timestamp("2000-01-01 00:01:00"),
            Timedelta(seconds=10),
            Timedelta(seconds=1),
            Timestamp("2000-01-01 00:01:09"),
            id="overlap_ends_after_end",
        ),
    ],
)
def test_last_window_end(
    begin: Timestamp,
    end: Timestamp,
    window_duration: Timedelta,
    window_hop: Timedelta,
    expected: Timestamp,
) -> None:
    assert (
        last_window_end(
            begin=begin,
            end=end,
            window_duration=window_duration,
            window_hop=window_hop,
        )
        == expected
    )
