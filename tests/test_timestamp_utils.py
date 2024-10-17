import pandas as pd
import pytest
from pandas import Timestamp
from OSmOSE.utils.timestamp_utils import *


@pytest.mark.unittest
@pytest.mark.parametrize(
    "datetime_template, expected",
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
def test_is_datetime_template_valid(datetime_template, expected):
    assert is_datetime_template_valid(datetime_template) == expected


@pytest.mark.unittest
@pytest.mark.parametrize(
    "datetime_template, expected",
    [
        pytest.param(
            "%y%m%d%H%M%S",
            "([0-9]{2})(0[1-9]|1[0-2])([0-2][0-9]|3[0-1])([0-1][0-9]|2[0-4])([0-5][0-9])([0-5][0-9])",
            id="simple_pattern",
        ),
        pytest.param(
            "%Y%y%m%d%H%M%S%I%p%f",
            "([12][0-9]{3})([0-9]{2})(0[1-9]|1[0-2])([0-2][0-9]|3[0-1])([0-1][0-9]|2[0-4])([0-5][0-9])([0-5][0-9])(0[1-9]|1[0-2])(AM|PM)([0-9]{6})",
            id="all_strftime_codes",
        ),
        pytest.param(
            "%y %m %d %H %M %S",
            "([0-9]{2}) (0[1-9]|1[0-2]) ([0-2][0-9]|3[0-1]) ([0-1][0-9]|2[0-4]) ([0-5][0-9]) ([0-5][0-9])",
            id="spaces_separated_codes",
        ),
        pytest.param(
            "%y:%m:%d%H.%M%S",
            "([0-9]{2}):(0[1-9]|1[0-2]):([0-2][0-9]|3[0-1])([0-1][0-9]|2[0-4]).([0-5][0-9])([0-5][0-9])",
            id="special_chars_separated_codes",
        ),
        pytest.param(
            "%y%m%d_at_%H%M%S",
            "([0-9]{2})(0[1-9]|1[0-2])([0-2][0-9]|3[0-1])_at_([0-1][0-9]|2[0-4])([0-5][0-9])([0-5][0-9])",
            id="alpha_letters_separated_codes",
        ),
        pytest.param(
            "{%y}%m%d(%H%M%S)",
            "{([0-9]{2})}(0[1-9]|1[0-2])([0-2][0-9]|3[0-1])\(([0-1][0-9]|2[0-4])([0-5][0-9])([0-5][0-9])\)",
            id="parentheses_separated_codes",
        ),
        pytest.param(
            "%y%m%d%z",
            "([0-9]{2})(0[1-9]|1[0-2])([0-2][0-9]|3[0-1])([\+-]\d{4})",
            id="utc_offset",
        ),
        pytest.param(
            "%y%m%d%Z",
            "([0-9]{2})(0[1-9]|1[0-2])([0-2][0-9]|3[0-1])((?:\w+)(?:[-/]\w+)*(?:[\+-]\d+)?)",
            id="timezone_name",
        ),
    ],
)
def test_build_regex_from_datetime_template(datetime_template: str, expected: str):
    assert build_regex_from_datetime_template(datetime_template) == expected


@pytest.mark.integtest
@pytest.mark.parametrize(
    "filename, datetime_template, expected",
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
    ],
)
def test_extract_timestamp_from_filename(
    filename: str, datetime_template: str, expected: Timestamp
):
    assert extract_timestamp_from_filename(filename, datetime_template) == expected


@pytest.mark.integtest
@pytest.mark.parametrize(
    "filename, datetime_template, expected",
    [
        pytest.param(
            "7189.230405144906.wav",
            "%y%m%d%H%M%%S",
            pytest.raises(
                ValueError, match="%y%m%d%H%M%%S is not a supported strftime template"
            ),
            id="%%_is_wrong_strftime_code",
        ),
        pytest.param(
            "7189.230405144906.wav",
            "%y%m%d%H%M%P%S",
            pytest.raises(
                ValueError, match="%y%m%d%H%M%P%S is not a supported strftime template"
            ),
            id="%P_is_wrong_strftime_code",
        ),
        pytest.param(
            "7189.230405144906.wav",
            "%y%m%d%H%M%52%S",
            pytest.raises(
                ValueError, match="%y%m%d%H%M%52%S is not a supported strftime template"
            ),
            id="%5_is_wrong_strftime_code",
        ),
        pytest.param(
            "7189.230405_144906.wav",
            "%y%m%d%H%M%S",
            pytest.raises(
                ValueError,
                match="7189.230405_144906.wav did not match the given %y%m%d%H%M%S template",
            ),
            id="unmatching_pattern",
        ),
        pytest.param(
            "7189.230405144906.wav",
            "%Y%m%d%H%M%S",
            pytest.raises(
                ValueError,
                match="7189.230405144906.wav did not match the given %Y%m%d%H%M%S template",
            ),
            id="%Y_should_have_4_digits",
        ),
        pytest.param(
            "7189.230405146706.wav",
            "%y%m%d%H%M%S",
            pytest.raises(
                ValueError,
                match="7189.230405146706.wav did not match the given %y%m%d%H%M%S template",
            ),
            id="%M_should_be_lower_than_60",
        ),
        pytest.param(
            "7189.230405146706_0200.wav",
            "%y%m%d%H%M%S_%z",
            pytest.raises(ValueError),
            id="incorrect_timezone_offset",
        ),
        pytest.param(
            "7189.230405146706_+2500.wav",
            "%y%m%d%H%M%S_%z",
            pytest.raises(ValueError),
            id="out_of_range_timezone_offset",
        ),
        pytest.param(
            "7189.230405146706_Brest.wav",
            "%y%m%d%H%M%S_%Z",
            pytest.raises(ValueError),
            id="incorrect_timezone_name",
        ),
    ],
)
def test_extract_timestamp_from_filename_errors(
    filename: str, datetime_template: str, expected: Timestamp
):
    with expected as e:
        assert extract_timestamp_from_filename(filename, datetime_template) == e


@pytest.fixture
def correct_series():
    series = pd.Series(
        {
            "something2345_2012_06_24__16:32:10.wav": pd.Timestamp(
                "2012-06-24 16:32:10"
            ),
            "something2345_2023_07_28__08:24:50.flac": pd.Timestamp(
                "2023-07-28 08:24:50"
            ),
            "something2345_2024_01_01__23:12:11.WAV": pd.Timestamp(
                "2024-01-01 23:12:11"
            ),
            "something2345_2024_02_02__02:02:02.FLAC": pd.Timestamp(
                "2024-02-02 02:02:02"
            ),
        },
        name="timestamp",
    )
    series.index.name = "filename"
    return series.reset_index()


@pytest.mark.integtest
def test_associate_timestamps(correct_series):
    input_files = list(correct_series["filename"])
    assert associate_timestamps((i for i in input_files), "%Y_%m_%d__%H:%M:%S").equals(
        correct_series
    )


@pytest.mark.integtest
def test_associate_timestamps_error_with_incorrect_datetime_format(correct_series):
    input_files = list(correct_series["filename"])
    mismatching_datetime_format = "%Y%m%d__%H:%M:%S"
    incorrect_datetime_format = "%y%m%d%H%M%P%S"

    with pytest.raises(
        ValueError,
        match=f"{input_files[0]} did not match the given {mismatching_datetime_format} template",
    ) as e:
        assert e == associate_timestamps(
            (i for i in input_files), mismatching_datetime_format
        )

    with pytest.raises(
        ValueError,
        match=f"{incorrect_datetime_format} is not a supported strftime template",
    ):
        assert e == associate_timestamps(
            (i for i in input_files), incorrect_datetime_format
        )


@pytest.mark.unittest
@pytest.mark.parametrize(
    "timestamp, expected",
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
def test_strftime_osmose_format(timestamp: Timestamp, expected: str):
    assert strftime_osmose_format(timestamp) == expected
