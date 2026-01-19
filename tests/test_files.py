from __future__ import annotations

from pathlib import Path

import pytest
import pytz
from pandas import Timestamp

from tests.test_core_api_base import DummyDataset, DummyFile


@pytest.mark.parametrize(
    (
        "file_name",
        "strptime_format",
        "timezone",
        "expected_begin",
    ),
    [
        pytest.param(
            "040503121212.foo",
            "%y%m%d%H%M%S",
            None,
            Timestamp("2004-05-03 12:12:12"),
            id="naive_strptime_naive_timezone",
        ),
        pytest.param(
            "040503121212.foo",
            "%y%m%d%H%M%S",
            "America/Chihuahua",
            Timestamp("2004-05-03 12:12:12-0600", tz="America/Chihuahua"),
            id="naive_strptime_localized_to_str_timezone",
        ),
        pytest.param(
            "040503121212.foo",
            "%y%m%d%H%M%S",
            pytz.timezone("America/Chihuahua"),
            Timestamp("2004-05-03 12:12:12-0600", tz="America/Chihuahua"),
            id="naive_strptime_localized_to_pytz_timezone",
        ),
        pytest.param(
            "040503121212.foo",
            "%y%m%d%H%M%S",
            "-0600",
            Timestamp("2004-05-03 12:12:12-0600"),
            id="naive_strptime_localized_to_utc_offset",
        ),
        pytest.param(
            "040503121212_-0600.foo",
            "%y%m%d%H%M%S_%z",
            None,
            Timestamp("2004-05-03 12:12:12-0600"),
            id="aware_strptime_no_provided_timezone",
        ),
        pytest.param(
            "040503121212_-0600.foo",
            "%y%m%d%H%M%S_%z",
            "-0600",
            Timestamp("2004-05-03 12:12:12-0600"),
            id="aware_strptime_same_provided_timezone",
        ),
        pytest.param(
            "040503121212_-0600.foo",
            "%y%m%d%H%M%S_%z",
            "UTC",
            Timestamp("2004-05-03 18:12:12", tz="UTC"),
            id="aware_strptime_converted_to_provided_timezone",
        ),
        pytest.param(
            "040503121212_-0600.foo",
            "%y%m%d%H%M%S_%z",
            "+0100",
            Timestamp("2004-05-03 19:12:12", tz="UTC+01:00"),
            id="aware_strptime_converted_to_provided_utc_offset",
        ),
    ],
)
def test_file_localization(
    file_name: str,
    strptime_format: str,
    timezone: str | pytz.timezone | None,
    expected_begin: Timestamp,
) -> None:
    file = DummyFile(
        path=Path(file_name),
        strptime_format=strptime_format,
        timezone=timezone,
    )

    assert expected_begin == file.begin


@pytest.mark.parametrize(
    (
        "file_names",
        "strptime_format",
        "timezone",
        "expected_begins",
    ),
    [
        pytest.param(
            [
                "040503121212",
                "040503131212",
                "040503141212",
            ],
            "%y%m%d%H%M%S",
            None,
            [
                Timestamp("2004-05-03 12:12:12"),
                Timestamp("2004-05-03 13:12:12"),
                Timestamp("2004-05-03 14:12:12"),
            ],
            id="naive_files",
        ),
        pytest.param(
            [
                "040503121212",
                "040503131212",
                "040503141212",
            ],
            "%y%m%d%H%M%S",
            "America/Chihuahua",
            [
                Timestamp("2004-05-03 12:12:12-0600", tz="America/Chihuahua"),
                Timestamp("2004-05-03 13:12:12-0600", tz="America/Chihuahua"),
                Timestamp("2004-05-03 14:12:12-0600", tz="America/Chihuahua"),
            ],
            id="naive_files_localized",
        ),
        pytest.param(
            [
                "040503121212_+0100",
                "040503131212_+0100",
                "040503141212_+0100",
            ],
            "%y%m%d%H%M%S_%z",
            None,
            [
                Timestamp("2004-05-03 12:12:12+0100"),
                Timestamp("2004-05-03 13:12:12+0100"),
                Timestamp("2004-05-03 14:12:12+0100"),
            ],
            id="already_aware_files",
        ),
        pytest.param(
            [
                "040503121212_+0000",
                "040503121212_-0100",
                "040503121212_-0200",
            ],
            "%y%m%d%H%M%S_%z",
            "UTC",
            [
                Timestamp("2004-05-03 12:12:12+0000", tz="UTC"),
                Timestamp("2004-05-03 13:12:12+0000", tz="UTC"),
                Timestamp("2004-05-03 14:12:12+0000", tz="UTC"),
            ],
            id="aware_files_are_converted",
        ),
    ],
)
def test_dataset_localization(
    tmp_path: Path,
    file_names: list[str],
    strptime_format: str,
    timezone: str | pytz.timezone | None,
    expected_begins: list[Timestamp],
) -> None:
    for file in file_names:
        (tmp_path / f"{file}").touch()

    dataset = DummyDataset.from_folder(
        tmp_path,
        strptime_format=strptime_format,
        timezone=timezone,
    )

    assert all(
        begin == expected
        for begin, expected in zip(
            expected_begins,
            [file.begin for file in sorted(dataset.files, key=lambda f: f.begin)],
            strict=False,
        )
    )
