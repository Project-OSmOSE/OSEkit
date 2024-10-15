import pytest
from pandas import Timestamp
from OSmOSE.utils.timestamp_utils import *


@pytest.mark.unittest
@pytest.mark.parametrize('datetime_template, expected', [
    ("%y%m%d%H%M%S", True),
    ("%Y%y%m%d%H%M%S%I%p%f", True),
    ("%y %m %d %H %M %S", True),
    ("%y:%m:%d%H.%M%S", True),
    ("%y%z%d%H%X%S", False),
    ("%y%z%d%H%%%S", False),
    ("%y%m%d_at_%H%M%S", True),
    ("%y%m%d%H%M%S%", False)
])
def test_is_datetime_template_valid(datetime_template, expected):
    assert is_datetime_template_valid(datetime_template) == expected


@pytest.mark.unittest
@pytest.mark.parametrize('datetime_template, expected', [
    ("%y%m%d%H%M%S", "([0-9]{2})(0[1-9]|1[0-2])([0-2][0-9]|3[0-1])([0-1][0-9]|2[0-4])([0-5][0-9])([0-5][0-9])"),
    ("%Y%y%m%d%H%M%S%I%p%f",
     "([12][0-9]{3})([0-9]{2})(0[1-9]|1[0-2])([0-2][0-9]|3[0-1])([0-1][0-9]|2[0-4])([0-5][0-9])([0-5][0-9])(0[1-9]|1[0-2])(AM|PM)([0-9]{6})"),
    ("%y %m %d %H %M %S",
     "([0-9]{2}) (0[1-9]|1[0-2]) ([0-2][0-9]|3[0-1]) ([0-1][0-9]|2[0-4]) ([0-5][0-9]) ([0-5][0-9])"),
    ("%y:%m:%d%H.%M%S", "([0-9]{2}):(0[1-9]|1[0-2]):([0-2][0-9]|3[0-1])([0-1][0-9]|2[0-4]).([0-5][0-9])([0-5][0-9])"),
    ("%y%m%d_at_%H%M%S", "([0-9]{2})(0[1-9]|1[0-2])([0-2][0-9]|3[0-1])_at_([0-1][0-9]|2[0-4])([0-5][0-9])([0-5][0-9])"),
    ("{%y}%m%d(%H%M%S)",
     "{([0-9]{2})}(0[1-9]|1[0-2])([0-2][0-9]|3[0-1])\(([0-1][0-9]|2[0-4])([0-5][0-9])([0-5][0-9])\)"),
])
def test_build_regex_from_datetime_template(datetime_template, expected):
    assert build_regex_from_datetime_template(datetime_template) == expected


@pytest.mark.integtest
@pytest.mark.parametrize('filename, datetime_template, expected', [
    ('7189.230405144906.wav', '%y%m%d%H%M%S',
     Timestamp(year=2023, month=4, day=5, hour=14, minute=49, second=6, microsecond=0)),
    ('7189.(230405)/(144906).wav', '(%y%m%d)/(%H%M%S)',
     Timestamp(year=2023, month=4, day=5, hour=14, minute=49, second=6, microsecond=0)),
    ('7189.23_04_05_14:49:06.wav', '%y_%m_%d_%H:%M:%S',
     Timestamp(year=2023, month=4, day=5, hour=14, minute=49, second=6, microsecond=0)),
    ('7189.230405_at_144906.wav', '%y%m%d_at_%H%M%S',
     Timestamp(year=2023, month=4, day=5, hour=14, minute=49, second=6, microsecond=0)),
    ('7189.202323040514490602PM000010.wav', '%Y%y%m%d%H%M%S%I%p%f',
     Timestamp(year=2023, month=4, day=5, hour=14, minute=49, second=6, microsecond=10)),
    ('7189.230405144906.wav', '%y%m%d%H%M%%S', pytest.raises(ValueError)),
    ('7189.230405144906.wav', '%y%m%d%H%M%P%S', pytest.raises(ValueError)),
    ('7189.230405144906.wav', '%y%m%d%H%M%52%S', pytest.raises(ValueError)),
])
def test_extract_timestamp_from_filename(filename: str, datetime_template: str, expected: Timestamp):
    assert extract_timestamp_from_filename(filename, datetime_template) == expected


@pytest.mark.integtest
@pytest.mark.parametrize('filename, datetime_template, expected', [
    ('7189.230405144906.wav', '%y%m%d%H%M%%S', pytest.raises(ValueError)),
    ('7189.230405144906.wav', '%y%m%d%H%M%P%S', pytest.raises(ValueError)),
    ('7189.230405144906.wav', '%y%m%d%H%M%52%S', pytest.raises(ValueError)),
    ('7189.230405_144906.wav', '%y%m%d%H%M%S', pytest.raises(ValueError)),
    ('7189.230405144906.wav', '%Y%m%d%H%M%S', pytest.raises(ValueError)),
    ('7189.230405146706.wav', '%Y%m%d%H%M%S', pytest.raises(ValueError)),
])
def test_extract_timestamp_from_filename(filename: str, datetime_template: str, expected: Timestamp):
    with expected as e:
        assert extract_timestamp_from_filename(filename, datetime_template) == e
