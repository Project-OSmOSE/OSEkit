import pytest
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
    ("%Y%y%m%d%H%M%S%I%p%f", "([12][0-9]{3})([0-9]{2})(0[1-9]|1[0-2])([0-2][0-9]|3[0-1])([0-1][0-9]|2[0-4])([0-5][0-9])([0-5][0-9])(0[1-9]|1[0-2])(AM|PM)([0-9]{6})"),
    ("%y %m %d %H %M %S", "([0-9]{2}) (0[1-9]|1[0-2]) ([0-2][0-9]|3[0-1]) ([0-1][0-9]|2[0-4]) ([0-5][0-9]) ([0-5][0-9])"),
    ("%y:%m:%d%H.%M%S", "([0-9]{2}):(0[1-9]|1[0-2]):([0-2][0-9]|3[0-1])([0-1][0-9]|2[0-4]).([0-5][0-9])([0-5][0-9])"),
    ("%y%m%d_at_%H%M%S", "([0-9]{2})(0[1-9]|1[0-2])([0-2][0-9]|3[0-1])_at_([0-1][0-9]|2[0-4])([0-5][0-9])([0-5][0-9])"),
    ("{%y}%m%d(%H%M%S)", "{([0-9]{2})}(0[1-9]|1[0-2])([0-2][0-9]|3[0-1])\(([0-1][0-9]|2[0-4])([0-5][0-9])([0-5][0-9])\)"),
])
def test_build_regex_from_datetime_template(datetime_template, expected):
    assert build_regex_from_datetime_template(datetime_template) == expected