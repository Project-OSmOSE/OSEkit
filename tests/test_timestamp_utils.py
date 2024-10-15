import pytest
from OSmOSE.utils.timestamp_utils import *

@pytest.mark.unittest
@pytest.mark.parametrize('datetime_template, expected', [
    ("%y%m%d%H%M%S", True),
    ("%y%m%d%H%M%S%I%p%M%S%f", True),
    ("%y %m %d %H %M %S", True),
    ("%y:%m:%d%H.%M%S", True),
    ("%y%z%d%H%X%S", False),
    ("%y%z%d%H%%%S", False),
    ("%y%m%d_on_%H%M%S", True),
    ("%y%m%d%H%M%S%", False)
])
def test_is_datetime_template_valid(datetime_template, expected):
    assert is_datetime_template_valid(datetime_template) == expected