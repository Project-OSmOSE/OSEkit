import os
import pytest
import shutil
from OSmOSE.utils import *
from OSmOSE.config import OSMOSE_PATH


def test_display_folder_storage_infos(monkeypatch):
    mock_usage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(
        shutil, "disk_usage", lambda: mock_usage(2048**4, 1536**4, 1862**4)
    )

    assert True


def test_read_header(input_dir):
    sr, frames, channels, sampwidth = read_header(input_dir.joinpath("test.wav"))

    assert sr == 44100
    assert frames == 132300.0
    assert channels == 1
    assert sampwidth == 2


def test_safe_read(input_dir):
    rate = 44100  # samples per second
    duration = 3
    rng = np.random.default_rng()

    data = rng.standard_normal(duration * rate)
    sf.write(input_dir.joinpath("nonan.wav"), data, rate)

    assert np.array_equal(data, safe_read(input_dir.joinpath("nonan.wav"))[0])

    nandata = data.copy()
    nandata[0], nandata[137], nandata[2055] = np.nan
    sf.write(input_dir.joinpath("nan.wav"), nandata, rate)

    assert np.array_equal(data, safe_read(input_dir.joinpath("nan.wav"))[0])
