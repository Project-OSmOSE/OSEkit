from io import StringIO
import os
import pytest
import shutil
from OSmOSE.utils.core_utils import read_header, safe_read
from OSmOSE.config import OSMOSE_PATH
import numpy as np
import soundfile as sf


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:3 NaN detected")
def test_safe_read(input_dir):
    rate = 44100  # samples per second
    duration = 3
    rng = np.random.default_rng()

    data = rng.standard_normal(duration * rate)
    sf.write(
        input_dir.joinpath("nonan.wav"), data, rate, format="WAV", subtype="DOUBLE"
    )

    assert np.array_equal(data, safe_read(input_dir.joinpath("nonan.wav"))[0])

    nandata = data.copy()
    expected = data.copy()
    nandata[0], nandata[137], nandata[2055] = np.nan, np.nan, np.nan
    expected[0], expected[137], expected[2055] = 0.0, 0.0, 0.0
    sf.write(
        input_dir.joinpath("nan.wav"), nandata, rate, format="WAV", subtype="DOUBLE"
    )

    assert np.array_equal(expected, safe_read(input_dir.joinpath("nan.wav"))[0])


@pytest.mark.unit
def test_read_header(input_dir):
    sr = 44100
    frames = float(sr * 3)
    channels = 1
    sampwidth = 4
    size = 529272

    assert (sr, frames, sampwidth, channels, size) == read_header(
        input_dir.joinpath("test.wav")
    )
