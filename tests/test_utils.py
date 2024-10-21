from io import StringIO
import os

import pandas as pd
import pytest
import shutil
from OSmOSE.utils.core_utils import read_header, safe_read
from OSmOSE.config import OSMOSE_PATH
import numpy as np
import soundfile as sf
from OSmOSE.utils.formatting_utils import aplose2raven


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


@pytest.fixture
def correct_dataframe():
    data = pd.DataFrame(
        {
            "dataset": ["dataset_test", "dataset_test", "dataset_test"],
            "filename": ["file1.wav", "file2.wav", "file3.wav"],
            "start_time": [0, 0, 5.9],
            "end_time": [60, 60, 8.1],
            "start_frequency": [0, 0, 18500.0],
            "end_frequency": [96000, 96000, 53000.0],
            "annotation": ["boat", "boat", "boat"],
            "annotator": ["bbjuni", "bbjuni", "bbjuni"],
            "start_datetime": [
                pd.Timestamp("2020-05-29T11:30:00.000+00:00"),
                pd.Timestamp("2020-05-29T11:31:00.000+00:00"),
                pd.Timestamp("2020-05-29T11:31:00.000+00:00"),
            ],
            "end_datetime": [
                pd.Timestamp("2020-05-29T11:31:00.000+00:00"),
                pd.Timestamp("2020-05-29T11:32:00.000+00:00"),
                pd.Timestamp("2020-05-29T11:32:00.000+00:00"),
            ],
            "is_box": [0, 0, 1],
        }
    )

    return data.reset_index(drop=True)


@pytest.mark.unit
def test_aplose2raven(correct_dataframe):
    df_raven = aplose2raven(df=correct_dataframe)

    expected_dataframe = pd.DataFrame(
        {
            "Selection": [1, 2, 3],
            "View": [1, 1, 1],
            "Channel": [1, 1, 1],
            "Begin Time (s)": [0.0, 60.0, 60.0],
            "End Time (s)": [60.0, 120.0, 68.1],
            "Low Freq (Hz)": [0.0, 0.0, 18500.0],
            "High Freq (Hz)": [96000.0, 96000.0, 53000.0],
        }
    )

    assert expected_dataframe.equals(df_raven)
