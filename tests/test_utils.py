import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from OSmOSE.utils.core_utils import read_header, safe_read
from OSmOSE.utils.formatting_utils import aplose2raven


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:3 NaN detected")
def test_safe_read(input_dir):
    rate = 44100  # samples per second
    duration = 3
    rng = np.random.default_rng()

    data = rng.standard_normal(duration * rate)
    sf.write(
        input_dir.joinpath("nonan.wav"),
        data,
        rate,
        format="WAV",
        subtype="DOUBLE",
    )

    assert np.array_equal(data, safe_read(input_dir.joinpath("nonan.wav"))[0])

    nandata = data.copy()
    expected = data.copy()
    nandata[0], nandata[137], nandata[2055] = np.nan, np.nan, np.nan
    expected[0], expected[137], expected[2055] = 0.0, 0.0, 0.0
    sf.write(
        input_dir.joinpath("nan.wav"),
        nandata,
        rate,
        format="WAV",
        subtype="DOUBLE",
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
        input_dir.joinpath("test.wav"),
    )


@pytest.fixture
def aplose_dataframe() -> pd.DataFrame:
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
                pd.Timestamp("2020-05-29T11:31:05.900+00:00"),
            ],
            "end_datetime": [
                pd.Timestamp("2020-05-29T11:31:00.000+00:00"),
                pd.Timestamp("2020-05-29T11:32:00.000+00:00"),
                pd.Timestamp("2020-05-29T11:32:08.100+00:00"),
            ],
            "is_box": [0, 0, 1],
        },
    )

    return data.reset_index(drop=True)


@pytest.fixture
def raven_timestamps() -> list:
    return list(
        pd.date_range(
            start="2020-05-29T11:30:00.000+00:00",
            end="2020-05-29T11:35:00.000+00:00",
            freq="1min",
        ),
    )


@pytest.fixture
def raven_durations(raven_timestamps: pytest.fixture) -> list:
    return [60] * len(raven_timestamps)


@pytest.mark.unit
def test_aplose2raven(
    aplose_dataframe: pytest.fixture,
    raven_timestamps: pytest.fixture,
    raven_durations: pytest.fixture,
) -> None:
    raven_dataframe = aplose2raven(
        aplose_result=aplose_dataframe,
        audio_datetimes=raven_timestamps,
        audio_durations=raven_durations,
    )

    expected_raven_dataframe = pd.DataFrame(
        {
            "Selection": [1, 2, 3],
            "View": [1, 1, 1],
            "Channel": [1, 1, 1],
            "Begin Time (s)": [0.0, 60.0, 65.9],
            "End Time (s)": [60.0, 120.0, 128.1],
            "Low Freq (Hz)": [0.0, 0.0, 18500.0],
            "High Freq (Hz)": [96000.0, 96000.0, 53000.0],
        },
    )

    assert expected_raven_dataframe.equals(raven_dataframe)
