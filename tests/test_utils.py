from io import StringIO
import os
import pytest
import shutil
from OSmOSE.utils import *
from OSmOSE.config import OSMOSE_PATH

@pytest.mark.unit
def test_display_folder_storage_infos(monkeypatch):
    mock_usage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(
        shutil, "disk_usage", lambda: mock_usage(2048**4, 1536**4, 1862**4)
    )

    assert True

@pytest.mark.unit
def test_read_header(input_dir):
    sr, frames, channels, sampwidth,size = read_header(input_dir.joinpath("test.wav"))
    print(size)

    assert sr == 44100
    assert frames == 132300.0
    assert channels == 1
    assert sampwidth == 4
    assert size == 529272

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

    assert (sr, frames, channels, sampwidth,size) == read_header(
        input_dir.joinpath("test.wav")
    )

@pytest.mark.unit
def test_check_n_files_ok_files(input_dir, output_dir):
    file_list = [input_dir.joinpath("test.wav")]
    for i in range(9):
        wav_file = input_dir.joinpath(f"test{i}.wav")
        shutil.copyfile(input_dir.joinpath("test.wav"), wav_file)
        file_list.append(wav_file)

    # False returned means there was no normalization
    assert not check_n_files(file_list=file_list, n=10, output_path=output_dir)

    assert len(os.listdir(output_dir)) == 0
    
