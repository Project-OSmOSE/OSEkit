import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import chirp
from unittest.mock import patch
import logging

from OSmOSE.config import OSMOSE_PATH


def capture_csv(monkeypatch):
    pass


@pytest.fixture(autouse=True)
def patch_filehandlers(monkeypatch, request):
    if "allow_log_write_to_file" in request.keywords:
        return

    def disabled_filewrite(self, record):
        pass

    monkeypatch.setattr(logging.FileHandler, "emit", disabled_filewrite)


@pytest.fixture
def input_dataset(tmp_path: Path):
    """Fixture to create an input dataset.

    Creates the basic structure of a dataset in a temporary direction, as well as 10 audio files (5 wav and 5 flac) of 3 seconds of random noise at a sample rate of 44100,
     as well as the timestamp.csv file, from 2022-01-01T12:00:00 to 2022-01-01T12:00:30

    Returns
    -------
        The paths to the dataset's folders, in order :
        - root directory
        - main audio directory
        - original audio sub-directory
        - main spectrogram directory.

    """
    main_dir = tmp_path.joinpath("sample_dataset")
    main_audio_dir = main_dir.joinpath(OSMOSE_PATH.raw_audio)
    orig_audio_dir = main_audio_dir.joinpath("original")
    process_dir = main_dir.joinpath(OSMOSE_PATH.spectrogram)

    folders_to_create = [main_dir, main_audio_dir, orig_audio_dir, process_dir]

    for folder in folders_to_create:
        folder.mkdir(parents=True)

    rate = 44100  # samples per second
    duration = 3
    rng = np.random.default_rng()

    for i in range(10):
        if i == 0:  # make first signal deterministic
            t = np.linspace(0, duration, int(duration * rate))
            data = chirp(t, f0=6, f1=1, t1=duration, method="linear")
        else:
            data = rng.standard_normal(duration * rate)

        data[data > 1] = 1
        data[data < -1] = -1

        if i < 5:
            wav_file = orig_audio_dir.joinpath(f"20220101_1200{str(3*i).zfill(2)}.wav")
            sf.write(wav_file, data, rate, format="wav", subtype="FLOAT")
        else:
            flac_file = orig_audio_dir.joinpath(
                f"20220101_1200{str(3*i).zfill(2)}.flac",
            )
            sf.write(flac_file, data, rate, format="flac", subtype="PCM_24")
    return dict(
        zip(
            ["main_dir", "main_audio_dir", "orig_audio_dir", "process_dir"],
            folders_to_create,
        ),
    )


@pytest.fixture
def input_dir(tmp_path):
    """Creates a temporary input directory with a single audio file.

    The file is 3 seconds of random noise at a sample rate of 44100.

    Returns
    -------
        input_dir: `Path`
            The path to the input directory.

    """
    # Parameters
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    rate = 44100  # samples per second
    duration = 3
    rng = np.random.default_rng()

    data = rng.standard_normal(duration * rate)
    data[data > 1] = 1
    data[data < -1] = -1
    wav_file = input_dir / "test.wav"
    sf.write(wav_file, data, rate, format="WAV", subtype="FLOAT")

    return input_dir


@pytest.fixture
def output_dir(tmp_path: Path):
    """Creates an empty temporary output directory.

    Returns
    -------
        The directory path

    """
    output_dir = tmp_path.joinpath("output")
    if output_dir.is_dir():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def input_spectrogram(input_dataset):
    """Creates an input dataset and analysis parameters.

    See input_dataset for the details of the input dataset.

    Returns
    -------
        input_dataset: `Path`
            The path to the dataset directory
        analysis_params: `dict`
            Dummy analysis parameters

    """
    analysis_params = {
        "nfft": 512,
        "winsize": 512,
        "overlap": 97,
        "spectro_colormap": "viridis",
        "zoom_levels": 2,
        "number_adjustment_spectrograms": 2,
        "dynamic_min": 0,
        "dynamic_max": 150,
        "spectro_duration": 5,
        "data_normalization": "instrument",
        "HPfilter_min_freq": 0,
        "sensitivity_dB": -164,
        "peak_voltage": 2.5,
        "spectro_normalization": "density",
        "gain_dB": 14.7,
        "zscore_duration": "original",
    }

    return input_dataset, analysis_params


@pytest.fixture
def input_reshape(input_dir: Path):
    """Creates 10 audio files in a temporary directory and the corresponding timestamp.csv.

    The files are all copies of the file created by the input_dir fixture.

    Returns
    -------
        input_dir: `Path`
            The path to the input directory.

    """
    for i in range(9):
        wav_file = input_dir.joinpath(f"test{i}.wav")
        shutil.copyfile(input_dir.joinpath("test.wav"), wav_file)

    return input_dir
