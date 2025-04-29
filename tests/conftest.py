from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from scipy.signal import chirp

from OSmOSE.config import OSMOSE_PATH, TIMESTAMP_FORMAT_TEST_FILES
from OSmOSE.core_api import AudioFileManager
from OSmOSE.core_api.base_dataset import BaseDataset
from OSmOSE.core_api.base_file import BaseFile
from OSmOSE.utils.audio_utils import generate_sample_audio


@pytest.fixture
def audio_files(
    tmp_path: Path,
    request: pytest.fixtures.Subrequest,
) -> tuple[list[Path], pytest.fixtures.Subrequest]:
    nb_files = request.param.get("nb_files", 1) if hasattr(request, "param") else 1

    if nb_files == 0:
        return [], request

    if hasattr(request, "param"):
        sample_rate = request.param.get("sample_rate", 48_000)
        duration = request.param.get("duration", 1.0)
        date_begin = request.param.get(
            "date_begin",
            pd.Timestamp("2000-01-01 00:00:00"),
        )
        inter_file_duration = request.param.get("inter_file_duration", 0)
        series_type = request.param.get("series_type", "repeat")
        sine_frequency = request.param.get("sine_frequency", 1000.0)
        magnitude = request.param.get("magnitude", 1.0)
        format = request.param.get("format", "wav")
        datetime_format = request.param.get(
            "datetime_format",
            TIMESTAMP_FORMAT_TEST_FILES,
        )
    else:
        sample_rate = 48_000
        duration = 1.0
        date_begin = pd.Timestamp("2000-01-01 00:00:00")
        inter_file_duration = 0
        series_type = "repeat"
        sine_frequency = 1000.0
        magnitude = 1.0
        format = "wav"
        datetime_format = TIMESTAMP_FORMAT_TEST_FILES

    nb_samples = int(round(duration * sample_rate))
    data = generate_sample_audio(
        nb_files=nb_files,
        nb_samples=nb_samples,
        series_type=series_type,
        sine_frequency=sine_frequency,
        max_value=magnitude,
        duration=duration,
    )
    files = []
    file_begin_timestamps = (
        list(
            pd.date_range(
                date_begin,
                periods=nb_files,
                freq=pd.Timedelta(seconds=duration + inter_file_duration),
            ),
        )
        if duration + inter_file_duration != 0
        else [date_begin] * nb_files
    )
    for index, begin_time in enumerate(file_begin_timestamps):
        time_str = begin_time.strftime(format=datetime_format)
        idx = 0
        while (file := tmp_path / f"audio_{time_str}_{idx}.{format}").exists():
            idx += 1
        files.append(file)
        kwargs = {
            "file": file,
            "data": data[index],
            "samplerate": sample_rate,
            "subtype": "DOUBLE" if format.lower() == "wav" else "PCM_24",
        }
        sf.write(**kwargs)
    return files, request


@pytest.fixture(autouse=True)
def patch_filehandlers(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    if "allow_log_write_to_file" in request.keywords:
        return

    def disabled_filewrite(self: any, record: any) -> None:
        pass

    monkeypatch.setattr(logging.FileHandler, "emit", disabled_filewrite)


@pytest.fixture
def patch_grp_module(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock the grp module.
    The grp.getgrnam() mocked function returns a mocked group.
    The mocked_group.gr_gid attribute returns the index of mocked_group in grp.groups.
    """

    groups = ["ensta", "gosmose", "other"]
    active_group = {"gid": 0}

    mocked_grp_module = sys.modules["grp"] = MagicMock()
    mocked_group = MagicMock()

    def mock_group_with_gid(name: str) -> MagicMock:
        if name not in groups:
            message = f"getgrnam(): name not found: '{name}'"
            raise KeyError(message)
        mocked_group.gr_gid = groups.index(name)
        return mocked_group

    mocked_grp_module.getgrnam = MagicMock(side_effect=mock_group_with_gid)

    def mock_chown(path: Path, uid: int, gid: int) -> None:
        sys.modules["grp"].active_group["gid"] = gid

    monkeypatch.setattr(os, "chown", mock_chown, raising=False)
    monkeypatch.setattr(Path, "group", lambda path: groups[active_group["gid"]])

    return mocked_grp_module


@pytest.fixture
def patch_afm_open(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Mock the AudioFileManager._open method in order to track the file openings."""

    opened_files = []
    open_func = AudioFileManager._open

    def mock_open(self: AudioFileManager, path: Path) -> None:
        opened_files.append(path)
        open_func(self, path)

    monkeypatch.setattr(AudioFileManager, "_open", mock_open)
    return opened_files


@pytest.fixture
def base_dataset(tmp_path: Path) -> BaseDataset:
    files = [tmp_path / f"file_{i}.txt" for i in range(5)]
    for file in files:
        file.touch()
    timestamps = pd.date_range(
        start=pd.Timestamp("2000-01-01 00:00:00"),
        freq="1s",
        periods=5,
    )

    bfs = [
        BaseFile(path=file, begin=timestamp, end=timestamp + pd.Timedelta(seconds=1))
        for file, timestamp in zip(files, timestamps)
    ]
    return BaseDataset.from_files(files=bfs, bound="files")


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
