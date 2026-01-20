from __future__ import annotations

import logging
import os
import sys
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from pandas import Timestamp

from osekit import config
from osekit.audio_backend.soundfile_backend import SoundFileBackend
from osekit.config import (
    TIMESTAMP_FORMAT_EXPORTED_FILES_LOCALIZED,
    TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
)
from osekit.core_api.audio_data import AudioData
from osekit.core_api.audio_file import AudioFile
from osekit.utils.audio_utils import generate_sample_audio


@pytest.fixture
def audio_files(
    tmp_path: Path,
    request: pytest.fixtures.Subrequest,
) -> tuple[list[AudioFile], pytest.fixtures.Subrequest]:
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
    else:
        sample_rate = 48_000
        duration = 1.0
        date_begin = pd.Timestamp("2000-01-01 00:00:00")
        inter_file_duration = 0
        series_type = "repeat"
        sine_frequency = 1000.0
        magnitude = 1.0
        format = "wav"

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

    datetime_format = (
        TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED
        if date_begin.tzinfo is None
        else TIMESTAMP_FORMAT_EXPORTED_FILES_LOCALIZED
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
    output = [AudioFile(path=f, strptime_format=datetime_format) for f in files]
    return output, request


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


@pytest.fixture(autouse=True)
def disable_tqdm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISABLE_TQDM", "1")


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
    open_func = SoundFileBackend._open

    def mock_open(self: SoundFileBackend, path: Path) -> None:
        opened_files.append(path)
        open_func(self, path)

    monkeypatch.setattr(SoundFileBackend, "_open", mock_open)
    return opened_files


@pytest.fixture
def patch_audio_data(monkeypatch: pytest.MonkeyPatch) -> None:
    original_init = AudioData.__init__
    original_get_raw_value = AudioData.get_raw_value
    original_length = AudioData.length

    def mocked_init(
        self: AudioData,
        *args: list,
        mocked_value: list[float] | np.ndarray | None = None,
        **kwargs: dict,
    ) -> None:
        defaults = {
            "begin": Timestamp("2000-01-01 00:00:00"),
            "end": Timestamp("2000-01-01 00:00:01"),
            "sample_rate": 48000,
        }
        for key, value in defaults.items():
            if key not in kwargs:
                kwargs.update(**{key: value})

        original_init(self, *args, **kwargs)
        if mocked_value is not None:
            self.mocked_value = mocked_value
            if type(mocked_value) is list or len(mocked_value.shape) == 1:
                self.mocked_value = np.array(self.mocked_value).reshape(
                    len(mocked_value),
                    1,
                )

    def mocked_length(self: AudioData) -> int:
        if hasattr(self, "mocked_value"):
            return len(self.mocked_value)
        return original_length.fget(self)

    def mocked_get_raw_value(self: AudioData) -> np.ndarray:
        if hasattr(self, "mocked_value"):
            return self.mocked_value
        return original_get_raw_value(self)

    monkeypatch.setattr(AudioData, "__init__", mocked_init)
    monkeypatch.setattr(AudioData, "length", property(mocked_length))
    monkeypatch.setattr(
        AudioData,
        "get_raw_value",
        mocked_get_raw_value,
    )


@pytest.fixture(autouse=True)
def restore_config() -> Generator:
    resample_quality_settings = {**config.resample_quality_settings}
    multiprocessing = {**config.multiprocessing}
    yield
    for key, value in resample_quality_settings.items():
        config.resample_quality_settings[key] = value
    for key, value in multiprocessing.items():
        config.multiprocessing[key] = value
