from __future__ import annotations

import logging
import os
import sys
import typing
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
import soundfile as sf

from osekit import config
from osekit.audio_backend.soundfile_backend import SoundFileBackend
from osekit.config import (
    TIMESTAMP_FORMAT_EXPORTED_FILES_LOCALIZED,
    TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
    TIMESTAMP_FORMATS_EXPORTED_FILES,
)
from osekit.core.audio_file import AudioFile
from osekit.public import export_transform
from osekit.public.project import Project
from osekit.public.transform import OutputType
from osekit.utils.audio import generate_sample_audio

if typing.TYPE_CHECKING:
    from collections.abc import Generator


def _generate_audio_files(
    tmp_path: Path,
    params: dict,
) -> list[AudioFile]:
    nb_files = params.get("nb_files", 1)
    if nb_files == 0:
        return []

    sample_rate = params.get("sample_rate", 48_000)
    duration = params.get("duration", 1.0)
    date_begin = params.get(
        "date_begin",
        pd.Timestamp("2000-01-01 00:00:00"),
    )
    inter_file_duration = params.get("inter_file_duration", 0)
    series_type = params.get("series_type", "repeat")
    sine_frequency = params.get("sine_frequency", 1000.0)
    magnitude = params.get("magnitude", 1.0)
    audio_format = params.get("format", "wav")

    nb_samples = round(duration * sample_rate)
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
        while (file := tmp_path / f"audio_{time_str}_{idx}.{audio_format}").exists():
            idx += 1
        files.append(file)
        sf.write(
            file=file,
            data=data[index],
            samplerate=sample_rate,
            subtype="DOUBLE" if audio_format.lower() == "wav" else "PCM_24",
        )
    return [AudioFile(path=f, strptime_format=datetime_format) for f in files]


@pytest.fixture(scope="module")
def sample_project(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.fixture.SubRequest,
) -> tuple[Project, pytest.fixtures.Subrequest]:
    tmp_path = tmp_path_factory.mktemp("sample_project")
    params = request.param if hasattr(request, "param") else {}

    _generate_audio_files(tmp_path=tmp_path, params=params)

    instrument = params.get("instrument", None)

    project = Project(
        folder=tmp_path,
        strptime_format=TIMESTAMP_FORMATS_EXPORTED_FILES,
        instrument=instrument,
    )
    project.build()
    return project, request


@pytest.fixture
def audio_files(
    tmp_path: Path,
    request: pytest.fixtures.Subrequest,
) -> tuple[list[AudioFile], pytest.fixtures.Subrequest]:
    params = request.param if hasattr(request, "param") else {}
    return _generate_audio_files(tmp_path=tmp_path, params=params), request


@pytest.fixture(autouse=True)
def patch_filehandlers(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    if "allow_log_write_to_file" in request.keywords:
        return

    def disabled_filewrite(self: typing.Any, record: typing.Any) -> None:
        """Prevent the logger from actually writing files."""

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


@pytest.fixture(autouse=True)
def restore_config() -> typing.Generator:
    resample_quality_settings = {**config.resample_quality_settings}
    multiprocessing = {**config.multiprocessing}
    yield
    for key, value in resample_quality_settings.items():
        config.resample_quality_settings[key] = value
    for key, value in multiprocessing.items():
        config.multiprocessing[key] = value


@pytest.fixture(autouse=True)
def reset_logging() -> typing.Generator[None, typing.Any, None]:
    """Reset the python logging module."""
    root = logging.getLogger()
    handlers_before = list(root.handlers)
    level_before = root.level

    # Snapshot of loggers before the test
    loggers_before = {
        name: list(logger.handlers)
        for name, logger in logging.Logger.manager.loggerDict.items()
        if isinstance(logger, logging.Logger)
    }

    yield
    root.handlers = handlers_before
    root.level = level_before

    # Cleaning of the handlers of each logger:
    for name, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            logger.handlers = loggers_before.get(name, [])

    logging.Logger.manager.loggerDict = {
        key: value
        for key, value in logging.Logger.manager.loggerDict.items()
        if key in loggers_before
    }


@pytest.fixture(autouse=True)
def check_logger() -> Generator[None, Any, None]:
    h1 = list(logging.root.handlers)
    yield
    h2 = list(logging.root.handlers)
    if h1 != h2:  # pragma: no cover
        msg = "This test changed the root logger handlers."
        raise ValueError(msg)


@pytest.fixture
def dummy_export_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create dummy files in the transform output(s) dataset(s) folders(s).

    This avoids actually computing the outputs of transforms in tests
    that don't need to."""

    def dummy_export(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
        output_type = kwargs["output_type"]
        ads = kwargs.get("ads")
        sds = kwargs.get("sds")
        spectrum_folder_path = kwargs.get("spectrum_folder_path")
        spectrogram_folder_path = kwargs.get("spectrogram_folder_path")
        welch_folder_path = kwargs.get("welch_folder_path")

        exports = []

        if ads:
            exports.append((OutputType.AUDIO, ads.folder / "dummy_audio.wav"))

        if sds:
            exports += [
                (
                    OutputType.SPECTROGRAM,
                    sds.folder / spectrogram_folder_path / "dummy_spectrogram.png",
                ),
                (
                    OutputType.SPECTRUM,
                    sds.folder / spectrum_folder_path / "dummy_spectrum.npz",
                ),
                (OutputType.WELCH, sds.folder / welch_folder_path / "dummy_welch.npz"),
            ]

        for output, file in exports:
            if output in output_type:
                Path.mkdir(file.parent, parents=True, exist_ok=True)
                file.touch()

    monkeypatch.setattr(export_transform, "write_transform_output", dummy_export)
