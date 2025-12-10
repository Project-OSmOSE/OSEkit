import argparse
import logging
import os
import shlex
from pathlib import Path

import pytest

from osekit import config
from osekit.core_api.audio_dataset import AudioDataset
from osekit.core_api.spectro_dataset import SpectroDataset
from osekit.public_api import export_analysis
from osekit.public_api.export_analysis import create_parser
from osekit.utils.job import Job


def test_parser_factory() -> None:
    parser = create_parser()

    assert parser.description

    expected_args = {
        "--analysis",
        "--ads-json",
        "--sds-json",
        "--subtype",
        "--matrix-folder-path",
        "--spectrogram-folder-path",
        "--welch-folder-path",
        "--first",
        "--last",
        "--downsampling-quality",
        "--upsampling-quality",
        "--umask",
        "--tqdm-disable",
        "--multiprocessing",
        "--use-logging-setup",
        "--nb-processes",
        "--dataset-json-path",
    }

    actual_args = {
        action.option_strings[0] for action in parser._actions if action.option_strings
    }
    assert expected_args.issubset(actual_args)


def test_argument_defaults() -> None:
    parser = create_parser()
    args = parser.parse_args(
        [
            "--analysis",
            "1",
        ],
    )

    assert args.analysis == 1
    assert args.ads_json is None
    assert args.sds_json is None
    assert args.subtype is None
    assert args.matrix_folder_path is None
    assert args.spectrogram_folder_path is None
    assert args.welch_folder_path is None
    assert args.first == 0
    assert args.last is None
    assert args.downsampling_quality is None
    assert args.upsampling_quality is None
    assert args.umask == 0o002  # noqa: PLR2004
    assert args.tqdm_disable
    assert not args.multiprocessing
    assert not args.use_logging_setup
    assert args.nb_processes is None
    assert args.dataset_json_path is None


@pytest.fixture
def script_arguments() -> dict:
    return {
        "analysis": 2,
        "ads-json": r"path/to/ads.json",
        "sds-json": r"path/to/ads.json",
        "subtype": "FLOAT",
        "matrix-folder-path": r"out/matrix",
        "spectrogram-folder-path": r"out/spectro",
        "welch-folder-path": r"out/welch",
        "first": 10,
        "last": 12,
        "downsampling-quality": "HQ",
        "upsampling-quality": "VHQ",
        "umask": 0o022,
        "tqdm-disable": False,
        "multiprocessing": True,
        "nb-processes": "3",  # String because it might be "None"
        "use-logging-setup": True,
        "dataset-json-path": r"path/to/dataset.json",
    }


def test_specified_arguments(script_arguments: dict) -> None:
    parser = create_parser()

    parsed_str = Job(Path(), script_arguments)._build_arg_string()

    args = parser.parse_args(shlex.split(parsed_str))

    assert args.analysis == script_arguments["analysis"]
    assert args.ads_json == script_arguments["ads-json"]
    assert args.sds_json == script_arguments["sds-json"]
    assert args.subtype == script_arguments["subtype"]
    assert args.matrix_folder_path == script_arguments["matrix-folder-path"]
    assert args.spectrogram_folder_path == script_arguments["spectrogram-folder-path"]
    assert args.welch_folder_path == script_arguments["welch-folder-path"]
    assert args.first == script_arguments["first"]
    assert args.last == script_arguments["last"]
    assert args.downsampling_quality == script_arguments["downsampling-quality"]
    assert args.upsampling_quality == script_arguments["upsampling-quality"]
    assert args.umask == script_arguments["umask"]
    assert args.tqdm_disable == script_arguments["tqdm-disable"]
    assert args.multiprocessing == script_arguments["multiprocessing"]
    assert args.use_logging_setup == script_arguments["use-logging-setup"]
    assert args.nb_processes == script_arguments["nb-processes"]
    assert args.dataset_json_path == script_arguments["dataset-json-path"]


def test_main_script(monkeypatch: pytest.MonkeyPatch, script_arguments: dict) -> None:
    class MockedArgs:
        def __init__(self, *args: list, **kwargs: dict) -> None:
            self.analysis = script_arguments["analysis"]
            self.ads_json = script_arguments["ads-json"]
            self.sds_json = script_arguments["sds-json"]
            self.subtype = script_arguments["subtype"]
            self.matrix_folder_path = script_arguments["matrix-folder-path"]
            self.spectrogram_folder_path = script_arguments["spectrogram-folder-path"]
            self.welch_folder_path = script_arguments["welch-folder-path"]
            self.first = script_arguments["first"]
            self.last = script_arguments["last"]
            self.downsampling_quality = script_arguments["downsampling-quality"]
            self.upsampling_quality = script_arguments["upsampling-quality"]
            self.umask = script_arguments["umask"]
            self.tqdm_disable = script_arguments["tqdm-disable"]
            self.multiprocessing = script_arguments["multiprocessing"]
            self.use_logging_setup = script_arguments["use-logging-setup"]
            self.nb_processes = script_arguments["nb-processes"]
            self.dataset_json_path = "none"

    def return_mocked_attr(*args: list, **kwargs: dict) -> MockedArgs:
        return MockedArgs(*args, **kwargs)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", return_mocked_attr)

    parser = create_parser()

    args = parser.parse_args()

    calls = {}

    def mock_ads_json(path: Path) -> Path:
        calls["ads_json"] = path
        return path

    monkeypatch.setattr(AudioDataset, "from_json", mock_ads_json)

    def mock_sds_json(path: Path) -> Path:
        calls["sds_json"] = path
        return path

    monkeypatch.setattr(SpectroDataset, "from_json", mock_sds_json)

    def mock_write_analysis(*args: list, **kwargs: dict) -> None:
        for k, v in kwargs.items():
            calls[k] = v  # noqa: PERF403

    monkeypatch.setattr(export_analysis, "write_analysis", mock_write_analysis)

    export_analysis.main()

    assert (
        os.environ["DISABLE_TQDM"].lower() in ("true", "1", "t")
    ) == args.tqdm_disable
    assert config.multiprocessing["is_active"]
    assert config.multiprocessing["nb_processes"] == 3  # noqa: PLR2004
    assert (
        config.resample_quality_settings["downsample"]
        == script_arguments["downsampling-quality"]
    )
    assert (
        config.resample_quality_settings["upsample"]
        == script_arguments["upsampling-quality"]
    )
    assert calls["ads_json"] == Path(script_arguments["ads-json"])
    assert calls["sds_json"] == Path(script_arguments["sds-json"])

    # write_analysis
    assert calls["analysis_type"].value == script_arguments["analysis"]
    assert calls["ads"] == Path(script_arguments["ads-json"])
    assert calls["sds"] == Path(script_arguments["sds-json"])
    assert calls["subtype"] == script_arguments["subtype"]
    assert calls["matrix_folder_path"] == Path(script_arguments["matrix-folder-path"])
    assert calls["spectrogram_folder_path"] == Path(
        script_arguments["spectrogram-folder-path"],
    )
    assert calls["welch_folder_path"] == Path(script_arguments["welch-folder-path"])
    assert calls["first"] == script_arguments["first"]
    assert calls["last"] == script_arguments["last"]
    assert calls["link"] is True
    assert calls["logger"] == logging.getLogger()
