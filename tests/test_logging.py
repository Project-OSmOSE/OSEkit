import importlib
import logging
import os
from pathlib import Path

import pytest
import yaml

from osekit import setup_logging
from osekit.config import TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED
from osekit.logging_context import LoggingContext
from osekit.public_api.dataset import Dataset


@pytest.fixture
def setup_module_logging() -> None:
    """Set up the osekit logging."""
    setup_logging()


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset the python logging module."""
    yield
    importlib.reload(logging)


@pytest.fixture
def temp_user_logging_config(tmp_path: Path) -> Path:
    """Writes a yaml logging config file in tmp_path, then returns its path.

    Parameters
    ----------
    tmp_path: pathlib.Path
        Path of the directory in which "logging_config.yaml" should be written.

    Returns
    -------
    pathlib.Path: path to the logging_config.yaml file.

    """
    config = {
        "version": 1,
        "loggers": {
            "test_user_logger": {
                "level": "DEBUG",
                "handlers": ["consoleHandler", "fileHandler"],
                "propagate": True,
            },
        },
        "handlers": {
            "consoleHandler": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "userFormatter",
                "stream": "ext://sys.stdout",
            },
            "fileHandler": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "userFormatter",
                "filename": f"{tmp_path}/logs.log",
                "mode": "w",
            },
        },
        "formatters": {
            "userFormatter": {"format": "%(message)s"},
        },
    }

    user_config_path = tmp_path / "logging_config.yaml"
    with open(user_config_path, "w") as f:
        yaml.dump(config, f)
    return user_config_path


@pytest.fixture
def set_user_config_env(temp_user_logging_config: Path) -> None:
    """Set the OSMOSE_USER_CONFIG environment variable to the pytest tmp_file directory.
    This will be read by osekit during import to find the simulated user logging_config.yaml
    During setup, the logging module is reloaded to clear all configs, then the osekit module is reloaded with a OSMOSE_USER_CONFIG environment key locating the mocked user config file.
    During teardown, both modules are reloaded again with the OSMOSE_USER_CONFIG key being reset, to clear the moked user config.

    Parameters
    ----------
    temp_user_logging_config: Path
        The path to the logging_config.yaml user config file.

    """
    importlib.reload(logging)

    original_config_env = os.getenv("OSMOSE_USER_CONFIG", None)
    os.environ["OSMOSE_USER_CONFIG"] = str(temp_user_logging_config.parent)
    setup_logging()
    yield
    if original_config_env:
        os.environ["OSMOSE_USER_CONFIG"] = original_config_env
    else:
        del os.environ["OSMOSE_USER_CONFIG"]
    importlib.reload(logging)


@pytest.mark.allow_log_write_to_file
def test_user_logging_config(
    set_user_config_env: pytest.fixture,
    caplog: pytest.fixture,
    tmp_path: Path,
) -> None:
    assert (
        len(logging.getLogger("test_user_logger").handlers) > 0
    )  # This is a tweaky way of checking if the test_user_logger logger has already been created
    assert len(logging.getLogger("dataset").handlers) == 0

    with caplog.at_level(logging.DEBUG, logger="test_user_logger"):
        logging.getLogger("test_user_logger").debug("User debug log")

    assert "User debug log" in caplog.text
    assert "User debug log" in open(f"{tmp_path}/logs.log").read()


def test_default_logging_config(
    setup_module_logging: pytest.fixture,
    caplog: pytest.fixture,
    tmp_path: Path,
) -> None:
    assert (
        len(logging.getLogger("dataset").handlers) > 0
    )  # This is a tweaky way of checking if the test_user_logger logger has already been created
    assert len(logging.getLogger("test_user_logger").handlers) == 0

    with caplog.at_level(logging.DEBUG):
        logging.getLogger().debug("Some debug log")

    assert "Some debug log" in caplog.text


@pytest.mark.unit
def test_logging_context(caplog: pytest.fixture) -> None:
    logging_context = LoggingContext()

    context_logger = logging.getLogger("context_logger")
    logging_context.logger = logging.getLogger("default_logger")

    with caplog.at_level(logging.DEBUG):
        logging_context.logger.debug("From default logger")

        with logging_context.set_logger(context_logger):
            logging_context.logger.debug("From context logger")

        logging_context.logger.debug("From default logger again")

    assert len(caplog.records) == 3

    assert caplog.records[0].name == "default_logger"
    assert caplog.records[0].message == "From default logger"
    assert caplog.records[1].name == "context_logger"
    assert caplog.records[1].message == "From context logger"
    assert caplog.records[2].name == "default_logger"
    assert caplog.records[2].message == "From default logger again"


def test_public_api_dataset_logger(
    audio_files: pytest.fixture, tmp_path: pytest.fixture
) -> None:
    dataset = Dataset(
        folder=tmp_path,
        strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
    )

    # Without setting the logging up, the PublicAPI should log directly to the root logger
    dataset.build()
    assert dataset.logger == logging.getLogger()
    dataset.reset()

    # With setting the logging up, the PublicAPI should log to the dataset's logger
    setup_logging()
    dataset.build()
    assert dataset.logger != logging.getLogger()
    assert dataset.logger.parent == logging.getLogger("dataset")
