"""Main class of the Public API.

The Dataset correspond to a collection of audio,
spectro and auxilary core_api datasets.
It has additionnal metadata that can be exported, e.g. to APLOSE.

"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, TypeVar

from OSmOSE.core_api.audio_dataset import AudioDataset
from OSmOSE.core_api.spectro_dataset import SpectroDataset
from OSmOSE.utils.path_utils import move_tree

if TYPE_CHECKING:
    from pathlib import Path


class Dataset:
    """Main class of the Public API.

    The Dataset correspond to a collection of audio,
    spectro and auxilary core_api datasets.
    It has additionnal metadata that can be exported, e.g. to APLOSE.

    """

    def __init__(
        self,
        folder: Path,
        strptime_format: str,
        gps_coordinates: str | list | tuple = (0, 0),
        depth: str | int = 0,
        timezone: str | None = None,
    ) -> None:
        """Initialize a Dataset."""
        self.folder = folder
        self.strptime_format = strptime_format
        self.gps_coordinates = gps_coordinates
        self.depth = depth
        self.timezone = timezone
        self.datasets = {}

    def build(self) -> None:
        """Build the Dataset.

        Building a dataset moves the original audio files to a specific folder
        and creates metadata csv used by APLOSE.

        """
        ads = AudioDataset.from_folder(
            self.folder,
            strptime_format=self.strptime_format,
            bound="files",
        )
        self.datasets["original"] = ads
        move_tree(
            self.folder,
            self.folder / "other",
            {file.path for file in self.datasets["original"].files},
        )
        self._sort_dataset(self.datasets["original"])

    def reset(self) -> None:
        """Reset the Dataset.

        Resetting a dataset will move back the original audio files and the content of
        the "other" folder to the root folder.
        WARNING: all other files and folders will be deleted.
        """
        files_to_remove = list(self.folder.iterdir())
        self.datasets["original"].folder = self.folder

        if self.folder / "other" in files_to_remove:
            move_tree(self.folder / "other", self.folder)

        for file in files_to_remove:
            shutil.rmtree(file)

    def _sort_dataset(self, dataset: type[DatasetChild]) -> None:
        if type(dataset) is AudioDataset:
            self._sort_audio_dataset(dataset)
            return
        if type(dataset) is SpectroDataset:
            self._sort_spectro_dataset(dataset)
            return

    def _sort_audio_dataset(self, dataset: AudioDataset) -> None:
        data_duration = dataset.data_duration
        sample_rate = dataset.sample_rate
        data_duration, sample_rate = (
            parameter if type(parameter) is not set else next(iter(parameter))
            for parameter in (data_duration, sample_rate)
        )
        dataset.folder = (
            self.folder
            / "data"
            / "audio"
            / f"{round(data_duration.total_seconds())}_{round(sample_rate)}"
        )

    def _sort_spectro_dataset(self, dataset: SpectroDataset) -> None:
        pass


DatasetChild = TypeVar("DatasetChild", bound=Dataset)
