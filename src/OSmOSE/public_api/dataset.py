from __future__ import annotations

import shutil
from pathlib import Path
from typing import TypeVar

from OSmOSE.core_api.audio_dataset import AudioDataset
from OSmOSE.core_api.spectro_dataset import SpectroDataset


class Dataset:
    def __init__(
        self,
        folder: Path,
        strptime_format: str,
        gps_coordinates: str | list | tuple = (0, 0),
        depth: str | int = 0,
        timezone: str | None = None,
    ):
        self.folder = folder
        self.strptime_format = strptime_format
        self.gps_coordinates = gps_coordinates
        self.depth = depth
        self.timezone = timezone
        self.dataset = {}

    def build(self):
        ads = AudioDataset.from_folder(
            self.folder,
            strptime_format=self.strptime_format,
        )
        self.dataset["original"] = ads
        self._sort_data(self.dataset["original"])

    def restore(self):
        files_to_remove = list(self.folder.iterdir())
        self.dataset["original"].move(self.folder)
        for file in files_to_remove:
            shutil.rmtree(file)

    def _sort_data(self, dataset: type[DatasetChild]):
        if type(dataset) is AudioDataset:
            self._sort_audio_data(dataset)
            return
        if type(dataset) is SpectroDataset:
            self._sort_spectro_data(dataset)
            return

    def _sort_audio_data(self, data: AudioDataset):
        data_duration = data.data_duration
        sample_rate = data.sample_rate
        data_duration, sample_rate = (parameter if type(parameter) is not set else next(iter(parameter)) for parameter in (data_duration, sample_rate))
        data.move(
            self.folder
            / "data"
            / "audio"
            / f"{round(data_duration.total_seconds())}_{round(sample_rate)}",
        )



DatasetChild = TypeVar("DatasetChild", bound=Dataset)
