from __future__ import annotations

from pathlib import Path

from pandas import Timedelta, Timestamp

from OSmOSE.data.audio_data import AudioData
from OSmOSE.data.audio_file import AudioFile
from OSmOSE.data.dataset_base import DatasetBase


class AudioDataset(DatasetBase[AudioData, AudioFile]):
    def __init__(self, data: list[AudioData]):
        super().__init__(data)

    @classmethod
    def from_folder(
        cls,
        folder: Path,
        strptime_format: str,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        data_duration: Timedelta | None = None,
    ) -> AudioDataset:
        files = [
            AudioFile(file, strptime_format=strptime_format)
            for file in folder.glob("*.wav")
        ]
        base_dataset = DatasetBase.from_files(files, begin, end, data_duration)
        return cls.from_base_dataset(base_dataset)

    @classmethod
    def from_base_dataset(cls, base_dataset: DatasetBase) -> AudioDataset:
        return cls([AudioData.from_base_data(data) for data in base_dataset.data])
