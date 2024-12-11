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
        begin: Timestamp,
        end: Timestamp,
        data_duration: Timedelta,
        strptime_format: str,
    ) -> AudioDataset:
        files = [
            AudioFile(file, strptime_format=strptime_format)
            for file in folder.glob("*.wav")
        ]
        data_base = DatasetBase.data_from_files(files, begin, end, data_duration)
        audio_data = [
            AudioData.from_files(files, data.begin, data.end) for data in data_base
        ]
        return cls(audio_data)
