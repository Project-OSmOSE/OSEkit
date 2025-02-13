"""AudioDataset is a collection of AudioData objects.

AudioDataset is a collection of AudioData, with methods
that simplify repeated operations on the audio data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from soundfile import LibsndfileError

from OSmOSE.config import global_logging_context as glc
from OSmOSE.core_api.audio_data import AudioData
from OSmOSE.core_api.audio_file import AudioFile
from OSmOSE.core_api.base_dataset import BaseDataset

if TYPE_CHECKING:
    from pathlib import Path

    from pandas import Timedelta, Timestamp


class AudioDataset(BaseDataset[AudioData, AudioFile]):
    """AudioDataset is a collection of AudioData objects.

    AudioDataset is a collection of AudioData, with methods
    that simplify repeated operations on the audio data.

    """

    def __init__(self, data: list[AudioData]) -> None:
        """Initialize an AudioDataset."""
        if (
            len(
                sample_rates := {
                    data.sample_rate for data in data if data.sample_rate is not None
                },
            )
            != 1
        ):
            logging.warning("Audio dataset contains different sample rates.")
        else:
            for empty_data in (data for data in data if data.sample_rate is None):
                empty_data.sample_rate = min(sample_rates)
        super().__init__(data)

    @property
    def sample_rate(self) -> set[float] | float:
        """Return the sample rate of the audio data."""
        sample_rates = {data.sample_rate for data in self.data}
        return sample_rates if len(list(sample_rates)) > 1 else next(iter(sample_rates))

    @sample_rate.setter
    def sample_rate(self, sample_rate: float) -> None:
        for data in self.data:
            data.sample_rate = sample_rate

    def write(self, folder: Path, subtype: str | None = None) -> None:
        """Write all data objects in the specified folder.

        Parameters
        ----------
        folder: Path
            Folder in which to write the data.
        subtype: str | None
            Subtype as provided by the soundfile module.
            Defaulted as the default 16-bit PCM for WAV audio files.


        """
        for data in self.data:
            data.write(folder, subtype)

    @classmethod
    def from_folder(
        cls,
        folder: Path,
        strptime_format: str,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        data_duration: Timedelta | None = None,
    ) -> AudioDataset:
        """Return an AudioDataset from a folder containing the audio files.

        Parameters
        ----------
        folder: Path
            The folder containing the audio files.
        strptime_format: str
            The strptime format of the timestamps in the audio file names.
        begin: Timestamp | None
            The begin of the audio dataset.
            Defaulted to the begin of the first file.
        end: Timestamp | None
            The end of the audio dataset.
            Defaulted to the end of the last file.
        data_duration: Timedelta | None
            Duration of the audio data objects.
            If provided, audio data will be evenly distributed between begin and end.
            Else, one data object will cover the whole time period.

        Returns
        -------
        Audiodataset:
            The audio dataset.

        """
        audio_files = []
        rejected_files = []
        for file in folder.iterdir():
            if file.suffix.lower() not in (".wav", ".flac"):
                continue
            try:
                af = AudioFile(file, strptime_format=strptime_format)
                audio_files.append(af)
            except (ValueError, LibsndfileError):
                rejected_files.append(file)

        if rejected_files:
            rejected_files = "\n\t".join(f.name for f in rejected_files)
            glc.logger.warn(
                f"The following files couldn't be parsed:\n\t{rejected_files}",
            )

        if not audio_files:
            raise FileNotFoundError(f"No valid audio file found in {folder}.")

        base_dataset = BaseDataset.from_files(audio_files, begin, end, data_duration)
        return cls.from_base_dataset(base_dataset)

    @classmethod
    def from_base_dataset(cls, base_dataset: BaseDataset) -> AudioDataset:
        """Return an AudioDataset object from a BaseDataset object."""
        return cls([AudioData.from_base_data(data) for data in base_dataset.data])
