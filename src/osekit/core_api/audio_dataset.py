"""AudioDataset is a collection of AudioData objects.

AudioDataset is a collection of AudioData, with methods
that simplify repeated operations on the audio data.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from tqdm import tqdm

from osekit.core_api.audio_data import AudioData
from osekit.core_api.audio_file import AudioFile
from osekit.core_api.base_dataset import BaseDataset
from osekit.core_api.json_serializer import deserialize_json

if TYPE_CHECKING:
    import pytz
    from pandas import Timedelta, Timestamp

    from osekit.core_api.instrument import Instrument


class AudioDataset(BaseDataset[AudioData, AudioFile]):
    """AudioDataset is a collection of AudioData objects.

    AudioDataset is a collection of AudioData, with methods
    that simplify repeated operations on the audio data.

    """

    def __init__(
        self,
        data: list[AudioData],
        name: str | None = None,
        suffix: str = "",
        folder: Path | None = None,
        instrument: Instrument | None = None,
    ) -> None:
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
        super().__init__(data=data, name=name, suffix=suffix, folder=folder)

        if instrument is not None:
            self.instrument = instrument
        else:
            self.instrument = next(
                (d.instrument for d in data if d.instrument is not None),
                None,
            )

    @property
    def sample_rate(self) -> set[float] | float:
        """Return the most frequent sample rate among those of this dataset data."""
        sample_rates = [data.sample_rate for data in self.data]
        return max(set(sample_rates), key=sample_rates.count)

    @sample_rate.setter
    def sample_rate(self, sample_rate: float) -> None:
        for data in self.data:
            data.sample_rate = sample_rate

    @property
    def instrument(self) -> Instrument | None:
        """Instrument that can be used to get acoustic pressure from wav audio data."""
        return self._instrument

    @instrument.setter
    def instrument(self, instrument: Instrument | None) -> None:
        self._instrument = instrument
        for data in self.data:
            data.instrument = instrument

    def write(
        self,
        folder: Path,
        subtype: str | None = None,
        link: bool = False,  # noqa: FBT001, FBT002,
        first: int = 0,
        last: int | None = None,
    ) -> None:
        """Write all data objects in the specified folder.

        Parameters
        ----------
        folder: Path
            Folder in which to write the data.
        subtype: str | None
            Subtype as provided by the soundfile module.
            Defaulted as the default 16-bit PCM for WAV audio files.
        link: bool
            If True, each AudioData will be bound to the corresponding written file.
            Their items will be replaced with a single item, which will match the whole
            new AudioFile.
        first: int
            Index of the first AudioData object to write.
        last: int | None
            Index after the last AudioData object to write.


        """
        last = len(self.data) if last is None else last
        for data in tqdm(
            self.data[first:last], disable=os.environ.get("DISABLE_TQDM", "")
        ):
            data.write(folder=folder, subtype=subtype, link=link)

    @classmethod
    def from_dict(cls, dictionary: dict) -> AudioDataset:
        """Deserialize an AudioDataset from a dictionary.

        Parameters
        ----------
        dictionary: dict
            The serialized dictionary representing the AudioDataset.

        Returns
        -------
        AudioDataset
            The deserialized AudioDataset.

        """
        return cls(
            [AudioData.from_dict(d) for d in dictionary["data"].values()],
            name=dictionary["name"],
            suffix=dictionary["suffix"],
            folder=Path(dictionary["folder"]),
        )

    @classmethod
    def from_folder(  # noqa: PLR0913
        cls,
        folder: Path,
        strptime_format: str,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        timezone: str | pytz.timezone | None = None,
        bound: Literal["files", "timedelta"] = "timedelta",
        data_duration: Timedelta | None = None,
        name: str | None = None,
        instrument: Instrument | None = None,
        **kwargs: any,
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
        timezone: str | pytz.timezone | None
            The timezone in which the file should be localized.
            If None, the file begin/end will be tz-naive.
            If different from a timezone parsed from the filename, the timestamps'
            timezone will be converted from the parsed timezone
            to the specified timezone.
        bound: Literal["files", "timedelta"]
            Bound between the original files and the dataset data.
            "files": one data will be created for each file.
            "timedelta": data objects of duration equal to data_duration will
            be created.
        data_duration: Timedelta | None
            Duration of the audio data objects.
            If bound is set to "files", this parameter has no effect.
            If provided, audio data will be evenly distributed between begin and end.
            Else, one data object will cover the whole time period.
        name: str|None
            Name of the dataset.
        instrument: Instrument | None
            Instrument that might be used to obtain acoustic pressure from
            the wav audio data.
        kwargs: any
            Keyword arguments passed to the BaseDataset.from_folder classmethod.

        Returns
        -------
        Audiodataset:
            The audio dataset.

        """
        kwargs.update(
            {"file_class": AudioFile, "supported_file_extensions": [".wav", ".flac"]},
        )
        base_dataset = BaseDataset.from_folder(
            folder=folder,
            strptime_format=strptime_format,
            begin=begin,
            end=end,
            timezone=timezone,
            bound=bound,
            data_duration=data_duration,
            **kwargs,
        )
        return cls.from_base_dataset(
            base_dataset=base_dataset,
            name=name,
            instrument=instrument,
        )

    @classmethod
    def from_files(  # noqa: PLR0913
        cls,
        files: list[AudioFile],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        bound: Literal["files", "timedelta"] = "timedelta",
        data_duration: Timedelta | None = None,
        name: str | None = None,
        instrument: Instrument | None = None,
    ) -> AudioDataset:
        """Return an AudioDataset object from a list of AudioFiles.

        Parameters
        ----------
        files: list[AudioFile]
            The list of files contained in the Dataset.
        begin: Timestamp | None
            Begin of the first data object.
            Defaulted to the begin of the first file.
        end: Timestamp | None
            End of the last data object.
            Defaulted to the end of the last file.
        bound: Literal["files", "timedelta"]
            Bound between the original files and the dataset data.
            "files": one data will be created for each file.
            "timedelta": data objects of duration equal to data_duration will
            be created.
        data_duration: Timedelta | None
            Duration of the data objects.
            If bound is set to "files", this parameter has no effect.
            If provided, data will be evenly distributed between begin and end.
            Else, one data object will cover the whole time period.
        name: str|None
            Name of the dataset.
        instrument: Instrument | None
            Instrument that might be used to obtain acoustic pressure from
            the wav audio data.

        Returns
        -------
        BaseDataset[TItem, TFile]:
        The DataBase object.

        """
        base = BaseDataset.from_files(
            files=files,
            begin=begin,
            end=end,
            bound=bound,
            data_duration=data_duration,
        )
        return cls.from_base_dataset(base, name=name, instrument=instrument)

    @classmethod
    def from_base_dataset(
        cls,
        base_dataset: BaseDataset,
        sample_rate: float | None = None,
        name: str | None = None,
        instrument: Instrument | None = None,
    ) -> AudioDataset:
        """Return an AudioDataset object from a BaseDataset object."""
        return cls(
            [AudioData.from_base_data(data, sample_rate) for data in base_dataset.data],
            name=name,
            instrument=instrument,
        )

    @classmethod
    def from_json(cls, file: Path) -> AudioDataset:
        """Deserialize an AudioDataset from a JSON file.

        Parameters
        ----------
        file: Path
            Path to the serialized JSON file representing the AudioDataset.

        Returns
        -------
        AudioDataset
            The deserialized AudioDataset.

        """
        # I have to redefine this method (without overriding it)
        # for the type hint to be correct.
        # It seems to be due to BaseData being a Generic class, following which
        # AudioData.from_json() is supposed to return a BaseData
        # without this duplicate definition...
        # I might look back at all this in the future
        return cls.from_dict(deserialize_json(file))
