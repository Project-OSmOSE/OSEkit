"""``AudioDataset`` is a collection of ``AudioData`` objects.

``AudioDataset`` is a collection of ``AudioData``, with methods
that simplify repeated operations on the audio data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Self

from osekit.core_api.audio_data import AudioData
from osekit.core_api.audio_file import AudioFile
from osekit.core_api.base_dataset import BaseDataset
from osekit.core_api.json_serializer import deserialize_json
from osekit.utils.audio_utils import Normalization
from osekit.utils.multiprocess_utils import multiprocess

if TYPE_CHECKING:
    from pathlib import Path

    import pytz
    from pandas import Timedelta, Timestamp

    from osekit.core_api.instrument import Instrument


class AudioDataset(BaseDataset[AudioData, AudioFile]):
    """``AudioDataset`` is a collection of ``AudioData`` objects.

    ``AudioDataset`` is a collection of ``AudioData``, with methods
    that simplify repeated operations on the audio data.

    """

    file_cls = AudioFile

    def __init__(
        self,
        data: list[AudioData],
        name: str | None = None,
        suffix: str = "",
        folder: Path | None = None,
        instrument: Instrument | None = None,
    ) -> None:
        """Initialize an ``AudioDataset``."""
        if (
            len(
                sample_rates := {
                    data.sample_rate for data in data if data.sample_rate is not None
                },
            )
            != 1
        ):
            msg = "Audio dataset contains different sample rates."
            logging.warning(msg)
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
    def normalization(self) -> Normalization:
        """Return the most frequent normalization among those of this dataset data."""
        normalizations = [data.normalization for data in self.data]
        return max(set(normalizations), key=normalizations.count)

    @normalization.setter
    def normalization(self, normalization: Normalization) -> None:
        for data in self.data:
            data.normalization = normalization

    @property
    def instrument(self) -> Instrument | None:
        """Instrument that can be used to get acoustic pressure from wav audio data."""
        return self._instrument

    @instrument.setter
    def instrument(self, instrument: Instrument | None) -> None:
        self._instrument = instrument
        for data in self.data:
            data.instrument = instrument

    def _write_audio(
        self,
        data: AudioData,
        folder: Path,
        subtype: str | None = None,
        link: bool = False,  # noqa: FBT001, FBT002,
    ) -> AudioData:
        """Write audio data to disk."""
        data.write(folder=folder, subtype=subtype, link=link)
        return data

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
            If ``True``, each ``AudioData`` will be bound to the corresponding written file.
            Their items will be replaced with a single item, which will match the whole
            new ``AudioFile``.
        first: int
            Index of the first ``AudioData`` object to write.
        last: int | None
            Index after the last ``AudioData`` object to write.


        """
        last = len(self.data) if last is None else last
        self.data[first:last] = multiprocess(
            func=self._write_audio,
            enumerable=self.data[first:last],
            folder=folder,
            subtype=subtype,
            link=link,
        )

    @classmethod
    def _data_from_dict(cls, dictionary: dict) -> list[AudioData]:
        """Return the list of ``AudioData`` objects from the serialized dictionary.

        Parameters
        ----------
        dictionary: dict
            Dictionary representing the serialized ``AudioDataset``.

        Returns
        -------
        list[AudioData]:
            The list of deserialized ``AudioData`` objects.

        """
        return [AudioData.from_dict(data) for data in dictionary.values()]

    @classmethod
    def from_folder(  # noqa: PLR0913
        cls,
        folder: Path,
        strptime_format: str | None,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        timezone: str | pytz.timezone | None = None,
        mode: Literal["files", "timedelta_total", "timedelta_file"] = "timedelta_total",
        overlap: float = 0.0,
        data_duration: Timedelta | None = None,
        sample_rate: float | None = None,
        name: str | None = None,
        instrument: Instrument | None = None,
        normalization: Normalization = Normalization.RAW,
        **kwargs,  # noqa: ANN003
    ) -> Self:
        """Return an ``AudioDataset`` from a folder containing the audio files.

        Parameters
        ----------
        folder: Path
            The folder containing the audio files.
        strptime_format: str | None
            The strptime format used in the filenames.
            It should use valid strftime codes (https://strftime.org/).
            If ``None``, the first audio file of the folder will start
            at ``first_file_begin``, and each following file will start
            at the end of the previous one.
        begin: Timestamp | None
            The begin of the audio dataset.
            Defaulted to the begin of the first file.
        end: Timestamp | None
            The end of the audio dataset.
            Defaulted to the end of the last file.
        timezone: str | pytz.timezone | None
            The timezone in which the file should be localized.
            If ``None``, the file begin/end will be tz-naive.
            If different from a timezone parsed from the filename, the timestamps'
            timezone will be converted from the parsed timezone
            to the specified timezone.
        mode: Literal["files", "timedelta_total", "timedelta_file"]
            Mode of creation of the dataset data from the original files.
            ``"files"``: one data will be created for each file.
            ``"timedelta_total"``: data objects of duration equal to ``data_duration``
            will be created from the begin timestamp to the end timestamp.
            ``"timedelta_file"``: data objects of duration equal to ``data_duration``
            will be created from the beginning of the first file that the begin
            timestamp is into, until it would resume in a data beginning between
            two files.
            Then, the next data object will be created from the
            beginning of the next original file and so on.
        overlap: float
            Overlap percentage between consecutive data.
        data_duration: Timedelta | None
            Duration of the audio data objects.
            If mode is set to ``"files"``, this parameter has no effect.
            If provided, audio data will be evenly distributed between begin and end.
            Else, one data object will cover the whole time period.
        sample_rate: float | None
            Sample rate of the audio data objects.
        name: str|None
            Name of the dataset.
        instrument: Instrument | None
            Instrument that might be used to obtain acoustic pressure from
            the wav audio data.
        normalization: Normalization
            The type of normalization to apply to the audio data.
        kwargs: any
            Keyword arguments passed to the ``BaseDataset.from_folder()`` classmethod.

        Returns
        -------
        Audiodataset:
            The audio dataset.

        """
        return super().from_folder(
            folder=folder,
            strptime_format=strptime_format,
            begin=begin,
            end=end,
            mode=mode,
            timezone=timezone,
            overlap=overlap,
            data_duration=data_duration,
            sample_rate=sample_rate,
            name=name,
            instrument=instrument,
            normalization=normalization,
        )

    @classmethod
    def from_files(  # noqa: PLR0913
        cls,
        files: list[AudioFile],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        name: str | None = None,
        mode: Literal["files", "timedelta_total", "timedelta_file"] = "timedelta_total",
        overlap: float = 0.0,
        data_duration: Timedelta | None = None,
        sample_rate: float | None = None,
        instrument: Instrument | None = None,
        normalization: Normalization = Normalization.RAW,
    ) -> AudioDataset:
        """Return an AudioDataset object from a list of AudioFiles.

        Parameters
        ----------
        files: list[AudioFile]
            The list of files contained in the Dataset.
        begin: Timestamp | None
            The begin of the audio dataset.
            Defaulted to the begin of the first file.
        end: Timestamp | None
            The end of the audio dataset.
            Defaulted to the end of the last file.
        mode: Literal["files", "timedelta_total", "timedelta_file"]
            Mode of creation of the dataset data from the original files.
            ``"files"``: one data will be created for each file.
            ``"timedelta_total"``: data objects of duration equal to ``data_duration``
            will be created from the begin timestamp to the end timestamp.
            ``"timedelta_file"``: data objects of duration equal to ``data_duration``
            will be created from the beginning of the first file that the begin
            timestamp is into, until it would resume in a data beginning between
            two files.
            Then, the next data object will be created from the
            beginning of the next original file and so on.
        overlap: float
            Overlap percentage between consecutive data.
        data_duration: Timedelta | None
            Duration of the audio data objects.
            If mode is set to ``"files"``, this parameter has no effect.
            If provided, audio data will be evenly distributed between begin and end.
            Else, one data object will cover the whole time period.
        sample_rate: float | None
            Sample rate of the audio data objects.
        name: str|None
            Name of the dataset.
        instrument: Instrument | None
            Instrument that might be used to obtain acoustic pressure from
            the wav audio data.
        normalization: Normalization
            The type of normalization to apply to the audio data.

        Returns
        -------
        AudioDataset:
        The ``AudioDataset`` object.

        """
        return super().from_files(
            files=files,
            begin=begin,
            end=end,
            name=name,
            instrument=instrument,
            normalization=normalization,
            sample_rate=sample_rate,
            mode=mode,
            overlap=overlap,
            data_duration=data_duration,
        )

    @classmethod
    def _data_from_files(
        cls,
        files: list[AudioFile],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        name: str | None = None,
        **kwargs,  # noqa: ANN003
    ) -> AudioData:
        """Return an ``AudioData`` object from a list of ``AudioFiles``.

        The ``AudioData`` starts at the begin and ends at end.

        Parameters
        ----------
        files: list[AudioFile]
            List of ``AudioFiles`` contained in the ``AudioData``.
        begin: Timestamp | None
            Begin of the ``AudioData``.
            Defaulted to the begin of the first ``AudioFile``.
        end: Timestamp | None
            End of the ``AudioData``.
            Defaulted to the end of the last ``AudioFile``.
        name: str|None
            Name of the ``AudioData``.
        kwargs:
            Keyword arguments to pass to the ``AudioData.from_files()`` method.

        Returns
        -------
        AudioData:
            The ``AudioData`` object.

        """
        return AudioData.from_files(
            files=files,
            begin=begin,
            end=end,
            name=name,
            **kwargs,
        )

    @classmethod
    def from_json(cls, file: Path) -> Self:
        """Deserialize an ``AudioDataset`` from a JSON file.

        Parameters
        ----------
        file: Path
            Path to the serialized JSON file representing the ``AudioDataset``.

        Returns
        -------
        AudioDataset
            The deserialized ``AudioDataset``.

        """
        # I have to redefine this method (without overriding it)
        # for the type hint to be correct.
        # It seems to be due to BaseData being a Generic class, following which
        # AudioData.from_json() is supposed to return a BaseData
        # without this duplicate definition...
        # I might look back at all this in the future
        return cls.from_dict(deserialize_json(file))
