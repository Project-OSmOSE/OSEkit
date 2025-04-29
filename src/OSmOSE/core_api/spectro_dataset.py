"""SpectroDataset is a collection of SpectroData objects.

SpectroDataset is a collection of SpectroData, with methods
that simplify repeated operations on the spectro data.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.signal import ShortTimeFFT

from OSmOSE.core_api.base_dataset import BaseDataset
from OSmOSE.core_api.json_serializer import deserialize_json
from OSmOSE.core_api.spectro_data import SpectroData
from OSmOSE.core_api.spectro_file import SpectroFile
from OSmOSE.utils.core_utils import locked

if TYPE_CHECKING:

    import pytz
    from pandas import Timedelta, Timestamp

    from OSmOSE.core_api.audio_dataset import AudioDataset


class SpectroDataset(BaseDataset[SpectroData, SpectroFile]):
    """SpectroDataset is a collection of SpectroData objects.

    SpectroDataset is a collection of SpectroData, with methods
    that simplify repeated operations on the spectro data.

    """

    def __init__(
        self,
        data: list[SpectroData],
        name: str | None = None,
        suffix: str = "",
        folder: Path | None = None,
    ) -> None:
        """Initialize a SpectroDataset."""
        super().__init__(data=data, name=name, suffix=suffix, folder=folder)

    @property
    def fft(self) -> ShortTimeFFT:
        """Return the fft of the spectro data."""
        return next(data.fft for data in self.data)

    @fft.setter
    def fft(self, fft: ShortTimeFFT) -> None:
        for data in self.data:
            data.fft = fft

    @property
    def folder(self) -> Path:
        """Folder in which the dataset files are located."""
        return self._folder if self._folder is not None else super().folder

    @folder.setter
    def folder(self, folder: Path) -> None:
        """Move the dataset to the specified destination folder.

        Parameters
        ----------
        folder: Path
            The folder in which the dataset will be moved.
            It will be created if it does not exist.

        """
        self._folder = folder
        for file in self.files:
            file.move(folder)

    def save_spectrogram(
        self,
        folder: Path,
        first: int = 0,
        last: int | None = None,
    ) -> None:
        """Export all spectrogram data as png images in the specified folder.

        Parameters
        ----------
        folder: Path
            Folder in which the spectrograms should be saved.
        first: int
            Index of the first SpectroData object to export.
        last: int|None
            Index after the last SpectroData object to export.


        """
        last = len(self.data) if last is None else last
        for data in self.data[first:last]:
            data.save_spectrogram(folder)

    def save_all(
        self,
        matrix_folder: Path,
        spectrogram_folder: Path,
        link: bool = False,
        first: int = 0,
        last: int | None = None,
    ) -> None:
        """Export both Sx matrices as npz files and spectrograms for each data.

        Parameters
        ----------
        matrix_folder: Path
            Path to the folder in which the Sx matrices npz files will be saved.
        spectrogram_folder: Path
            Path to the folder in which the spectrograms png files will be saved.
        link: bool
            If True, the SpectroData will be bound to the written npz file.
            Its items will be replaced with a single item, which will match the whole
            new SpectroFile.
        first: int
            Index of the first SpectroData object to export.
        last: int|None
            Index after the last SpectroData object to export.

        """
        last = len(self.data) if last is None else last
        for data in self.data[first:last]:
            sx = data.get_value()
            data.write(folder=matrix_folder, sx=sx, link=link)
            data.save_spectrogram(folder=spectrogram_folder, sx=sx)

    def link_audio_dataset(
        self,
        audio_dataset: AudioDataset,
        first: int = 0,
        last: int | None = None,
    ) -> None:
        """Link the SpectroData of the SpectroDataset to the AudioData of the AudioDataset.

        Parameters
        ----------
        audio_dataset: AudioDataset
            The AudioDataset which data will be linked to the SpectroDataset data.

        """
        if len(audio_dataset.data) != len(self.data):
            raise ValueError(
                "The audio dataset doesn't contain the same number of data as the spectro dataset.",
            )

        last = len(self.data) if last is None else last

        for sd, ad in list(
            zip(
                sorted(self.data, key=lambda d: (d.begin, d.end)),
                sorted(audio_dataset.data, key=lambda d: (d.begin, d.end)),
            ),
        )[first:last]:
            sd.link_audio_data(ad)

    def update_json_audio_data(self, first: int, last: int) -> None:
        """Update the serialized JSON file with the spectro data from first to int.

        The update is done while using the locked decorator.
        That way, if a SpectroDataset is processed through multiple jobs,
        each one can update the JSON file safely.

        Parameters
        ----------
        first: int
            Index of the first data to update.
        last: int
            Index of the last data to update.

        """
        json_file = self.folder / f"{self.name}.json"

        @locked(lock_file=self.folder / "lock.lock")
        def update(first: int, last: int) -> None:
            sds_to_update = SpectroDataset.from_json(file=json_file)
            sds_to_update.data[first:last] = self.data[first:last]
            sds_to_update.write_json(folder=self.folder)

        update(first=first, last=last)

    def to_dict(self) -> dict:
        """Serialize a SpectroDataset to a dictionary.

        Returns
        -------
        dict:
            The serialized dictionary representing the SpectroDataset.

        """
        sft_dict = {}
        for data in self.data:
            sft = next(
                (
                    sft
                    for name, sft in sft_dict.items()
                    if np.array_equal(data.fft.win, sft["win"])
                    and data.fft.hop == sft["hop"]
                    and data.fft.fs == sft["fs"]
                    and data.fft.mfft == sft["mfft"]
                    and data.fft.scaling == sft["scale_to"]
                ),
                None,
            )
            if sft is None:
                sft_dict[str(data.fft)] = {
                    "win": list(data.fft.win),
                    "hop": data.fft.hop,
                    "fs": data.fft.fs,
                    "mfft": data.fft.mfft,
                    "spectro_data": [str(data)],
                    "scale_to": data.fft.scaling,
                }
                continue
            sft["spectro_data"].append(str(data))
        spectro_data_dict = {str(d): d.to_dict(embed_sft=False) for d in self.data}
        return {
            "data": spectro_data_dict,
            "sft": sft_dict,
            "name": self._name,
            "suffix": self.suffix,
            "folder": str(self.folder),
        }

    @classmethod
    def from_dict(cls, dictionary: dict) -> SpectroDataset:
        """Deserialize a SpectroDataset from a dictionary.

        Parameters
        ----------
        dictionary: dict
            The serialized dictionary representing the SpectroDataset.

        Returns
        -------
        AudioData
            The deserialized SpectroDataset.

        """
        sfts = [
            (
                ShortTimeFFT(
                    win=np.array(sft["win"]),
                    hop=sft["hop"],
                    fs=sft["fs"],
                    mfft=sft["mfft"],
                ),
                sft["spectro_data"],
            )
            for name, sft in dictionary["sft"].items()
        ]
        sd = [
            SpectroData.from_dict(
                params,
                sft=next(sft for sft, linked_data in sfts if name in linked_data),
            )
            for name, params in dictionary["data"].items()
        ]
        return cls(
            data=sd,
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
        **kwargs: any,
    ) -> SpectroDataset:
        """Return a SpectroDataset from a folder containing the audio files.

        Parameters
        ----------
        folder: Path
            The folder containing the spectro files.
        strptime_format: str
            The strptime format of the timestamps in the spectro file names.
        begin: Timestamp | None
            The begin of the spectro dataset.
            Defaulted to the begin of the first file.
        end: Timestamp | None
            The end of the spectro dataset.
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
            Duration of the spectro data objects.
            If bound is set to "files", this parameter has no effect.
            If provided, spectro data will be evenly distributed between begin and end.
            Else, one data object will cover the whole time period.
        name: str|None
            Name of the dataset.
        kwargs: any
            Keyword arguments passed to the BaseDataset.from_folder classmethod.

        Returns
        -------
        Spectrodataset:
            The audio dataset.

        """
        kwargs.update(
            {"file_class": SpectroFile, "supported_file_extensions": [".npz"]},
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
        sft = next(iter(base_dataset.files)).get_fft()
        return cls.from_base_dataset(base_dataset=base_dataset, fft=sft, name=name)

    @classmethod
    def from_base_dataset(
        cls,
        base_dataset: BaseDataset,
        fft: ShortTimeFFT,
        name: str | None = None,
    ) -> SpectroDataset:
        """Return a SpectroDataset object from a BaseDataset object."""
        return cls(
            [SpectroData.from_base_data(data, fft) for data in base_dataset.data],
            name=name,
        )

    @classmethod
    def from_audio_dataset(
        cls,
        audio_dataset: AudioDataset,
        fft: ShortTimeFFT,
        name: str | None = None,
    ) -> SpectroDataset:
        """Return a SpectroDataset object from an AudioDataset object.

        The SpectroData is computed from the AudioData using the given fft.
        """
        return cls(
            data=[SpectroData.from_audio_data(d, fft) for d in audio_dataset.data],
            name=name,
        )

    @classmethod
    def from_json(cls, file: Path) -> SpectroDataset:
        """Deserialize a SpectroDataset from a JSON file.

        Parameters
        ----------
        file: Path
            Path to the serialized JSON file representing the SpectroDataset.

        Returns
        -------
        SpectroDataset
            The deserialized SpectroDataset.

        """
        # I have to redefine this method (without overriding it)
        # for the type hint to be correct.
        # It seems to be due to BaseData being a Generic class, following which
        # AudioData.from_json() is supposed to return a BaseData
        # without this duplicate definition...
        # I might look back at all this in the future
        return cls.from_dict(deserialize_json(file))
