"""``SpectroDataset`` is a collection of ``SpectroData`` objects.

``SpectroDataset`` is a collection of ``SpectroData``, with methods
that simplify repeated operations on the ``SpectroData``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from pandas import DataFrame
from scipy.signal import ShortTimeFFT

from osekit.config import DPDEFAULT
from osekit.core_api.base_dataset import BaseDataset, TFile
from osekit.core_api.frequency_scale import Scale
from osekit.core_api.json_serializer import deserialize_json
from osekit.core_api.spectro_data import SpectroData
from osekit.core_api.spectro_file import SpectroFile
from osekit.utils.core_utils import locked
from osekit.utils.multiprocess_utils import multiprocess

if TYPE_CHECKING:
    import pytz
    from pandas import Timedelta, Timestamp

    from osekit.core_api.audio_dataset import AudioDataset


class SpectroDataset(BaseDataset[SpectroData, SpectroFile]):
    """``SpectroDataset`` is a collection of ``SpectroData`` objects.

    ``SpectroDataset`` is a collection of ``SpectroData``, with methods
    that simplify repeated operations on the ``SpectroData``.

    """

    sentinel_value = object()
    _bypass_multiprocessing_on_dataset = False
    data_cls = SpectroData
    file_cls = SpectroFile

    def __init__(
        self,
        data: list[SpectroData],
        name: str | None = None,
        suffix: str = "",
        folder: Path | None = None,
        scale: Scale | None = None,
        v_lim: tuple[float, float] | None | object = sentinel_value,
    ) -> None:
        """Initialize a ``SpectroDataset``."""
        super().__init__(data=data, name=name, suffix=suffix, folder=folder)
        self.scale = scale

        if v_lim is not self.sentinel_value:
            # the sentinel value allows to differentiate between
            # a specified None value (resets the v_lim to the default values)
            # from an unspecified v_lim (in that case, the data v_lim are unchanged)
            self.v_lim = v_lim

    @property
    def fft(self) -> ShortTimeFFT:
        """Return the fft of the spectro data."""
        return next(data.fft for data in self.data)

    @fft.setter
    def fft(self, fft: ShortTimeFFT) -> None:
        for data in self.data:
            data.fft = fft

    @property
    def colormap(self) -> str:
        """Return the most frequent ``colormap`` of the spectro dataset."""
        return max(
            {d.colormap for d in self.data},
            key=[d.colormap for d in self.data].count,
        )

    @colormap.setter
    def colormap(self, colormap: str) -> None:
        for d in self.data:
            d.colormap = colormap

    @property
    def v_lim(self) -> tuple[float, float] | None:
        """Return the most frequent ``v_lim`` of the spectro dataset."""
        return max(
            {d.v_lim for d in self.data},
            key=[d.v_lim for d in self.data].count,
        )

    @v_lim.setter
    def v_lim(self, v_lim: tuple[float, float] | None) -> None:
        """Set the spectrogram color scale limits (in ``dB``).

        Parameters
        ----------
        v_lim: tuple[float, float] | None
            Limits (in ``dB``) of the colormap used for plotting the spectrogram.

        """
        for d in self.data:
            d.v_lim = v_lim

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

    def _save_spectrogram(self, sd: SpectroData, folder: Path) -> None:
        """Save the spectrogram data."""
        sd.save_spectrogram(folder=folder, scale=self.scale)

    def save_spectrogram(
        self,
        folder: Path,
        first: int = 0,
        last: int | None = None,
    ) -> None:
        """Export all spectrogram data as ``png`` images in the specified folder.

        Parameters
        ----------
        folder: Path
            Folder in which the spectrograms should be saved.
        first: int
            Index of the first ``SpectroData`` object to export.
        last: int|None
            Index after the last ``SpectroData`` object to export.

        """
        last = len(self.data) if last is None else last
        multiprocess(
            self._save_spectrogram,
            self.data[first:last],
            bypass_multiprocessing=type(self)._bypass_multiprocessing_on_dataset,
            folder=folder,
        )

    def _get_welch(
        self,
        sd: SpectroData,
        nperseg: int | None = None,
        detrend: str | callable | False = "constant",
        scaling: Literal["density", "spectrum"] = "density",
        average: Literal["mean", "median"] = "mean",
        *,
        return_onesided: bool = True,
    ) -> tuple[SpectroData, np.ndarray]:
        """Get the welch value of each ``SpectroData``."""
        return sd, sd.get_welch(
            nperseg=nperseg,
            detrend=detrend,
            return_onesided=return_onesided,
            scaling=scaling,
            average=average,
        )

    def get_welch(
        self,
        first: int = 0,
        last: int | None = None,
        nperseg: int | None = None,
        detrend: str | callable | False = "constant",
        scaling: Literal["density", "spectrum"] = "density",
        average: Literal["mean", "median"] = "mean",
        *,
        return_onesided: bool = True,
    ) -> DataFrame:
        """Return the welch values of the ``SpectroDataset``.

        Each ``SpectroData`` of the ``SpectroDataset`` represent
        one column of the welch values.

        Parameters
        ----------
        first: int
            Index of the first ``SpectroData`` object to include in the welch.
        last: int
            Index after the last ``SpectroData`` object to include in the welch.
        nperseg: int|None
            Length of each segment. Defaults to ``None``, but if window is ``str`` or
            ``tuple``, is set to ``256``, and if window is ``array_like``,
            is set to the length of the window.
        detrend: str | callable | False
            Specifies how to detrend each segment. If detrend is a ``str``,
            it is passed as the type argument to the detrend function.
            If it is a ``func``, it takes a segment and returns a detrended segment.
            If detrend is ``False``, no detrending is done.
            Defaults to ``'constant'``.
        return_onesided: bool
            If ``True``, return a one-sided spectrum for real data.
            If ``False``, return a two-sided spectrum.
            Defaults to ``True``, but for complex data,
            a two-sided spectrum is always returned.
        scaling: Literal["density", "spectrum"]
            Selects between computing the power spectral density (``'density'``) where
            ``Pxx`` has units of ``V**2/Hz`` and computing the squared magnitude
            spectrum (``'spectrum'``) where ``Pxx`` has units of ``V**2``,
            if ``x`` is measured in ``V`` and ``fs`` is measured in ``Hz``.
            Defaults to ``'density'``.
        average: Literal["mean", "median"]
            Method to use when averaging periodograms.
            Defaults to ``'mean'``.
        return_onesided: bool
            If True, return a one-sided spectrum for real data. If False return a two-sided spectrum. Defaults to True, but for complex data, a two-sided spectrum is always returned.

        """
        output = {}
        for data, welch in multiprocess(
            self._get_welch,
            self.data[first:last],
            bypass_multiprocessing=type(self)._bypass_multiprocessing_on_dataset,
            nperseg=nperseg,
            detrend=detrend,
            return_onesided=return_onesided,
            scaling=scaling,
            average=average,
        ):
            timestamp = f"{data.begin!s}_{data.end!s}"
            output[timestamp] = welch
        return DataFrame(output)

    def write_welch(
        self,
        folder: Path,
        first: int = 0,
        last: int | None = None,
        nperseg: int | None = None,
        detrend: str | callable | False = "constant",
        scaling: Literal["density", "spectrum"] = "density",
        average: Literal["mean", "median"] = "mean",
        pxs: DataFrame | None = None,
        *,
        return_onesided: bool = True,
    ) -> None:
        """Write the welch values of the SpectroDataset in a npz file.

        Parameters
        ----------
        folder: Path
            The folder in which the ``npz`` file will be written.
        first: int
            Index of the first ``SpectroData`` object to include in the welch.
        last: int
            Index after the last ``SpectroData`` object to include in the welch.
        nperseg: int|None
            Length of each segment.
            Defaults to ``None``, but if window is ``str`` or
            ``tuple``, is set to ``256``, and if window is ``array_like``,
            is set to the length of the window.
        detrend: str | callable | False
            Specifies how to detrend each segment. If detrend is a ``str``,
            it is passed as the type argument to the detrend function.
            If it is a ``func``, it takes a segment and returns a detrended segment.
            If detrend is ``False``, no detrending is done.
            Defaults to ``'constant'``.
        return_onesided: bool
            If ``True``, return a one-sided spectrum for real data.
            If ``False``, return a two-sided spectrum.
            Defaults to ``True``, but for complex data,
            a two-sided spectrum is always returned.
        scaling: Literal["density", "spectrum"]
            Selects between computing the power spectral density (``'density'``) where
            ``Pxx`` has units of ``V**2/Hz`` and computing the squared magnitude
            spectrum (``'spectrum'``) where ``Pxx`` has units of ``V**2``,
            if ``x`` is measured in ``V`` and ``fs`` is measured in ``Hz``.
            Defaults to ``'density'``.
        average: Literal["mean", "median"]
            Method to use when averaging periodograms.
            Defaults to ``'mean'``.
        pxs: DataFrame | None
            Welch values as returned by ``SpectroDataset.get_welch()``.
            If provided, the computation will be skipped and the provided
            pxs values will be written to disk.
        return_onesided: bool
            If ``True``, return a one-sided spectrum for real data.
            If ``False`` return a two-sided spectrum.
            Defaults to ``True``, but for complex data,
            a two-sided spectrum is always returned.

        """
        folder.mkdir(parents=True, exist_ok=True, mode=DPDEFAULT)
        pxs = (
            self.get_welch(
                first=first,
                last=last,
                nperseg=nperseg,
                detrend=detrend,
                scaling=scaling,
                average=average,
                return_onesided=return_onesided,
            )
            if pxs is None
            else pxs
        )
        np.savez(
            file=folder / f"{self.data[first]}.npz",
            timestamps=list(pxs.columns),
            pxs=pxs.to_numpy().T,
            freq=self.fft.f,
        )

    def _save_all_(
        self,
        data: SpectroData,
        matrix_folder: Path,
        spectrogram_folder: Path,
        *,
        link: bool,
    ) -> SpectroData:
        """Save the data matrix and spectrogram to disk."""
        sx = data.get_value()
        data.write(folder=matrix_folder, sx=sx, link=link)
        data.save_spectrogram(folder=spectrogram_folder, sx=sx, scale=self.scale)
        return data

    def save_all(
        self,
        matrix_folder: Path,
        spectrogram_folder: Path,
        first: int = 0,
        last: int | None = None,
        *,
        link: bool = False,
    ) -> None:
        """Export both Sx matrices as ``npz`` files and spectrograms for each data.

        Parameters
        ----------
        matrix_folder: Path
            Path to the folder in which the Sx matrices ``npz`` files will be saved.
        spectrogram_folder: Path
            Path to the folder in which the spectrograms ``png`` files will be saved.
        link: bool
            If ``True``, the ``SpectroData`` will be bound to the written ``npz`` file.
            Its items will be replaced with a single item, which will match the whole
            new ``SpectroFile``.
        first: int
            Index of the first ``SpectroData`` object to export.
        last: int|None
            Index after the last ``SpectroData`` object to export.

        """
        last = len(self.data) if last is None else last
        self.data[first:last] = multiprocess(
            func=self._save_all_,
            enumerable=self.data[first:last],
            bypass_multiprocessing=type(self)._bypass_multiprocessing_on_dataset,
            matrix_folder=matrix_folder,
            spectrogram_folder=spectrogram_folder,
            link=link,
        )

    def link_audio_dataset(
        self,
        audio_dataset: AudioDataset,
        first: int = 0,
        last: int | None = None,
    ) -> None:
        """Link the ``SpectroData`` of the ``SpectroDataset`` to the ``AudioData`` of the ``AudioDataset``.

        Parameters
        ----------
        audio_dataset: AudioDataset
            The ``AudioDataset`` which data will be linked to the ``SpectroDataset`` data.
        first: int
            Index of the first ``SpectroData`` and ``AudioData`` to link.
        last: int
            Index of the last ``SpectroData`` and ``AudioData`` to link.

        """
        if len(audio_dataset.data) != len(self.data):
            msg = (
                "The audio dataset doesn't contain the same number of data"
                " as the spectro dataset."
            )
            raise ValueError(msg)

        last = len(self.data) if last is None else last

        for sd, ad in list(
            zip(
                sorted(self.data, key=lambda d: (d.begin, d.end)),
                sorted(audio_dataset.data, key=lambda d: (d.begin, d.end)),
                strict=False,
            ),
        )[first:last]:
            sd.link_audio_data(ad)

    def update_json_audio_data(self, first: int, last: int) -> None:
        """Update the serialized ``json`` file with the spectro data from first to last.

        The update is done while using the locked decorator.
        That way, if a ``SpectroDataset`` is processed through multiple jobs,
        each one can update the ``json`` file safely.

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
            sds_to_update = type(self).from_json(file=json_file)
            sds_to_update.data[first:last] = self.data[first:last]
            sds_to_update.write_json(folder=self.folder)

        update(first=first, last=last)

    def to_dict(self) -> dict:
        """Serialize a ``SpectroDataset`` to a dictionary.

        Returns
        -------
        dict:
            The serialized dictionary representing the ``SpectroDataset``.

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
            "scale": self.scale.to_dict_value() if self.scale is not None else None,
            "name": self._name,
            "suffix": self.suffix,
            "folder": str(self.folder),
        }

    @classmethod
    def from_dict(cls, dictionary: dict) -> SpectroDataset:
        """Deserialize a ``SpectroDataset`` from a dictionary.

        Parameters
        ----------
        dictionary: dict
            The serialized dictionary representing the ``SpectroDataset``.

        Returns
        -------
        SpectroDataset
            The deserialized ``SpectroDataset``.

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
            cls.data_cls.from_dict(
                params,
                sft=next(sft for sft, linked_data in sfts if name in linked_data),
            )
            for name, params in dictionary["data"].items()
        ]
        scale = dictionary["scale"]
        if dictionary["scale"] is not None:
            scale = Scale.from_dict_value(scale)
        return cls(
            data=sd,
            name=dictionary["name"],
            suffix=dictionary["suffix"],
            folder=Path(dictionary["folder"]),
            scale=scale,
        )

    @classmethod
    def from_folder(  # noqa: PLR0913
        cls,
        folder: Path,
        strptime_format: str,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        timezone: str | pytz.timezone | None = None,
        mode: Literal["files", "timedelta_total", "timedelta_file"] = "timedelta_total",
        overlap: float = 0.0,
        data_duration: Timedelta | None = None,
        name: str | None = None,
        **kwargs,  # noqa: ANN003
    ) -> Self:
        """Return a ``SpectroDataset`` from a folder containing the spectro files.

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
            If ``None``, the file begin/end will be tz-naive.
            If different from a timezone parsed from the filename, the timestamps'
            timezone will be converted from the parsed timezone
            to the specified timezone.
        mode: Literal["files", "timedelta_total", "timedelta_file"]
            Mode of creation of the dataset data from the original files.
            ``"files"``: one data will be created for each file.
            ``"timedelta_total"``: data objects of duration equal to ``data_duration``
            will be created from the ``begin`` timestamp to the ``end`` timestamp.
            ``"timedelta_file"``: data objects of duration equal to ``data_duration``
            will be created from the beginning of the first file that the ``begin``
            timestamp is into, until it would resume in a data beginning between
            two files. Then, the next data object will be created from the beginning of
            the next original file and so on.
        overlap: float
            Overlap percentage between consecutive data.
        data_duration: Timedelta | None
            Duration of the spectro data objects.
            If mode is set to ``"files"``, this parameter has no effect.
            If provided, spectro data will be evenly distributed between
            ``begin`` and ``end``.
            Else, one data object will cover the whole time period.
        name: str|None
            Name of the dataset.
        kwargs:
            None

        Returns
        -------
        Spectrodataset:
            The ``SpectroDataset`` instance.

        """
        return super().from_folder(
            folder=folder,
            strptime_format=strptime_format,
            begin=begin,
            end=end,
            timezone=timezone,
            mode=mode,
            overlap=overlap,
            data_duration=data_duration,
            name=name,
        )

    @classmethod
    def from_files(  # noqa: PLR0913
        cls,
        files: list[SpectroFile],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        name: str | None = None,
        mode: Literal["files", "timedelta_total", "timedelta_file"] = "timedelta_total",
        overlap: float = 0.0,
        data_duration: Timedelta | None = None,
        **kwargs,  # noqa: ANN003
    ) -> AudioDataset:
        """Return an ``SpectroDataset`` object from a list of ``SpectroFiles``.

        Parameters
        ----------
        files: list[SpectroFile]
            The list of files contained in the Dataset.
        begin: Timestamp | None
            Begin of the first data object.
            Defaulted to the begin of the first file.
        end: Timestamp | None
            End of the last data object.
            Defaulted to the end of the last file.
        mode: Literal["files", "timedelta_total", "timedelta_file"]
            Mode of creation of the dataset data from the original files.
            ``"files"``: one data will be created for each file.
            ``"timedelta_total"``: data objects of duration equal to ``data_duration``
            will be created from the ``begin`` timestamp to the ``end`` timestamp.
            ``"timedelta_file"``: data objects of duration equal to ``data_duration``
            will be created from the beginning of the first file that the ``begin``
            timestamp is into, until it would resume in a data beginning between
            two files. Then, the next data object will be created from the beginning of
            the next original file and so on.
        overlap: float
            Overlap percentage between consecutive data.
        data_duration: Timedelta | None
            Duration of the spectro data objects.
            If mode is set to ``"files"``, this parameter has no effect.
            If provided, spectro data will be evenly distributed between
            ``begin`` and ``end``.
            Else, one data object will cover the whole time period.
        sample_rate: float | None
            Sample rate of the audio data objects.
        name: str|None
            Name of the dataset.
        instrument: Instrument | None
            Instrument that might be used to obtain acoustic pressure from
            the ``wav`` audio data.
        normalization: Normalization
            The type of normalization to apply to the audio data.
        kwargs:
            None

        Returns
        -------
        SpectroDataset:
        The ``SpectroDataset`` object.

        """
        return super().from_files(
            files=files,
            begin=begin,
            end=end,
            name=name,
            mode=mode,
            overlap=overlap,
            data_duration=data_duration,
        )

    @classmethod
    def _data_from_dict(cls, dictionary: dict) -> list[SpectroData]:
        """Return the list of ``SpectroData`` objects from the serialized dictionary.

        Parameters
        ----------
        dictionary: dict
            Dictionary representing the serialized ``SpectroDataset``.

        Returns
        -------
        list[SpectroData]:
            The list of deserialized ``SpectroData`` objects.

        """
        return [SpectroData.from_dict(dictionary=data) for data in dictionary.values()]

    @classmethod
    def _data_from_files(
        cls,
        files: list[TFile],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        name: str | None = None,
        **kwargs,  # noqa: ANN003
    ) -> SpectroData:
        """Return a ``SpectroData`` object from a list of ``SpectroFiles``.

        The ``SpectroData`` starts at ``begin`` and ends at ``end``.

        Parameters
        ----------
        files: list[SpectroFile]
            List of ``SpectroFiles`` contained in the ``SpectroData``.
        begin: Timestamp | None
            Begin of the ``SpectroData``.
            Defaulted to the begin of the first ``SpectroFile``.
        end: Timestamp | None
            End of the ``SpectroData``.
            Defaulted to the end of the last ``SpectroFile``.
        name: str|None
            Name of the ``SpectroData``.
        kwargs:
            Keyword arguments to pass to the ``SpectroData.from_files()`` method.

        Returns
        -------
        SpectroData:
            The ``SpectroData`` object.

        """
        return SpectroData.from_files(
            files=files,
            begin=begin,
            end=end,
            name=name,
            **kwargs,
        )

    @classmethod
    def from_audio_dataset(
        cls,
        audio_dataset: AudioDataset,
        fft: ShortTimeFFT,
        name: str | None = None,
        colormap: str | None = None,
        v_lim: tuple[float, float] | None = sentinel_value,
        scale: Scale | None = None,
    ) -> SpectroDataset:
        """Return a ``SpectroDataset`` object from an ``AudioDataset`` object.

        The ``SpectroData`` is computed from the ``AudioData`` using ``fft``.
        """
        return cls(
            data=[
                SpectroData.from_audio_data(
                    data=d,
                    fft=fft,
                    colormap=colormap,
                )
                for d in audio_dataset.data
            ],
            name=name,
            scale=scale,
            v_lim=v_lim,
        )

    @classmethod
    def from_json(cls, file: Path) -> SpectroDataset:
        """Deserialize a ``SpectroDataset`` from a ``json`` file.

        Parameters
        ----------
        file: Path
            Path to the serialized ``json`` representing the ``SpectroDataset``.

        Returns
        -------
        SpectroDataset
            The deserialized ``SpectroDataset``.

        """
        # I have to redefine this method (without overriding it)
        # for the type hint to be correct.
        # It seems to be due to BaseData being a Generic class, following which
        # AudioData.from_json() is supposed to return a BaseData
        # without this duplicate definition...
        # I might look back at all this in the future
        return cls.from_dict(deserialize_json(file))
