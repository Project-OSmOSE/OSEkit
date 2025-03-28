"""Main class of the Public API.

The Dataset correspond to a collection of audio,
spectro and auxilary core_api datasets.
It has additionnal metadata that can be exported, e.g. to APLOSE.

"""

from __future__ import annotations

import shutil
from enum import Flag, auto
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from OSmOSE.core_api.audio_dataset import AudioDataset
from OSmOSE.core_api.base_dataset import BaseDataset
from OSmOSE.core_api.json_serializer import deserialize_json, serialize_json
from OSmOSE.core_api.spectro_dataset import SpectroDataset
from OSmOSE.utils.path_utils import move_tree

if TYPE_CHECKING:
    from pandas import Timedelta, Timestamp
    from scipy.signal import ShortTimeFFT

    from OSmOSE.core_api.audio_file import AudioFile


class Analysis(Flag):
    """Enum of flags that should be use to specify the type of analysis to run.

    AUDIO:
        Will add an AudioDataset to the datasets and write the reshaped audio files
        to disk.
        The new AudioDataset will be linked to the reshaped audio files rather than to
        the original files.
    MATRIX:
        Will write the npz SpectroFiles to disk and link the SpectroDataset to
        these files.
    SPECTROGRAM:
        Will export the spectrogram png images.

    Multiple flags can be enabled thanks to the logical or | operator:
    Analysis.AUDIO | Analysis.SPECTROGRAM will export both audio files and
    spectrogram images.

    >>> # Exporting both the reshaped audio and the spectrograms
    >>> # (without the npz matrices):
    >>> export = Analysis.AUDIO | Analysis.SPECTROGRAM
    >>> Analysis.AUDIO in export
    True
    >>> Analysis.SPECTROGRAM in export
    True
    >>> Analysis.MATRIX in export
    False

    """

    AUDIO = auto()
    MATRIX = auto()
    SPECTROGRAM = auto()


class Dataset:
    """Main class of the Public API.

    The Dataset correspond to a collection of audio,
    spectro and auxilary core_api datasets.
    It has additionnal metadata that can be exported, e.g. to APLOSE.

    """

    def __init__(  # noqa: PLR0913
        self,
        folder: Path,
        strptime_format: str,
        gps_coordinates: str | list | tuple = (0, 0),
        depth: str | int = 0,
        timezone: str | None = None,
        datasets: dict | None = None,
    ) -> None:
        """Initialize a Dataset."""
        self.folder = folder
        self.strptime_format = strptime_format
        self.gps_coordinates = gps_coordinates
        self.depth = depth
        self.timezone = timezone
        self.datasets = datasets if datasets is not None else {}

    @property
    def origin_files(self) -> set[AudioFile]:
        """Return the original audio files from which this Dataset has been built."""
        return None if self.origin_dataset is None else self.origin_dataset.files

    @property
    def origin_dataset(self) -> AudioDataset:
        """Return the AudioDataset from which this Dataset has been built."""
        return self.get_dataset("original")

    def build(self) -> None:
        """Build the Dataset.

        Building a dataset moves the original audio files to a specific folder
        and creates metadata csv used by APLOSE.

        """
        ads = AudioDataset.from_folder(
            self.folder,
            strptime_format=self.strptime_format,
            bound="files",
            timezone=self.timezone,
            name="original",
        )
        self.datasets[ads.name] = {"class": type(ads).__name__, "dataset": ads}
        move_tree(
            self.folder,
            self.folder / "other",
            {file.path for file in ads.files},
        )
        self._sort_dataset(ads)
        ads.write_json(ads.folder)
        self.write_json()

    def reset(self) -> None:
        """Reset the Dataset.

        Resetting a dataset will move back the original audio files and the content of
        the "other" folder to the root folder.
        WARNING: all other files and folders will be deleted.
        """
        files_to_remove = list(self.folder.iterdir())
        self.get_dataset("original").folder = self.folder

        if self.folder / "other" in files_to_remove:
            move_tree(self.folder / "other", self.folder)

        for file in files_to_remove:
            if file.is_dir():
                shutil.rmtree(file)
            else:
                file.unlink()

        self.datasets = {}

    def create_analysis(
        self,
        analysis: Analysis,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        data_duration: Timedelta | None = None,
        sample_rate: float | None = None,
        name: str | None = None,
        subtype: str | None = None,
        fft: ShortTimeFFT | None = None,
    ) -> None:
        """Create a new analysis dataset from the original audio files.

        The analysis parameter sets which type(s) of core_api dataset(s) will be
        created and added to the Dataset.datasets property, plus which output
        files will be written to disk (reshaped audio files, npz spectra matrices,
        png spectrograms...).

        Parameters
        ----------
        analysis: Analysis
            Flags that should be use to specify the type of analysis to run.
            See Dataset.Analysis docstring for more info.
        begin: Timestamp | None
            The begin of the analysis dataset.
            Defaulted to the begin of the original dataset.
        end: Timestamp | None
            The end of the analysis dataset.
            Defaulted to the end of the original dataset.
        data_duration: Timedelta | None
            Duration of the data within the analysis dataset.
            If provided, audio data will be evenly distributed between begin and end.
            Else, one data object will cover the whole time period.
        sample_rate: float | None
            Sample rate of the new analysis data.
            Audio data will be resampled if provided, else the sample rate
            will be set to the one of the original dataset.
        name: str | None
            Name of the analysis dataset.
            Defaulted as the begin timestamp of the analysis dataset.
            If both audio and spectro analyses are selected, the audio
            analysis dataset name will be suffixed with "_audio".
        subtype: str | None
            Subtype of the written audio files as provided by the soundfile module.
            Defaulted as the default 16-bit PCM for WAV audio files.
            This parameter has no effect if Analysis.AUDIO is not in analysis.
        fft: ShortTimeFFT | None
            FFT to use for computing the spectra.
            This parameter is mandatory if either Analysis.MATRIX or Analysis.SPECTROGRAM
            is in analysis.
            This parameter has no effect if neither Analysis.MATRIX nor Analysis.SPECTROGRAM
            is in the analysis.

        """
        is_spectro = any(
            flag in analysis for flag in (Analysis.MATRIX, Analysis.SPECTROGRAM)
        )

        if is_spectro and fft is None:
            raise ValueError(
                "FFT parameter should be given if spectra outputs are selected.",
            )

        ads = AudioDataset.from_files(
            files=list(self.origin_files),
            begin=begin,
            end=end,
            data_duration=data_duration,
            name=name,
        )

        if sample_rate is not None:
            ads.sample_rate = sample_rate

        if Analysis.AUDIO in analysis:
            if is_spectro and name is not None:
                ads.name = f"{ads.name}_audio"
            self._add_audio_dataset(ads=ads, subtype=subtype)

        if is_spectro:
            sds = SpectroDataset.from_audio_dataset(audio_dataset=ads, fft=fft)
            self._add_spectro_dataset(sds=sds, export=analysis)

    def _add_audio_dataset(
        self,
        ads: AudioDataset,
        subtype: str | None = None,
    ) -> None:
        ads_folder = self._get_audio_dataset_subpath(ads=ads)
        ads.write(ads_folder, link=True, subtype=subtype)

        self.datasets[ads.name] = {"class": type(ads).__name__, "dataset": ads}

        ads.write_json(folder=ads.folder)
        self.write_json()

    def _get_audio_dataset_subpath(
        self,
        ads: AudioDataset,
    ) -> Path:
        return (
            self.folder
            / "data"
            / "audio"
            / (
                f"{round(ads.data_duration.total_seconds())}_{round(ads.sample_rate)}"
                if ads.has_default_name
                else ads.name
            )
        )

    def _add_spectro_dataset(
        self,
        sds: SpectroDataset,
        export: Analysis,
    ) -> None:
        sds.folder = self._get_spectro_dataset_subpath(sds=sds)

        if Analysis.MATRIX in export and Analysis.SPECTROGRAM in export:
            sds.save_all(
                matrix_folder=sds.folder / "welch",
                spectrogram_folder=sds.folder / "spectrogram",
                link=True,
            )
        elif Analysis.SPECTROGRAM in export:
            sds.save_spectrogram(
                folder=sds.folder / "spectrogram",
            )
        elif Analysis.MATRIX in export:
            sds.write(
                folder=sds.folder / "welch",
                link=True,
            )

        self.datasets[sds.name] = {"class": type(sds).__name__, "dataset": sds}

        sds.write_json(folder=sds.folder)
        self.write_json()

    def _get_spectro_dataset_subpath(
        self,
        sds: SpectroDataset,
    ) -> Path:
        ads_folder = Path(
            f"{round(sds.data_duration.total_seconds())}_{round(sds.fft.fs)}",
        )
        fft_folder = f"{sds.fft.mfft}_{sds.fft.win.shape[0]}_{sds.fft.hop}_linear"
        return (
            self.folder
            / "processed"
            / (ads_folder / fft_folder if sds.has_default_name is None else sds.name)
        )

    def _sort_dataset(self, dataset: type[DatasetChild]) -> None:
        if type(dataset) is AudioDataset:
            self._sort_audio_dataset(dataset)
            return
        if type(dataset) is SpectroDataset:
            self._sort_spectro_dataset(dataset)
            return

    def _sort_audio_dataset(self, dataset: AudioDataset) -> None:
        dataset.folder = self._get_audio_dataset_subpath(dataset)

    def _sort_spectro_dataset(self, dataset: SpectroDataset) -> None:
        raise NotImplementedError

    def get_dataset(self, dataset_name: str) -> type[DatasetChild] | None:
        """Get an analysis dataset from its name.

        Parameters
        ----------
        dataset_name: str
            Name of the analysis dataset.

        Returns
        -------
        type[DatasetChild]:
            Analysis dataset from the dataset.datasets property.

        """
        if dataset_name not in self.datasets:
            return None
        return self.datasets[dataset_name]["dataset"]

    def to_dict(self) -> dict:
        """Serialize a dataset to a dictionary.

        Returns
        -------
        dict:
            The serialized dictionary representing the dataset.

        """
        return {
            "datasets": {
                name: {
                    "class": dataset["class"],
                    "json": str(dataset["dataset"].folder / f"{name}.json"),
                }
                for name, dataset in self.datasets.items()
            },
            "depth": self.depth,
            "folder": str(self.folder),
            "gps_coordinates": self.gps_coordinates,
            "strptime_format": self.strptime_format,
            "timezone": self.timezone,
        }

    """
        folder: Path,
        strptime_format: str,
        gps_coordinates: str | list | tuple = (0, 0),
        depth: str | int = 0,
        timezone: str | None = None,
    """

    @classmethod
    def from_dict(cls, dictionary: dict) -> Dataset:
        """Deserialize a dataset from a dictionary.

        Parameters
        ----------
        dictionary: dict
            The serialized dictionary representing the dataset.

        Returns
        -------
        Dataset
            The deserialized dataset.

        """
        datasets = {}
        for name, dataset in dictionary["datasets"].items():
            dataset_class = (
                AudioDataset
                if dataset["class"] == "AudioDataset"
                else (
                    SpectroDataset
                    if dataset["class"] == "SpectroDataset"
                    else BaseDataset
                )
            )
            datasets[name] = {
                "class": dataset["class"],
                "dataset": dataset_class.from_json(Path(dataset["json"])),
            }
        return cls(
            folder=Path(dictionary["folder"]),
            strptime_format=dictionary["strptime_format"],
            gps_coordinates=dictionary["gps_coordinates"],
            depth=dictionary["depth"],
            timezone=dictionary["timezone"],
            datasets=datasets,
        )

    def write_json(self, folder: Path | None = None) -> None:
        """Write a serialized Dataset to a JSON file."""
        folder = folder if folder is not None else self.folder
        serialize_json(folder / "dataset.json", self.to_dict())

    @classmethod
    def from_json(cls, file: Path) -> Dataset:
        """Deserialize a Dataset from a JSON file.

        Parameters
        ----------
        file: Path
            Path to the serialized JSON file representing the Dataset.

        Returns
        -------
        Dataset
            The deserialized BaseDataset.

        """
        return cls.from_dict(deserialize_json(file))


DatasetChild = TypeVar("DatasetChild", bound=BaseDataset)
