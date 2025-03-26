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


class SpectroOutput(Flag):
    """Enum of flags that should be use to specify the outputs of a spectra generation.

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
    SpectroOutput.AUDIO | SpectroOutput.SPECTROGRAM will export both audio files and
    spectrogram images.

    >>> # Exporting both the reshaped audio and the spectrograms (without the npz matrices):
    >>> export = SpectroOutput.AUDIO | SpectroOutput.SPECTROGRAM
    >>> SpectroOutput.AUDIO in export
    True
    >>> SpectroOutput.SPECTROGRAM in export
    True
    >>> SpectroOutput.MATRIX in export
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
        )
        self.datasets["original"] = {"class": type(ads).__name__, "dataset": ads}
        move_tree(
            self.folder,
            self.folder / "other",
            {file.path for file in self.datasets["original"]["dataset"].files},
        )
        self._sort_dataset(self.datasets["original"]["dataset"])
        ads.write_json(ads.folder, name="original")
        self.write_json()

    def reset(self) -> None:
        """Reset the Dataset.

        Resetting a dataset will move back the original audio files and the content of
        the "other" folder to the root folder.
        WARNING: all other files and folders will be deleted.
        """
        files_to_remove = list(self.folder.iterdir())
        self.datasets["original"]["dataset"].folder = self.folder

        if self.folder / "other" in files_to_remove:
            move_tree(self.folder / "other", self.folder)

        for file in files_to_remove:
            if file.is_dir():
                shutil.rmtree(file)
            else:
                file.unlink()

        self.datasets = {}

    def reshape(
        self,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        data_duration: Timedelta | None = None,
        sample_rate: float | None = None,
        name: str | None = None,
        subtype: str | None = None,
    ) -> None:
        """Create and write a new AudioDataset from the original audio files.

        The parameters of this method allow for a reshaping and resampling
        of the audio data.
        The created AudioDataset's files will be written to disk along with
        a JSON serialized file.

        Parameters
        ----------
        begin: Timestamp | None
            The begin of the audio dataset.
            Defaulted to the begin of the original dataset.
        end: Timestamp | None
            The end of the audio dataset.
            Defaulted to the end of the original dataset.
        data_duration: Timedelta | None
            Duration of the audio data within the new dataset.
            If provided, audio data will be evenly distributed between begin and end.
            Else, one data object will cover the whole time period.
        sample_rate: float | None
            Sample rate of the new audio data.
            Audio data will be resampled if provided, else the sample rate
            will be set to the one of the original dataset.
        name: str | None
            Name of the new dataset.
            Defaulted as the begin timestamp of the new dataset.
        subtype: str | None
            Subtype of the written audio files as provided by the soundfile module.
            Defaulted as the default 16-bit PCM for WAV audio files.

        """
        ads = AudioDataset.from_files(
            files=list(self.origin_files),
            begin=begin,
            end=end,
            data_duration=data_duration,
        )

        if sample_rate is not None:
            ads.sample_rate = sample_rate

        self._add_audio_dataset(ads=ads, name=name, subtype=subtype)

    def _add_audio_dataset(
        self,
        ads: AudioDataset,
        name: str | None = None,
        subtype: str | None = None,
    ) -> None:
        ads_folder = self._get_audio_dataset_subpath(ads=ads, name=name)
        ads.write(ads_folder, link=True, subtype=subtype)

        dataset_name = str(ads) if name is None else name
        self.datasets[dataset_name] = {"class": type(ads).__name__, "dataset": ads}

        ads.write_json(folder=ads.folder, name=dataset_name)
        self.write_json()

    def _get_audio_dataset_subpath(
        self,
        ads: AudioDataset,
        name: str | None = None,
    ) -> Path:
        return (
            self.folder
            / "data"
            / "audio"
            / (
                f"{round(ads.data_duration.total_seconds())}_{round(ads.sample_rate)}"
                if name is None
                else name
            )
        )

    def generate_spectra(
        self,
        fft: ShortTimeFFT,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        data_duration: Timedelta | None = None,
        sample_rate: float | None = None,
        name: str | None = None,
        export: SpectroOutput = SpectroOutput.SPECTROGRAM,
    ) -> None:
        ads = AudioDataset.from_files(
            files=list(self.origin_files),
            begin=begin,
            end=end,
            data_duration=data_duration,
        )
        if sample_rate is not None:
            ads.sample_rate = sample_rate

        if SpectroOutput.AUDIO in export:
            self._add_audio_dataset(ads=ads, name=f"{name}_audio")

        sds = SpectroDataset.from_audio_dataset(audio_dataset=ads, fft=fft)
        self._add_spectro_dataset(sds=sds, name=name, export=export)

    def _add_spectro_dataset(
        self,
        sds: SpectroDataset,
        export: SpectroOutput,
        name: str | None = None,
    ) -> None:
        sds.folder = self._get_spectro_dataset_subpath(sds=sds, name=name)

        if SpectroOutput.MATRIX in export and SpectroOutput.SPECTROGRAM in export:
            sds.save_all(
                matrix_folder=sds.folder / "welch",
                spectrogram_folder=sds.folder / "spectrogram",
                link=True,
            )
        elif SpectroOutput.SPECTROGRAM in export:
            sds.save_spectrogram(
                folder=sds.folder / "spectrogram",
            )
        elif SpectroOutput.MATRIX in export:
            sds.write(
                folder=sds.folder / "welch",
                link=True,
            )

        sds_name = str(sds) if name is None else name
        self.datasets[sds_name] = {"class": type(sds).__name__, "dataset": sds}

        sds.write_json(folder=sds.folder, name=sds_name)
        self.write_json()

    def _get_spectro_dataset_subpath(
        self,
        sds: SpectroDataset,
        name: str | None = None,
    ) -> Path:
        ads_folder = Path(
            f"{round(sds.data_duration.total_seconds())}_{round(sds.fft.fs)}",
        )
        fft_folder = f"{sds.fft.mfft}_{sds.fft.win.shape[0]}_{sds.fft.hop}_linear"
        return (
            self.folder
            / "processed"
            / (ads_folder / fft_folder if name is None else name)
        )

    def _sort_dataset(self, dataset: type[DatasetChild]) -> None:
        if type(dataset) is AudioDataset:
            self._sort_audio_dataset(dataset)
            return
        if type(dataset) is SpectroDataset:
            self._sort_spectro_dataset(dataset)
            return

    def _sort_audio_dataset(self, dataset: AudioDataset) -> None:
        dataset.folder = self._get_audio_dataset_subpath(dataset, name="original")

    def _sort_spectro_dataset(self, dataset: SpectroDataset) -> None:
        raise NotImplementedError

    def get_dataset(self, dataset_name: str) -> type[DatasetChild] | None:
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
