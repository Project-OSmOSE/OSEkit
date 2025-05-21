"""Main class of the Public API.

The Dataset correspond to a collection of audio,
spectro and auxilary core_api datasets.
It has additionnal metadata that can be exported, e.g. to APLOSE.

"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from OSmOSE.config import resample_quality_settings
from OSmOSE.core_api.audio_dataset import AudioDataset
from OSmOSE.core_api.base_dataset import BaseDataset
from OSmOSE.core_api.instrument import Instrument
from OSmOSE.core_api.json_serializer import deserialize_json, serialize_json
from OSmOSE.core_api.spectro_data import SpectroData
from OSmOSE.core_api.spectro_dataset import SpectroDataset
from OSmOSE.public_api.analysis import Analysis, AnalysisType
from OSmOSE.utils.core_utils import (
    file_indexes_per_batch,
    get_umask,
)
from OSmOSE.utils.path_utils import move_tree

if TYPE_CHECKING:

    from OSmOSE.core_api.audio_file import AudioFile
    from OSmOSE.job import Job_builder


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
        job_builder: Job_builder | None = None,
        instrument: Instrument | None = None,
    ) -> None:
        """Initialize a Dataset."""
        self.folder = folder
        self.strptime_format = strptime_format
        self.gps_coordinates = gps_coordinates
        self.depth = depth
        self.timezone = timezone
        self.datasets = datasets if datasets is not None else {}
        self.job_builder = job_builder
        self.instrument = instrument

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
            instrument=self.instrument,
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
        self.get_dataset("original").move_files(self.folder)

        if self.folder / "other" in files_to_remove:
            move_tree(self.folder / "other", self.folder)

        for file in files_to_remove:
            if file.is_dir():
                shutil.rmtree(file)
            else:
                file.unlink()

        self.datasets = {}

    def sample_spectra(
        self,
        analysis: Analysis,
        nb_spectra: int = 1,
    ) -> list[SpectroData]:
        """Return a list of sample SpectroData.

        These SpectroData can be plotted to check the validity of the
        parameters before running a full analysis.

        Parameters
        ----------
        analysis: Analysis
            Analysis for which to generate sample SpectroData objects.
            See the public_api.Analysis.Analysis docstring for more info.
        nb_spectra: int
            The number of sample SpectroData to return.

        Returns
        -------
        list[SpectroData]:
            List of nb_spectra sample SpectroData objects.

        """
        ads = AudioDataset.from_files(
            files=list(self.origin_files),
            begin=analysis.begin,
            end=analysis.end,
            data_duration=analysis.data_duration,
            instrument=self.instrument,
        )
        ads.sample_rate = analysis.sample_rate
        ads = random.sample(ads.data, nb_spectra)
        return [
            SpectroData.from_audio_data(data=ad, fft=analysis.fft, v_lim=analysis.v_lim)
            for ad in ads
        ]

    def run_analysis(
        self,
        analysis: Analysis,
    ) -> None:
        """Create a new analysis dataset from the original audio files.

        The analysis parameter sets which type(s) of core_api dataset(s) will be
        created and added to the Dataset.datasets property, plus which output
        files will be written to disk (reshaped audio files, npz spectra matrices,
        png spectrograms...).

        Parameters
        ----------
        analysis: Analysis
            Analysis to run.
            Contains the analysis type and required info.
            See the public_api.Analysis.Analysis docstring for more info.

        """
        ads = AudioDataset.from_files(
            files=list(self.origin_files),
            begin=analysis.begin,
            end=analysis.end,
            data_duration=analysis.data_duration,
            name=analysis.name,
            instrument=self.instrument,
        )

        if analysis.sample_rate is not None:
            ads.sample_rate = analysis.sample_rate

        if AnalysisType.AUDIO in analysis.analysis_type:
            if analysis.is_spectro:
                ads.suffix = "audio"
            self._add_audio_dataset(ads=ads)

        sds = None
        if analysis.is_spectro:
            sds = SpectroDataset.from_audio_dataset(
                audio_dataset=ads,
                fft=analysis.fft,
                name=analysis.name,
                v_lim=analysis.v_lim,
            )
            self._add_spectro_dataset(sds=sds)

        self.export_analysis(
            analysis_type=analysis.analysis_type,
            ads=ads,
            sds=sds,
            link=True,
            subtype=analysis.subtype,
        )

        self.write_json()

    def _add_audio_dataset(
        self,
        ads: AudioDataset,
    ) -> None:
        ads.folder = self._get_audio_dataset_subpath(ads=ads)
        self.datasets[ads.name] = {"class": type(ads).__name__, "dataset": ads}
        ads.write_json(ads.folder)

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

    def export_analysis(
        self,
        analysis_type: AnalysisType,
        ads: AudioDataset | None = None,
        sds: SpectroDataset | None = None,
        link: bool = False,
        subtype: str | None = None,
        matrix_folder_name: str = "welches",
        spectrogram_folder_name: str = "spectrogram",
    ) -> None:
        """Perform an analysis and write the results on disk.

        An analysis is defined as a manipulation of the original audio files:
        reshaping the audio, exporting spectrograms or npz matrices (or a mix of
        those three) are examples of analyses.
        The tasks will be distributed to jobs if self.job_builder
        is not None, else it will be distributed on self.job_builder.nb_jobs jobs.

        Parameters
        ----------
        spectrogram_folder_name
            The name of the folder in which the png spectrograms will be
            exported (relative to sds.folder)
        matrix_folder_name:
            The name of the folder in which the npz matrices will be
            exported (relative to sds.folder)
        sds: SpectroDataset
            The SpectroDataset on which the data should be written.
        analysis_type : AnalysisType
            Type of the analysis to be performed.
            AudioDataset and SpectroDataset instances will be
            created depending on the flags.
            See OSmOSE.public_api.analysis.AnalysisType docstring for more information.
        ads: AudioDataset
            The AudioDataset on which the data should be written.
        link: bool
            If set to True, the ads data will be linked to the exported files.
        subtype: str | None
            The subtype of the audio files as provided by the soundfile module.

        """
        # Import here to avoid circular imports since the script needs to import Dataset
        from OSmOSE.public_api import export_analysis

        if self.job_builder is None:
            export_analysis.write_analysis(
                analysis_type=analysis_type,
                ads=ads,
                sds=sds,
                link=link,
                subtype=subtype,
                matrix_folder_name=matrix_folder_name,
                spectrogram_folder_name=spectrogram_folder_name,
            )
            return

        batch_indexes = file_indexes_per_batch(
            total_nb_files=len(ads.data),
            nb_batches=self.job_builder.nb_jobs,
        )

        for start, stop in batch_indexes:
            self.job_builder.build_job_file(
                script_path=export_analysis.__file__,
                script_args=f"--dataset-json-path {self.folder / 'dataset.json'} "
                f"--analysis {sum(v.value for v in list(analysis_type))} "
                f"--ads-name {ads.name if ads is not None else ''} "
                f"--sds-name {sds.name if sds is not None else ''} "
                f"--subtype {subtype} "
                f"--matrix-folder-name {matrix_folder_name} "
                f"--spectrogram-folder-name {spectrogram_folder_name} "
                f"--first {start} "
                f"--last {stop} "
                f"--downsampling-quality {resample_quality_settings['downsample']} "
                f"--upsampling-quality {resample_quality_settings['upsample']} "
                f"--umask {get_umask()} ",
                jobname="OSmOSE_Analysis",
                preset="low",
                env_name=sys.executable.replace("/bin/python", ""),
                mem="32G",
                walltime="01:00:00",
                logdir=self.folder / "log",
            )
        self.job_builder.submit_job()

    def _add_spectro_dataset(
        self,
        sds: SpectroDataset,
    ) -> None:
        sds.folder = self._get_spectro_dataset_subpath(sds=sds)
        self.datasets[sds.name] = {"class": type(sds).__name__, "dataset": sds}
        sds.write_json(sds.folder)

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
            / (ads_folder / fft_folder if sds.has_default_name else sds.name)
        )

    def _sort_dataset(self, dataset: type[DatasetChild]) -> None:
        if type(dataset) is AudioDataset:
            self._sort_audio_dataset(dataset)
            return
        if type(dataset) is SpectroDataset:
            self._sort_spectro_dataset(dataset)
            return

    def _sort_audio_dataset(self, dataset: AudioDataset) -> None:
        dataset.move_files(self._get_audio_dataset_subpath(dataset))

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
            "instrument": (
                None if self.instrument is None else self.instrument.to_dict()
            ),
            "depth": self.depth,
            "folder": str(self.folder),
            "gps_coordinates": self.gps_coordinates,
            "strptime_format": self.strptime_format,
            "timezone": self.timezone,
        }

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
            instrument=Instrument.from_dict(dictionary["instrument"]),
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
