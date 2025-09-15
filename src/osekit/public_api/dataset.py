"""Main class of the Public API.

The Dataset correspond to a collection of audio,
spectro and auxilary core_api datasets.
It has additionnal metadata that can be exported, e.g. to APLOSE.

"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from osekit import config
from osekit.config import DPDEFAULT, resample_quality_settings
from osekit.core_api import audio_file_manager as afm
from osekit.core_api.audio_dataset import AudioDataset
from osekit.core_api.base_dataset import BaseDataset
from osekit.core_api.instrument import Instrument
from osekit.core_api.json_serializer import deserialize_json, serialize_json
from osekit.core_api.ltas_dataset import LTASDataset
from osekit.core_api.spectro_dataset import SpectroDataset
from osekit.public_api.analysis import Analysis, AnalysisType
from osekit.utils.core_utils import (
    file_indexes_per_batch,
    get_umask,
)
from osekit.utils.path_utils import move_tree

if TYPE_CHECKING:
    from osekit.core_api.audio_file import AudioFile
    from osekit.job import Job_builder


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
        depth: float = 0.0,
        timezone: str | None = None,
        datasets: dict | None = None,
        job_builder: Job_builder | None = None,
        instrument: Instrument | None = None,
    ) -> None:
        """Initialize a Dataset.

        Parameters
        ----------
        folder: Path
            Path to the folder containing the original audio files.
        strptime_format: str
            The strptime format used in the filenames.
            It should use valid strftime codes (https://strftime.org/).
        gps_coordinates: str | list | tuple
            GPS coordinates of the location were audio files were recorded.
        depth: float
            Depth at which the audio files were recorded.
        timezone: str | None
            Timezone in which the audio data will be located.
            If the audio file timestamps are parsed with a tz-aware strptime_format
            (%z or %Z code), the AudioFiles will be converted from the parsed timezone
            to the specified timezone.
        datasets: dict | None
            Core API datasets that already belong to this dataset.
            Mainly used for deserialization.
        job_builder: Job_builder | None
            If None, analyses from this Dataset will be run locally.
            Otherwise, PBS job files will be created and submitted when
            analyses are run.
            See the osekit.job module for more info.
        instrument: Instrument | None
            Instrument that might be used to obtain acoustic pressure from
            the wav audio data.
            See the osekit.core_api.instrument module for more info.

        """
        self.folder = folder
        self.strptime_format = strptime_format
        self.gps_coordinates = gps_coordinates
        self.depth = depth
        self.timezone = timezone
        self.datasets = datasets if datasets is not None else {}
        self.job_builder = job_builder
        self.instrument = instrument

    @property
    def origin_files(self) -> list[AudioFile] | None:
        """Return the original audio files from which this Dataset has been built."""
        return (
            None
            if self.origin_dataset is None
            else sorted(self.origin_dataset.files, key=lambda f: f.begin)
        )

    @property
    def origin_dataset(self) -> AudioDataset:
        """Return the AudioDataset from which this Dataset has been built."""
        return self.get_dataset("original")

    @property
    def analyses(self) -> list[str]:
        """Return the list of the names of the analyses ran with this Dataset."""
        return list({dataset["analysis"] for dataset in self.datasets.values()})

    def build(self) -> None:
        """Build the Dataset.

        Building a dataset moves the original audio files to a specific folder
        and creates metadata csv used by APLOSE.

        """
        self._create_logger()

        self.logger.info("Building the dataset...")

        self.logger.info("Analyzing original audio files...")
        ads = AudioDataset.from_folder(
            self.folder,
            strptime_format=self.strptime_format,
            mode="files",
            timezone=self.timezone,
            name="original",
            instrument=self.instrument,
        )
        self.datasets[ads.name] = {
            "class": type(ads).__name__,
            "analysis": "original",
            "dataset": ads,
        }

        self.logger.info("Organizing dataset folder...")
        move_tree(
            source=self.folder,
            destination=self.folder / "other",
            excluded_paths={file.path for file in ads.files}
            | set(
                (self.folder / "log").iterdir()
                if (self.folder / "log").exists()
                else ()
            )
            | {self.folder / "log"},
        )
        self._sort_dataset(ads)
        ads.write_json(ads.folder)
        self.write_json()

        self.logger.info("Build done!")

    def _create_logger(self) -> None:
        if not logging.getLogger("dataset").handlers:
            message = (
                "Logging has not been configured. "
                "The dataset will use the root logger. "
                "Use osekit.setup_logging() if wanted."
            )
            logging.warning(message)
            self.logger = logging.getLogger()
            return

        logs_directory = self.folder / "log"
        if not logs_directory.exists():
            logs_directory.mkdir(mode=DPDEFAULT)
        self.logger = logging.getLogger("dataset").getChild(self.folder.name)
        file_handler = logging.FileHandler(logs_directory / "logs.log", mode="w")
        file_handler.setFormatter(
            logging.getLogger("dataset").handlers[0].formatter,
        )
        self.logger.setLevel(logging.DEBUG)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

    def reset(self) -> None:
        """Reset the Dataset.

        Resetting a dataset will move back the original audio files and the content of
        the "other" folder to the root folder.
        WARNING: all other files and folders will be deleted.
        """
        afm.close()

        files_to_remove = list(self.folder.iterdir())
        self.get_dataset("original").move_files(self.folder)

        if self.folder / "other" in files_to_remove:
            move_tree(self.folder / "other", self.folder)

        self.logger.handlers.clear()

        for file in files_to_remove:
            if file.is_dir():
                shutil.rmtree(file)
            else:
                file.unlink()

        self.datasets = {}

    def get_analysis_audiodataset(self, analysis: Analysis) -> AudioDataset:
        """Return an AudioDataset created from the analysis parameters.

        Parameters
        ----------
        analysis: Analysis
            Analysis for which to generate an AudioDataset object.

        Returns
        -------
        AudioDataset:
            The AudioDataset that match the analysis parameters.
            This AudioDataset can be used either to have a peek at the
            analysis output, or to edit the analysis (adding/removing data)
            by editing it and passing it as a parameter to the
            Dataset.run_analysis() method.

        """
        self.logger.info("Creating the audio data...")

        ads = AudioDataset.from_files(
            files=list(self.origin_files),
            begin=analysis.begin,
            end=analysis.end,
            data_duration=analysis.data_duration,
            mode=analysis.mode,
            name=analysis.name,
            instrument=self.instrument,
        )

        if analysis.sample_rate is not None:
            ads.sample_rate = analysis.sample_rate

        if analysis.is_spectro:
            ads.suffix = "audio"

        return ads

    def get_analysis_spectrodataset(
        self,
        analysis: Analysis,
        audio_dataset: AudioDataset | None = None,
    ) -> SpectroDataset | LTASDataset:
        """Return a SpectroDataset (or LTASDataset) created from analysis parameters.

        Parameters
        ----------
        analysis: Analysis
            Analysis for which to generate an AudioDataset object.
        audio_dataset: AudioDataset|None
            If provided, the SpectroDataset will be initialized from this AudioDataset.
             This can be used to edit the analysis (e.g. adding/removing data)
            before running it.

        Returns
        -------
        SpectroDataset | LTASDataset:
            The SpectroDataset that match the analysis parameters.
            This SpectroDataset can be used, for example, to have a peek at the
            analysis output before running it.
            If Analysis.is_ltas is True, a LTASDataset is returned.

        """
        if analysis.fft is None:
            msg = "FFT parameter should be given if spectra outputs are selected."
            raise ValueError(msg)

        ads = (
            self.get_analysis_audiodataset(analysis=analysis)
            if audio_dataset is None
            else audio_dataset
        )

        sds = SpectroDataset.from_audio_dataset(
            audio_dataset=ads,
            fft=analysis.fft,
            name=ads.base_name,
            v_lim=analysis.v_lim,
            colormap=analysis.colormap,
            scale=analysis.scale,
        )

        if analysis.nb_ltas_time_bins is not None:
            sds = LTASDataset.from_spectro_dataset(
                sds=sds,
                nb_time_bins=analysis.nb_ltas_time_bins,
            )

        return sds

    def run_analysis(
        self,
        analysis: Analysis,
        audio_dataset: AudioDataset | None = None,
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
        audio_dataset: AudioDataset
            If provided, the analysis will be run on this AudioDataset.
            Else, an AudioDataset will be created from the analysis parameters.
            This can be used to edit the analysis AudioDataset (adding/removing
            AudioData etc.)

        """
        if analysis.name in self.analyses:
            message = (
                f"Analysis {analysis.name} already exists."
                f"Please choose a different name,"
                f"or delete it with the Dataset.delete_analysis() method."
            )
            raise ValueError(message)

        ads = (
            self.get_analysis_audiodataset(analysis=analysis)
            if audio_dataset is None
            else audio_dataset
        )

        if AnalysisType.AUDIO in analysis.analysis_type:
            self._add_audio_dataset(ads=ads, analysis_name=analysis.name)

        sds = None
        if analysis.is_spectro:
            sds = self.get_analysis_spectrodataset(
                analysis=analysis,
                audio_dataset=ads,
            )
            self._add_spectro_dataset(sds=sds, analysis_name=analysis.name)

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
        analysis_name: str,
    ) -> None:
        ads.folder = self._get_audio_dataset_subpath(ads=ads)
        self.datasets[ads.name] = {
            "class": type(ads).__name__,
            "analysis": analysis_name,
            "dataset": ads,
        }
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
        sds: SpectroDataset | LTASDataset | None = None,
        link: bool = False,
        subtype: str | None = None,
        matrix_folder_name: str = "matrix",
        spectrogram_folder_name: str = "spectrogram",
        welch_folder_name: str = "welch",
    ) -> None:
        """Perform an analysis and write the results on disk.

        An analysis is defined as a manipulation of the original audio files:
        reshaping the audio, exporting spectrograms or npz matrices (or a mix of
        those three) are examples of analyses.
        The tasks will be distributed to jobs if self.job_builder
        is not None, else it will be distributed on self.job_builder.nb_jobs jobs.

        Parameters
        ----------
        spectrogram_folder_name:
            The name of the folder in which the png spectrograms will be
            exported (relative to sds.folder)
        matrix_folder_name:
            The name of the folder in which the npz matrices will be
            exported (relative to sds.folder)
        welch_folder_name:
            The name of the folder in which the npz welch files will be
            exported (relative to sds.folder)
        sds: SpectroDataset | LTASDataset
            The SpectroDataset on which the data should be written.
        analysis_type : AnalysisType
            Type of the analysis to be performed.
            AudioDataset and SpectroDataset instances will be
            created depending on the flags.
            See osekit.public_api.analysis.AnalysisType docstring for more information.
        ads: AudioDataset
            The AudioDataset on which the data should be written.
        link: bool
            If set to True, the ads data will be linked to the exported files.
        subtype: str | None
            The subtype of the audio files as provided by the soundfile module.

        """
        # Import here to avoid circular imports since the script needs to import Dataset
        from osekit.public_api import export_analysis  # noqa: PLC0415

        if self.job_builder is None:
            export_analysis.write_analysis(
                analysis_type=analysis_type,
                ads=ads,
                sds=sds,
                link=link,
                subtype=subtype,
                matrix_folder_name=matrix_folder_name,
                spectrogram_folder_name=spectrogram_folder_name,
                welch_folder_name=welch_folder_name,
                logger=self.logger,
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
                f"--analysis {analysis_type.value} "
                f"--ads-name {ads.name if ads is not None else 'None'} "
                f"--sds-name {sds.name if sds is not None else 'None'} "
                f"--subtype {subtype} "
                f"--matrix-folder-name {matrix_folder_name} "
                f"--spectrogram-folder-name {spectrogram_folder_name} "
                f"--welch-folder-name {welch_folder_name} "
                f"--first {start} "
                f"--last {stop} "
                f"--downsampling-quality {resample_quality_settings['downsample']} "
                f"--upsampling-quality {resample_quality_settings['upsample']} "
                f"--umask {get_umask()} "
                f"--multiprocessing {config.multiprocessing['is_active']} "
                f"--nb-processes {config.multiprocessing['nb_processes']} ",
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
        sds: SpectroDataset | LTASDataset,
        analysis_name: str,
    ) -> None:
        sds.folder = self._get_spectro_dataset_subpath(sds=sds)
        self.datasets[sds.name] = {
            "class": type(sds).__name__,
            "dataset": sds,
            "analysis": analysis_name,
        }
        sds.write_json(sds.folder)

    def _get_spectro_dataset_subpath(
        self,
        sds: SpectroDataset | LTASDataset,
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
        if type(dataset) is SpectroDataset | LTASDataset:
            self._sort_spectro_dataset(dataset)
            return

    def _sort_audio_dataset(self, dataset: AudioDataset) -> None:
        dataset.move_files(self._get_audio_dataset_subpath(dataset))

    def _sort_spectro_dataset(self, dataset: SpectroDataset | LTASDataset) -> None:
        raise NotImplementedError

    def _delete_dataset(self, dataset_name: str) -> None:
        """Delete an analysis dataset.

        WARNING: all the analysis output files will be deleted.
        WARNING: removing linked datasets (e.g. an AudioDataset to which a SpectroDataset is
        linked) might lead to errors.

        Parameters
        ----------
        dataset_name: str
            Name of the dataset to remove.

        """
        dataset_to_remove = self.get_dataset(dataset_name)
        if dataset_to_remove is None:
            return
        self.datasets.pop(dataset_to_remove.name)

        afm.close()
        shutil.rmtree(str(dataset_to_remove.folder))
        self.write_json()

    def get_datasets_by_analysis(self, analysis_name: str) -> list[type[DatasetChild]]:
        """Get all output datasets from a given analysis.

        Parameters
        ----------
        analysis_name: str
            Name of the analysis of which to get the output datasets.

        Returns
        -------
        list[type[DatasetChild]]
        List of the analysis output datasets.

        """
        return [
            dataset["dataset"]
            for dataset in self.datasets.values()
            if dataset["analysis"] == analysis_name
        ]

    def rename_analysis(self, analysis_name: str, new_analysis_name: str) -> None:
        """Rename an already ran analysis.

        Parameters
        ----------
        analysis_name: str
            Name of the analysis to rename.
        new_analysis_name: str
            New name of the analysis to rename.

        """
        if analysis_name == "original":
            raise ValueError("You can't rename the original dataset.")
        if analysis_name not in self.datasets:
            raise ValueError(f"Unknown analysis {analysis_name}.")
        if analysis_name == new_analysis_name:
            return

        keys_to_rename = {}
        for analysis_dataset in self.datasets.values():
            if analysis_dataset["analysis"] == analysis_name:
                analysis_dataset["analysis"] = new_analysis_name
                ds = analysis_dataset["dataset"]
                old_name, new_name = (
                    ds.name,
                    new_analysis_name + (f"_{ds.suffix}" if ds.suffix else ""),
                )
                ds.base_name = new_analysis_name
                old_folder = ds.folder
                new_folder = ds.folder.parent / new_name
                keys_to_rename[old_name] = new_name

                ds.move_files(new_folder)
                move_tree(
                    old_folder, new_folder, excluded_paths=old_folder.glob("*.json")
                )  # Moves exported files
                shutil.rmtree(str(old_folder))
                ds.write_json(ds.folder)

        for old_name, new_name in keys_to_rename.items():
            self.datasets[new_name] = self.datasets.pop(old_name)

        self.write_json()

    def delete_analysis(self, analysis_name: str) -> None:
        """Delete all output datasets from an analysis.

        WARNING: all the analysis output files will be deleted.

        """
        for dataset_to_delete in self.get_datasets_by_analysis(analysis_name):
            self._delete_dataset(dataset_to_delete.name)

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
            message = f"Dataset '{dataset_name}' not found."
            self.logger.warning(message)
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
                    "analysis": dataset["analysis"],
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
                else SpectroDataset
                if dataset["class"] == "SpectroDataset"
                else LTASDataset
                if dataset["class"] == "LTASDataset"
                else BaseDataset
            )
            datasets[name] = {
                "class": dataset["class"],
                "analysis": dataset["analysis"],
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
        instance = cls.from_dict(deserialize_json(file))
        instance._create_logger()  # noqa: SLF001
        return instance


DatasetChild = TypeVar("DatasetChild", bound=BaseDataset)
