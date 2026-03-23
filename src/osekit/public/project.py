"""Main class of the Public API.

The ``Project`` is the class that stores the original audio dataset,
and from which transforms are ran from this dataset to generate spectro
datasets, reshaped audio datasets, etc.
It has additional metadata that can be exported, e.g. to APLOSE.

"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from osekit import config
from osekit.config import DPDEFAULT, resample_quality_settings
from osekit.core import audio_file_manager as afm
from osekit.core.audio_dataset import AudioDataset
from osekit.core.base_dataset import BaseDataset
from osekit.core.instrument import Instrument
from osekit.core.json_serializer import deserialize_json, serialize_json
from osekit.core.ltas_dataset import LTASDataset
from osekit.core.spectro_dataset import SpectroDataset
from osekit.public.transform import OutputType, Transform
from osekit.utils.core_utils import (
    file_indexes_per_batch,
    get_umask,
)
from osekit.utils.path_utils import move_tree

if TYPE_CHECKING:
    from collections.abc import Iterable
    from os import PathLike

    from pandas import Timestamp

    from osekit.core.audio_file import AudioFile
    from osekit.utils.job import JobBuilder


class Project:
    """Main class of the Public API.

    The ``Project`` is the class that stores the original audio dataset,
    and from which transforms are ran from this dataset to generate spectro
    datasets, reshaped audio datasets, etc.
    It has additionnal metadata that can be exported, e.g. to APLOSE.

    """

    def __init__(  # noqa: PLR0913
        self,
        folder: Path,
        strptime_format: str | None,
        gps_coordinates: str | list | tuple = (0, 0),
        depth: float = 0.0,
        timezone: str | None = None,
        output_datasets: dict | None = None,
        job_builder: JobBuilder | None = None,
        instrument: Instrument | None = None,
        first_file_begin: Timestamp | None = None,
    ) -> None:
        """Initialize a ``Project``.

        Parameters
        ----------
        folder: Path
            Path to the folder containing the original audio files.
        strptime_format: str | None
            The strptime format used in the filenames.
            It should use valid strftime codes (https://strftime.org/).
            If ``None``, the first audio file of the folder will start
            at ``first_file_begin``, and each following file will start
            at the end of the previous one.
        gps_coordinates: str | list | tuple
            GPS coordinates of the location were audio files were recorded.
        depth: float
            Depth at which the audio files were recorded.
        timezone: str | None
            Timezone in which the audio data will be located.
            If the audio file timestamps are parsed with a tz-aware strptime_format
            (``%z`` or ``%Z`` code), the ``AudioFiles`` will be converted from
            the parsed timezone to the specified timezone.
        output_datasets: dict | None
            Core API Datasets that have been exported in this project.
            Mainly used for deserialization.
        job_builder: Job_builder | None
            If ``None``, outputs from this ``Project`` will be run locally.
            Otherwise, PBS job files will be created and submitted when
            transforms are run.
            See the ``osekit.job`` module for more info.
        instrument: Instrument | None
            Instrument that might be used to obtain acoustic pressure from
            the ``wav`` audio data.
            See the ``osekit.core.instrument`` module for more info.
        first_file_begin: Timestamp | None
            Timestamp of the first audio file being processed.
            Will be ignored if ``striptime_format`` is specified.

        """
        self.folder = folder
        self.strptime_format = strptime_format
        self.gps_coordinates = gps_coordinates
        self.depth = depth
        self.timezone = timezone
        self.output_datasets = output_datasets if output_datasets is not None else {}
        self.job_builder = job_builder
        self.instrument = instrument
        self.first_file_begin = first_file_begin
        self.logger = None

    @property
    def origin_files(self) -> list[AudioFile] | None:
        """Return the original audio files from which this ``Project`` has been built."""
        return (
            None
            if self.origin_dataset is None
            else sorted(self.origin_dataset.files, key=lambda f: f.begin)
        )

    @property
    def origin_dataset(self) -> AudioDataset:
        """Return the ``AudioDataset`` from which this ``Project`` has been built."""
        return self.deserialize_output_dataset("original")

    @property
    def transforms(self) -> list[str]:
        """Return the list of the names of the transforms ran with this ``Project``."""
        return list({dataset["transform"] for dataset in self.output_datasets.values()})

    def build(
        self,
    ) -> None:
        """Build the ``Project``.

        Building a ``Project`` moves the original audio files to a specific folder
        and creates serialized ``json`` files used by APLOSE.

        """
        self._create_logger()

        self.logger.info("Building the project...")

        self.logger.info("Analyzing original audio files...")
        ads = AudioDataset.from_folder(
            self.folder,
            strptime_format=self.strptime_format,
            first_file_begin=self.first_file_begin,
            mode="files",
            timezone=self.timezone,
            name="original",
            instrument=self.instrument,
        )

        self.output_datasets[ads.name] = {
            "class": type(ads).__name__,
            "transform": "original",
            "dataset": ads,
        }

        self.logger.info("Organizing project folder...")
        afm.close()
        move_tree(
            source=self.folder,
            destination=self.folder / "other",
            excluded_paths={file.path for file in ads.files}
            | set(
                (self.folder / "log").iterdir()
                if (self.folder / "log").exists()
                else (),
            )
            | {self.folder / "log"},
        )
        self._sort_dataset(ads)
        ads.write_json(ads.folder)
        self.write_json()

        self.logger.info("Build done!")

    def build_from_files(
        self,
        files: Iterable[PathLike | str],
        *,
        move_files: bool = False,
    ) -> None:
        """Build the ``Project`` from the specified files.

        The files will be copied (or moved) to the ``project.folder`` folder.

        Parameters
        ----------
        files: Iterable[PathLike|str]
            Files that are included in the project.
        move_files: bool
            If set to ``True``, the files will be moved (rather than copied) in
            the project folder.

        """
        self._create_logger()

        msg = f"{'Moving' if move_files else 'Copying'} files to the project folder."
        self.logger.info(msg)

        if not self.folder.exists():
            self.folder.mkdir(mode=DPDEFAULT)

        for file in map(Path, files):
            destination = self.folder / file.name
            if move_files:
                file.replace(destination)
            else:
                shutil.copyfile(file, destination)

        self.build()

    def _create_logger(self) -> None:
        if self.logger:
            return
        if not logging.getLogger("project").handlers:
            message = (
                "Logging has not been configured. "
                "The project will use the root logger. "
                "Use osekit.setup_logging() if wanted."
            )
            logging.warning(message)
            self.logger = logging.getLogger()
            return

        logs_directory = self.folder / "log"
        if not logs_directory.exists():
            logs_directory.mkdir(mode=DPDEFAULT, parents=True)
        self.logger = logging.getLogger("project").getChild(self.folder.name)
        file_handler = logging.FileHandler(logs_directory / "logs.log", mode="w")
        file_handler.setFormatter(
            logging.getLogger("project").handlers[0].formatter,
        )
        self.logger.setLevel(logging.DEBUG)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

    def reset(self) -> None:
        """Reset the ``Project``.

        Resetting a project will move back the original audio files and the content of
        the ``other`` folder to the root folder.
        WARNING: all other files and folders will be deleted.
        """
        afm.close()

        files_to_remove = list(self.folder.iterdir())
        self.get_output("original").move_files(self.folder)

        if self.folder / "other" in files_to_remove:
            move_tree(self.folder / "other", self.folder)

        self.logger.handlers.clear()

        for file in files_to_remove:
            if file.is_dir():
                shutil.rmtree(file)
            else:
                file.unlink()

        self.output_datasets = {}
        self.logger = None

    def prepare_audio(self, transform: Transform) -> AudioDataset:
        """Return an ``AudioDataset`` created from the transform parameters.

        Parameters
        ----------
        transform: Transform
            ``Transform`` for which to generate an ``AudioDataset`` object.

        Returns
        -------
        AudioDataset:
            The ``AudioDataset`` that match the transform parameters.
            This ``AudioDataset`` can be used either to have a peek at the
            transform output, or to edit the transform (adding/removing data)
            by editing it and passing it as a parameter to the
            ``Project.run()`` method.

        """
        self.logger.info("Creating the audio data...")

        ads = AudioDataset.from_files(
            files=list(self.origin_files),
            begin=transform.begin,
            end=transform.end,
            data_duration=transform.data_duration,
            mode=transform.mode,
            overlap=transform.overlap,
            normalization=transform.normalization,
            name=transform.name,
            instrument=self.instrument,
        )

        if transform.sample_rate is not None:
            ads.sample_rate = transform.sample_rate

        if transform.is_spectro:
            ads.suffix = "audio"

        return ads

    def prepare_spectro(
        self,
        transform: Transform,
        audio_dataset: AudioDataset | None = None,
    ) -> SpectroDataset | LTASDataset:
        """Return a ``SpectroDataset`` (or ``LTASDataset``) created from transform parameters.

        Parameters
        ----------
        transform: Transform
            ``Transform`` for which to generate an ``AudioDataset`` object.
        audio_dataset: AudioDataset|None
            If provided, the ``SpectroDataset`` will be initialized from
            this ``AudioDataset``.
            This can be used to edit the transform (e.g. adding/removing data)
            before running it.

        Returns
        -------
        SpectroDataset | LTASDataset:
            The ``SpectroDataset`` that match the transform parameters.
            This ``SpectroDataset`` can be used, for example, to have a peek at the
            transform output before running it.
            If ``Transform.is_ltas is True``, a ``LTASDataset`` is returned.

        """
        if transform.fft is None:
            msg = "FFT parameter should be given if spectra outputs are selected."
            raise ValueError(msg)

        ads = (
            self.prepare_audio(transform=transform)
            if audio_dataset is None
            else audio_dataset
        )

        sds = SpectroDataset.from_audio_dataset(
            audio_dataset=ads,
            fft=transform.fft,
            name=transform.name,
            v_lim=transform.v_lim,
            colormap=transform.colormap,
            scale=transform.scale,
        )

        if transform.nb_ltas_time_bins is not None:
            sds = LTASDataset.from_spectro_dataset(
                sds=sds,
                nb_time_bins=transform.nb_ltas_time_bins,
            )

        return sds

    def run(
        self,
        transform: Transform,
        audio_dataset: AudioDataset | None = None,
        spectro_dataset: SpectroDataset | None = None,
        nb_jobs: int = 1,
    ) -> None:
        """Create a new transform dataset from the original audio files.

        The transform parameter sets which type(s) of ``core`` dataset(s) will be
        created and added to the ``Project.output_datasets`` property, plus which output
        files will be written to disk (reshaped audio files, ``npz`` spectra matrices,
        ``png`` spectrograms...).

        Parameters
        ----------
        transform: Transform
            ``Transform`` to run.
            Contains the transform type and required info.
            See the ``public.transform.Transform`` docstring for more info.
        audio_dataset: AudioDataset
            If provided, the transform will be run on this ``AudioDataset``.
            Else, an ``AudioDataset`` will be created from the transform parameters.
            This can be used to edit the transform ``AudioDataset`` (adding, removing,
            renaming ``AudioData`` etc.)
        spectro_dataset: SpectroDataset
            If provided, the spectral transform will be run on this ``SpectroDataset``.
            Else, a ``SpectroDataset`` will be created from the ``audio_dataset``
            if provided, or from the transform parameters.
            This can be used to edit the transform ``SpectroDataset`` (adding, removing,
            renaming ``SpectroData`` etc.)
        nb_jobs: int
            Number of jobs to run in parallel.

        """
        if transform.name in self.transforms:
            message = (
                f"Transform {transform.name} already exists."
                f"Please choose a different name,"
                f"or delete it with the Project.delete_output() method."
            )
            raise ValueError(message)

        ads = (
            self.prepare_audio(transform=transform)
            if audio_dataset is None
            else audio_dataset
        )

        if OutputType.AUDIO in transform.output_type:
            self._add_audio_dataset(ads=ads, transform_name=transform.name)

        sds = None
        if transform.is_spectro:
            sds = (
                self.prepare_spectro(
                    transform=transform,
                    audio_dataset=ads,
                )
                if spectro_dataset is None
                else spectro_dataset
            )
            self._add_spectro_dataset(sds=sds, transform_name=transform.name)

        self.export(
            output_type=transform.output_type,
            ads=ads,
            sds=sds,
            link=True,
            subtype=transform.subtype,
            nb_jobs=nb_jobs,
            name=transform.name,
        )

        self.write_json()

    def _add_audio_dataset(
        self,
        ads: AudioDataset,
        transform_name: str,
    ) -> None:
        ads.folder = self._get_audio_dataset_subpath(ads=ads)
        self.output_datasets[ads.name] = {
            "class": type(ads).__name__,
            "transform": transform_name,
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

    def export(
        self,
        output_type: OutputType,
        ads: AudioDataset | None = None,
        sds: SpectroDataset | LTASDataset | None = None,
        subtype: str | None = None,
        matrix_folder_name: str = "matrix",
        spectrogram_folder_name: str = "spectrogram",
        welch_folder_name: str = "welch",
        nb_jobs: int = 1,
        name: str = "OSEkit_transform",
        *,
        link: bool = False,
    ) -> None:
        """Perform a transform and write the results on disk.

        An transform is defined as a manipulation of the original audio files:
        reshaping the audio, exporting ``png`` spectrograms or ``npz`` matrices
        (or a combination of those three) are examples of transforms.
        The tasks will be distributed to jobs if ``self.job_builder``
        is not ``None``, else it will be distributed on
        ``self.job_builder.nb_jobs`` jobs.

        Parameters
        ----------
        spectrogram_folder_name:
            The name of the folder in which the ``png`` spectrograms will be
            exported (relative to ``sds.folder``)
        matrix_folder_name:
            The name of the folder in which the ``npz`` matrices will be
            exported (relative to ``sds.folder``)
        welch_folder_name:
            The name of the folder in which the ``npz`` welch files will be
            exported (relative to ``sds.folder``)
        sds: SpectroDataset | LTASDataset
            The ``SpectroDataset`` on which the data should be written.
        output_type : OutputType
            Type of the transform to be performed.
            ``AudioDataset`` and ``SpectroDataset`` instances will be
            created depending on the flags.
            See ``osekit.public.transform.OutputType`` docstring
            for more information.
        ads: AudioDataset
            The ``AudioDataset`` on which the data should be written.
        subtype: str | None
            The subtype of the audio files as provided by the soundfile module.
        nb_jobs: int
            The number of jobs to run in parallel.
        name: str
            The name of the transform being performed.
        link: bool
            If ``True``, the ads data will be linked to the exported files.

        """
        # Import here to avoid circular imports since the script needs to import Project
        from osekit.public import export_transform  # noqa: PLC0415

        matrix_folder_path, spectrogram_folder_path, welch_folder_path = (
            (
                sds.folder / name
                for name in (
                    matrix_folder_name,
                    spectrogram_folder_name,
                    welch_folder_name,
                )
            )
            if sds is not None
            else ("None", "None", "None")
        )

        if self.job_builder is None:
            export_transform.write_transform_output(
                output_type=output_type,
                ads=ads,
                sds=sds,
                link=link,
                subtype=subtype,
                matrix_folder_path=matrix_folder_path,
                spectrogram_folder_path=spectrogram_folder_path,
                welch_folder_path=welch_folder_path,
                logger=self.logger,
            )
            return

        batch_indexes = file_indexes_per_batch(
            total_nb_files=len(ads.data),
            nb_batches=nb_jobs,
        )

        ads_json = (
            ads.folder / f"{ads.name}.json"
            if OutputType.AUDIO in output_type
            else "None"
        )
        sds_json = sds.folder / f"{sds.name}.json" if sds is not None else "None"

        for index, (start, stop) in enumerate(batch_indexes):
            self.job_builder.create_job(
                script_path=Path(export_transform.__file__),
                script_args={
                    "output-type": output_type.value,
                    "ads-json": ads_json,
                    "sds-json": sds_json,
                    "subtype": subtype,
                    "matrix-folder-path": matrix_folder_path,
                    "spectrogram-folder-path": spectrogram_folder_path,
                    "welch-folder-path": welch_folder_path,
                    "first": start,
                    "last": stop,
                    "downsampling-quality": resample_quality_settings["downsample"],
                    "upsampling-quality": resample_quality_settings["upsample"],
                    "umask": get_umask(),
                    "multiprocessing": config.multiprocessing["is_active"],
                    "nb-processes": config.multiprocessing["nb_processes"],
                    "use-logging-setup": True,
                    "dataset-json-path": self.folder / "dataset.json",
                },
                name=name + (f"_{index}" if len(batch_indexes) > 1 else ""),
                output_folder=self.folder / "log",
            )
        self.job_builder.submit_pbs()

    def _add_spectro_dataset(
        self,
        sds: SpectroDataset | LTASDataset,
        transform_name: str,
    ) -> None:
        sds.folder = self._get_spectro_dataset_subpath(sds=sds)
        self.output_datasets[sds.name] = {
            "class": type(sds).__name__,
            "dataset": sds,
            "transform": transform_name,
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

    def _sort_audio_dataset(self, dataset: AudioDataset) -> None:
        dataset.move_files(self._get_audio_dataset_subpath(dataset))

    def _sort_spectro_dataset(self, dataset: SpectroDataset | LTASDataset) -> None:
        raise NotImplementedError

    def _delete_output(self, output_dataset_name: str) -> None:
        """Delete a transform dataset.

        WARNING: all the transform output files will be deleted.
        WARNING: removing linked output_datasets (e.g. an ``AudioDataset`` to which a
        ``SpectroDataset`` is linked) might lead to errors.

        Parameters
        ----------
        output_dataset_name: str
            Name of the dataset to remove.

        """
        dataset_to_remove = self.get_output(output_dataset_name)
        if dataset_to_remove is None:
            return
        self.output_datasets.pop(dataset_to_remove.name)

        afm.close()
        shutil.rmtree(str(dataset_to_remove.folder))
        self.write_json()

    def get_output_dataset_by_transform_name(
        self,
        transform_name: str,
    ) -> list[type[DatasetChild]]:
        """Get all output output_datasets from a given transform.

        Parameters
        ----------
        transform_name: str
            Name of the transform of which to get the output_datasets.

        Returns
        -------
        list[type[DatasetChild]]
        List of the output_datasets.

        """
        return [
            self.deserialize_output_dataset(output_dataset_name=dataset_name)
            for dataset_name, dataset_values in self.output_datasets.items()
            if dataset_values["transform"] == transform_name
        ]

    def rename_output(self, output_name: str, new_output_name: str) -> None:
        """Rename an already ran transform.

        Parameters
        ----------
        output_name: str
            Name of the transform to rename.
        new_output_name: str
            New name of the transform to rename.

        """
        if output_name == new_output_name:
            return
        if output_name == "original":
            msg = "You can't rename the original dataset."
            raise ValueError(msg)
        if output_name not in self.output_datasets:
            msg = f"Unknown output {output_name}."
            raise ValueError(msg)
        if new_output_name in self.output_datasets:
            msg = f"{new_output_name} already exists."
            raise ValueError(msg)

        keys_to_rename = {}
        for output_dataset in self.output_datasets.values():
            if output_dataset["transform"] == output_name:
                output_dataset["transform"] = new_output_name
                ds = output_dataset["dataset"]
                old_name, new_name = (
                    ds.name,
                    new_output_name + (f"_{ds.suffix}" if ds.suffix else ""),
                )
                ds.base_name = new_output_name
                old_folder = ds.folder
                new_folder = ds.folder.parent / new_name
                keys_to_rename[old_name] = new_name

                ds.move_files(new_folder)
                move_tree(
                    old_folder,
                    new_folder,
                    excluded_paths=old_folder.glob("*.json"),
                )  # Moves exported files
                shutil.rmtree(str(old_folder))
                ds.write_json(ds.folder)

        for old_name, new_name in keys_to_rename.items():
            self.output_datasets[new_name] = self.output_datasets.pop(old_name)

        self.write_json()

    def delete_output(self, output_name: str) -> None:
        """Delete all output_datasets from a given ran transform name.

        WARNING: all the output files will be deleted.


        Parameters
        ----------
        output_name: str
            Name of the transform whose output to delete.

        """
        for dataset_to_delete in self.get_output_dataset_by_transform_name(
            output_name,
        ):
            self._delete_output(dataset_to_delete.name)

    def get_output(self, output_name: str) -> type[DatasetChild] | None:
        """Get an output dataset from its name.

        Parameters
        ----------
        output_name: str
            Name of the output dataset.

        Returns
        -------
        type[DatasetChild]:
            Output dataset from the ``project.output_datasets`` property.

        """
        if output_name not in self.output_datasets:
            message = f"Dataset '{output_name}' not found."
            raise ValueError(message)
        return self.deserialize_output_dataset(output_dataset_name=output_name)

    def to_dict(self) -> dict:
        """Serialize a project to a dictionary.

        Returns
        -------
        dict:
            The serialized dictionary representing the project.

        """
        return {
            "output_datasets": {
                name: {
                    "class": dataset["class"],
                    "transform": dataset["transform"],
                    "json": str(dataset["dataset"])
                    if isinstance(dataset["dataset"], Path)
                    else str(dataset["dataset"].folder / f"{name}.json"),
                }
                for name, dataset in self.output_datasets.items()
            },
            "instrument": (
                None if self.instrument is None else self.instrument.to_dict()
            ),
            "depth": self.depth,
            "gps_coordinates": self.gps_coordinates,
            "strptime_format": self.strptime_format,
            "timezone": self.timezone,
        }

    @classmethod
    def from_dict(cls, dictionary: dict) -> Project:
        """Deserialize a project from a dictionary.

        Parameters
        ----------
        dictionary: dict
            The serialized dictionary representing the project.

        Returns
        -------
        Project
            The deserialized project.

        """
        datasets = {}
        for name, dataset in dictionary["output_datasets"].items():
            datasets[name] = {
                "class": dataset["class"],
                "transform": dataset["transform"],
                "dataset": Path(dataset["json"]),
            }
        return cls(
            folder=Path(),
            instrument=Instrument.from_dict(dictionary["instrument"]),
            strptime_format=dictionary["strptime_format"],
            gps_coordinates=dictionary["gps_coordinates"],
            depth=dictionary["depth"],
            timezone=dictionary["timezone"],
            output_datasets=datasets,
        )

    def write_json(self, folder: Path | None = None) -> None:
        """Write a serialized Project to a JSON file."""
        folder = folder if folder is not None else self.folder
        serialize_json(folder / "dataset.json", self.to_dict())

    @classmethod
    def from_json(cls, file: Path) -> Project:
        """Deserialize a ``Project`` from a ``json`` file.

        Parameters
        ----------
        file: Path
            Path to the serialized ``json`` file representing the ``Project``.

        Returns
        -------
        Project
            The deserialized ``Project``.

        """
        instance = cls.from_dict(deserialize_json(file))
        instance.folder = file.parent
        instance._create_logger()  # noqa: SLF001
        return instance

    def deserialize_output_dataset(
        self,
        output_dataset_name: str,
    ) -> type[DatasetChild]:
        """Deserialize an output dataset from its json file.

        The self.output_datasets property will be updated so that it stores the deserialized
        dataset rather than the json file so that it is deserialized only once.

        Parameters
        ----------
        output_dataset_name: str
            Name of the output dataset.

        Returns
        -------
        type[DatasetChild]:
            The deserialized output dataset.

        """
        output_dataset = self.output_datasets[output_dataset_name]
        dataset_classes = {
            "AudioDataset": AudioDataset,
            "SpectroDataset": SpectroDataset,
            "LTASDataset": LTASDataset,
        }
        if isinstance(output_dataset["dataset"], Path):
            output_dataset_class = dataset_classes[output_dataset["class"]]
            output_dataset["dataset"] = output_dataset_class.from_json(
                output_dataset["dataset"],
            )
        return output_dataset["dataset"]


DatasetChild = TypeVar("DatasetChild", bound=BaseDataset)
