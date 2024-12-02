from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path
from statistics import fmean as mean
from typing import List, Tuple, Union

import pandas as pd

from OSmOSE.config import (
    DPDEFAULT,
    FPDEFAULT,
    OSMOSE_PATH,
    TIMESTAMP_FORMAT_AUDIO_FILE,
)
from OSmOSE.config import global_logging_context as glc
from OSmOSE.utils.audio_utils import (
    check_audio,
    get_all_audio_files,
    get_audio_metadata,
)
from OSmOSE.utils.core_utils import (
    change_owner_group,
    chmod_if_needed,
)
from OSmOSE.utils.formatting_utils import clean_filenames, clean_forbidden_characters
from OSmOSE.utils.timestamp_utils import (
    adapt_timestamp_csv_to_osmose,
    check_epoch,
    parse_timestamps_csv,
)


class Dataset:
    """Super class used to create dataset compatible with the rest of the package.

    A dataset is a set of audio files located in a folder whose name is the dataset name.
    The files must be in the `raw/audio/original`subfolder. and be alongside a `timestamp.csv` file, which includes
    the name of the file and the associated timestamp, in the `%Y-%m-%dT%H:%M:%S.%fZ` strftime format.

    This file can be created using the `OSmOSE.write_timestamp` function.
    """

    def __init__(
        self,
        dataset_path: str,
        *,
        gps_coordinates: str | list | tuple = (0, 0),
        depth: str | int = 0,
        timezone: str | None = None,
        owner_group: str | None = None,
        original_folder: str | None = None,
        local: bool = True,
    ) -> None:
        """Instanciate the dataset with at least its path.

        Parameters
        ----------
        dataset_path : `str`
            The absolute path to the dataset folder. The last folder in the path will be considered as the name of the dataset.

        gps_coordinates : `str` or `list` or `Tuple`, optional, keyword-only
            The GPS coordinates of the listening location. If it is of type `str`, it must be the name of a csv file located in `data/auxiliary`,
            otherwise a list or a tuple with the first element being the latitude coordinates and second the longitude coordinates.

        owner_group : `str`, optional, keyword-only
            The name of the group using the OsmOSE package. All files created using this dataset will be accessible by the osmose group.
            Will not work on Windows.

        original_folder : `str`, optional, keyword-only
            The path to the folder containing the original audio files. It can be set right away, passed in the build() function or automatically detected.

        Example
        -------
        >>> from pathlib import Path
        >>> from OSmOSE import Dataset
        >>> dataset = Dataset(Path("home","user","my_dataset"), coordinates = [49.2, -5], owner_group = "gosmose")

        """
        assert isinstance(dataset_path, Path) or isinstance(
            dataset_path,
            str,
        ), f"Expected value to be a Path or a string, but got {type(dataset_path).__name__}"
        # assert gps_coordinates

        self.__path = Path(dataset_path)
        self.__name = self.__path.stem
        self._create_logger()
        self.owner_group = owner_group
        self.__local = local
        self.timezone = timezone

        self.gps_coordinates = gps_coordinates
        self.depth = depth

        self.__original_folder = original_folder or self.__path

        pd.set_option("display.float_format", lambda x: "%.0f" % x)

    @property
    def name(self):
        """str: The Dataset name. It is readonly."""
        return self.__name

    @property
    def path(self):
        """Path: The Dataset path. It is readonly."""
        return self.__path

    @property
    def original_folder(self):
        """Path: The folder containing the original audio file."""
        return (
            self.__original_folder
            if self.__original_folder
            else self._get_original_after_build()
        )

    @property
    def gps_coordinates(
        self,
    ) -> Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]]:
        """The GPS coordinates of the listening location. First element is latitude, second is longitude.

        GPS coordinates are used to localize the dataset and required for some utilities, like the
        weather and environment utility.

        Parameter
        ---------
        coordinates: `str` or `list` or `tuple`
            If the coordinates are a string, it must be the name of a csv file located in `data/auxiliary/instrument/`, containing two columns: 'lat' and 'lon'
            Else, they can be either a list or a tuple of two float, the first being the latitude and second the longitude; or a
            list or a tuple containing two lists or tuples respectively of floats. In this case, the coordinates are not treated as a point but
            as an area.

        Returns
        -------
        The GPS coordinates as a tuple.

        """
        return self.__gps_coordinates

    @gps_coordinates.setter
    def gps_coordinates(
        self,
        new_coordinates: Union[
            str,
            List[float],
            List[List[float]],
            Tuple[float, float],
            Tuple[Tuple[float, float], Tuple[float, float]],
        ],
    ):
        match new_coordinates:
            case str():
                aux_data_path = next(self.path.rglob(new_coordinates), False)

                if aux_data_path:
                    self.__gps_coordinates = check_epoch(pd.read_csv(aux_data_path))
                    """
                    csvFileArray = pd.read_csv(aux_data_path)
                    self.__gps_coordinates = [
                        np.mean(csvFileArray["lat"]),
                        np.mean(csvFileArray["lon"]),
                    ]"""
                else:
                    raise FileNotFoundError(
                        f"The {new_coordinates} has been found no where within {self.path}",
                    )

            case tuple():
                self.__gps_coordinates = new_coordinates
            case list():
                self.__gps_coordinates = new_coordinates
            case _:
                raise TypeError(
                    f"GPS coordinates must be either a list of coordinates or the name of csv containing the coordinates, but {type(new_coordinates)} found.",
                )

    @property
    def depth(
        self,
    ) -> int:
        """The depth of the hydrophone, in meter.

        Parameter
        ---------
        depth: `str` or `int`
            If the depth is a string, it must be the name of a csv file located in `data/auxiliary/instrument/`, containing at least a column 'depth'

        Returns
        -------
        The depth as an int.

        """
        return self.__depth

    @depth.setter
    def depth(
        self,
        new_depth: Union[
            str,
            int,
        ],
    ):
        match new_depth:
            case str():
                aux_data_path = next(self.path.rglob(new_depth), False)
                if aux_data_path:
                    self.__depth = check_epoch(pd.read_csv(aux_data_path))
                    """
                    csvFileArray = pd.read_csv(aux_data_path)
                    self.__depth = int(np.mean(csvFileArray["depth"]))"""
                else:
                    raise FileNotFoundError(
                        f"The {new_depth} has been found no where within {self.path}",
                    )

            case int():
                self.__depth = new_depth
            case _:
                raise TypeError(
                    "Variable depth must be either an int value for fixed hydrophone or a csv filename for moving hydrophone",
                )

    @property
    def owner_group(self) -> str:
        """str: The Unix group able to interact with the dataset."""
        if self.__group is None:
            self.logger.warning(
                "The OSmOSE group name is not defined. Please specify the group name before trying to build the dataset.",
            )
        return self.__group

    @owner_group.setter
    def owner_group(self, value: str) -> None:
        self.__group = value

    def build(
        self,
        *,
        original_folder: str | None = None,
        owner_group: str | None = None,
        date_template: str = TIMESTAMP_FORMAT_AUDIO_FILE,
        auto_normalization: bool = False,
        force_upload: bool = False,
        number_test_bad_files: int = 1,
        dico_aux_substring: dict = {
            "instrument": ["depth", "gps"],
            "environment": ["insitu"],
        },
    ) -> None:
        """Set up the architecture of the dataset.

        The following operations will be performed on the dataset. None of them are destructive:
            - open and read the header of audio files located in `raw/audio/original/`.
            - rename files containing illegal characters.
            - generate some stastics regarding the files and dataset durations.
            - write the raw/metadata.csv file.
            - Identify and record files with anomalies (short duration, unreadable header...).
            - Set the permission of the dataset to the osmose group.

        Parameters
        ----------
            original_folder: `str`, optional, keyword-only
                The name of the folder containing the original audio file. It is named "original" by convention.
                If none is passed, the program expects to find either only one folder in the dataset audio folder, or
                a folder named `original`
            owner_group: `str`, optional, keyword_only
                The name of the group using the osmose dataset. It will have all permissions over the dataset.
            date_template: `str`, optional, keyword_only
                the date template in strftime format. For example, `2017/02/24` has the template `%Y/%m/%d`.
                It is used to generate automatically the timestamp.csv file. Alternatively, you can call the script to create the timestamp file first.
                If no template is provided, will assume that the file already exists. In future versions, the template will be guessed automatically.
                For more information on strftime template, see https://strftime.org/.
            auto_normalization: `bool`, optional, keyword_only
                If true, automatically normalize audio files if the data would cause issues downstream. The default is False.
            force_upload: `bool`, optional, keyword_only
                If true, ignore the file anomalies and build the dataset anyway. The default is False.

        Returns
        -------
            dataset: `Dataset`
                The dataset object.

        Example
        -------
            >>> from pathlib import Path
            >>> from OSmOSE import Dataset
            >>> dataset = Dataset(Path("home","user","my_dataset"))
            >>> dataset.build()

            DONE ! your dataset is on OSmOSE platform !

        """
        if self._is_built() and not force_upload:
            self.logger.warning(
                "This dataset has already been built. To run the build() method on an "
                "already built dataset, you have to use the force_upload parameter.",
            )
            return

        audio_path = self._find_original_folder(original_folder)

        self.dico_aux_substring = dico_aux_substring
        # TODO: rework the auxiliary management with a specific class.

        if not self.__local:
            with glc.set_logger(self.logger):
                change_owner_group(
                    path=self.path,
                    owner_group=owner_group or self.owner_group,
                )
                chmod_if_needed(path=self.path, mode=DPDEFAULT)

        file_metadata = self._build_audio(
            audio_path=audio_path,
            date_template=date_template,
            force_upload=force_upload,
        )

        self._write_metadata(file_metadata=file_metadata)
        self._move_other_files()

        self.logger.info("DONE ! your dataset is on OSmOSE platform !")

    def _find_original_folder(
        self,
        original_folder: PathLike | str | None = None,
    ) -> Path:
        """Find the original folder in which the audio are stored.

        The original folder is either, in this order:
            - The specified original_folder
            - The original_folder found for a built dataset if the dataset is built
            - The first found folder with the name "original"
            - The root folder of the dataset

        Parameters
        ----------
        original_folder: str|PathLike, optional, keyword-only
            The original_folder containing the audio to use for the build.

        Returns
        -------
        Path:
            The path of the original_folder containing the audio to use for the build.

        """
        if original_folder is not None:
            if (folder := Path(original_folder)).exists():
                return folder
            if (folder := self.path / Path(original_folder)).exists():
                return folder

        if self._is_built():
            return self._get_original_after_build()

        return self.path

    def _create_logger(self) -> None:
        logs_directory = self.__path / "log"
        if not logs_directory.exists():
            logs_directory.mkdir(mode=DPDEFAULT)
        self.logger = logging.getLogger("dataset").getChild(self.__name)
        self.file_handler = logging.FileHandler(logs_directory / "logs.log", mode="w")
        self.file_handler.setFormatter(
            logging.getLogger("dataset").handlers[0].formatter,
        )
        self.logger.setLevel(logging.DEBUG)
        self.file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(self.file_handler)

    def _is_built(self) -> bool:
        metadata_path = next(
            (self.path / OSMOSE_PATH.raw_audio).rglob("metadata.csv"),
            None,
        )
        if metadata_path is None:
            return False
        metadata = pd.read_csv(metadata_path)
        if "is_built" not in metadata.columns:
            return False
        return metadata["is_built"][0]

    def _build_audio(
        self,
        audio_path: Path,
        date_template: str,
        force_upload: bool,
    ) -> pd.DataFrame:
        """Move all audio to the raw_audio folder, along with a timestamp.csv file.

        If no timestamp.csv is found, it is created by parsing audio file names.
        If the found timestamp.csv:
            is in the OSmOSE format:
                It is moved to the audio folder.
            is not in the OSmOSE format:
                A copy of the timestamp.csv is formatted and moved to the audio folder.
        """
        raw_audio_files = get_all_audio_files(
            audio_path,
        )  # TODO: manage built dataset with reshape audio folders ?

        audio_files = clean_filenames(raw_audio_files)
        for old, new in zip(raw_audio_files, audio_files):
            old.replace(new)
        date_template = clean_forbidden_characters(date_template)

        timestamps = self._parse_timestamp_df(
            audio_files=audio_files,
            date_template=date_template,
            path=audio_path,
        )
        audio_metadata = pd.DataFrame.from_records(
            get_audio_metadata(file) for file in audio_files
        )

        try:
            check_audio(audio_metadata=audio_metadata, timestamps=timestamps)
        except FileNotFoundError as e:
            if not force_upload:
                self.logger.exception(
                    "Please fix the following error or set the force_upload parameter to True: \n",
                    exc_info=e,
                )
                raise
            self.logger.warning(
                "Timestamp.csv and audio files didn't match. Creating new timestamp.csv files from audio. Detail: \n",
                exc_info=e,
            )
            timestamps = parse_timestamps_csv(
                filenames=[file.name for file in audio_files],
                datetime_template=date_template,
                timezone=self.timezone,
            )
        except ValueError as e:
            if not force_upload:
                self.logger.exception(
                    "Please fix the following error or set the force_upload parameter to True: \n",
                    exc_info=e,
                )
                raise
            self.logger.warning(
                "Your audio files failed the following test(s):\n",
                exc_info=e,
            )

        file_metadata = self._create_file_metadata(audio_metadata, timestamps)

        folder_name = (
            f'{round(mean(audio_metadata["duration"].values))}'
            "_"
            f'{round(mean(audio_metadata["origin_sr"].values))}'
        )
        destination_folder = self.path / OSMOSE_PATH.raw_audio / folder_name
        destination_folder.mkdir(parents=True, exist_ok=True)

        timestamps.to_csv(
            destination_folder / "timestamp.csv",
            index=False,
        )

        file_metadata.to_csv(
            destination_folder / "file_metadata.csv",
            index=False,
        )

        for file in audio_files:
            file.replace(destination_folder / file.name)

        return file_metadata

    def _parse_timestamp_df(
        self,
        audio_files: list[Path],
        date_template: str,
        path: Path | None,
    ) -> pd.DataFrame:
        timestamp_file = None

        if path is not None:
            timestamp_file = list(path.rglob("timestamp.csv"))

        if not timestamp_file:
            self.logger.debug("Creating timestamp.csv file from scratch.")
            return parse_timestamps_csv(
                filenames=[file.name for file in audio_files],
                datetime_template=date_template,
                timezone=self.timezone,
            )

        if len(timestamp_file) > 1:
            warning_message = (
                "More than one timestamp file found in the dataset. "
                f"Only {timestamp_file[0]} has been considered."
            )
            self.logger.warning(warning_message)
        else:
            message = f"Creating timestamps.csv file from {timestamp_file[0]}"
            self.logger.debug(message)

        timestamps = pd.read_csv(timestamp_file[0])
        return adapt_timestamp_csv_to_osmose(
            timestamps=timestamps,
            date_template=date_template,
            timezone=self.timezone,
        )

    def _create_file_metadata(
        self,
        audio_metadata: pd.DataFrame,
        timestamps: pd.DataFrame,
    ) -> pd.DataFrame:
        file_metadata = audio_metadata.merge(timestamps, on="filename")
        file_metadata["duration_inter_file"] = audio_metadata["duration"].diff()
        return file_metadata

    def _write_metadata(self, file_metadata: pd.DataFrame) -> None:
        metadata = pd.Series()
        metadata["origin_sr"] = round(mean(file_metadata["origin_sr"].values))
        metadata["sample_bits"] = list(set(file_metadata["sampwidth"]))
        metadata["channel_count"] = round(mean(file_metadata["channel_count"].values))
        metadata["audio_file_count"] = len(file_metadata["filename"].values)
        metadata["start_date"] = file_metadata["timestamp"].iloc[0]
        metadata["end_date"] = file_metadata["timestamp"].iloc[-1]
        metadata["audio_file_origin_duration"] = round(
            mean(file_metadata["duration"].values),
        )
        metadata["audio_file_origin_volume"] = round(
            mean(file_metadata["size"].values),
            1,
        )
        metadata["dataset_origin_volume"] = max(
            1,
            round(sum(file_metadata["size"].values) / 1_000),
        )  # cannot be inferior to 1 GB
        metadata["dataset_origin_duration"] = round(
            sum(file_metadata["duration"].values),
        )
        metadata["is_built"] = True
        metadata["audio_file_dataset_overlap"] = 0
        metadata["lat"] = self.gps_coordinates[0]
        metadata["lon"] = self.gps_coordinates[1]
        metadata["depth"] = self.depth
        metadata["dataset_sr"] = metadata["origin_sr"]
        metadata["audio_file_dataset_duration"] = metadata["audio_file_origin_duration"]
        audio_origin_duration = metadata["audio_file_origin_duration"]
        origin_sr = metadata["origin_sr"]
        metadata_file_path = (
            self.path
            / OSMOSE_PATH.raw_audio
            / f"{audio_origin_duration}_{origin_sr}"
            / "metadata.csv"
        )
        metadata = metadata.to_frame().T
        metadata.to_csv(metadata_file_path, index=False)
        chmod_if_needed(path=metadata_file_path, mode=FPDEFAULT)

    def _move_other_files(self) -> None:
        build_folders = (
            self.path / "log",
            self.path / "other",
            self._get_original_after_build(),
        )
        nb_moved_files = 0
        for file in self.path.rglob("*"):
            if file.is_dir() and any(file.iterdir()):
                continue
            if file in (file for folder in build_folders for file in folder.rglob("*")):
                continue
            if not file.is_dir() or any(file.iterdir()):
                relative_path = file.relative_to(self.path)
                destination_folder = (self.path / "other" / relative_path).parent
                if not destination_folder.exists():
                    destination_folder.mkdir(parents=True, exist_ok=True)
                file.replace(self.path / "other" / relative_path)
                nb_moved_files += 1
            folder_to_remove = file if file.is_dir() else file.parent
            while not any(folder_to_remove.iterdir()):
                folder_to_remove.rmdir()
                folder_to_remove = folder_to_remove.parent
        if nb_moved_files > 0:
            self.logger.info("Moved %i file(s) to the 'other' folder.", nb_moved_files)

    def _get_original_after_build(self) -> Path:
        """Find the original folder path after the dataset has been built.

        Returns
        -------
            original_folder: `Path`
                The path to the folder containing the original audio file.

        Raises
        ------
            ValueError
                If no metadata.csv has been found and the original folder is not able to be found.

        """
        # First, grab any metadata.csv
        all_datasets = self.path.joinpath(OSMOSE_PATH.raw_audio).iterdir()
        while True:
            audio_dir = next(all_datasets)
            try:
                metadata_path = audio_dir.resolve().joinpath("metadata.csv")
            except StopIteration:
                # If we get to the end of the generator, it means that no metadata file has been found, so we raise a more explicit error.
                raise ValueError(
                    f"No metadata file found in {self.path.joinpath(OSMOSE_PATH.raw_audio, audio_dir)}. Impossible to find the original data file.",
                )
            if metadata_path.exists():
                break

        metadata = pd.read_csv(metadata_path)
        # Catch the parameters inscribed in the original folder name
        audio_file_origin_duration = int(metadata["audio_file_origin_duration"][0])
        origin_sr = int(metadata["origin_sr"][0])

        self.__original_folder = self.path.joinpath(
            OSMOSE_PATH.raw_audio,
            f"{audio_file_origin_duration}_{origin_sr}",
        )

        return self.original_folder

    def __str__(self):
        metadata = pd.read_csv(self.original_folder.joinpath("metadata.csv"))
        displayed_metadata = {
            "audio_file_origin_duration": "(s)",
            "origin_sr": "(Hz)",
            "start_date": "",
            "end_date": "",
            "audio_file_count": "",
            "audio_file_origin_volume": "(MB)",
            "dataset_origin_volume": "(GB)",
        }  # restrain metadata to a shorter list of variables to be displayed, with their respective units
        preamble = f"Metadata of {self.name} :"
        metadata_str = "\n".join(
            f"- {key} : {metadata[key][0]} {unit}"
            for key, unit in displayed_metadata.items()
        )
        return f"{preamble}\n{metadata_str}"
