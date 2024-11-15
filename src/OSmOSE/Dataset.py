from __future__ import annotations

import logging
import os
from pathlib import Path
from statistics import fmean as mean
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from OSmOSE.config import (
    DPDEFAULT,
    FPDEFAULT,
    OSMOSE_PATH,
    TIMESTAMP_FORMAT_AUDIO_FILE,
)
from OSmOSE.config import global_logging_context as glc
from OSmOSE.utils.audio_utils import get_all_audio_files, get_audio_metadata
from OSmOSE.utils.core_utils import (
    change_owner_group,
    chmod_if_needed,
)
from OSmOSE.utils.formatting_utils import clean_filenames
from OSmOSE.utils.path_utils import make_path
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

        self.__original_folder = original_folder

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
        if not force_upload and self._is_built():
            self.logger.warning(
                "This dataset has already been built. To run the build() method on an "
                "already built dataset, you have to use the force_upload parameter.",
            )
            return

        self.dico_aux_substring = dico_aux_substring
        # TODO: rework the auxiliary management with a specific class.

        with glc.set_logger(self.logger):
            if not self.__local:
                change_owner_group(
                    path=self.path,
                    owner_group=owner_group or self.owner_group,
                )
                chmod_if_needed(path=self.path, mode=DPDEFAULT)

        self._build_audio(date_template=date_template)
        audio_metadata = self._create_audio_metadata(self.path / OSMOSE_PATH.raw_audio)
        audio_metadata.to_csv(
            self.path / OSMOSE_PATH.raw_audio / "file_metadata.csv", index=False
        )

        return

        for ind_dt in tqdm(range(len(timestamp_csv)), desc="Scanning audio files"):

            try:
                # origin_sr, frames, sampwidth, channel_count, size = read_header(
                #     path_raw_audio / cur_filename
                # )
                channel_count = sf.info(path_raw_audio / cur_filename).channels
                size = (path_raw_audio / cur_filename).stat().st_size
                # with wave.open(str(path_raw_audio / cur_filename), 'rb') as wav:
                #     sampwidth = wav.getsampwidth()
                sampwidth = sf.info(path_raw_audio / cur_filename).subtype

                sf_meta = sf.info(path_raw_audio / cur_filename)

            except Exception as e:
                self.logger.error(
                    f"error message making status read header False : \n {e}",
                )
                # append audio metadata read from header for files with corrupted headers
                audio_metadata = pd.concat(
                    [
                        audio_metadata,
                        pd.DataFrame(
                            {
                                "filename": cur_filename,
                                "timestamp": cur_timestamp,
                                "duration": np.nan,
                                "origin_sr": np.nan,
                                "sampwidth": None,
                                "size": None,
                                "duration_inter_file": None,
                                "channel_count": None,
                                "status_read_header": False,
                            },
                            index=[0],
                        ),
                    ],
                    axis=0,
                )
                continue

            # append audio metadata read from header in the dataframe audio_metadata
            new_data = pd.DataFrame(
                {
                    "filename": cur_filename,
                    "timestamp": cur_timestamp,
                    "duration": sf_meta.duration,
                    "origin_sr": int(sf_meta.samplerate),
                    "sampwidth": sampwidth,
                    "size": size / 1e6,
                    "duration_inter_file": None,
                    "channel_count": channel_count,
                    "status_read_header": True,
                },
                index=[ind_dt],
            )
            new_data = new_data.dropna(axis=1, how="all")
            audio_metadata = pd.concat(
                [audio_metadata if not audio_metadata.empty else None, new_data],
                axis=0,
            )



        # write file_metadata.csv
        audio_metadata.to_csv(path_raw_audio.joinpath("file_metadata.csv"), index=False)
        chmod_if_needed(path=path_raw_audio / "file_metadata.csv", mode=FPDEFAULT)

        # define anomaly tests of level 0 and 1
        test_level0_1 = (
            len(
                np.unique(
                    audio_metadata["origin_sr"].values[
                        ~pd.isna(audio_metadata["origin_sr"].values)
                    ],
                ),
            )
            == 1
        )
        test_level0_2 = number_bad_files == 0
        test_level0_3 = sum(audio_metadata["status_read_header"].values) == len(
            timestamp_csv,
        )
        test_level1_1 = (
            len(
                np.unique(
                    audio_metadata["duration"].values[
                        ~pd.isna(audio_metadata["duration"].values)
                    ],
                ),
            )
            == 1
        )
        list_tests_level0 = [test_level0_1, test_level0_2, test_level0_3]
        list_tests_level1 = [test_level1_1]

        # write resume_test_anomalies.txt
        if resume_test_anomalies.exists():
            status_text = "w"
        else:
            status_text = "a"
        lines = [
            "Anomalies of level 0",
            f"- Test 1 : {test_level0_1}",
            f"- Test 2 : {test_level0_2}",
            f"- Test 3 : {test_level0_3}",
            "---------------------",
            "Anomalies of level 1",
            f"- Test 1 : {test_level1_1}",
        ]
        lines = [
            ll.replace("False", "FAILED").replace("True", "PASSED") for ll in lines
        ]

        with open(resume_test_anomalies, status_text) as f:
            f.write("\n".join(lines))

        # write messages in prompt for user
        if (
            len(list_tests_level0) - sum(list_tests_level0) > 0
        ):  # if presence of anomalies of level 0
            self.logger.warning(
                f"Your dataset failed {len(list_tests_level0)-sum(list_tests_level0)} anomaly test of level 0 (over {len(list_tests_level0)}); see details below. \n Anomalies of level 0 block dataset uploading as long as they are present. Please correct your anomalies first, and try uploading it again after. \n You can inspect your metadata saved here {path_raw_audio.joinpath('file_metadata.csv')} using the notebook /home/datawork-osmose/osmose-datarmor/notebooks/metadata_analyzer.ipynb.",
            )

            if (
                len(list_tests_level1) - sum(list_tests_level1) > 0
            ):  # if also presence of anomalies of level 1
                self.logger.warning(
                    f"Your dataset also failed {len(list_tests_level1)-sum(list_tests_level1)} anomaly test of level 1 (over {len(list_tests_level1)}).",
                )

            with open(resume_test_anomalies) as f:
                self.logger.warning(f.read())

            # we remove timestamp.csv here to force its recreation as we may have changed the filenames during a first pass (eg - transformed into _)
            if (
                not user_timestamp
            ):  # in case where the user did not bring its own timestamp.csv file
                os.remove(path_raw_audio.joinpath("timestamp.csv"))

        elif (
            len(list_tests_level1) - sum(list_tests_level1) > 0
        ) and not force_upload:  # if presence of anomalies of level 1

            self.logger.warning(
                f"Your dataset failed {len(list_tests_level1)-sum(list_tests_level1)} anomaly test of level 1 (over {len(list_tests_level1)}); see details below. \n  Anomalies of level 1 block dataset uploading, but anyone can force it by setting the variable `force_upload` to True. \n You can inspect your metadata saved here {path_raw_audio.joinpath('file_metadata.csv')} using the notebook  /home/datawork-osmose/osmose-datarmor/notebooks/metadata_analyzer.ipynb.",
            )

            with open(resume_test_anomalies) as f:
                self.logger.warning(f.read())

            if not user_timestamp:
                os.remove(path_raw_audio.joinpath("timestamp.csv"))

        else:  # no anomalies
            # rebuild the timestamp.csv file (necessary as we might have changed filenames) and set permissions
            df = pd.DataFrame(
                {
                    "filename": audio_metadata["filename"].values,
                    "timestamp": audio_metadata["timestamp"].values,
                },
            )
            df.sort_values(by=["timestamp"], inplace=True)
            df.to_csv(path_raw_audio.joinpath("timestamp.csv"), index=False)

            chmod_if_needed(path=path_raw_audio / "timestamp.csv", mode=FPDEFAULT)

            # change name of the original wav folder
            new_folder_name = path_raw_audio.parent.joinpath(
                str(int(mean(audio_metadata["duration"].values)))
                + "_"
                + str(int(mean(audio_metadata["origin_sr"].values))),
            )

            if new_folder_name.exists():
                new_folder_name.rmdir()

            path_raw_audio = path_raw_audio.rename(new_folder_name)
            self.__original_folder = path_raw_audio

            for subpath in OSMOSE_PATH:
                if "data" in str(subpath):
                    make_path(self.path.joinpath(subpath), mode=DPDEFAULT)

            # rename filenames in the subset_files.csv if any to replace -' by '_'
            subset_path = OSMOSE_PATH.processed.joinpath("subset_files.csv")
            if subset_path.is_file():
                xx = pd.read_csv(subset_path, header=None).values
                pd.DataFrame(
                    [ff[0].replace("-", "_").replace(":", "_") for ff in xx],
                ).to_csv(subset_path, index=False, header=None)
                chmod_if_needed(path=subset_path, mode=FPDEFAULT)

            # write summary metadata.csv
            data = {
                "origin_sr": int(mean(audio_metadata["origin_sr"].values)),
                "sample_bits": list(set(audio_metadata["sampwidth"])),
                "channel_count": int(mean(audio_metadata["channel_count"].values)),
                "audio_file_count": len(audio_metadata["filename"].values),
                "start_date": timestamp_csv[0],
                "end_date": timestamp_csv[-1],
                "audio_file_origin_duration": int(
                    mean(audio_metadata["duration"].values),
                ),
                "audio_file_origin_volume": round(
                    mean(audio_metadata["size"].values),
                    1,
                ),
                "dataset_origin_volume": max(
                    1,
                    round(sum(audio_metadata["size"].values) / 1000),
                ),  # cannot be inferior to 1 GB
                "dataset_origin_duration": round(
                    sum(audio_metadata["duration"].values),
                ),
                "is_built": True,
                "audio_file_dataset_overlap": 0,
            }
            df = pd.DataFrame.from_records([data])
            df["lat"] = self.gps_coordinates[0]
            df["lon"] = self.gps_coordinates[1]
            df["depth"] = self.depth
            df["dataset_sr"] = int(mean(audio_metadata["origin_sr"].values))
            df["audio_file_dataset_duration"] = int(
                mean(audio_metadata["duration"].values),
            )
            df.to_csv(path_raw_audio.joinpath("metadata.csv"), index=False)
            chmod_if_needed(path=path_raw_audio / "metadata.csv", mode=FPDEFAULT)

            for path, _, files in os.walk(self.path.joinpath(OSMOSE_PATH.auxiliary)):
                for f in files:
                    if f.endswith(".csv"):
                        self.logger.debug(
                            f"\n Checking your timestamp format in {Path(path,f).name}",
                        )
                        self._format_timestamp(Path(path, f), date_template, False)

            self.logger.info("DONE ! your dataset is on OSmOSE platform !")

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

    def _build_audio(self, date_template: str) -> None:
        """Move all audio to the raw_audio folder, along with a timestamp.csv file.

        If no timestamp.csv is found, it is created by parsing audio file names.
        If the found timestamp.csv:
            is in the OSmOSE format:
                It is moved to the audio folder.
            is not in the OSmOSE format:
                A copy of the timestamp.csv is formatted and moved to the audio folder.
        """
        audio_files = get_all_audio_files(
            self.path
        )  # TODO: manage built dataset with reshape audio folders
        timestamp_file = list(self.path.rglob("timestamp.csv"))
        timestamp_filepath = self.path / OSMOSE_PATH.raw_audio / "timestamp.csv"
        (self.path / OSMOSE_PATH.raw_audio).mkdir(parents=True, exist_ok=True)

        audio_files = clean_filenames(audio_files)

        for file in audio_files:
            file.replace(self.path / OSMOSE_PATH.raw_audio / file.name)

        if not timestamp_file:
            self.logger.debug("Creating timestamp.csv file from scratch.")
            timestamps = parse_timestamps_csv(
                filenames=[file.name for file in audio_files],
                datetime_template=date_template,
                timezone=self.timezone,
            )
            timestamps.to_csv(timestamp_filepath, index=False)
            return

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
        timestamps = adapt_timestamp_csv_to_osmose(
            timestamps=timestamps,
            date_template=date_template,
            timezone=self.timezone,
        )

        timestamps.to_csv(
            self.path / OSMOSE_PATH.raw_audio / "timestamp.csv",
            index=False,
        )

    def _create_audio_metadata(self, path: Path) -> pd.DataFrame:
        if not (path / "timestamp.csv").exists():
            message = f"There is no timestamp.csv file in {path}."
            self.logger.error(message)
            raise FileNotFoundError(message)

        timestamps = pd.read_csv(path / "timestamp.csv")
        audio_files = get_all_audio_files(path)

        if any(
            (unlisted_file := file.name) not in timestamps["filename"].unique()
            for file in audio_files
        ):
            message = f"{unlisted_file} has not been found in timestamp.csv"
            self.logger.error(message)
            raise FileNotFoundError(message)

        if any(
            (missing_file := filename) not in [file.name for file in audio_files]
            for filename in timestamps["filename"]
        ):
            message = f"{missing_file} is listed in timestamp.csv but hasn't be found."
            self.logger.error(message)
            raise FileNotFoundError(message)

        audio_metadata = pd.DataFrame(
            columns=[
                "filename",
                "timestamp",
                "duration",
                "origin_sr",
                "duration_inter_file",
                "size",
                "sampwidth",
                "channel_count",
                "status_read_header",
            ],
        )

        for file in audio_files:
            file_metadata = {
                **get_audio_metadata(file),
                "timestamp": timestamps.loc[
                    timestamps["filename"] == file.name, "timestamp"
                ].iloc[0],
            }
            audio_metadata.loc[len(audio_metadata)] = file_metadata
        audio_metadata["duration_inter_file"] = audio_metadata["duration"].diff()
        return pd.DataFrame(audio_metadata)

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
