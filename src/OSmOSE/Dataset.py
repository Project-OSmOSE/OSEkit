import os
from pathlib import Path
from typing import Union, Tuple, List
from datetime import datetime
from warnings import warn
from statistics import fmean as mean

try:
    import grp

    skip_perms = False
except ModuleNotFoundError:
    skip_perms = True

import pandas as pd
import numpy as np
from tqdm import tqdm
from OSmOSE.utils import read_header, check_n_files
from OSmOSE import _osmose_path_nt as osm_path


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
        gps_coordinates: Union[str, list, Tuple] = None,
        owner_group: str = None,
    ) -> None:
        """Instanciate the dataset with at least its path.

        Parameters
        ----------
        dataset_path : `str`
            The absolute path to the dataset folder. The last folder in the path will be considered as the name of the dataset.

        gps_coordinates : `str` or `list` or `Tuple`, optional, keyword-only
            The GPS coordinates of the listening location. If it is of type `str`, it must be the name of a csv file located in `raw/auxiliary`,
            otherwise a list or a tuple with the first element being the latitude coordinates and second the longitude coordinates.

        owner_group : `str`, optional, keyword-only
            The name of the group using the OsmOSE package. All files created using this dataset will be accessible by the osmose group.
            Will not work on Windows.

        Example
        -------
        >>> from pathlib import Path
        >>> from OSmOSE import Dataset
        >>> dataset = Dataset(Path("home","user","my_dataset"), coordinates = [49.2, -5], owner_group = "gosmose")
        """
        self.__path = Path(dataset_path)
        self.__name = self.__path.stem
        self.__group = owner_group
        self.__gps_coordinates = []
        if gps_coordinates is not None:
            self.gps_coordinates = gps_coordinates

        self.list_abnormal_filenames = []

        if skip_perms:
            print(
                "It seems you are on a non-Unix operating system (probably Windows). The build_dataset() method will not work as intended and permission might be uncorrectly set."
            )

        pd.set_option("display.float_format", lambda x: "%.0f" % x)

    # region Properties
    @property
    def name(self):
        """str: The Dataset name. It is readonly."""
        return self.__name

    @property
    def path(self):
        """str: The Dataset path. It is readonly."""
        return self.__path

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
            If the coordinates are a string, it must be the name of a csv file located in `raw/auxiliary/`, containing two columns: 'lat' and 'long'
            Else, they can be either a list or a tuple of two float, the first being the latitude and second the longitude; or a
            list or a tuple containing two lists or tuples respectively of floats. In this case, the coordinates are not treated as a point but
            as an area.

        Returns
        -------
        The GPS coordinates as a tuple.
        """
        if not self.__gps_coordinates:
            print("This dataset has no GPS coordinates.")
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
        # TODO: Allow any iterator?
        match type(new_coordinates):
            case str():
                csvFileArray = pd.read_csv(
                    Path(self.path, osm_path.auxiliary, new_coordinates)
                )
                self.__gps_coordinates = [
                    (np.min(csvFileArray["lat"]), np.max(csvFileArray["lat"])),
                    (np.min(csvFileArray["lon"]), np.max(csvFileArray["lon"])),
                ]
            case tuple():
                self.__gps_coordinates = new_coordinates
            case list():
                if all(isinstance(coord, list) for coord in new_coordinates):
                    self.__gps_coordinates = (
                        (new_coordinates[0][0], new_coordinates[0][1]),
                        (new_coordinates[1][0], new_coordinates[1][1]),
                    )
                elif all(isinstance(coord, float) for coord in new_coordinates):
                    self.__gps_coordinates = (new_coordinates[0], new_coordinates[1])
                else:
                    raise ValueError(
                        f"The coordinates list must contain either only floats or only sublists of two elements."
                    )
            case _:
                raise TypeError(
                    f"GPS coordinates must be either a list of coordinates or the name of csv containing the coordinates, but {type(new_coordinates)} found."
                )

    @property
    def owner_group(self):
        """str: The Unix group able to interact with the dataset."""
        if self.__group is None:
            print(
                "The OSmOSE group name is not defined. Please specify the group name before trying to build the dataset."
            )
        return self.__group

    @owner_group.setter
    def owner_group(self, value):
        if skip_perms:
            print("Cannot set osmose group on a non-Unix operating system.")
            return
        try:
            grp.getgrnam(value)
        except KeyError as e:
            raise KeyError(
                f"The group {value} does not exist on the system. Full error trace: {e}"
            )

        self.__group = value

    @property
    def is_built(self):
        """Checks if self.path/raw/audio contains at least one folder and none called "original"."""
        return (
            len(Path(self.path, osm_path.raw_audio).iterdir()) > 0
            and not Path(self.path, osm_path, "original").exists()
        )

    # endregion

    def build(
        self,
        *,
        original_folder: str = None,
        owner_group: str = None,
        bare_check: bool = False,
        auto_normalization: bool = False,
        force_upload: bool = False,
    ) -> None:
        """
        Set up the architecture of the dataset.

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
            bare_check : `bool`, optional, keyword_only
                Only do the checks and build steps that requires low resource usage. If you build the dataset on a login node or
                if you are sure it is already good to use, set to True. Otherwise, it should be inside a job. Default is False.
            auto_normalization: `bool`, optional, keyword_only
                If true, automatically normalize audio files if the data would cause issues downstream. The default is False.
            force_upload: `bool`, optional, keyword_only
                If true, ignore the file anomalies and build the dataset anyway. The default is False.

        Example
        -------
            >>> from pathlib import Path
            >>> from OSmOSE import Dataset
            >>> dataset = Dataset(Path("home","user","my_dataset"))
            >>> dataset.build()

            DONE ! your dataset is on OSmOSE platform !
        """
        if owner_group is None:
            owner_group = self.owner_group

        if original_folder:
            path_raw_audio = Path(self.path, osm_path.raw_audio, original_folder)
        elif Path(self.path, osm_path.raw_audio, "original").is_dir():
            path_raw_audio = Path(self.path, osm_path.raw_audio, "original")
        elif len(list(Path(self.path, osm_path.raw_audio).iterdir())) == 1:
            path_raw_audio = Path(
                self.path,
                osm_path.raw_audio,
                next(Path(self.path, osm_path.raw_audio).iterdir()),
            )

        path_timestamp_formatted = Path(original_folder, "timestamp.csv")

        csvFileArray = pd.read_csv(path_timestamp_formatted, header=None)

        timestamp_csv = csvFileArray[1].values
        filename_csv = csvFileArray[0].values

        list_filename_abnormal_duration = []

        list_file_problem = []
        timestamp = []
        filename_rawaudio = []
        list_duration = []
        list_samplingRate = []
        list_interWavInterval = []
        list_size = []
        list_sampwidth = []
        list_filename = []
        lost_levels = False

        audio_file_list = [Path(path_raw_audio, indiv) for indiv in filename_csv]

        if not bare_check:
            lost_levels = check_n_files(
                audio_file_list,
                10,
                output_folder=self.path.joinpath(
                    osm_path.raw_audio, "normalized_original"
                ),
                auto_normalization=auto_normalization,
            )

        for ind_dt in tqdm(range(len(timestamp_csv))):
            if ind_dt < len(timestamp_csv) - 1:
                diff = datetime.strptime(
                    timestamp_csv[ind_dt + 1], "%Y-%m-%dT%H:%M:%S.%fZ"
                ) - datetime.strptime(timestamp_csv[ind_dt], "%Y-%m-%dT%H:%M:%S.%fZ")
                list_interWavInterval.append(diff.total_seconds())

            audio_file = audio_file_list[ind_dt]

            list_filename.append(filename_csv[ind_dt])

            try:
                sr, frames, sampwidth, channels = read_header(audio_file)

            except Exception as e:
                list_file_problem.append(audio_file)
                print(
                    f"The audio file {audio_file} could not be loaded, its importation has been canceled.\nDescription of the error: {e}"
                )
                list_filename_abnormal_duration.append(audio_file)

            list_size.append(audio_file.stat().st_size / 1e6)

            list_duration.append(frames / float(sr))
            #     list_volumeFile.append( np.round(sr * params.nchannels * (sampwidth) * frames / float(sr) /1024 /1000))
            list_samplingRate.append(float(sr))
            list_sampwidth.append(sampwidth)

            # reformat timestamp.csv
            date_obj = datetime.strptime(timestamp_csv[ind_dt], "%Y-%m-%dT%H:%M:%S.%fZ")
            dates = datetime.strftime(date_obj, "%Y-%m-%dT%H:%M:%S.%f")
            # simply chopping !
            dates_final = dates[:-3] + "Z"
            timestamp.append(dates_final)

            # we remove the sign '-' in filenames (because of our qsub_resample.sh)
            if "-" in filename_csv[ind_dt]:
                cur_filename = filename_csv[ind_dt].replace("-", "_")
                Path(path_raw_audio, filename_csv[ind_dt]).rename(
                    path_raw_audio.joinpath(cur_filename)
                )
            else:
                cur_filename = filename_csv[ind_dt]
            filename_rawaudio.append(cur_filename)

        if list_filename_abnormal_duration:
            print(
                "Please see list of audio files above that canceled your dataset importation (maybe corrupted files with OkB volume ?). You can also find it in the list list_filename_abnormal_duration, and execute following cell to directly delete them. Those filenames have been written in the file ./raw/audio/files_not_loaded.csv"
            )

            with open(path_raw_audio.joinpath("files_not_loaded.csv"), "w") as fp:
                fp.write("\n".join(list_filename_abnormal_duration))

            return list_filename_abnormal_duration

        dd = pd.DataFrame(list_interWavInterval).describe()
        print("Summary statistics on your INTER-FILE DURATION")
        print(dd[0].to_string())
        if dd[0]["std"] < 1e-10:
            dutyCycle_percent = round(
                100 * mean(list_duration) / mean(list_interWavInterval),
                1,
            )
        else:
            dutyCycle_percent = np.nan

        # write raw/metadata.csv
        data = {
            "sr_origin": mean(list_samplingRate),
            "sample_bits": int(8 * mean(list_sampwidth)),
            "nchannels": int(channels),
            "audio_file_number": len(filename_csv),
            "start_date": timestamp_csv[0],
            "end_date": timestamp_csv[-1],
            "duty_cycle": dutyCycle_percent,
            "audio_file_origin_duration": round(mean(list_duration), 2),
            "audio_file_origin_volume": mean(list_size),
            "dataset_origin_volume": round(
                sum(list_size) / 1000,
                1,
            ),
            "dataset_origin_duration": round(
                sum(list_duration) / 60,  # miiiiight break smth. We'll see.
                2,
            ),
            "lat": self.gps_coordinates[0],
            "lon": self.gps_coordinates[1],
            "lost_levels_in_normalization": lost_levels,
        }
        df = pd.DataFrame.from_records([data])

        df["dataset_sr"] = float(mean(list_samplingRate))
        df["dataset_fileDuration"] = round(mean(list_duration), 2)
        df.to_csv(
            path_raw_audio.joinpath("metadata.csv"),
            index=False,
        )

        # get files with too small duration
        nominalVal_duration = int(np.percentile(list_duration, 10))
        print("\n Summary statistics on your file DURATION")
        dd_duration = pd.DataFrame(list_duration).describe()
        print(dd_duration[0].to_string())
        # go through the duration and check whether abnormal files
        ct_abnormal_duration = 0
        self.list_abnormal_filenames = []
        list_abnormalFilename_duration = []

        for name, duration in zip(list_filename, list_duration):
            if int(duration) < int(nominalVal_duration):
                ct_abnormal_duration += 1
                self.list_abnormal_filenames.append(name)
                list_abnormalFilename_duration.append(duration)

        if ct_abnormal_duration > 0 and not force_upload:
            print(
                "\n \n SORRY but your dataset contains files with different durations, especially",
                str(len(self.list_abnormal_filenames)),
                "files that have durations smaller than the 10th percentile of all your file durations.. \n",
            )

            print(
                "Here are their summary stats:",
                pd.DataFrame(list_abnormalFilename_duration).describe()[0].to_string(),
                "\n",
            )

            print(
                "So YOUR DATASET HAS NOT BEEN IMPORTED ON OSMOSE PLATFORM, but you have the choice now : either 1) you can force the upload using the variable force_upbload , or 2) you can first delete those files with small durations, they have been put into the variable list_abnormalFilename_name and can be removed from your dataset using the cell below"
            )

        else:
            df = pd.DataFrame({"filename": filename_rawaudio, "timestamp": timestamp})
            df.sort_values(by=["timestamp"], inplace=True)
            df.to_csv(
                path_raw_audio.joinpath("timestamp.csv"),
                index=False,
                na_rep="NaN",
                header=None,
            )

            # change name of the original wav folder
            new_folder_name = path_raw_audio.joinpath(
                str(int(mean(list_duration))) + "_" + str(int(mean(list_samplingRate)))
            )

            path_raw_audio = path_raw_audio.rename(new_folder_name)

            # rename filenames in the subset_files.csv if any to replace -' by '_'
            subset_path = osm_path.processed.joinpath("subset_files.csv")
            if subset_path.is_file():
                xx = pd.read_csv(subset_path, header=None).values
                pd.DataFrame([ff[0].replace("-", "_") for ff in xx]).to_csv(
                    subset_path,
                    index=False,
                    header=None,
                )

            # save lists of metadata in metadata_file
            f = open(path_raw_audio.joinpath("metadata.csv"), "w")
            for i in range(len(list_duration)):
                f.write(
                    f"{filename_rawaudio[i]} {list_duration[i]} {list_samplingRate[i]}\n"
                )
            f.close()

            # change permission on the dataset
            if force_upload:
                print("\n Well you have anomalies but you choose to FORCE UPLOAD")
            print("\n Now setting OSmOSE permissions ; wait a bit ...")
            gid = grp.getgrnam(owner_group).gr_gid

            os.chown(self.path, -1, gid)
            os.chmod(self.path, 0o770)
            for path in self.path.rglob("*"):
                os.chown(path, -1, gid)
                os.chmod(path, 0o770)
            print("\n DONE ! your dataset is on OSmOSE platform !")

    def delete_abnormal_files(self) -> None:
        """Delete all files with abnormal durations in the dataset, and rewrite the timestamps.csv file to reflect the changes.
        If no abnormal file is detected, then it does nothing."""

        if not self.list_abnormal_filenames:
            warn(
                "No abnormal file detected. You need to run the Dataset.build() method in order to detect abnormal files before using this method."
            )
            return

        path_raw_audio = os.path.join(self.path, "raw", "audio", "original")

        csvFileArray = pd.read_csv(
            os.path.join(path_raw_audio, "timestamp.csv"), header=None
        )

        for abnormal_file in self.list_abnormal_filenames:
            audio_file = os.path.join(path_raw_audio, abnormal_file)

            csvFileArray = csvFileArray.drop(
                csvFileArray[
                    csvFileArray[0].values == os.path.basename(abnormal_file)
                ].index
            )

            print(f"removing : {os.path.basename(abnormal_file)}")
            os.remove(audio_file)

        csvFileArray.sort_values(by=[1], inplace=True)
        csvFileArray.to_csv(
            os.path.join(path_raw_audio, "timestamp.csv"),
            index=False,
            na_rep="NaN",
            header=None,
        )

        print(
            "\n ALL ABNORMAL FILES REMOVED ! you can now re-run the build() method to finish importing it on OSmOSE platform"
        )
