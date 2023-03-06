import glob
import os
import random
from typing import Union, Tuple, List
from datetime import datetime
from warnings import warn

try:
    import grp

    skip_perms = False
except ModuleNotFoundError:
    skip_perms = True

import pandas as pd
import numpy as np
from tqdm import tqdm
from OSmOSE.utils import read_header, safe_read
import soundfile as sf


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
        >>> from OSmOSE import Dataset
        >>> dataset = Dataset(os.path.join("home","user","my_dataset"), coordinates = [49.2, -5], owner_group = "gosmose")
        """
        self.__name = os.path.basename(dataset_path)
        self.__path = dataset_path
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
    def Name(self):
        """str: The Dataset name. It is readonly."""
        return self.__name

    @property
    def Path(self):
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
        coordinates: Union[
            str,
            List[float],
            List[List[float]],
            Tuple[float, float],
            Tuple[Tuple[float, float], Tuple[float, float]],
        ],
    ):
        # TODO: Allow any iterator?
        match type(coordinates):
            case str():
                csvFileArray = pd.read_csv(
                    os.path.join(self.Path, "raw", "auxiliary", coordinates)
                )
                self.__gps_coordinates = [
                    (np.min(csvFileArray["lat"]), np.max(csvFileArray["lat"])),
                    (np.min(csvFileArray["lon"]), np.max(csvFileArray["lon"])),
                ]
            case tuple():
                self.__gps_coordinates = coordinates
            case list():
                if all(isinstance(coord, list) for coord in coordinates):
                    self.__gps_coordinates = (
                        (coordinates[0][0], coordinates[0][1]),
                        (coordinates[1][0], coordinates[1][1]),
                    )
                elif all(isinstance(coord, float) for coord in coordinates):
                    self.__gps_coordinates = (coordinates[0], coordinates[1])
                else:
                    raise ValueError(
                        f"The coordinates list must contain either only floats or only sublists of two elements."
                    )
            case _:
                raise TypeError(
                    f"GPS coordinates must be either a list of coordinates or the name of csv containing the coordinates, but {type(coordinates)} found."
                )

    @property
    def Owner_Group(self):
        """str: The Unix group able to interact with the dataset."""
        if self.__group is None:
            print(
                "The OSmOSE group name is not defined. Please specify the group name before trying to build the dataset."
            )
        return self.__group

    @Owner_Group.setter
    def Owner_Group(self, value):
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
        """Checks if self.Path/raw/audio contains at least one folder and none called "original"."""
        return len(
            os.listdir(os.path.join(self.Path, "raw", "audio"))
        ) > 0 and not os.path.exists(
            os.path.join(self.Path, "raw", "audio", "original")
        )

    # endregion

    def build(
        self,
        *,
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
            >>> from OSmOSE import Dataset
            >>> dataset = Dataset(os.path.join("home","user","my_dataset"))
            >>> dataset.build()

            DONE ! your dataset is on OSmOSE platform !
        """
        if owner_group is None:
            owner_group = self.Owner_Group

        path_timestamp_formatted = os.path.join(
            self.Path, "raw", "audio", "original", "timestamp.csv"
        )  # TODO: turn original into wildcard
        path_raw_audio = os.path.join(self.Path, "raw", "audio", "original")

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

        audio_file_list = [
            os.path.join(path_raw_audio, indiv) for indiv in filename_csv
        ]

        if not bare_check:
            lost_levels = self.check_n_files(
                audio_file_list, 10, auto_normalization=auto_normalization
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

            list_size.append(os.path.getsize(audio_file) / 1e6)

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
                os.rename(
                    os.path.join(path_raw_audio, filename_csv[ind_dt]),
                    os.path.join(path_raw_audio, cur_filename),
                )
            else:
                cur_filename = filename_csv[ind_dt]
            filename_rawaudio.append(cur_filename)

        if list_filename_abnormal_duration:
            print(
                "Please see list of audio files above that canceled your dataset importation (maybe corrupted files with OkB volume ?). You can also find it in the list list_filename_abnormal_duration, and execute following cell to directly delete them. Those filenames have been written in the file ./raw/audio/files_not_loaded.csv"
            )

            with open(
                os.path.join(self.Path, "raw", "audio", "files_not_loaded.csv"), "w"
            ) as fp:
                fp.write("\n".join(list_filename_abnormal_duration))

            return list_filename_abnormal_duration

        dd = pd.DataFrame(list_interWavInterval).describe()
        print("Summary statistics on your INTER-FILE DURATION")
        print(dd[0].to_string())
        if dd[0]["std"] < 1e-10:
            dutyCycle_percent = round(
                100
                * pd.DataFrame(list_duration).values.flatten().mean()
                / pd.DataFrame(list_interWavInterval).values.flatten().mean(),
                1,
            )
        else:
            dutyCycle_percent = np.nan

        # write raw/metadata.csv
        data = {
            "orig_fs": float(pd.DataFrame(list_samplingRate).values.flatten().mean()),
            "sound_sample_size_in_bits": int(
                8 * pd.DataFrame(list_sampwidth).values.flatten().mean()
            ),
            "nchannels": int(channels),
            "audio_file_number": len(filename_csv),
            "start_date": timestamp_csv[0],
            "end_date": timestamp_csv[-1],
            "dutyCycle_percent": dutyCycle_percent,
            "origin_audio_file_duration": round(
                pd.DataFrame(list_duration).values.flatten().mean(), 2
            ),
            "origin_audio_file_volume": pd.DataFrame(list_size).values.flatten().mean(),
            "origin_dataset_volume": round(
                pd.DataFrame(list_size).values.flatten().mean()
                * len(filename_csv)
                / 1000,
                1,
            ),
            "origin_dataset_duration": round(
                pd.DataFrame(list_duration).values.flatten().mean()
                * len(filename_csv)
                / 60,
                2,
            ),
            "lat": self.gps_coordinates[0],
            "lon": self.gps_coordinates[1],
            "lost_levels_in_normalization": lost_levels,
        }
        df = pd.DataFrame.from_records([data])
        df.to_csv(os.path.join(self.Path, "raw", "metadata.csv"), index=False)

        # write raw/audio/original/metadata.csv
        df["dataset_fs"] = float(
            pd.DataFrame(list_samplingRate).values.flatten().mean()
        )
        df["dataset_fileDuration"] = round(
            pd.DataFrame(list_duration).values.flatten().mean(), 2
        )
        df.to_csv(
            os.path.join(self.Path, "raw", "audio", "original", "metadata.csv"),
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
                os.path.join(self.Path, "raw", "audio", "original", "timestamp.csv"),
                index=False,
                na_rep="NaN",
                header=None,
            )

            # change name of the original wav folder
            new_folder_name = os.path.join(
                self.Path,
                "raw",
                "audio",
                str(int(pd.DataFrame(list_duration).values.flatten().mean()))
                + "_"
                + str(
                    int(float(pd.DataFrame(list_samplingRate).values.flatten().mean()))
                ),
            )
            os.rename(
                os.path.join(self.Path, "raw", "audio", "original"), new_folder_name
            )

            # rename filenames in the subset_files.csv if any to replace -' by '_'
            if os.path.isfile(os.path.join(self.Path, "analysis/subset_files.csv")):
                xx = pd.read_csv(
                    os.path.join(self.Path, "analysis/subset_files.csv"), header=None
                ).values
                pd.DataFrame([ff[0].replace("-", "_") for ff in xx]).to_csv(
                    os.path.join(self.Path, "analysis/subset_files.csv"),
                    index=False,
                    header=None,
                )

            # save lists of metadata in metadata_file
            f = open(os.path.join(new_folder_name, "metadata_file.csv"), "w")
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

            os.chown(self.Path, -1, gid)
            os.chmod(self.Path, 0o770)
            for dirpath, dirnames, filenames in os.walk(self.Path):
                for filename in filenames:
                    os.chown(os.path.join(dirpath, filename), -1, gid)
                    os.chmod(os.path.join(dirpath, filename), 0o770)
            print("\n DONE ! your dataset is on OSmOSE platform !")

    def check_n_files(
        self,
        file_list: list,
        n: int,
        *,
        threshold_percent: float = 0.1,
        auto_normalization: bool = False,
    ) -> bool:
        """Check n files at random for anomalies and may normalize them.

        Currently, check if the data for wav in PCM float format are between -1.0 and 1.0. If the number of files that
        fail the test is higher than the threshold (which is 10% of n by default, with an absolute minimum of 1), all the
        dataset will be normalized and written in another file.

        Parameters
        ----------
            file_list: `list`
                The list of files to be evaluated. It must be equal or longer than n.
            n: `int`
                The number of files to evaluate. To lower resource consumption, it is advised to check only a subset of the dataset.
                10 files taken at random should provide an acceptable idea of the whole dataset.
            threshold_percent: `float`, optional, keyword-only
                The maximum acceptable percentage of evaluated files that can contain anomalies. Understands fraction and whole numbers. Default is 0.1, or 10%
            auto_normalization: `bool`, optional, keyword_only
                Whether the normalization should proceed automatically or not if the threshold is reached. As a safeguard, the default is False.
        Returns
        -------
            normalized: `bool`
                Indicates whether or not the dataset has been normalized.
        """
        if threshold_percent > 1:
            threshold_percent = threshold_percent / 100

        if "float" in str(sf.info(file_list[0])):
            threshold = max(threshold_percent * n, 1)
            bad_files = []
            for audio_file in random.sample(file_list, n):
                data, sr = safe_read(audio_file)
                if not (np.max(data) < 1.0 and np.min(data) > -1.0):
                    bad_files.append(audio_file)

                    if len(bad_files) > threshold:
                        print(
                            "The treshold has been exceeded, too many files unadequately recorded."
                        )
                        if not auto_normalization:
                            raise ValueError(
                                "You need to set auto_normalization to True to normalize your dataset automatically."
                            )
                        if not os.path.exists(
                            os.path.join(
                                self.Path, "raw", "audio", "normalized_original"
                            )
                        ):
                            os.makedirs(
                                os.path.join(
                                    self.Path, "raw", "audio", "normalized_original"
                                )
                            )
                        for audio_file in file_list:
                            data, sr = safe_read(audio_file)
                            data = (
                                (data - np.mean(data)) / np.std(data)
                            ) * 0.063  # = -24dB
                            data[data > 1] = 1
                            data[data < -1] = -1

                            sf.write(
                                os.path.join(
                                    self.Path,
                                    "raw",
                                    "audio",
                                    "normalized_original",
                                    os.path.basename(audio_file),
                                ),
                                data=data,
                                samplerate=sr,
                            )
                            # TODO: lock in spectrum mode
                        print(
                            "All files have been normalized. Spectrograms created from them will be locked in spectrum mode."
                        )
                        return True
        return False

    def delete_abnormal_files(self) -> None:
        """Delete all files with abnormal durations in the dataset, and rewrite the timestamps.csv file to reflect the changes.
        If no abnormal file is detected, then it does nothing."""

        if not self.list_abnormal_filenames:
            warn(
                "No abnormal file detected. You need to run the Dataset.build() method in order to detect abnormal files before using this method."
            )
            return

        path_raw_audio = os.path.join(self.Path, "raw", "audio", "original")

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
