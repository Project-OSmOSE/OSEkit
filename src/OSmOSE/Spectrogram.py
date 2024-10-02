from functools import partial
import inspect
import os
import shutil
import sys
from typing import Tuple, Union, Literal
from math import log10
from pathlib import Path
import multiprocessing as mp
import glob
from IPython.display import Image, display

from tqdm import tqdm
import itertools

from datetime import timedelta, datetime

import pandas as pd
import numpy as np
from scipy import signal
from termcolor import colored
from matplotlib import pyplot as plt
from OSmOSE.job import Job_builder
from OSmOSE.cluster import (
    reshape,
    compute_stats,
    merge_timestamp_csv,
)
from OSmOSE.Dataset import Dataset
from OSmOSE.utils.path_utils import make_path
from OSmOSE.utils.core_utils import (
    safe_read,
    set_umask,
    get_timestamp_of_audio_file,
)
from OSmOSE.utils.audio_utils import get_all_audio_files
from OSmOSE.config import OSMOSE_PATH, FPDEFAULT, DPDEFAULT
from OSmOSE.frequency_scales.frequency_scale_serializer import FrequencyScaleSerializer


class Spectrogram(Dataset):
    """Main class for spectrogram-related computations. Can resample, reshape and normalize audio files before generating spectrograms."""

    def __init__(
        self,
        dataset_path: str,
        *,
        gps_coordinates: Union[str, list, Tuple] = None,
        depth: Union[str, int] = None,
        dataset_sr: int = None,
        owner_group: str = None,
        analysis_params: dict = None,
        batch_number: int = 5,
        local: bool = True,
        verbose: bool = True,
        concat: bool = True,
    ) -> None:
        """Instanciates a spectrogram object.

        The characteristics of the dataset are essential to input for the generation of the spectrograms. There is three ways to input them:
            - Use the existing `analysis/analysis_sheet.csv` file. If one exist, it will take priority over the other methods. Note that
            when using this file, some attributes will be locked in read-only mode.
            - Fill the `analysis_params` argument. More info on the expected value below.
            - Don't initialize the attributes in the constructor, and assign their values manually.

        In any case, all attributes must have a value for the spectrograms to be generated. If it does not exist, `analysis/analysis_sheet.csv`
        will be written at the end of the `Spectrogram.initialize()` method.

        Parameters
        ----------
        dataset_path : `str`
            The absolute path to the dataset folder. The last folder in the path will be considered as the name of the dataset.
        dataset_sr : `int`, keyword-only
            The sample rate used for the generation of the spectrograms.
        coordinates : `str` or `list` or `tuple`, optional, keyword-only
            The GPS coordinates of the listening location. If it is of type `str`, it must be the name of a csv file located in `raw/auxiliary`,
            otherwise a list or a tuple with the first element being the latitude coordinates and second the longitude coordinates.
        osmose_group_name : `str`, optional, keyword-only
            The name of the group using the OsmOSE package. All files created using this dataset will be accessible by the osmose group.
            Will not work on Windows.
        analysis_params : `dict`, optional, keyword-only
            If `analysis/analysis_sheet.csv` does not exist, the analysis parameters can be submitted in the form of a dict,
            with keys matching what is expected:
                - nfft : `int`
                - window_size : `int`
                - overlap : `int`
                - colormap : `str`
                - zoom_level : `int`
                - dynamic_min : `int`
                - dynamic_max : `int`
                - number_adjustment_spectrogram : `int`
                - spectro_duration : `int`
                - zscore_duration : `float` or `str`
                - hp_filter_min_freq : `int`
                - sensitivity_dB : `int`
                - peak_voltage : `float`
                - spectro_normalization : `str`
                - data_normalization : `str`
                - gain_dB : `int`
            If additional information is given, it will be ignored. Note that if there is an `analysis/analysis_sheet.csv` file, it will
            always have the priority.
        batch_number : `int`, optional, keyword_only
            The number of batches the dataset files will be split into when submitting parallel jobs (the default is 5).
        local : `bool`, optional, keyword_only
            Indicates whether or not the program is run locally. If it is the case, it will not create jobs and will handle the paralelisation
            alone. The default is False.
        """
        super().__init__(
            dataset_path=dataset_path,
            owner_group=owner_group,
        )

        self.__local = local

        orig_metadata = pd.read_csv(
            self._get_original_after_build() / "metadata.csv", header=0
        )

        processed_path = self.path / OSMOSE_PATH.spectrogram
        metadata_path = processed_path / "adjustment_spectros" / "adjust_metadata.csv"

        if analysis_params:
            self.__analysis_file = False
            # We put the value in a list so that values[0] returns the right value below.
            analysis_sheet = {key: [value] for (key, value) in analysis_params.items()}
        elif metadata_path.exists():
            self.__analysis_file = True
            analysis_sheet = pd.read_csv(metadata_path, header=0)
        else:
            analysis_sheet = {}
            self.__analysis_file = False
            print(
                "No valid processed/adjust_metadata.csv found and no parameters provided. All attributes will be initialized to default values..  \n"
            )

        self.batch_number: int = batch_number

        self.dataset_sr: int = (
            dataset_sr if dataset_sr is not None else orig_metadata["origin_sr"][0]
        )

        self.nfft: int = analysis_sheet["nfft"][0] if "nfft" in analysis_sheet else 2048
        self.window_size: int = (
            analysis_sheet["window_size"][0]
            if "window_size" in analysis_sheet
            else 2048
        )
        self.overlap: int = (
            analysis_sheet["overlap"][0] if "overlap" in analysis_sheet else 0
        )
        self.colormap: str = (
            analysis_sheet["colormap"][0] if "colormap" in analysis_sheet else "viridis"
        )
        self.zoom_level: int = (
            analysis_sheet["zoom_level"][0] if "zoom_level" in analysis_sheet else 0
        )
        self.dynamic_min: int = (
            analysis_sheet["dynamic_min"][0] if "dynamic_min" in analysis_sheet else -30
        )
        self.dynamic_max: int = (
            analysis_sheet["dynamic_max"][0] if "dynamic_max" in analysis_sheet else 30
        )
        self.number_adjustment_spectrogram: int = (
            analysis_sheet["number_adjustment_spectrogram"][0]
            if "number_adjustment_spectrogram" in analysis_sheet
            else 1
        )
        self.spectro_duration: int = (
            int(analysis_sheet["spectro_duration"][0])
            if analysis_sheet is not None and "spectro_duration" in analysis_sheet
            else int(orig_metadata["audio_file_origin_duration"][0])
        )

        self.zscore_duration: Union[float, str] = (
            analysis_sheet["zscore_duration"][0]
            if "zscore_duration" in analysis_sheet
            and isinstance(analysis_sheet["zscore_duration"][0], float)
            else "original"
        )

        # fmin cannot be 0 in butterworth. If that is the case, it takes the smallest value possible, epsilon
        self.hp_filter_min_freq: int = (
            analysis_sheet["hp_filter_min_freq"][0]
            if "hp_filter_min_freq" in analysis_sheet
            else 0
        )

        self.sensitivity: float = (
            analysis_sheet["sensitivity_dB"][0]
            if "sensitivity_dB" in analysis_sheet
            else 0
        )

        self.peak_voltage: float = (
            analysis_sheet["peak_voltage"][0] if "peak_voltage" in analysis_sheet else 1
        )
        self.spectro_normalization: str = (
            analysis_sheet["spectro_normalization"][0]
            if "spectro_normalization" in analysis_sheet
            else "spectrum"
        )
        self.data_normalization: str = (
            analysis_sheet["data_normalization"][0]
            if "data_normalization" in analysis_sheet
            else "none"
        )
        self.gain_dB: float = (
            analysis_sheet["gain_dB"][0]
            if "gain_dB" in analysis_sheet is not None
            else 0
        )

        self.window_type: str = (
            analysis_sheet["window_type"][0]
            if "window_type" in analysis_sheet
            else "hamming"
        )

        self.audio_file_overlap: int = (
            analysis_sheet["audio_file_overlap"][0]
            if "audio_file_overlap" in analysis_sheet
            else 0
        )

        self.custom_frequency_scale: str = (
            analysis_sheet["custom_frequency_scale"][0]
            if "custom_frequency_scale" in analysis_sheet
            else "linear"
        )

        self.verbose: bool = verbose
        self.concat: bool = concat
        self.jb = Job_builder()

        plt.switch_backend("agg")

        fontsize = 16
        ticksize = 12
        plt.rc("font", size=fontsize)  # controls default text sizes
        plt.rc("axes", titlesize=fontsize)  # fontsize of the axes title
        plt.rc("axes", labelsize=fontsize)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=ticksize)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=ticksize)  # fontsize of the tick labels
        plt.rc("legend", fontsize=ticksize)  # legend fontsize
        plt.rc("figure", titlesize=ticksize)  # fontsize of the figure title

        self.__build_path(dry=True)

    @property
    def dataset_sr(self):
        """int: The sampling frequency of the dataset."""
        return self.__dataset_sr

    @dataset_sr.setter
    def dataset_sr(self, value: int):
        self.__dataset_sr = value

    @property
    def nfft(self):
        """int: The number of points used in the Fast Fourier Transform."""
        return self.__nfft

    @nfft.setter
    def nfft(self, value: int):
        self.__nfft = value

    @property
    def window_size(self):
        """int: Size of the window applied to the signal."""
        return self.__window_size

    @window_size.setter
    def window_size(self, value: int):
        self.__window_size = value

    @property
    def overlap(self):
        """int: The overlap percentage between two successive windows."""
        return self.__overlap

    @overlap.setter
    def overlap(self, value: int):
        self.__overlap = value

    @property
    def colormap(self):
        """str: The type of colormap of the spectrograms."""
        return self.__colormap

    @colormap.setter
    def colormap(self, value: str):
        self.__colormap = value

    @property
    def zoom_level(self):
        """int: Number of zoom levels."""
        return self.__zoom_level

    @zoom_level.setter
    def zoom_level(self, value: int):
        self.__zoom_level = value

    @property
    def dynamic_min(self):
        """int: Minimum value of the colormap."""
        return self.__dynamic_min

    @dynamic_min.setter
    def dynamic_min(self, value: int):
        self.__dynamic_min = value

    @property
    def dynamic_max(self):
        """int: Maximum value of the colormap."""
        return self.__dynamic_max

    @dynamic_max.setter
    def dynamic_max(self, value: int):
        self.__dynamic_max = value

    @property
    def number_adjustment_spectrogram(self):
        """int: Number of spectrograms used to adjust the parameters."""
        return self.__number_adjustment_spectrogram

    @number_adjustment_spectrogram.setter
    def number_adjustment_spectrogram(self, value: int):
        self.__number_adjustment_spectrogram = value

    @property
    def audio_file_overlap(self):
        """int: Overlap between segmented audio files."""
        return self.__audio_file_overlap

    @property
    def spectro_duration(self):
        """int: Duration of the spectrogram (at the lowest zoom level) in seconds."""
        return self.__spectro_duration

    @spectro_duration.setter
    def spectro_duration(self, value: int):
        self.__spectro_duration = value

    @audio_file_overlap.setter
    def audio_file_overlap(self, value: int):
        if value < self.spectro_duration:
            self.__audio_file_overlap = value
        else:
            raise ValueError(
                "Segmented audio file overlapping value must be smaller than spectro_duration"
            )

    @property
    def zscore_duration(self):
        return self.__zscore_duration

    @zscore_duration.setter
    def zscore_duration(self, value: int):
        self.__zscore_duration = value

    @property
    def hp_filter_min_freq(self):
        """float: Floor frequency for the High Pass Filter."""
        return self.__hp_filter_min_freq

    @hp_filter_min_freq.setter
    def hp_filter_min_freq(self, value: int):
        self.__hp_filter_min_freq = value

    @property
    def sensitivity(self):
        """int: Numeric sensitivity of the recording device."""
        return self.__sensitivity

    @sensitivity.setter
    def sensitivity(self, value):
        """Always assume the sensitivity is given in dB."""
        self.__sensitivity = 10 ** (value / 20) * 1e6

    @property
    def peak_voltage(self):
        """float: The maximum voltage of the device."""
        return self.__peak_voltage

    @peak_voltage.setter
    def peak_voltage(self, value: float):
        self.__peak_voltage = value

    @property
    def spectro_normalization(self):
        """str: Type of normalization used to generate the spectrograms."""
        return self.__spectro_normalization

    @spectro_normalization.setter
    def spectro_normalization(self, value: Literal["spectrum", "density"]):
        self.__spectro_normalization = value

    @property
    def data_normalization(self):
        """str: Type of normalization applied to the data."""
        return self.__data_normalization

    @data_normalization.setter
    def data_normalization(self, value: Literal["instrument", "zscore", "none"]):
        self.__data_normalization = value

    @property
    def gain_dB(self):
        """float: Gain of the device in decibels."""
        return self.__gain_dB

    @gain_dB.setter
    def gain_dB(self, value: float):
        self.__gain_dB = value

    @property
    def window_type(self):
        """str: Type of the window used to generate the spectrograms."""
        return self.__window_type

    @window_type.setter
    def window_type(self, value: Literal["hamming"]):
        self.__window_type = value

    @property
    def frequency_resolution(self) -> float:
        """Frequency resolution of the spectrogram, calculated by dividing the samplerate by nfft."""
        return self.dataset_sr / self.nfft

    @property
    def time_resolution(self):
        return self.__time_resolution

    @time_resolution.setter
    def time_resolution(self, value):
        self.__time_resolution = value

    @property
    def custom_frequency_scale(self):
        return self.__custom_frequency_scale

    @custom_frequency_scale.setter
    def custom_frequency_scale(self, value):
        self.__custom_frequency_scale = value

    def __build_path(
        self, adjust: bool = False, dry: bool = False, force_init: bool = False
    ):
        """Build some internal paths according to the expected architecture and might create them.

        Parameter
        ---------
            adjust : `bool`, optional
                Whether or not the paths are used to adjust spectrogram parameters.
            dry: `bool`, optional
                If set to True, will not create the folders and just return the file path.
        """
        set_umask()
        processed_path = self.path / OSMOSE_PATH.spectrogram
        audio_foldername = f"{str(self.spectro_duration)}_{str(self.dataset_sr)}"
        self.audio_path = self.path / OSMOSE_PATH.raw_audio / audio_foldername

        self.__spectro_foldername = f"{str(self.nfft)}_{str(self.window_size)}_{str(self.overlap)}_{self.custom_frequency_scale}"

        self.path_output_spectrogram = (
            processed_path / audio_foldername / self.__spectro_foldername / "image"
        )
        self.path_output_spectrogram_matrix = (
            processed_path / audio_foldername / self.__spectro_foldername / "matrix"
        )

        self.path_output_welch = self.path / OSMOSE_PATH.welch
        self.path_output_LTAS = self.path / OSMOSE_PATH.LTAS
        self.path_output_EPD = self.path / OSMOSE_PATH.EPD
        self.path_output_SPLfiltered = self.path / OSMOSE_PATH.SPLfiltered

        make_path(
            self.path / OSMOSE_PATH.spectrogram / "adjustment_spectros",
            mode=DPDEFAULT,
        )

        if not dry:
            if self.audio_path.exists() and force_init:
                print(
                    f"removing existing directory {self.audio_path}.. this can take a bit of time"
                )
                shutil.rmtree(self.audio_path)
            make_path(self.audio_path, mode=DPDEFAULT)

            if self.path_output_spectrogram.exists() and force_init:
                print(
                    f"removing existing directory {self.path_output_spectrogram}.. this can take a bit of time"
                )
                shutil.rmtree(self.path_output_spectrogram)
            make_path(self.path_output_spectrogram, mode=DPDEFAULT)

            if not adjust:
                make_path(self.path_output_spectrogram_matrix, mode=DPDEFAULT)
                make_path(self.path / OSMOSE_PATH.statistics, mode=DPDEFAULT)

    def extract_spectro_params(self):
        tile_duration = self.spectro_duration / 2 ** (self.zoom_level)
        data = np.zeros([int(tile_duration * self.dataset_sr), 1])
        Noverlap = int(self.window_size * self.overlap / 100)
        Nbech = np.size(data)
        Noffset = self.window_size - Noverlap
        Nbwin = int((Nbech - self.window_size) / Noffset)
        Freq = np.fft.rfftfreq(self.nfft, d=1 / self.dataset_sr)
        Time = np.linspace(0, Nbech / self.dataset_sr, Nbwin)

        temporal_resolution = round(Time[1] - Time[0], 3)
        frequency_resolution = round(Freq[1] - Freq[0], 3)

        return temporal_resolution, frequency_resolution, Nbwin

    def check_spectro_size(self):
        """Verify if the parameters will generate a spectrogram that can fit one screen properly"""
        if self.nfft > 2048:
            print(f"Your spectra contain more than 1024 bin (ie {self.nfft/2}). \n")
            print(
                colored(
                    "Note that unless you have a 4K screen, unwanted numerical compression might occur when visualizing your spectrograms..",
                    "red",
                )
            )

        temporal_resolution, frequency_resolution, Nbwin = self.extract_spectro_params()

        print(
            f"your smallest tile has a duration of: {self.spectro_duration / 2 ** (self.zoom_level)} (s), with a number of spectra of {Nbwin} \n"
        )

        if Nbwin > 3500:
            print(
                colored(
                    "Note that unless you have a 4K screen, unwanted numerical compression might occur when visualizing your spectrograms..",
                    "red",
                )
            )

        print("\n")
        print(
            "your resolutions : time = ",
            temporal_resolution,
            "(s) / frequency = ",
            frequency_resolution,
            "(Hz)",
        )

    def initialize(
        self,
        *,
        datetime_begin: str = None,
        datetime_end: str = None,
        dataset_sr: int = None,
        batch_ind_min: int = 0,
        batch_ind_max: int = -1,
        force_init: bool = False,
        env_name: str = None,
        threshold: int = 5,
    ) -> None:
        """Prepares everything (path, variables, files) for spectrogram generation. This needs to be run before the spectrograms are generated.
        If the dataset has not yet been build, it is before the rest of the functions are initialized.

        Parameters
        ----------
        datetime_begin : `Timestamp`,
            The begin datetime of the audio files used to generate the spectrograms.
        datetime_end : `Timestamp`,
            The end datetime of the audio files used to generate the spectrograms.
        dataset_sr : `int`, optional, keyword-only
            The sampling frequency of the audio files used to generate the spectrograms. If set, will overwrite the Spectrogram.dataset_sr attribute.
        batch_ind_min : `int`, optional, keyword-only
            The index of the first file to consider. Both this parameter and `batch_ind_max` are not commonly used and are
            for very specific use cases. Most of the time, you want to initialize the whole dataset (the default is 0).
        batch_ind_max : `int`, optional, keyword-only
            The index of the last file to consider (the default is -1, meaning consider every file).
        force_init : `bool`, optional, keyword-only
            Force every parameter of the initialization.
        threshold : int, optional
            Integer from 0 to 100 to filter out segments with a number of sample inferior to (threshold * spectrogram duration * segment_sample_rate)
        """

        # remove temp directories from adjustment spectrograms
        for path in glob.glob(str(self.path / OSMOSE_PATH.raw_audio / "temp_*")):
            shutil.rmtree(path)

        # remove existing log directory and create a new empty one
        if os.path.exists(self.path / OSMOSE_PATH.log):
            shutil.rmtree(self.path / OSMOSE_PATH.log)
            os.mkdir(self.path / OSMOSE_PATH.log)

        # remove welch directory if existing
        if self.path_output_welch.exists():
            shutil.rmtree(self.path_output_welch)
            make_path(
                self.path_output_welch,
                mode=DPDEFAULT,
            )

        # create some internal paths
        self.__build_path(force_init=force_init)

        if not (
            self.path
            / OSMOSE_PATH.spectrogram
            / "adjustment_spectros"
            / "adjust_metadata.csv"
        ).exists():
            self.save_spectro_metadata(True)

        if dataset_sr:
            self.dataset_sr = dataset_sr

        self.path_input_audio_file = self._get_original_after_build()

        list_audio_withEvent_comp = get_all_audio_files(self.path_input_audio_file)

        if batch_ind_max == -1:
            batch_ind_max = len(list_audio_withEvent_comp)

        list_audio_withEvent = list_audio_withEvent_comp[batch_ind_min:batch_ind_max]

        self.list_audio_to_process = [
            audio_file.name for audio_file in list_audio_withEvent
        ]

        # Load variables from metadata origin files
        metadata = pd.read_csv(self.path_input_audio_file / "metadata.csv")
        file_metadata = pd.read_csv(
            self.path_input_audio_file / "file_metadata.csv",
            parse_dates=["timestamp"],
        )
        audio_metadata_path = (
            self.path
            / OSMOSE_PATH.raw_audio
            / f"{str(self.spectro_duration)}_{str(self.dataset_sr)}"
            / "metadata.csv"
        )
        audio_file_origin_duration = file_metadata["duration"]
        origin_sr = metadata["origin_sr"][0]

        # List containing the last job ids to grab outside of the class
        self.pending_jobs = []
        self.finished_jobs = []

        if audio_metadata_path.exists():
            print(
                "It seems these spectrogram parameters are already initialized. If it is an error or you want to rerun the initialization, add the `force_init` argument."
            )
            return

        # process only a subset of the dataset
        if (self.path / OSMOSE_PATH.processed / "subset_files.csv").is_file():
            subset = pd.read_csv(
                self.path / OSMOSE_PATH.processed / "subset_files.csv",
                header=None,
            )[0].values
            self.list_audio_to_process = list(
                set(subset).intersection(set(self.list_audio_to_process))
            )

        # if datetime begin/end are not provided, takes the datetime of first audio file/datetime + duration of last audio file
        if not datetime_begin:
            datetime_begin = (file_metadata["timestamp"].iloc[0]).strftime(
                "%Y-%m-%dT%H:%M:%S%z"
            )

        if not datetime_end:
            datetime_end = (
                file_metadata["timestamp"].iloc[-1]
                + pd.Timedelta(file_metadata["duration"].iloc[-1], unit="s")
            ).strftime("%Y-%m-%dT%H:%M:%S%z")

        # check datetimes
        if not pd.Timestamp(datetime_begin) < pd.Timestamp(datetime_end):
            raise ValueError(
                f"'datetime_begin' must be anterior to 'datetime_end'.\ndatetime_begin is set to {pd.Timestamp(datetime_begin)} and datetime_end is set to {pd.Timestamp(datetime_end)}"
            )

        # new timestamps calculation to determine the size of a batch
        if self.concat:
            new_file = pd.date_range(
                start=pd.Timestamp(datetime_begin),
                end=pd.Timestamp(datetime_end),
                freq=f"{self.spectro_duration}s",
            ).to_list()[:-1]
        else:
            selected_file_metadata = file_metadata[
                (file_metadata["timestamp"] >= datetime_begin)
                & (file_metadata["timestamp"] <= datetime_end)
            ]
            new_file = []
            for i, ts in enumerate(selected_file_metadata["timestamp"]):
                counter = 0
                current_ts = ts
                original_timedelta = pd.Timedelta(
                    seconds=selected_file_metadata["duration"].iloc[i]
                )

                while current_ts <= ts + original_timedelta:
                    new_file.append(current_ts)
                    current_ts += pd.Timedelta(seconds=self.spectro_duration)
                    counter += 1
            new_file = new_file[: -counter + 1] if counter > 1 else new_file

        # size of a batch
        batch_size = len(new_file) // self.batch_number
        if batch_size == 0:
            self.batch_number = 1
            batch_size = len(new_file) // self.batch_number

        # if a resample or a segmentation or both is necessary
        if (int(self.spectro_duration) != int(audio_file_origin_duration.mean())) or (
            self.dataset_sr != origin_sr
        ):
            print(
                f"Automatically reshaping audio files to fit the spectro duration value. Files will be {self.spectro_duration} seconds long. Parameter 'concat' is set to {self.concat}."
            )

            input_files = self.path_input_audio_file
            processes = []
            reshape_job_id_list = []
            i_max = -1

            for batch in range(self.batch_number):

                i_min = i_max + 1
                i_max = (
                    i_min + batch_size
                    if batch < self.batch_number - 1
                    and i_min + batch_size < len(new_file)
                    else len(new_file) - 1
                )

                if self.__local:
                    process = mp.Process(
                        target=reshape,
                        kwargs={
                            "input_files": input_files,
                            "file_metadata_path": self.path_input_audio_file
                            / "file_metadata.csv",
                            "timestamp_path": self.path_input_audio_file
                            / "timestamp.csv",
                            "output_dir_path": self.audio_path,
                            "datetime_begin": datetime_begin,
                            "datetime_end": datetime_end,
                            "segment_size": int(self.spectro_duration),
                            "new_sr": int(self.dataset_sr),
                            "threshold": threshold,
                            "batch": batch,
                            "batch_num": self.batch_number,
                            "batch_ind_min": i_min,
                            "batch_ind_max": i_max,
                            "concat": self.concat,
                            "verbose": self.verbose,
                        },
                    )

                    process.start()
                    processes.append(process)

                else:
                    self.jb.build_job_file(
                        script_path=Path(inspect.getfile(reshape)).resolve(),
                        script_args=f"--input-files {input_files}\
                                --file-metadata-path {self.path_input_audio_file / 'file_metadata.csv'}\
                                --timestamp-path {self.path_input_audio_file / 'timestamp.csv'}\
                                --output-dir {self.audio_path}\
                                --datetime-begin {datetime_begin}\
                                --datetime-end {datetime_end}\
                                --segment-size {int(self.spectro_duration)}\
                                --new-sr {int(self.dataset_sr)}\
                                --threshold {threshold}\
                                --batch {batch}\
                                --batch-num {self.batch_number}\
                                --batch-ind-min {i_min}\
                                --batch-ind-max {i_max}\
                                --concat {self.concat}\
                                {'--verbose' if self.verbose else ''}",
                        jobname=f"reshape_{batch}",
                        preset="low",
                        mem="30G",
                        walltime="04:00:00",
                        logdir=self.path / OSMOSE_PATH.log,
                        env_name=env_name,
                    )

                    job_id = self.jb.submit_job()
                    reshape_job_id_list += job_id

            for process in processes:
                process.join()

        # merge timestamps_*.csv after reshaping
        if not self.__local:
            self.jb.build_job_file(
                script_path=Path(inspect.getfile(merge_timestamp_csv)).resolve(),
                script_args=f"--input-files {self.audio_path}",
                jobname="merge_timestamp",
                preset="low",
                mem="30G",
                walltime="04:00:00",
                logdir=self.path / "log",
                env_name=env_name,
            )
            job_id = self.jb.submit_job(dependency=reshape_job_id_list)

            self.pending_jobs = job_id

        else:
            input_dir_path = self.audio_path

            list_conca_timestamps = []
            list_conca_filename = []
            for ll in list(input_dir_path.glob("timestamp_*")):
                print(f"read and remove file {ll}")
                list_conca_timestamps.append(list(pd.read_csv(ll)["timestamp"].values))
                list_conca_filename.append(list(pd.read_csv(ll)["filename"].values))
                os.remove(ll)

            print(f"save file {str(input_dir_path / 'timestamp.csv')}")
            df = pd.DataFrame(
                {
                    "filename": list(
                        itertools.chain.from_iterable(list_conca_filename)
                    ),
                    "timestamp": list(
                        itertools.chain.from_iterable(list_conca_timestamps)
                    ),
                }
            )
            df.sort_values(by=["timestamp"], inplace=True)
            df.to_csv(input_dir_path / "timestamp.csv", index=False)

        # ZSCORE NORMALIZATION
        norma_job_id_list = []
        if (
            self.data_normalization == "zscore"
            and self.zscore_duration != "original"
            and (len(os.listdir(self.path / OSMOSE_PATH.statistics)) == 0 or force_init)
        ):
            shutil.rmtree(self.path / OSMOSE_PATH.statistics, ignore_errors=True)
            make_path(self.path / OSMOSE_PATH.statistics, mode=DPDEFAULT)
            for batch in range(self.batch_number):
                i_min = batch * batch_size
                i_max = (
                    i_min + batch_size
                    if batch < self.batch_number - 1
                    else len(self.list_audio_to_process)
                )  # If it is the last batch, take all files

                if self.__local:
                    process = mp.Process(
                        target=compute_stats,
                        kwargs={
                            "input_dir": self.audio_path,
                            "output_file": self.path
                            / OSMOSE_PATH.statistics
                            / f"SummaryStats_{i_min}.csv",
                            "hp_filter_min_freq": self.hp_filter_min_freq,
                            "batch_ind_min": i_min,
                            "batch_ind_max": i_max,
                        },
                    )

                    process.start()
                    processes.append(process)
                else:
                    jobfile = self.jb.build_job_file(
                        script_path=Path(inspect.getfile(compute_stats)).resolve(),
                        script_args=f"--input-dir {self.path_input_audio_file}\
                            --hp-filter-min-freq {self.hp_filter_min_freq}\
                            --batch-ind-min {i_min}\
                            --batch-ind-max {i_max}\
                            --output-file {self.path / OSMOSE_PATH.statistics / f'SummaryStats_{i_min}.csv'}",
                        jobname="OSmOSE_get_zscore_params",
                        preset="low",
                        logdir=self.path / OSMOSE_PATH.log,
                    )

                    job_id = self.jb.submit_job()
                    norma_job_id_list += job_id

            for process in processes:
                process.join()

        # write metadata.csv for segmented audio files
        metadata["audio_file_dataset_duration"] = self.spectro_duration
        metadata["dataset_sr"] = self.dataset_sr
        metadata["audio_file_dataset_overlap"] = self.audio_file_overlap

        origin_dt = pd.read_csv(
            self.path_input_audio_file / "timestamp.csv", parse_dates=["timestamp"]
        )["timestamp"]

        metadata["audio_file_count"] = sum(
            1 for dt in new_file if origin_dt.iloc[0] <= dt <= origin_dt.iloc[-1]
        )

        metadata["start_date"] = datetime_begin
        metadata["end_date"] = datetime_end
        new_meta_path = self.audio_path / "metadata.csv"

        if new_meta_path.exists():
            new_meta_path.unlink()
        metadata.to_csv(new_meta_path, index=False)
        os.chmod(new_meta_path, mode=FPDEFAULT)

    def save_spectro_metadata(self, adjust_bool: bool):
        temporal_resolution, frequency_resolution, Nbwin = self.extract_spectro_params()

        data = {
            "dataset_name": self.name,
            "dataset_sr": self.dataset_sr,
            "nfft": self.nfft,
            "window_size": self.window_size,
            "overlap": self.overlap,
            "colormap": self.colormap,
            "zoom_level": self.zoom_level,
            "number_adjustment_spectrogram": self.number_adjustment_spectrogram,
            "dynamic_min": self.dynamic_min,
            "dynamic_max": self.dynamic_max,
            "spectro_duration": self.spectro_duration,
            "audio_file_folder_name": self.audio_path.name,
            "data_normalization": self.data_normalization,
            "hp_filter_min_freq": self.hp_filter_min_freq,
            "sensitivity_dB": 20 * log10(self.sensitivity / 1e6),
            "peak_voltage": self.peak_voltage,
            "spectro_normalization": self.spectro_normalization,
            "gain_dB": self.gain_dB,
            "zscore_duration": self.zscore_duration,
            "window_type": self.window_type,
            "number_spectra": Nbwin,
            "frequency_resolution": frequency_resolution,
            "temporal_resolution": temporal_resolution,
            "audio_file_dataset_overlap": self.audio_file_overlap,
            "custom_frequency_scale": self.custom_frequency_scale,
        }
        analysis_sheet = pd.DataFrame.from_records([data])

        if adjust_bool:
            meta_path = (
                self.path
                / OSMOSE_PATH.spectrogram
                / "adjustment_spectros"
                / "adjust_metadata.csv"
            )
        else:
            meta_path = (
                self.path
                / OSMOSE_PATH.spectrogram
                / f"{str(self.spectro_duration)}_{str(self.dataset_sr)}"
                / f"{str(self.nfft)}_{str(self.window_size)}_{str(self.overlap)}_{self.custom_frequency_scale}"
                / "metadata.csv"
            )

        if meta_path.exists():
            meta_path.unlink()  # Always overwrite this file

        analysis_sheet.to_csv(meta_path, index=False)
        os.chmod(meta_path, mode=FPDEFAULT)

    def audio_file_list_csv(self) -> Path:
        list_audio = get_all_audio_files(self.path_input_audio_file)

        csv_path = self.audio_path / f"wav_list_{len(list_audio)}.csv"

        if csv_path.exists():
            return csv_path
        else:
            with open(csv_path, "w") as f:
                f.write("\n".join([str(audio) for audio in list_audio]))

            os.chmod(csv_path, mode=FPDEFAULT)

            return csv_path

    def update_parameters(self, filename: Path) -> bool:
        """Read the csv file filename and compare it to the spectrogram parameters. If any parameter is different, the file will be replaced and the fields changed.

        If there is nothing to update, the file won't be changed.

        Parameter
        ---------
        filename: Path
            The path to the csv file to be updated.

        Returns
        -------
            True if the parameters were updated else False."""

        if not filename.suffix == ".csv":
            raise ValueError("The file must be a .csv file to be updated.")

        temporal_resolution, frequency_resolution, Nbwin = self.extract_spectro_params()

        new_params = {
            "dataset_name": self.name,
            "dataset_sr": self.dataset_sr,
            "nfft": self.nfft,
            "window_size": self.window_size,
            "overlap": self.overlap,
            "colormap": self.colormap,
            "zoom_level": self.zoom_level,
            "number_adjustment_spectrogram": self.number_adjustment_spectrogram,
            "dynamic_min": self.dynamic_min,
            "dynamic_max": self.dynamic_max,
            "spectro_duration": self.spectro_duration,
            "audio_file_folder_name": self.audio_path.name,
            "data_normalization": self.data_normalization,
            "hp_filter_min_freq": self.hp_filter_min_freq,
            "sensitivity_dB": 20 * log10(self.sensitivity / 1e6),
            "peak_voltage": self.peak_voltage,
            "spectro_normalization": self.spectro_normalization,
            "gain_dB": self.gain_dB,
            "zscore_duration": self.zscore_duration,
            "window_type": self.window_type,
            "number_spectra": Nbwin,
            "frequency_resolution": frequency_resolution,
            "temporal_resolution": temporal_resolution,
            "audio_file_dataset_overlap": self.audio_file_overlap,
        }

        if not filename.exists():
            pd.DataFrame.from_records([new_params]).to_csv(filename, index=False)

            os.chmod(filename, mode=DPDEFAULT)
            return True

        orig_params = pd.read_csv(filename, header=0)

        if any(
            [
                param not in orig_params
                or str(orig_params[param]) != str(new_params[param])
                for param in new_params
            ]
        ):
            filename.unlink()
            pd.DataFrame.from_records([new_params]).to_csv(filename, index=False)

            os.chmod(filename, mode=DPDEFAULT)
            return True
        return False

    def process_file(
        self,
        audio_file: Union[str, Path],
        *,
        adjust: bool = False,
        save_matrix: bool = False,
        save_for_LTAS: bool = True,
        overwrite: bool = True,
        clean_adjust_folder: bool = False,
    ) -> None:
        """Read an audio file and generate the associated spectrogram.

        Parameters
        ----------
        audio_file : `str` or `Path`
            The name of the audio file to be processed
        adjust : `bool`, optional, keyword-only
            Indicates whether the file should be processed alone to adjust the spectrogram parameters (the default is False)
        save_matrix : `bool`, optional, keyword-only
            Whether to save the spectrogram matrices or not. Note that activating this parameter might increase greatly the volume of the project. (the default is False)
        overwrite: `bool`, optional, keyword-only
            If set to False, will skip the processing if all output files already exist. If set to True, will first delete the existing files before processing.
        clean_adjust_folder: `bool`, optional, keyword-only
            Whether the adjustment folder should be deleted.
        """
        if adjust:
            audio_file = Path(audio_file)

            self.path_output_spectrogram = (
                self.path / OSMOSE_PATH.spectrogram / "adjustment_spectros" / "image"
            )

            output_file = self.path_output_spectrogram / audio_file.name

            make_path(self.path_output_spectrogram, mode=DPDEFAULT)

            self.audio_path = audio_file.parent

            self.save_matrix = save_matrix
            self.save_for_LTAS = save_for_LTAS

        else:
            set_umask()
            try:
                if clean_adjust_folder and (
                    (
                        self.path_output_spectrogram.parents[1] / "adjustment_spectros"
                    ).exists()
                ):
                    shutil.rmtree(
                        self.path_output_spectrogram.parents[1] / "adjustment_spectros",
                        ignore_errors=True,
                    )
                    print("adjustment_spectros folder deleted.")
            except Exception as e:
                print(
                    f"Cannot remove adjustment_spectros folder. Description of the error : {str(e.value)}"
                )
                pass

            self.save_matrix = save_matrix
            self.save_for_LTAS = save_for_LTAS
            self.adjust = adjust

            self.__build_path(self.adjust)

            Zscore = self.zscore_duration if not adjust else "original"

            audio_file = Path(audio_file).name
            output_file = self.path_output_spectrogram / audio_file

            if any(self.path_output_spectrogram.glob(f"{Path(audio_file).stem}*")):
                if not overwrite:
                    print(
                        f"The spectrograms for the file {audio_file} have already been generated, skipping..."
                    )
                    return
                else:
                    print(
                        f"Existing spectrogram files detected for audio file {audio_file}! 'overwrite' is set to {overwrite}, they will be overwritten."
                    )
                    for old_file in self.path_output_spectrogram.glob(
                        f"{Path(audio_file).stem}*"
                    ):
                        old_file.unlink()
                    if save_matrix:
                        for old_matrix in self.path_output_spectrogram_matrix.glob(
                            f"{Path(audio_file).stem}*"
                        ):
                            old_matrix.unlink()

            if audio_file not in os.listdir(self.audio_path):
                raise FileNotFoundError(
                    f"The file {audio_file} must be in {self.audio_path} in order to be processed."
                )

            """
            #Useless with new spectro normalization
            #if self.data_normalization in ["zscore", "none"]:
            #    self.spectro_normalization = "spectrum"
            """

            #! Determination of zscore normalization parameters
            if self.data_normalization == "zscore" and Zscore != "original":
                df = pd.DataFrame()
                for dd in (self.path / OSMOSE_PATH.statistics).glob("SummaryStats*"):
                    df = pd.concat([df, pd.read_csv(dd, header=0)])

                df.set_index("timestamp", inplace=True)
                df.index = pd.to_datetime(df.index)

                if Zscore == "all":
                    df["mean_avg"] = df["mean"].mean()
                    df["std_avg"] = df["std"].pow(2).apply(np.sqrt, raw=True)
                else:
                    df["mean_avg"] = (
                        df["mean"].groupby(df.index.to_period(Zscore)).transform("mean")
                    )
                    df["std_avg"] = (
                        df["std"]
                        .pow(2)
                        .groupby(df.index.to_period(Zscore))
                        .transform("mean")
                        .apply(np.sqrt, raw=True)
                    )

                self.__summStats = df
                self.__zscore_mean = self.__summStats[
                    self.__summStats["filename"] == audio_file
                ]["mean_avg"].values[0]
                self.__zscore_std = self.__summStats[
                    self.__summStats["filename"] == audio_file
                ]["std_avg"].values[0]

            audio_file = self.audio_path / audio_file

        #! File processing
        data, sample_rate = safe_read(audio_file)

        if self.data_normalization == "instrument":
            data = (
                (data * self.peak_voltage)
                / self.sensitivity
                / 10 ** (self.gain_dB / 20)
            )

        bpcoef = signal.butter(
            20,
            np.array(
                [
                    max(self.hp_filter_min_freq, sys.float_info.epsilon),
                    sample_rate / 2 - 1,
                ]
            ),
            fs=sample_rate,
            output="sos",
            btype="bandpass",
        )
        data = signal.sosfilt(bpcoef, data)

        if self.verbose:
            print(f"Generating spectrograms for {output_file.name}")

        self.gen_tiles(
            data=data, sample_rate=sample_rate, output_file=output_file, adjust=adjust
        )

    def gen_tiles(
        self, *, data: np.ndarray, sample_rate: int, output_file: Path, adjust: bool
    ):
        """Generate spectrogram tiles corresponding to the zoom levels.

        Parameters
        ----------
        data : `np.ndarray`
            The audio data from which the tiles will be generated.
        sample_rate : `int`
            The sample rate of the audio data.
        output_file : `str`
            The name of the output spectrogram."""

        if self.data_normalization == "zscore" and self.zscore_duration:
            if (len(self.zscore_duration) > 0) and (self.zscore_duration != "original"):
                data = (data - self.__zscore_mean) / self.__zscore_std
            elif self.zscore_duration == "original":
                print("Apply zscore original")
                data = (data - np.mean(data)) / np.std(data)
        if self.verbose:
            print(
                f"- data min : {np.min(data):.3f}\n"
                f"- data max : {np.max(data):.3f}\n"
                f"- data mean : {np.mean(data):.3f}\n"
                f"- data std : {np.std(data):.3f}\n"
            )

        duration = len(data) / int(sample_rate)

        nber_tiles_lowest_zoom_level = 2**self.zoom_level
        tile_duration = duration / nber_tiles_lowest_zoom_level

        if not adjust:
            audio_file_name = output_file.stem
            audio_file_ext = output_file.suffixes[-1]
            current_timestamp = pd.to_datetime(
                get_timestamp_of_audio_file(
                    self.audio_path / "timestamp.csv",
                    audio_file_name + audio_file_ext,
                )
            )
            list_timestamps = []

        Sxx_complete_lowest_level = np.empty((int(self.nfft / 2) + 1, 1))
        Sxx_mean_lowest_tuile = np.empty((1, int(self.nfft / 2) + 1))

        for tile in range(0, nber_tiles_lowest_zoom_level):
            start = tile * tile_duration
            end = start + tile_duration

            if not adjust:
                list_timestamps.append(
                    current_timestamp + timedelta(seconds=int(start))
                )

            sample_data = data[int(start * sample_rate) : int(end * sample_rate) - 1]

            Sxx, Freq = self.gen_spectro(
                data=sample_data,
                sample_rate=sample_rate,
                output_file=output_file.parent
                / f"{output_file.stem}_{nber_tiles_lowest_zoom_level}_{str(tile)}.png",
            )

            Sxx_complete_lowest_level = np.hstack((Sxx_complete_lowest_level, Sxx))
            Sxx_mean_lowest_tuile = np.vstack(
                (Sxx_mean_lowest_tuile, Sxx.mean(axis=1)[np.newaxis, :])
            )

        Sxx_complete_lowest_level = Sxx_complete_lowest_level[:, 1:]
        Sxx_mean_lowest_tuile = Sxx_mean_lowest_tuile[1:, :]

        segment_times = np.linspace(
            0, len(data) / sample_rate, Sxx_complete_lowest_level.shape[1]
        )[np.newaxis, :]

        # lowest tuile resolution
        if not adjust and self.save_for_LTAS:
            # whatever the file duration, we send all welch in folder self.spectro_duration_dataset_sr;
            # OLD SOLUTION : here we use duration (read from current audio files) rather than self.spectro_duration to have
            # the exact audio file duration; so that when different audio file durations are present,
            # their respective welch spectra will be put into different folders
            output_path_welch_resolution = (
                self.path_output_welch / f"{self.spectro_duration}_{self.dataset_sr}"
            )

            if not output_path_welch_resolution.exists():
                make_path(output_path_welch_resolution, mode=DPDEFAULT)

            output_matrix = (
                output_path_welch_resolution / output_file.name
            ).with_suffix(".npz")

            if not output_matrix.exists():
                np.savez(
                    output_matrix,
                    Sxx=Sxx_mean_lowest_tuile,
                    Freq=Freq,
                    Time=list_timestamps,
                )

                os.chmod(output_matrix, mode=FPDEFAULT)

        # loop over the zoom levels from the second lowest to the highest one
        for zoom_level in range(self.zoom_level + 1)[::-1]:
            nberspec = Sxx_complete_lowest_level.shape[1] // (2**zoom_level)

            # loop over the tiles at each zoom level
            for tile in range(2**zoom_level):
                Sxx_int = Sxx_complete_lowest_level[
                    :, tile * nberspec : (tile + 1) * nberspec
                ][:, :: 2 ** (self.zoom_level - zoom_level)]

                segment_times_int = segment_times[
                    :, tile * nberspec : (tile + 1) * nberspec
                ][:, :: 2 ** (self.zoom_level - zoom_level)]

                if self.spectro_normalization == "density":
                    log_spectro = 10 * np.log10((Sxx_int + 1e-100) / (1e-12))
                if self.spectro_normalization == "spectrum":
                    log_spectro = 10 * np.log10(Sxx_int + 1e-100)

                self.generate_and_save_figures(
                    time=segment_times_int,
                    freq=Freq,
                    log_spectro=log_spectro,
                    output_file=output_file.parent
                    / f"{output_file.stem}_{str(2 ** zoom_level)}_{str(tile)}.png",
                    adjust=adjust,
                )

        # highest tuile resolution
        if False:
            if not adjust and self.save_for_LTAS and (nber_tiles_lowest_zoom_level > 1):
                # whatever the file duration , we send all welch in folder self.spectro_duration_dataset_sr  ;
                # OLD SOLUTION : here we use duration (read from current audio files) rather than self.spectro_duration to have
                # the exact audio file duration; so that when different audio file durations are present,
                # their respective welch spectra will be put into different folders
                output_path_welch_resolution = (
                    self.path_output_welch
                    / f"{int(self.spectro_duration)}_{int(self.dataset_sr)}"
                )
                if not output_path_welch_resolution.exists():
                    make_path(output_path_welch_resolution, mode=DPDEFAULT)

                output_matrix = (
                    output_path_welch_resolution / output_file.name
                ).with_suffix(".npz")

                if not output_matrix.exists():
                    np.savez(
                        output_matrix,
                        Sxx=Sxx_int.mean(axis=1),
                        Freq=Freq,
                        Time=current_timestamp,
                    )

                    os.chmod(output_matrix, mode=FPDEFAULT)

    def gen_spectro(
        self, *, data: np.ndarray, sample_rate: int, output_file: Path
    ) -> Tuple[np.ndarray, np.ndarray[float]]:
        """Generate the spectrograms

        Parameters
        ----------
        data : `np.ndarray`
            The audio data from which the tiles will be generated.
        sample_rate : `int`
            The sample rate of the audio data.
        output_file : `str`
            The name of the output spectrogram.

        Returns
        -------
        Sxx : `np.NDArray[float64]`
        Freq : `np.NDArray[float]`
        """

        Noverlap = int(self.window_size * self.overlap / 100)

        win = np.hamming(self.window_size)
        if self.nfft < (self.window_size):
            if self.spectro_normalization == "density":
                scale_psd = 1.0
            if self.spectro_normalization == "spectrum":
                scale_psd = 1.0
        else:
            if self.spectro_normalization == "density":
                scale_psd = 2.0 / (((win * win).sum()) * sample_rate)
            if self.spectro_normalization == "spectrum":
                scale_psd = 2.0 / (win.sum() ** 2)

        Nbech = np.size(data)
        Noffset = self.window_size - Noverlap
        Nbwin = int((Nbech - self.window_size) / Noffset)
        Freq = np.fft.rfftfreq(self.nfft, d=1 / sample_rate)

        Sxx = np.zeros([np.size(Freq), Nbwin])
        Time = np.linspace(0, Nbech / sample_rate, Nbwin)
        for idwin in range(Nbwin):
            if self.nfft < (self.window_size):
                x_win = data[idwin * Noffset : idwin * Noffset + self.window_size]
                _, Sxx[:, idwin] = signal.welch(
                    x_win,
                    fs=sample_rate,
                    window="hamming",
                    nperseg=int(self.nfft),
                    noverlap=int(self.nfft / 2),
                    scaling=self.spectro_normalization,
                )
            else:
                x_win = data[idwin * Noffset : idwin * Noffset + self.window_size] * win
                Sxx[:, idwin] = np.abs(np.fft.rfft(x_win, n=self.nfft)) ** 2
            Sxx[:, idwin] *= scale_psd

        if self.data_normalization == "instrument":
            log_spectro = 10 * np.log10((Sxx / (1e-12)) + (1e-20))

        if self.data_normalization == "zscore":
            if self.spectro_normalization == "density":
                Sxx *= sample_rate / 2  # value around 0dB
                log_spectro = 10 * np.log10(Sxx + (1e-20))
            if self.spectro_normalization == "spectrum":
                Sxx *= self.window_size / 2  # value around 0dB
                log_spectro = 10 * np.log10(Sxx + (1e-20))

        # save spectrogram matrices (intensity, time and freq) in a npz file
        if self.save_matrix:
            make_path(self.path_output_spectrogram_matrix, mode=DPDEFAULT)

            output_matrix = (
                self.path_output_spectrogram_matrix / output_file.name
            ).with_suffix(".npz")

            if not output_matrix.exists():
                np.savez(
                    output_matrix,
                    Sxx=Sxx,
                    log_spectro=log_spectro,
                    Freq=Freq,
                    Time=Time,
                )

                os.chmod(output_matrix, mode=FPDEFAULT)

        return Sxx, Freq

    def generate_and_save_figures(
        self,
        *,
        time: np.ndarray[float],
        freq: np.ndarray[float],
        log_spectro: np.ndarray[int],
        output_file: Path,
        adjust: bool,
    ):
        """Write the spectrogram figures to the output file.

        Parameters
        ----------
        time : `np.NDArray[floating]`
        freq : `np.NDArray[floating]`
        log_spectro : `np.NDArray[signed int]`
        output_file : `str`
            The name of the spectrogram file."""

        # Plotting spectrogram
        my_dpi = 100
        fact_x = 1.3
        fact_y = 1.3
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(fact_x * 1800 / my_dpi, fact_y * 512 / my_dpi),
            dpi=my_dpi,
        )
        if self.verbose:
            print(
                f"- min log spectro : {np.amin(log_spectro):.3f}\n"
                f"- max log spectro : {np.amax(log_spectro):.3f}\n"
            )

        color_map = plt.get_cmap(self.colormap)

        if self.custom_frequency_scale == "linear":
            plt.pcolormesh(time, freq, log_spectro, cmap=color_map)
        else:
            if self.custom_frequency_scale == "log":
                plt.pcolormesh(time, freq, log_spectro, cmap=color_map)
                plt.yscale("log")
                plt.ylim(freq[freq > 0].min(), self.dataset_sr / 2)
            else:
                custom_frequency_scale = FrequencyScaleSerializer().get_scale(
                    self.custom_frequency_scale, self.dataset_sr
                )
                freq_custom = np.vectorize(custom_frequency_scale.map_freq2scale)(freq)
                plt.pcolormesh(time, freq_custom, log_spectro, cmap=color_map)

        plt.clim(vmin=self.dynamic_min, vmax=self.dynamic_max)

        if adjust:
            fig.axes[0].get_xaxis().set_visible(True)
            fig.axes[0].get_yaxis().set_visible(True)
            ax.set_ylabel("Frequency (Hz)")
            ax.set_xlabel("Time (s)")
            plt.colorbar()
        else:
            fig.axes[0].get_xaxis().set_visible(False)
            fig.axes[0].get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(False)

        # Saving spectrogram plot to file
        plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
        plt.close()

        os.chmod(output_file, mode=FPDEFAULT)

        if adjust:
            display(Image(output_file))

    def process_all_files(
        self,
        *,
        save_matrix: bool = False,
        save_for_LTAS: bool = True,
        list_audio_to_process: list = [],
    ):
        """Process all the files in the dataset and generates the spectrograms. It uses the python multiprocessing library
        to parallelise the computation, so it is less efficient to use this method rather than the job scheduler if run on a cluster.
        """

        if len(list_audio_to_process) > 0:
            self.list_audio_to_process = list_audio_to_process

        kwargs = {
            "save_matrix": save_matrix,
            "save_for_LTAS": save_for_LTAS,
            "overwrite": True,
        }

        map_process_file = partial(self.process_file, **kwargs)

        for ll in self.list_audio_to_process:
            self.process_file(ll, **kwargs)

        # with mp.Pool(processes=mp.cpu_count()) as pool:
        #     pool.map(map_process_file, self.list_audio_to_process)

    def save_all_welch(self, list_npz_files: list, path_all_welch: Path):
        if isinstance(list_npz_files, list):
            Time = []
            ct = 0
            for file_npz in tqdm(list_npz_files):
                current_matrix = np.load(file_npz, allow_pickle=True)
                os.remove(file_npz)
                if ct == 0:
                    Sxx = np.empty((1, current_matrix["Sxx"].shape[1]))

                Sxx = np.vstack((Sxx, current_matrix["Sxx"]))
                Time.append(current_matrix["Time"])
                ct += 1
            Sxx = Sxx[1:, :]
            Freq = current_matrix["Freq"]

            if len(list_npz_files) > 1:
                Time = list(itertools.chain(*Time))

            np.savez(
                path_all_welch, Sxx=Sxx, Time=Time, Freq=Freq, allow_pickle=True
            )  # careful data not sorted here! we should save them based on dataframe df below

        else:
            os.rename(list_npz_files, path_all_welch)
            os.remove(list_npz_files)

        return Sxx, Time, Freq

    def build_LTAS(
        self,
        time_resolution: int,
        sample_rate: int,
        time_scale: str = "D",
        show_fig: bool = False,
    ):
        list_npz_files = list(
            (self.path_output_welch / f"{time_resolution}_{sample_rate}").glob("*npz")
        )

        if len(list_npz_files) == 0:
            raise FileNotFoundError(
                f"No intermediary welch spectra to aggregate in the folder {self.path_output_welch / f'{time_resolution}_{sample_rate}'} ; please run a complete generation of spectrograms first!"
            )

        else:
            if not self.path_output_LTAS.exists():
                make_path(self.path_output_LTAS, mode=DPDEFAULT)

            path_all_welch = (
                self.path_output_welch
                / f"{time_resolution}_{sample_rate}"
                / "all_welch.npz"
            )

            if os.path.exists(path_all_welch):
                data = np.load(path_all_welch, allow_pickle=True)
                Sxx = data["Sxx"]
                Time = data["Time"]
                Freq = data["Freq"]
            else:
                Sxx, Time, Freq = self.save_all_welch(list_npz_files, path_all_welch)

            # convert numpy arrays to dataframe, more convenient for time operations
            df = pd.DataFrame(Sxx, dtype=float)
            df["Time"] = Time
            # sort by time, and make time variable as dataframe index
            df.sort_values(by=["Time"], inplace=True)
            df.set_index("Time", inplace=True, drop=True)
            df.index = pd.to_datetime(df.index)

            if time_scale == "all":
                cur_LTAS = df.values

                if self.spectro_normalization == "density":
                    log_spectro = 10 * np.log10((cur_LTAS / (1e-12)) + (1e-100))
                if self.spectro_normalization == "spectrum":
                    log_spectro = 10 * np.log10(cur_LTAS + (1e-100))

                self.generate_and_save_LTAS(
                    start_time=df.index[0],
                    end_time=df.index[-1],
                    freq=Freq,
                    log_spectro=log_spectro.T,
                    output_file=self.path / OSMOSE_PATH.LTAS / "LTAS_all.png",
                    time_scale="all",
                    raw_time_vector=df.index,
                    show_fig=show_fig,
                )

            else:
                time_vector = pd.date_range(Time[0], Time[-1], freq=time_scale)

                for ind_period in range(len(time_vector) - 1):
                    current_df = df[
                        (df.index > time_vector[ind_period])
                        & (df.index <= time_vector[ind_period + 1])
                    ]

                    current_time_period = current_df.index
                    cur_LTAS = current_df.values.T

                    if self.spectro_normalization == "density":
                        log_spectro = 10 * np.log10((cur_LTAS / (1e-12)) + (1e-20))
                    if self.spectro_normalization == "spectrum":
                        log_spectro = 10 * np.log10(cur_LTAS + (1e-20))

                    self.generate_and_save_LTAS(
                        start_time=current_time_period[0],
                        end_time=current_time_period[-1],
                        freq=Freq,
                        log_spectro=log_spectro,
                        output_file=self.path
                        / OSMOSE_PATH.LTAS
                        / f"LTAS_{datetime.strftime(time_vector[ind_period], '%Y_%m_%dT%H_%M_%S')}.png",
                        time_scale=time_scale,
                        raw_time_vector=current_time_period,
                        show_fig=show_fig,
                    )

    def generate_and_save_LTAS(
        self,
        start_time: pd._libs.tslibs.timestamps.Timestamp,
        end_time: pd._libs.tslibs.timestamps.Timestamp,
        freq: np.ndarray[float],
        log_spectro: np.ndarray[float],
        output_file: Path,
        time_scale: str,
        raw_time_vector,
        show_fig: bool = False,
    ):
        # Plotting spectrogram
        my_dpi = 100
        fact_x = 1.3
        fact_y = 1.3
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(fact_x * 1800 / my_dpi, fact_y * 512 / my_dpi),
            dpi=my_dpi,
        )

        im = ax.pcolormesh(
            np.arange(0, log_spectro.shape[1]),
            freq,
            log_spectro,
            cmap=plt.get_cmap(self.colormap),
        )
        plt.colorbar(im, ax=ax)
        ax.set_ylabel("Frequency (Hz)")

        # make timestamps proper xitck_labels
        nber_xticks = min(10, log_spectro.shape[1])
        label_smoother = {"all": "D", "Y": "M", "M": "D", "D": "T", "H": "T"}
        time_vector = pd.date_range(start_time, end_time, periods=log_spectro.shape[1])
        date = [
            p.to_period(label_smoother[time_scale])
            for p in time_vector.tz_localize(None)
        ]

        int_sep = int(len(date) / nber_xticks)
        plt.xticks(np.arange(0, len(date), int_sep), date[::int_sep])
        ax.tick_params(axis="x", rotation=20)

        # Saving spectrogram plot to file
        print("Saving", output_file, "\nNumber of welch:", str(log_spectro.shape[1]))
        plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
        plt.close()

        output_file_npz = output_file.with_suffix(".npz")
        np.savez(
            output_file_npz,
            LTAS=log_spectro,
            time=time_vector,
            Freq=freq,
            raw_time=raw_time_vector,
            allow_pickle=True,
        )

        if show_fig:
            display(Image(filename=output_file))

    def build_SPL_filtered(
        self,
        time_resolution: int,
        sample_rate: int,
        freq_min: Union[list, int] = [0],
        show_fig: bool = False,
    ):
        list_npz_files = list(
            (self.path_output_welch / f"{time_resolution}_{sample_rate}").glob("*npz")
        )

        if len(list_npz_files) == 0:
            raise FileNotFoundError(
                "No intermediary welch spectra to aggregate, run a complete generation of spectrograms first!"
            )

        else:

            # assign default value for Freq_max, equivalent to no HF filtering
            freq_max = [int(sample_rate / 2)]
            if not isinstance(freq_min, list):
                freq_min = [freq_min]

            if not self.path_output_SPLfiltered.exists():
                make_path(self.path_output_SPLfiltered, mode=DPDEFAULT)

            path_all_welch = (
                self.path_output_welch
                / f"{time_resolution}_{sample_rate}"
                / "all_welch.npz"
            )

            if path_all_welch.exists():
                data = np.load(path_all_welch, allow_pickle=True)
                Sxx = data["Sxx"]
                Time = data["Time"]
                Freq = data["Freq"]
            else:
                Sxx, Time, Freq = self.save_all_welch(list_npz_files, path_all_welch)

            # convert numpy arrays to dataframe, more convenient for time operations
            df = pd.DataFrame(Sxx, dtype=float)
            df["Time"] = Time
            # sort by time, and make time variable as dataframe index
            df.sort_values(by=["Time"], inplace=True)
            df.set_index("Time", inplace=True, drop=True)
            df.index = pd.to_datetime(df.index)

            time_vector = pd.date_range(
                df.index[0], df.index[-1], freq=str(time_resolution) + "s"
            )

            # Plotting SPL
            my_dpi = 100
            fact_x = 1.3
            fact_y = 1.3
            fig, ax = plt.subplots(
                nrows=1,
                ncols=1,
                figsize=(fact_x * 1800 / my_dpi, fact_y * 512 / my_dpi),
                dpi=my_dpi,
            )

            lst_legend = []
            for cur_freq_min, cur_freq_max in zip(freq_min, freq_max):
                pre_SPL = np.mean(
                    df.values[
                        :,
                        np.argmin(abs(Freq - cur_freq_min)) : np.argmin(
                            abs(Freq - cur_freq_max)
                        ),
                    ],
                    1,
                )

                if self.spectro_normalization == "density":
                    SPL_filtered = 10 * np.log10((pre_SPL / (1e-12)) + (1e-20))
                if self.spectro_normalization == "spectrum":
                    SPL_filtered = 10 * np.log10(pre_SPL + (1e-20))

                plt.plot(np.arange(0, len(SPL_filtered)), SPL_filtered)
                plt.autoscale(enable=True, axis="x", tight=True)

                lst_legend.append(f"[{cur_freq_min}-{cur_freq_max}] Hz")

                output_file_npz = (
                    self.path
                    / OSMOSE_PATH.SPLfiltered
                    / f"SPLfiltered_{cur_freq_min}_{cur_freq_max}.npz"
                )

                np.savez(
                    output_file_npz,
                    SPL=SPL_filtered,
                    time=time_vector,
                    allow_pickle=True,
                )

            plt.ylabel("SPL (dB)")
            plt.legend(lst_legend)

            # write proper timestamps as xitck_labels
            nber_xticks = min(10, len(SPL_filtered))
            if (df.index[-1] - df.index[0]).days > 7:
                label_smoother = "D"
            else:
                label_smoother = "h"
            date = [p.to_period(label_smoother) for p in time_vector.tz_localize(None)]

            int_sep = int(len(date) / nber_xticks)
            plt.xticks(np.arange(0, len(date), int_sep), date[::int_sep])
            ax.tick_params(axis="x", rotation=20)

            # save as png figure
            output_file = self.path / OSMOSE_PATH.SPLfiltered / "SPLfiltered.png"

            print(
                "Saving",
                output_file,
                "\nNumber of time points:",
                str(len(SPL_filtered)),
            )
            plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
            plt.close()

            if show_fig:
                display(Image(filename=output_file))

    def build_EPD(self, time_resolution: str, sample_rate: int, show_fig: bool = False):
        list_npz_files = list(
            (self.path_output_welch / f"{time_resolution}_{sample_rate}").glob("*npz")
        )
        if len(list_npz_files) == 0:
            raise FileNotFoundError(
                "No intermediary welch spectra to aggregate, run a complete generation of spectrograms first!"
            )

        else:
            if not self.path_output_EPD.exists():
                make_path(self.path_output_EPD, mode=DPDEFAULT)

            path_all_welch = (
                self.path_output_welch
                / f"{time_resolution}_{sample_rate}"
                / "all_welch.npz"
            )

            if path_all_welch.exists():
                data = np.load(path_all_welch, allow_pickle=True)
                Sxx = data["Sxx"]
                Time = data["Time"]
                Freq = data["Freq"]
            else:
                Sxx, Time, Freq = self.save_all_welch(list_npz_files, path_all_welch)

            all_welch = 10 * np.log10(Sxx)

            RMSlevel = 10 * np.log10(np.nanmean(10 ** (all_welch / 10), axis=0))

            # Plotting SPL
            my_dpi = 100
            fact_x = 1.3
            fact_y = 1.3
            fig, ax = plt.subplots(
                nrows=1,
                ncols=1,
                figsize=(fact_x * 1800 / my_dpi, fact_y * 512 / my_dpi),
                dpi=my_dpi,
            )

            ax.plot(Freq, RMSlevel, color="k", label="RMS level")

            percen = [1, 5, 50, 95, 99]
            p = np.nanpercentile(all_welch, percen, 0, interpolation="linear")
            for i in range(len(p)):
                plt.plot(
                    Freq, p[i, :], linewidth=2, label="%s %% percentil" % percen[i]
                )

            ax.semilogx()
            plt.legend()
            plt.autoscale(enable=True, axis="y", tight=True)
            plt.autoscale(enable=True, axis="x", tight=True)
            plt.ylabel("relative SPL (dB)")
            plt.xlabel("Frequency (Hz)")

            # save as png figure
            output_file = self.path / OSMOSE_PATH.EPD / "EPD.png"
            print(f"Saving {output_file}\nNumber of welch: {all_welch.shape[0]}")
            plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
            plt.close()

            output_file_npz = self.path / OSMOSE_PATH.EPD / f"EPD.npz"

            np.savez(
                output_file_npz,
                Freq=Freq,
                RMSlevel=RMSlevel,
                percentiles=p,
                allow_pickle=True,
            )

            if show_fig:
                display(Image(filename=output_file))
