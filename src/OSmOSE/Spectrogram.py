import os
import sys
import shutil
from typing import Tuple, Union, Literal
from math import log10
from glob import glob

import pandas as pd
import numpy as np
import soundfile as sf
from scipy import signal
from termcolor import colored
from matplotlib import pyplot as plt
from OSmOSE.job_factory import Job_builder
from OSmOSE.cluster.audio_reshaper import reshape # Not used for now; will be when local execution will be a thing.
from OSmOSE.Dataset import Dataset

class Spectrogram(Dataset):
    """Main class for spectrogram-related computations. Can resample, reshape and normalize audio files before generating spectrograms."""

    def __init__(self, dataset_path: str, *, analysis_fs: int, coordinates: Union[str, list, tuple] = None, osmose_group_name: str = None, analysis_params: dict = None, batch_number : int = 10) -> None:
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
        analysis_fs : `int`, keyword-only
            The sampling frequency used for the generation of the spectrograms.
        coordinates : `str` or `list` or `tuple`, optional, keyword-only
            The GPS coordinates of the listening location. If it is of type `str`, it must be the name of a csv file located in `raw/auxiliary`,
            otherwise a list or a tuple with the first element being the latitude coordinates and second the longitude coordinates.
        osmose_group_name : `str`, optional, keyword-only
            The name of the group using the OsmOSE package. All files created using this dataset will be accessible by the osmose group. 
            Will not work on Windows.
        analysis_params : `dict`, optional, keyword-only
            If `analysis/analysis_sheet.csv` does not exist, the analysis parameters can be submitted in the form of a dict,
            with keys matching what is expected:
                - fileScale_nfft : `int`
                - fileScale_winsize : `int`
                - fileScale_overlap : `int`
                - colmapspectros : `str`
                - nber_zoom_levels : `int`
                - min_color_val : `int`
                - max_color_val : `int`
                - nberAdjustSpectros : `int`
                - max_time_display_spectro : `int`
                - zscore_duration : `float` or `str`
                - fmin_HighPassFilter : `int`
                - sensitivity_dB : `int`
                - peak_voltage : `float`
                - spectro_normalization : `str`
                - data_normalization : `str`
                - gain_dB : `int`
            If additional information is given, it will be ignored. Note that if there is an `analysis/analysis_sheet.csv` file, it will
            always have the priority.
        batch_number : `int`, optional
            The number of batches the dataset files will be split into when submitting parallel jobs (the default is 10).
"""
        super().__init__(dataset_path, coordinates=coordinates, osmose_group_name=osmose_group_name)

        analysis_path = os.path.join(self.Path, "analysis", "analysis_sheet.csv")
        if os.path.exists(analysis_path):
            self.__analysis_file = True
            analysis_sheet = pd.read_csv(analysis_path,header=0)
        elif analysis_params:
            self.__analysis_file = False
            analysis_sheet = {key: [value] for (key, value) in analysis_params.items()}
        else:
            analysis_sheet = None
            self.__analysis_file = False
            print("No valid analysis/analysis_sheet.csv found and no parameters provided. All attributes will be None.")

        self.Batch_number : int = batch_number
        self.__analysis_fs : int = analysis_fs

        self.__fileScale_nfft : int = analysis_sheet['fileScale_nfft'][0] if analysis_sheet is not None else None
        self.__fileScale_winsize : int = analysis_sheet['fileScale_winsize'][0] if analysis_sheet is not None else None
        self.__fileScale_overlap : int = analysis_sheet['fileScale_overlap'][0] if analysis_sheet is not None else None
        self.__colmapspectros : str = analysis_sheet['colmapspectros'][0] if analysis_sheet is not None else None
        self.__nber_zoom_levels : int = analysis_sheet['nber_zoom_levels'][0] if analysis_sheet is not None else None
        self.__min_color_val : int = analysis_sheet['min_color_val'][0] if analysis_sheet is not None else None
        self.__max_color_val : int = analysis_sheet['max_color_val'][0] if analysis_sheet is not None else None
        self.__nberAdjustSpectros : int = analysis_sheet['nberAdjustSpectros'][0] if analysis_sheet is not None else None
        self.__maxtime_display_spectro : int = analysis_sheet['max_time_display_spectro'][0] if analysis_sheet is not None and "max_time_display_spectro" in analysis_sheet else -1

        self.__zscore_duration : Union[float, str] = analysis_sheet['zscore_duration'][0] if analysis_sheet is not None and isinstance(analysis_sheet['zscore_duration'][0], float) else None

        # fmin cannot be 0 in butterworth. If that is the case, it takes the smallest value possible, epsilon
        self.__fmin_HighPassFilter : int = analysis_sheet['fmin_HighPassFilter'][0] if analysis_sheet is not None and analysis_sheet['fmin_HighPassFilter'][0] != 0 else sys.float_info.epsilon
        sensitivity_dB : int = analysis_sheet['sensitivity_dB'][0] if analysis_sheet is not None else None
        self.__sensitivity : float = 10**(sensitivity_dB/20) * 1e6 if analysis_sheet is not None else None
        self.__peak_voltage : float  = analysis_sheet['peak_voltage'][0] if analysis_sheet is not None else None
        self.__spectro_normalization : str = analysis_sheet['spectro_normalization'][0] if analysis_sheet is not None else None
        self.__data_normalization : str = analysis_sheet['data_normalization'][0] if analysis_sheet is not None else None
        self.__gain_dB : float = analysis_sheet['gain_dB'][0] if analysis_sheet is not None else None

        self.Jb = Job_builder()

        plt.switch_backend('agg')

        fontsize = 16
        ticksize = 12
        plt.rc('font', size=fontsize)  # controls default text sizes
        plt.rc('axes', titlesize=fontsize)  # fontsize of the axes title
        plt.rc('axes', labelsize=fontsize)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=ticksize)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=ticksize)  # fontsize of the tick labels
        plt.rc('legend', fontsize=ticksize)  # legend fontsize
        plt.rc('figure', titlesize=ticksize)  # fontsize of the figure title


    #region Spectrogram properties

    @property
    def Analysis_fs(self):
        """The sampling frequency of the dataset."""
        return self.__analysis_fs

    @Analysis_fs.setter
    def Analysis_fs(self, value: int):
        self.__analysis_fs = value

    @property
    def Nfft(self):
        """The Nonequispaced Fast Fourier Transform of the dataset."""
        return self.__fileScale_nfft
    
    @Nfft.setter
    def Nfft(self, value):
        if not self.__analysis_file:
            self.__fileScale_nfft = value
        else:
            raise ValueError("This parameter cannot be changed since it has been initialized using the analysis sheet.")

    @property
    def Window_size(self):
        """The window size """
        return self.__fileScale_winsize
    
    @Window_size.setter
    def Window_size(self, value):
        if not self.__analysis_file:
            self.__fileScale_winsize = value
        else:
            raise ValueError("This parameter cannot be changed since it has been initialized using the analysis sheet.")

    @property
    def Overlap(self):
        return self.__fileScale_overlap
    
    @Overlap.setter
    def Overlap(self, value):
        if not self.__analysis_file:
            self.__fileScale_overlap = value
        else:
            raise ValueError("This parameter cannot be changed since it has been initialized using the analysis sheet.")

    @property
    def Colmap(self):
        return self.__colmapspectros

    @Colmap.setter
    def Colmap(self, value):
        self.__colmapspectros = value
    
    @property
    def Zoom_levels(self):
        return self.__nber_zoom_levels
    
    @Zoom_levels.setter
    def Zoom_levels(self, value):
        self.__nber_zoom_levels = value

    @property
    def Min_color_value(self):
        return self.__min_color_val
    
    @Min_color_value.setter
    def Min_color_value(self, value):
        self.__min_color_val = value

    @property
    def Max_color_value(self):
        return self.__max_color_val
    
    @Max_color_value.setter
    def Max_color_value(self, value):
        self.__max_color_val = value

    @property
    def Number_adjustment_spectrograms(self):
        return self.__nberAdjustSpectros
    
    @Number_adjustment_spectrograms.setter
    def Number_adjustment_spectrograms(self, value):
        self.__nberAdjustSpectros = value

    @property
    def Max_time_display_spectro(self):
        return self.__maxtime_display_spectro
    
    @Max_time_display_spectro.setter
    def Max_time_display_spectro(self, value):
        self.__maxtime_display_spectro = value

    @property
    def Zscore_duration(self):
        return self.__zscore_duration

    @Zscore_duration.setter
    def Zscore_duration(self, value):
        self.__zscore_duration = value

    @property
    def Fmin_HighPassFilter(self):
        return self.__fmin_HighPassFilter

    @Fmin_HighPassFilter.setter
    def Fmin_HighPassFilter(self, value):
        if not self.__analysis_file:
            self.__fmin_HighPassFilter = value
        else:
            raise ValueError("This parameter cannot be changed since it has been initialized using the analysis sheet.")

    @property
    def Sensitivity(self):
        return self.__sensitivity

    @Sensitivity.setter
    def Sensitivity(self, value):
        """Always assume the sensitivity is given in dB"""
        if not self.__analysis_file:
            self.__sensitivity = 10**(value/20) * 1e6
        else:
            raise ValueError("This parameter cannot be changed since it has been initialized using the analysis sheet.")

    @property
    def Peak_voltage(self):
        return self.__peak_voltage

    @Peak_voltage.setter
    def Peak_voltage(self, value):
        if not self.__analysis_file:
            self.__peak_voltage = value
        else:
            raise ValueError("This parameter cannot be changed since it has been initialized using the analysis sheet.")

    @property
    def Spectro_normalization(self):
        return self.__spectro_normalization

    @Spectro_normalization.setter
    def Spectro_normalization(self, value):
        if not self.__analysis_file:
            self.__spectro_normalization = value
        else:
            raise ValueError("This parameter cannot be changed since it has been initialized using the analysis sheet.")

    @property
    def Data_normalization(self):
        return self.__data_normalization
    
    @Data_normalization.setter
    def Data_normalization(self, value):
        if not self.__analysis_file:
            self.__data_normalization = value
        else:
            raise ValueError("This parameter cannot be changed since it has been initialized using the analysis sheet.")

    @property
    def Gain_dB(self):
        return self.__gain_dB

    @Gain_dB.setter
    def Gain_dB(self, value):
        if not self.__analysis_file:
            self.__gain_dB = value
        else:
            raise ValueError("This parameter cannot be changed since it has been initialized using the analysis sheet.")

    #endregion

    def __build_path(self, adjust:bool = False):
        """Build some internal paths according to the expected architecture. Not path is created.
        
        Parameter
        ---------
            adjust : `bool`
                Whether or not the paths are used to adjust spectrogram parameters."""
        analysis_path = os.path.join(self.Path, "analysis")
        audio_foldername = str(self.Max_time_display_spectro)+'_'+str(self.Analysis_fs)
        self.audio_path = os.path.join(self.Path, "raw", "audio", audio_foldername)

        self.__path_output_spectrograms = os.path.join(analysis_path, "spectrograms", audio_foldername)
        self.__path_summstats = os.path.join(analysis_path, "normaParams", audio_foldername)

        if adjust:
            self.__spectro_foldername = "spectro_adjustParams"
        else:
            self.__spectro_foldername = f"nfft={str(self.Nfft)}_winsize={str(self.Window_size)}_overlap={str(self.Overlap)} \
                                _cvr={str(self.Min_color_value)}-{str(self.Max_color_value)}"

        self.__path_output_spectrogram_matrices = os.path.join(analysis_path, "spectrograms_mat", audio_foldername, self.__spectro_foldername)


    def check_spectro_size(self):
        """Verify if the parameters will generate a spectrogram that can fit one screen properly"""
        if self.Nfft > 2048:
            print('your nfft is :',self.Nfft)
            print( colored('PLEASE REDUCE IT UNLESS YOU HAVE A VERY HD SCREEN WITH MORE THAN 1k pixels vertically !!!! ', 'red') )           
        
        tile_duration = self.Max_time_display_spectro / 2**(self.Zoom_levels-1)

        data = np.zeros([int(tile_duration * self.Analysis_fs),1 ])

        Noverlap = int(self.Window_size * self.Overlap / 100)
        
        win = np.hamming(self.Window_size)
        if self.Nfft < (0.5*self.Window_size):
            #? What is scale_psd ?
            scale_psd = 2.0 * self.Window_size / (self.Nfft * ((win * win).sum() / self.Window_size))
        else:
            scale_psd = 2.0 / ( (win * win).sum())
        Nbech = np.size(data)
        Noffset = self.Window_size-Noverlap
        Nbwin = int((Nbech-self.Window_size) / Noffset)
        Freq =  np.fft.rfftfreq(self.Nfft, d=1/self.Analysis_fs)
        Time = np.linspace(0, Nbech/self.Analysis_fs, Nbwin)

        print('your smallest tile has a duration of:',tile_duration,'(s)')
        print('\n')

        if Nbwin>3500:
            print( colored('PLEASE REDUCE IT UNLESS YOU HAVE A VERY HD SCREEN WITH MORE THAN 2k pixels horizontally !!!! ', 'red') )

        print('\n')
        print('your number of time windows in this tile is:',Nbwin )        
        print('\n')
        print('your resolutions : time = ',round(Time[1]-Time[0],3),'(s) / frequency = ', round(Freq[1]-Freq[0],3) ,'(Hz)' )    


    # TODO: some cleaning | Rename available reshape methods ? (something like legacy, classic)
    def initialize(self, *, analysis_fs: int = None, reshape_method: Literal["resample","reshape","none"] = "none", 
    ind_min: int = 0, ind_max: int = -1, pad_silence: bool = False) -> None:
        """Prepares everything (path, variables, files) for spectrogram generation. This needs to be run only once per dataset.
        
        Parameters
        ----------
        analysis_fs : `int`, optional, keyword-only
            The sampling frequency of the audio files used to generate the spectrograms. If set, will overwrite the Spectrogram.Analysis_fs attribute.
        reshape_method : {"resample", "reshape", "none"}, optional
            Which method to use if the desired size of the spectrogram is different from the audio file duration.
            - resample : Legacy method, use bash and sox software to trim the audio files and fill the empty space with nothing.
            Unpractical when the audio file duration is longer than the desired spectrogram size.
            - reshape : Classic method, use python and sox library to cut and concatenate the audio files to fit the desired duration.
            Will rewrite the `timestamp.csv` file, thus timestamps may have unexpected behavior if the concatenated files are not chronologically
            subsequent.
            - none : Don't reshape, will throw an error if the file duration is different than the desired spectrogram size. (It is the default behavior)

        ind_min : `int`, optional
            The index of the first file to consider. Both this parameter and `ind_max` are not commonly used and are 
            for very specific use cases. Most of the time, you want to initialize the whole dataset (the default is 0).
        ind_max : `int`, optional
            The index of the last file to consider (the default is -1, meaning consider every file).
        pad_silence : `bool`, optinoal
            When using the legacy reshaping method, whether there should be a silence padding or not (default is False).
            """
        if analysis_fs:
            self.Analysis_fs = analysis_fs

        self.__build_path()

        audio_foldername = str(self.Max_time_display_spectro)+'_'+str(self.Analysis_fs)
        # Load variables from raw metadata
        metadata = pd.read_csv(os.path.join(self.Path, "raw","metadata.csv"))
        orig_fileDuration = metadata['orig_fileDuration'][0]
        orig_fs = metadata['orig_fs'][0]
        total_nber_audio_files = metadata['nberWavFiles'][0]

        input_audio_foldername = str(orig_fileDuration)+'_'+str(int(orig_fs))
        analysis_path = os.path.join(self.Path, "analysis")
        self.path_input_audio_file = os.path.join(self.Path, "raw", "audio", input_audio_foldername)

        list_wav_withEvent_comp = sorted(glob.glob(os.path.join(self.path_input_audio_file , '*wav')))

        if ind_max == -1:
            ind_max = len(list_wav_withEvent_comp)
        list_wav_withEvent = list_wav_withEvent_comp[ind_min:ind_max]
        
        self.list_wav_to_process = [os.path.basename(x) for x in list_wav_withEvent]

        if os.path.isfile(os.path.join(analysis_path,"subset_files.csv")):
            subset = pd.read_csv(os.path.join(self.Path , 'analysis', 'subset_files.csv'),header=None)[0].values
            self.list_wav_to_process = list(set(subset).intersection(set(self.list_wav_to_process)))
        
        batch_size = len(self.list_wav_to_process) // self.Batch_number

        #! RESAMPLING
        resample_job_id_list = []

        if self.Analysis_fs != orig_fs and not os.listdir(self.audio_path):
            for batch in range(self.Batch_number):
                i_min = batch * batch_size
                i_max = (i_min + batch_size if batch < self.Batch_number - 1 else len(self.list_wav_to_process)) # If it is the last batch, take all files
                jobfile = self.Jb.build_job_file(script_path=os.path.join(os.dirname(__file__), "cluster", "resample.py"), \
                            script_args=f"--input-dir {self.path_input_audio_file} --target-fs {self.Analysis_fs} --ind-min {i_min} --ind-max {i_max} --output-dir {self.audio_path}", \
                            jobname="OSmOSE_resample", preset="medium")
                            #TODO: use importlib.resources
                            
                job_id = self.Jb.submit_job(jobfile)
                resample_job_id_list.append(job_id)

        #! ZSCORE NORMALIZATION
        isnorma = any([cc in self.Zscore_duration for cc in ['D','M','H','S','W']]) 
        normaDir = os.path.join(self.audio_path, "normaParams", audio_foldername)
        norma_job_id_list = []
        if os.listdir(normaDir) and self.Data_normalization == "zscore" and isnorma:

            for batch in range(self.Batch_number):
                i_min = batch * batch_size
                i_max = (i_min + batch_size if batch < self.Batch_number - 1 else len(self.list_wav_to_process)) # If it is the last batch, take all files
                jobfile = self.Jb.build_job_file(script_path=os.path.join(os.dirname(__file__), "cluster", "get_zscore_params.py"), \
                            script_args=f"--input-dir {self.path_input_audio_file} --fmin-highpassfilter {self.Fmin_HighPassFilter} \
                                --ind-min {i_min} --ind-max {i_max} --output-file {os.path.join(normaDir, 'SummaryStats_' + str(i_min) + '.csv')}", \
                            jobname="OSmOSE_get_zscore_params", preset="low")

                job_id = self.Jb.submit_job(jobfile, dependency=resample_job_id_list)
                norma_job_id_list.append(job_id)

        #! RESHAPING
        # Reshape audio files to fit the maximum spectrogram size, whether it is greater or smaller.
        reshape_job_id_list = []

        if self.Max_time_display_spectro != int(orig_fileDuration):
            # We might reshape the files and create the folder. Note: reshape function might be memory-heavy and deserve a proper qsub job. 
            if self.Max_time_display_spectro > int(orig_fileDuration) and reshape_method in ["none", "resample"]:
                raise ValueError("Spectrogram size cannot be greater than file duration. If you want to automatically reshape your audio files to fit the spectrogram size, consider setting the reshape method to 'reshape'.")
            
            print(f"Automatically reshaping audio files to fit the Maxtime display spectro value. Files will be {self.Max_time_display_spectro} seconds long.")

            if reshape_method == "reshape":
                # build job, qsub, stuff
                nb_reshaped_files = (orig_fileDuration * total_nber_audio_files) / self.Max_time_display_spectro
                files_for_one_reshape = total_nber_audio_files / nb_reshaped_files
                next_offset_beginning = 0
                offset_end = 0
                i_max = -1
                for batch in range(self.Batch_number):
                    if i_max >= len(self.list_wav_to_process) -1: continue

                    offset_beginning = next_offset_beginning
                    next_offset_beginning = 0

                    i_min = i_max + 1
                    i_max = (i_min + batch_size if batch < self.Batch_number - 1 and i_min + batch_size < len(self.list_wav_to_process) else len(self.list_wav_to_process)) # If it is the last batch, take all files
                    
                    while (i_max - i_min + offset_end) % files_for_one_reshape > 1 and i_max < len(self.list_wav_to_process):
                        i_max += 1

                    offset_end = (i_max - i_min + offset_end) % files_for_one_reshape
                    if offset_end:
                        next_offset_beginning = orig_fileDuration - offset_end
                    else: offset_end = 0 #? ack

                    jobfile = self.Jb.build_job_file(script_path=os.path.join(os.dirname(__file__), "cluster", "audio_reshaper.py"), \
                                script_args=f"--input-files {self.path_input_audio_file} --chunk-size {self.Max_time_display_spectro} --ind-min {i_min}\
                                     --ind-max {i_max} --output-dir {self.audio_path} --offset-beginning {offset_beginning} --offset-end {offset_end}", \
                                jobname="OSmOSE_reshape_py", preset="medium")

                    job_id = self.Jb.submit_job(jobfile, dependency=norma_job_id_list)
                    reshape_job_id_list.append(job_id)
                
            elif reshape_method == "resample":
                silence_arg = "-s" if pad_silence else ""
                for batch in range(self.Batch_number):
                    i_min = batch * batch_size
                    i_max = (i_min + batch_size if batch < self.Batch_number - 1 else len(self.list_wav_to_process)) # If it is the last batch, take all files
                    jobfile = self.Jb.build_job_file(script_path=os.path.join(os.dirname(__file__), "cluster", "reshaper.sh"), \
                                script_args=f"-d {self.Path} -i {os.path.basename(self.path_input_audio_file)} -t {analysis_fs} \
                                    -m {i_min} -x {i_max} -o {self.audio_path} -n {self.Max_time_display_spectro} {silence_arg}", \
                                jobname="OSmOSE_reshape_bash", preset="medium")

                    job_id = self.Jb.submit_job(jobfile, dependency=resample_job_id_list)
                    reshape_job_id_list.append(job_id)


        metadata["dataset_fileDuration"] = self.Max_time_display_spectro
        metadata["dataset_fs"] = self.Analysis_fs
        new_meta_path = os.path.join(self.Path , 'raw', 'audio', str(int(self.Max_time_display_spectro))+'_'+str(self.Analysis_fs), "metadata.csv")
        metadata.to_csv(new_meta_path)

        
        #? Useful or deprecated?
        #region Maybe deprecated
            # if int(max_time_display_spectro) != int(orig_fileDuration):
            #     tt=pd.read_csv(os.path.join(path_osmose_dataset, dataset_ID, 'raw','audio', str(int(orig_fileDuration))+'_'+str(int(orig_fs)) ,'metadata_file.csv'),delimiter=' ',header=None)

            #     # quite complicated stuff, but we intersect audio files from the subset_files, recovering the indices of this intersection to filter the list of file durations
            #     if os.path.isfile( os.path.join(path_osmose_dataset , dataset_ID , 'analysis/subset_files.csv') ):
            #         dd=np.array([os.path.basename(x) for x in glob.glob(os.path.join(self.__path_audio_files,'*wav'))])
            #         pp=pd.read_csv(os.path.join(path_osmose_dataset , dataset_ID , 'analysis/subset_files.csv'),header=None)[0].values
            #         nber_audioFiles_after_segmentation=0
            #         list_files_subset=np.intersect1d(dd , pp )
            #         for file_in_subset in list_files_subset:
            #             ind=np.where(tt[0].values==file_in_subset)[0][0]
            #             nber_audioFiles_after_segmentation += np.ceil(tt[1][ind]/max_time_display_spectro)
            #         nber_audioFiles_after_segmentation=int(nber_audioFiles_after_segmentation)
            #     else:
            #         nber_audioFiles_after_segmentation = int(sum(np.ceil(tt[1].values/max_time_display_spectro)))
                                
            # else:        
            #     nber_audioFiles_after_segmentation = orig_total_nber_audio_files
        #endregion

        for path in [self.__path_output_spectrograms, self.__path_output_spectrogram_matrices]:
            if os.path.exists(path): shutil.rmtree(path)
            os.makedirs(path)

        self.to_csv(os.path.join(self.__path_output_spectrograms, "spectrograms.csv"))

        if not self.__analysis_file:
            #? Standalone method?
            data = {'dataset_ID' : self.Name,'self.Analysis_fs' :float(self.Analysis_fs),'fileScale_nfft' : self.Nfft,
                'fileScale_winsize' : self.Window_size,'fileScale_overlap' : self.Overlap,'colmapspectros' : self.Colmap,
                'nber_zoom_levels' : self.Zoom_levels,'nberAdjustSpectros':self.Number_adjustment_spectrograms,
                'min_color_val':self.Min_color_value,'max_color_val':self.Max_color_value,'max_time_display_spectro':self.Max_time_display_spectro, 
                'folderName_audioFiles':os.path.basename(self.audio_path), 'data_normalization':self.Data_normalization,'fmin_HighPassFilter':self.Fmin_HighPassFilter,
                'sensitivity_dB':20 * log10(self.Sensitivity / 1e6), 'peak_voltage':self.Peak_voltage,'spectro_normalization':self.Spectro_normalization,
                'gain_dB':self.Gain_dB,'zscore_duration':self.Zscore_duration}
            analysis_sheet = pd.DataFrame.from_records([data])
            analysis_sheet.to_csv( os.path.join(analysis_path ,'analysis_sheet.csv') )

    def to_csv(self, filename: str) -> None:
        """Outputs the characteristics of the spectrogram the specified file in csv format.
        
        Parameter
        ---------
        filename: str
            The name of the file to be written."""

        data = {'name' :self.__spectro_foldername , 'nfft':self.Nfft , 'window_size' : self.Window_size , \
             'overlap' : self.Overlap /100 , 'zoom_level': 2**(self.Zoom_levels-1)}
            #TODO: readd `, 'cvr_max':self.Max_color_value, 'cvr_min':self.Min_color_value` above when ok with Aplose
        df = pd.DataFrame.from_records([data])
        df.to_csv(filename , index=False)  


    #region On cluster

    def process_file(self, audio_file: str, *, adjust: bool = False) -> None:
        """Read an audio file and generate the associated spectrogram.
        
        Parameters
        ----------
        audio_file : `str`
            The name of the audio file to be processed
        adjust : `bool`, optional, keyword-only
            Indicates whether the file should be processed alone to adjust the spectrogram parameters (the default is False)"""
        self.__build_path(adjust)

        Zscore = self.Zscore_duration if not adjust else "original"

        #! Determination of zscore normalization parameters
        if Zscore and self.Data_normalization == "zscore" and Zscore != "original":
            average_over_H = int(round(pd.to_timedelta(Zscore).total_seconds() / self.Max_time_display_spectro))

            df=pd.DataFrame()
            for dd in glob(os.path.join(self.__path_summstats,'summaryStats*')):        
                df = pd.concat([ df , pd.read_csv(dd,header=0) ])
                
            df['mean_avg'] = df['mean'].rolling(average_over_H, min_periods=1).mean()
            df['std_avg'] = df['std'].rolling(average_over_H, min_periods=1).std()

            self.__summStats = df

        if audio_file not in os.listdir(self.audio_path):
            raise FileNotFoundError(f"The file {audio_file} must be in {self.audio_path} in order to be processed.")

        if Zscore and Zscore != "original" and self.Data_normalization=="zscore":
            self.__zscore_mean = self.__summStats[self.__summStats['filename'] == audio_file]['mean_avg'].values[0]
            self.__zscore_std = self.__summStats[self.__summStats['filename'] == audio_file]['std_avg'].values[0]


        #! File processing
        data, sample_rate = sf.read(os.path.join(self.audio_path, audio_file))        

        if self.Data_normalization=='instrument':
            data =(data*self.Peak_voltage)/self.Sensitivity / 10**(self.Gain_dB/20)
        
        bpcoef = signal.butter(20, np.array([self.Fmin_HighPassFilter, sample_rate / 2 - 1]), fs=sample_rate, output='sos',
                               btype='bandpass')
        data = signal.sosfilt(bpcoef, data)

        if not os.path.exists(os.path.join(self.__path_output_spectrograms, self.__spectro_foldername, os.path.splitext(audio_file)[0])):
            os.makedirs(os.path.join(self.__path_output_spectrograms, self.__spectro_foldername, os.path.splitext(audio_file)[0]))
        elif adjust:
            shutil.rmtree(os.path.join(self.__path_output_spectrograms, self.__spectro_foldername))
            os.makedirs(os.path.join(self.__path_output_spectrograms, self.__spectro_foldername, os.path.splitext(audio_file)[0]))

        output_file = os.path.join(self.__path_output_spectrograms, self.__spectro_foldername, os.path.splitext(audio_file)[0], audio_file)

        self.gen_tiles(data=data, sample_rate=sample_rate, output_file=output_file)

    def gen_tiles(self, *, data: np.ndarray, sample_rate: int, output_file: str):
        """Generate spectrogram tiles corresponding to the zoom levels.
        
        Parameters
        ----------
        data : `np.ndarray`
            The audio data from which the tiles will be generated.
        sample_rate : `int`
            The sample rate of the audio data.
        output_file : `str`
            The name of the output spectrogram."""
        if self.Data_normalization=='zscore' and self.Zscore_duration:
            if (len(self.Zscore_duration)>0) and (self.Zscore_duration != 'original'):
                data = (data - self.__zscore_mean) / self.__zscore_std
            elif self.Zscore_duration == 'original':
                data = (data - np.mean(data)) / np.std(data)
            
        duration = len(data) / int(sample_rate)
        max_w = 0

        nber_tiles_lowest_zoom_level = 2 ** (self.Zoom_levels - 1)
        tile_duration = duration / nber_tiles_lowest_zoom_level

        Sxx_2 = np.empty((int(self.Nfft / 2) + 1, 1))
        for tile in range(0, nber_tiles_lowest_zoom_level):
            start = tile * tile_duration
            end = start + tile_duration
            
            sample_data = data[int(start * sample_rate):int((end + 1) * sample_rate)]
            
            output_file = f"{os.path.splitext(output_file)[0]}_{str(nber_tiles_lowest_zoom_level)}_{str(tile)}.png"

        Sxx, Freq = self.gen_spectro(data=sample_data, sample_rate=sample_rate, output_file=output_file)
        
        Sxx_2 = np.hstack((Sxx_2, Sxx))

        Sxx_lowest_level = Sxx_2[:, 1:]

        segment_times = np.linspace(0, len(data) / sample_rate, Sxx_lowest_level.shape[1])[np.newaxis, :]

        # loop over the zoom levels from the second lowest to the highest one 
        for zoom_level in range(self.Zoom_levels)[::-1]:

            nberspec = Sxx_lowest_level.shape[1] // (2 ** zoom_level)

            # loop over the tiles at each zoom level
            for tile in range(2 ** zoom_level):
                
                Sxx_int = Sxx_lowest_level[:, tile * nberspec: (tile + 1) * nberspec][:, ::2 ** (self.Zoom_levels - zoom_level)]

                segment_times_int = segment_times[:, tile * nberspec: (tile + 1) * nberspec][:, ::2 ** (self.Zoom_levels - zoom_level)]

                if self.Spectro_normalization == 'density':
                    log_spectro = 10 * np.log10(Sxx_int/(1e-12))
                if self.Spectro_normalization == 'spectrum':
                    log_spectro = 10 * np.log10(Sxx_int)
                
                self.generate_and_save_figures(time=segment_times_int, freq=Freq, log_spectro=log_spectro, output_file=f"{os.path.splitext(output_file)[0]}_{str(2 ** zoom_level)}_{str(tile)}.png")

    def gen_spectro(self, *, data: np.ndarray, sample_rate: int, output_file: str) -> Tuple[np.ndarray, np.ndarray[float]]:
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
        Noverlap = int(self.Window_size * self.Overlap / 100)

        win = np.hamming(self.Window_size)
        if self.Nfft < (0.5 * self.Window_size):
            if self.Spectro_normalization == 'density':
                scale_psd = 2.0 
            if self.Spectro_normalization == 'spectrum':
                scale_psd = 2.0 *sample_rate
        else:
            if self.Spectro_normalization == 'density':
                scale_psd = 2.0 / (((win * win).sum())*sample_rate)
            if self.Spectro_normalization == 'spectrum':
                scale_psd = 2.0 / ((win * win).sum())
            
        Nbech = np.size(data)
        Noffset = self.Window_size - Noverlap
        Nbwin = int((Nbech - self.Window_size) / Noffset)
        Freq = np.fft.rfftfreq(self.Nfft, d=1 / sample_rate)

        Sxx = np.zeros([np.size(Freq), Nbwin])
        Time = np.linspace(0, Nbech / sample_rate, Nbwin)
        for idwin in range(Nbwin):
            if self.Nfft < (0.5 * self.Window_size):
                x_win = data[idwin * Noffset:idwin * Noffset + self.Window_size]
                _, Sxx[:, idwin] = signal.welch(x_win, fs=sample_rate, window='hamming', nperseg=int(self.Nfft),
                                                noverlap=int(self.Nfft / 2) , scaling='density')
            else:
                x_win = data[idwin * Noffset:idwin * Noffset + self.Window_size] * win
                Sxx[:, idwin] = (np.abs(np.fft.rfft(x_win, n=self.Nfft)) ** 2)
            Sxx[:, idwin] *= scale_psd
            
        if self.Spectro_normalization == 'density':
            log_spectro = 10 * np.log10(Sxx/(1e-12))
        if self.Spectro_normalization == 'spectrum':
            log_spectro = 10 * np.log10(Sxx)

        # save spectrogram as a png image
        self.generate_and_save_figures( time=Time, freq=Freq, log_spectro=log_spectro, output_file=output_file)
        
        # save spectrogram matrices (intensity, time and freq) in a npz file
        if not os.path.exists( os.path.dirname(output_file.replace('.png','.npz').replace('spectrograms','spectrograms_mat')) ):
            os.makedirs( os.path.dirname(output_file.replace('.png','.npz').replace('spectrograms','spectrograms_mat')) )
        np.savez(output_file.replace('.png','.npz').replace('spectrograms','spectrograms_mat'), Sxx=Sxx,log_spectro=log_spectro , Freq=Freq, Time=Time)

        return Sxx, Freq

    def generate_and_save_figures(self, *, time: np.ndarray[float], freq: np.ndarray[float], log_spectro: np.ndarray[int], output_file: str):
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
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fact_x * 1800 / my_dpi, fact_y * 512 / my_dpi), dpi=my_dpi)
        color_map = plt.cm.get_cmap(self.Colmap)  # .reversed()
        plt.pcolormesh(time, freq, log_spectro, cmap=color_map)
        plt.clim(vmin=self.Min_color_value, vmax=self.Max_color_value)
        #plt.colorbar()
        
        # If generate all
        fig.axes[0].get_xaxis().set_visible(True)
        fig.axes[0].get_yaxis().set_visible(True)
        ax.set_frame_on(True)
        
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['top'].set_visible(True)        
        
        # For test
        fig.axes[0].get_xaxis().set_visible(True)
        fig.axes[0].get_yaxis().set_visible(True)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        plt.colorbar()

        # Saving spectrogram plot to file
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()
    #endregion
