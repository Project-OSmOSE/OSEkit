from math import log10
import shutil
from typing import Union, List
from Dataset import Dataset
import pandas as pd
import os
import sys
from glob import glob
from audio_reshaper import reshape

class Spectrogram(Dataset):

    def __init__(self, config: str, analysis_params: dict = None, audio_files: List[str] = None) -> None:
        super().__init__(config)

        if os.path.exists(os.path.join(self.Path, "analysis", "analysis_sheet.csv")):
            self.__analysis_file = True
            analysis_sheet = pd.read_csv(self.__config.analysis_path,header=0)
        elif analysis_params:
            self.__analysis_file = False
            analysis_sheet = {key: [value] for (key, value) in analysis_params.items()}
        else:
            raise ValueError("You need to either have a valid analysis/analysis_sheet.csv file or provide the analysis metadatas as a dict.")

        self.__fileScale_nfft : int = analysis_sheet['fileScale_nfft'][0]
        self.__fileScale_winsize : int = analysis_sheet['fileScale_winsize'][0]
        self.__fileScale_overlap : int = analysis_sheet['fileScale_overlap'][0]
        self.__colmapspectros : str = analysis_sheet['colmapspectros'][0]
        self.__nber_zoom_levels : int = analysis_sheet['nber_zoom_levels'][0]
        self.__min_color_val : int = analysis_sheet['min_color_val'][0]
        self.__max_color_val : int = analysis_sheet['max_color_val'][0]
        self.__nberAdjustSpectros : int = analysis_sheet['nberAdjustSpectros'][0] #???
        self.__maxtime_display_spectro : int = analysis_sheet['max_time_display_spectro'][0]

        self.__zscore_duration : Union[float, str] = analysis_sheet['zscore_duration'][0] if isinstance(analysis_sheet['zscore_duration'][0], float) else None

        # fmin cannot be 0 in butterworth. If that is the case, it takes the smallest value possible, epsilon
        self.__fmin_HighPassFilter : int = analysis_sheet['fmin_HighPassFilter'][0] if analysis_sheet['fmin_HighPassFilter'][0] != 0 else sys.float_info.epsilon
        sensitivity_dB : int = analysis_sheet['sensitivity_dB'][0]
        self.__sensitivity : float = 10**(sensitivity_dB/20) * 1e6
        self.__peak_voltage : float  = analysis_sheet['peak_voltage'][0]
        self.__spectro_normalization : str = analysis_sheet['spectro_normalization'][0]
        self.__data_normalization : str = analysis_sheet['data_normalization'][0]
        self.__gain_dB : float = analysis_sheet['gain_dB'][0]


    #region Spectrogram properties

    @property
    def Nfft(self):
        """"""
        return self.__fileScale_nfft
    
    @Nfft.setter
    def Nfft(self, value):
        if self.__analysis_file:
            self.__fileScale_nfft = value
        else:
            raise ValueError("This parameter cannot be changed since it has been initialized with the analysis sheet.")

    @property
    def Window_size(self):
        return self.__fileScale_winsize
    
    @Window_size.setter
    def Window_size(self, value):
        if self.__analysis_file:
            self.__fileScale_winsize = value
        else:
            raise ValueError("This parameter cannot be changed since it has been initialized with the analysis sheet.")

    @property
    def Overlap(self):
        return self.__fileScale_overlap
    
    @Overlap.setter
    def Overlap(self, value):
        if self.__analysis_file:
            self.__fileScale_overlap = value
        else:
            raise ValueError("This parameter cannot be changed since it has been initialized with the analysis sheet.")

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
        if self.__analysis_file:
            self.__fmin_HighPassFilter = value
        else:
            raise ValueError("This parameter cannot be changed since it has been initialized with the analysis sheet.")

    @property
    def Sensitivity(self):
        return self.__sensitivity

    @Sensitivity.setter
    def Sensitivity(self, value):
        if self.__analysis_file:
            self.__sensitivity = value
        else:
            raise ValueError("Cannot change attribute as analysis_path is not empty.")

    @property
    def Peak_voltage(self):
        return self.__peak_voltage

    @Peak_voltage.setter
    def Peak_voltage(self, value):
        if self.__analysis_file:
            self.__peak_voltage = value
        else:
            raise ValueError("Cannot change attribute as analysis_path is not empty.")

    @property
    def Spectro_normalization(self):
        return self.__spectro_normalization

    @Spectro_normalization.setter
    def Spectro_normalization(self, value):
        if self.__analysis_file:
            self.__spectro_normalization = value
        else:
            raise ValueError("Cannot change attribute as analysis_path is not empty.")

    @property
    def Data_normalization(self):
        return self.__data_normalization
    
    @Data_normalization.setter
    def Data_normalization(self, value):
        if self.__analysis_file:
            self.__data_normalization = value
        else:
            raise ValueError("Cannot change attribute as analysis_path is not empty.")

    @property
    def Gain_dB(self):
        return self.__gain_dB

    @Gain_dB.setter
    def Gain_dB(self, value):
        if self.__analysis_file:
            self.__gain_dB = value
        else:
            raise ValueError("Cannot change attribute as analysis_path is not empty.")

    #endregion

    # TODO: some cleaning
    def initialize(self, analysis_fs: float, ind_min: int, ind_max: int, auto_reshape: bool = False) -> None:
        
        # Load variables from raw metadata
        metadata = pd.read_csv(os.path.join(self.Path, "raw","metadata.csv"))
        orig_fileDuration = metadata['orig_fileDuration'][0]
        orig_fs = metadata['orig_fs'][0]
        total_nber_audio_files = metadata['nberWavFiles'][0]

        input_audio_foldername = str(orig_fileDuration)+'_'+str(int(orig_fs))

        path_input_audio_file = os.path.join(self.Path, "raw", "audio", input_audio_foldername)

        # Reshape audio files to fit the maximum spectrogram size, whether it is greater or smaller.
        #? Quite I/O intensive and monothread, might need to rework to allow qsub.
        if self.Max_time_display_spectro != int(orig_fileDuration):
            # We might reshape the files and create the folder. Note: reshape function might be memory-heavy and deserve a proper qsub job. 
            if self.Max_time_display_spectro > int(orig_fileDuration) and not auto_reshape:
                raise ValueError("Spectrogram size cannot be greater than file duration. If you want to automatically reshape your audio files to fit the spectrogram size, consider adding auto_reshape=True as parameter.")
            
            reshaped_path = os.path.join(self.Path , 'raw', 'audio', str(self.Max_time_display_spectro)+'_'+str(analysis_fs))
            print(f"Automatically reshaping audio files to fit the Maxtime display spectro value. Files will be {self.Max_time_display_spectro} seconds long.")

            reshaped_files = reshape(self.Max_time_display_spectro, path_input_audio_file, reshaped_path)
            metadata["dataset_totalDuration"] = len(reshaped_files) * self.Max_time_display_spectro

        metadata["dataset_fileDuration"] = self.Max_time_display_spectro
        metadata["dataset_fs"] = analysis_fs
        new_meta_path = os.path.join(self.Path , 'raw', 'audio', str(int(self.Max_time_display_spectro))+'_'+str(analysis_fs), "metadata.csv")
        metadata.to_csv(new_meta_path)

        audio_foldername = str(self.Max_time_display_spectro)+'_'+str(analysis_fs)
        self.__audio_path = os.path.join(self.Path, "raw", "audio", audio_foldername)
        analysis_path = os.path.join(analysis_path)

        self.__path_output_spectrograms = os.path.join(analysis_path, "spectrograms", audio_foldername)
        self.__path_summstats = os.path.join(analysis_path, "normaParams", audio_foldername)

        self.__spectro_foldername = f"nfft={str(self.Nfft)}_winsize={str(self.Window_size)}_overlap={str(self.Overlap)} \
                                _cvr={str(self.Min_color_value)}-{str(self.Max_color_value)}"

        self.__path_output_spectrogram_matrices = os.path.join(analysis_path, "spectrograms_mat", audio_foldername, self.__spectro_foldername)


        if self.Data_normalization == "zscore" and self.Zscore_duration != "original" and self.Zscore_duration:
            average_over_H = int(round(pd.to_timedelta(self.Zscore_duration).total_seconds() / self.Max_time_display_spectro))

            df=pd.DataFrame()
            for dd in glob(os.path.join(self.__path_summstats,'summaryStats*')):        
                df = pd.concat([ df , pd.read_csv(dd,header=0) ])
                
            df['mean_avg'] = df['mean'].rolling(average_over_H, min_periods=1).mean()
            df['std_avg'] = df['std'].rolling(average_over_H, min_periods=1).std()

            self.__summStats = df


        list_wav_withEvent_comp = glob.glob(os.path.join(path_input_audio_file , '*wav'))
        list_wav_withEvent = list_wav_withEvent_comp[ind_min:ind_max]
        
        list_wav_withEvent = [os.path.basename(x) for x in list_wav_withEvent]

        if os.path.isfile(os.path.join(analysis_path,"subset_files.csv")):
            subset = pd.read_csv(os.path.join(self.Path , 'analysis', 'subset_files.csv'),header=None)[0].values
            list_wav_withEvent = list(set(subset).intersection(set(list_wav_withEvent)))

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
            if os.path.exists(path): shutil.remtree(path)
            os.makedirs(path)

        self.to_csv(os.path.join(self.__path_output_spectrograms, "spectrograms.csv"))

        if not self.__analysis_file:
            #? Standalone method?
            data = {'dataset_ID' : self.Name,'analysis_fs' :float(analysis_fs),'fileScale_nfft' : self.Nfft,
                'fileScale_winsize' : self.Window_size,'fileScale_overlap' : self.Overlap,'colmapspectros' : self.Colmap,
                'nber_zoom_levels' : self.Zoom_levels,'nberAdjustSpectros':self.Number_adjustment_spectrograms,
                'min_color_val':self.Min_color_value,'max_color_val':self.Max_color_value,'max_time_display_spectro':self.Max_time_display_spectro, 
                'folderName_audioFiles':audio_foldername, 'data_normalization':self.Data_normalization,'fmin_HighPassFilter':self.Fmin_HighPassFilter,
                'sensitivity_dB':20 * log10(self.Sensitivity / 1e6), 'peak_voltage':self.Peak_voltage,'spectro_normalization':self.Spectro_normalization,
                'gain_dB':self.Gain_dB,'zscore_duration':self.Zscore_duration}
            analysis_sheet = pd.DataFrame.from_records([data])
            analysis_sheet.to_csv( os.path.join(analysis_path ,'analysis_sheet.csv') )

    def to_csv(self, filename: str) -> None:
        """Outputs the characteristics of the spectrogram the specified file in csv format.
        
        Parameter:
        ----------
            filename: The name of the file to be written."""

        data = {'name' :self.__spectro_foldername , 'nfft':self.Nfft , 'window_size' : self.Window_size , \
             'overlap' : self.Overlap /100 , 'zoom_level': 2**(self.Zoom_levels-1) , 'cvr_max':self.Max_color_value, 'cvr_min':self.Min_color_value}
        df = pd.DataFrame.from_records([data])
        df.to_csv(filename , index=False)  

    

    def process_file(self):
        pass