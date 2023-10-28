"""
==============================================
Generate spectrograms
==============================================

This code will show you how to compute spectrograms
"""

from pathlib import Path
from OSmOSE import Spectrogram
import glob

#####################################################
# Define dataset path and name

path_osmose_dataset = "/home6/cazaudo/Bureau/osmose_sample_datasets/"
dataset_name = "MPSU"
campaign_name = ""

#####################################################
# Define standard parameters for spectrogram computation

spectrogram = Spectrogram(dataset_path = Path(path_osmose_dataset, campaign_name, dataset_name))

spectrogram.dataset_sr = 18000
spectrogram.spectro_duration = 15

spectrogram.nfft = 1024
spectrogram.window_size = 1024
spectrogram.overlap = 0


#####################################################
# Check size of spectrogram

spectrogram.check_spectro_size()

#####################################################
# Initialize spectrogram computation ; this method will prepare audio data 

reshape_method = "classic" # Automatically reshape the audio files to fit the spectro_duration value. Available methods : "classic" or "legacy"
merge_on_reshape = False # Set to False if fyou don't want to merge audio files while reshaping them (if they do not follow each other chronologically for example)
force_init = False # Force every initialization parameter, including force_reshape and other computing jobs. It is best to avoid using it.
spectrogram.initialize(reshape_method=reshape_method, force_init=force_init, merge_on_reshape=merge_on_reshape)#, offset_overlap=offset_overlap)

list_wav_to_process = glob.glob(path_osmose_dataset+f'{dataset_name}/data/audio/{spectrogram.spectro_duration}_{spectrogram.dataset_sr}/*wav')

spectrogram.process_all_files(list_wav_to_process=list_wav_to_process)




