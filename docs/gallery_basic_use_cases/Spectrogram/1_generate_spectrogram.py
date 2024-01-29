"""
==============================================
Generation with default parameters
==============================================

This code will show you how to compute spectrograms
"""

# %%
# Prerequisites
# ================
# Your dataset must be built before you can do any processing on it ; see the use case :ref:`sphx_glr_gallery_basic_use_cases_Dataset_1_build_dataset.py` to do this.

# %%
# Spectrogram size : a subject of matter ?
# ===============================================
# Whatever you intend to do with it, spectrogram size is most often a subject of matter. For example, to perform manual annotation on the spectrograms of our example here, containing
# more than 40k spectra, the user should be aware that numerical compression during image generation and/or display on your screen will occur. To avoid this, it is recommended that to make this number of spectra as close as your horizontal screen resolution (ie approximately 2000 pixels, as a classical screen resolution is 1920x1080 pixels (horizontal pixels) x (vertical pixels) ).
# It is also good to know that over-resoluted spectrograms are obtained at a higher memory cost, and sometimes may not fit at all in memory.


# %%
# Codes
# ===============

from pathlib import Path
from OSmOSE import Spectrogram
import glob


# %%
# Path and dataset names
# ------------------------

path_osmose_dataset = "/home6/cazaudo/Bureau/osmose_sample_datasets/"
dataset_name = "SPM"
campaign_name = ""

#####################################################
# Run the class :class:`OSmOSE.Spectrogram.Spectrogram`
# ----------------------------------------------------------

spectrogram = Spectrogram(
    dataset_path=Path(path_osmose_dataset, campaign_name, dataset_name)
)

#####################################################
# Printing the spectrogram instance will give you all useful metadata of the dataset from which you want to compute spectrograms

print(spectrogram)

#####################################################
# You can check the size of spectrograms that will be generated using :meth:`OSmOSE.Spectrogram.Spectrogram.check_spectro_size`. This spectrogram is very over-resoluted, we will see in
# :ref:`sphx_glr_gallery_basic_use_cases_Spectrogram_2_tune_parameters.py` how we can spectrogram parameters to reduce this number.

spectrogram.check_spectro_size()

#####################################################
# Initialize spectrogram parameters
# ----------------------------------------------------------
# Initialize everything needed for spectrogram computation ; in particular, this method will segment and/or resample audio files if needed

spectrogram.initialize()

#####################################################
# In this first tutorial we will see only one spectrogram parameter, namely the min / max values (in dB) of the spectrogram corlobar . We will see other parameters (for the moment set to their default values) in :ref:`sphx_glr_gallery_basic_use_cases_Spectrogram_2_tune_parameters.py`)
spectrogram.dynamic_min = -80
spectrogram.dynamic_max = 10

#####################################################
# Launch processing
# ----------------------------------------------------------
# The method :meth:`OSmOSE.Spectrogram.Spectrogram.process_all_files` will generate spectrograms for the different pre segmented and/or resampled (if necessary) audio files

wav_path = glob.glob(
    path_osmose_dataset
    + f"{dataset_name}/data/audio/{spectrogram.spectro_duration}_{spectrogram.dataset_sr}/*wav"
)
spectrogram.save_spectro_metadata(False)
spectrogram.process_all_files(list_wav_to_process=wav_path)

#####################################################
# Visualize an example of spectrogram
# ---------------------------------------------

spectrogram_path = glob.glob(
    path_osmose_dataset
    + f"{dataset_name}/processed/spectrogram/{spectrogram.spectro_duration}_{spectrogram.dataset_sr}/{spectrogram.nfft}_{spectrogram.window_size}_{spectrogram.overlap}/image/*png"
)

from matplotlib import pyplot as plt
from matplotlib import image as mpimg

image = mpimg.imread(spectrogram_path[2])
plt.imshow(image, interpolation="nearest", aspect="auto")

ax = plt.gca()
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_xticks([])
ax.set_yticks([])

plt.show()
