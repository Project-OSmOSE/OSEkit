"""
==============================================
Tuning some parameters
==============================================

This code will show you how to tune more spectrogram parameters
"""

from pathlib import Path
from OSmOSE import Spectrogram
import glob

#####################################################
# Define dataset path and name

path_osmose_dataset = "/home6/cazaudo/Bureau/osmose_sample_datasets/"
dataset_name = "SPM"
campaign_name = ""

spectrogram = Spectrogram(
    dataset_path=Path(path_osmose_dataset, campaign_name, dataset_name)
)

print(spectrogram)


#####################################################
# ^^^^^^^^^^^^^^^^^^^^^^^
# Segment and resample
# ^^^^^^^^^^^^^^^^^^^^^^^
# Note that we have already seen how performing these operations in :ref:`sphx_glr_gallery_basic_use_cases_Dataset_3_segment_resample_dataset.py`. The two following parameters `spectro_duration` (in s) and `dataset_sr` (in Hz) will allow you to process your data using different file durations (ie segmentation) and/or sampling rate (ie resampling) parameters. `spectro_duration` is the maximal duration of the spectrogram display window. To process audio files from your original folder (ie without any segmentation and/or resampling operations), use the original audio file duration and sample rate parameters estimated at your dataset uploading (they are printed in the previous cell).

spectrogram.dataset_sr = 4000
spectrogram.spectro_duration = 60

#####################################################
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Nfft, window size and overlap
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# After that, you can set the following classical spectrogram parameters : `nfft` (in samples), `winsize` (in samples), `overlap` (in \%). Note that with those parameters you set the resolution of your spectrogram display window with the smallest duration, obtained with the highest zoom level.

spectrogram.nfft = 1024
spectrogram.window_size = 512
spectrogram.overlap = 80

#####################################################
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Zoom levels
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Then, you can set the value of `zoom_levels`, which is the number of zoom levels you want (they are used in our web-based annotation tool APLOSE). With `zoom_levels = 0`, your shortest spectrogram display window has a duration of `spectro_duration` seconds (that is no zoom at all) ; with `zoom_levels = 1`, a duration of `spectro_duration`/2 seconds ; with `zoom_levels = 2`, a duration of `spectro_duration`/4 seconds ...

spectrogram.zoom_level = 0  # int


#####################################################
# Normalization of audio data and/or spectra
# ===========================================
# Normalization over raw data samples with the variable `data_normalization` (default value `'none'`, i.e. no normalization) :
# - instrument-based normalization with the three parameters `sensitivity_dB` (in dB, default value = 0), `gain` (in dB, default value = 0) and `peak_voltage` (in V, default value = 1). Using default values, no normalization will be performed ;
# - z-score normalization over a given time period through the variable `zscore_duration`, applied directly on your raw timeseries. The possible values are:
#     - `zscore_duration = 'original'` : the audio file duration will be used as time period ;
#     - `zscore_duration = '10H'` : any time period put as a string using classical [time alias](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases). This period should be higher than your file duration.
# Normalization over spectra with the variable `spectro_normalization` (default value `'density'`, see OSmOSEanalytics/documentation/theory_spectrogram.pdf for details) :
# - density-based normalization by setting `spectro_normalization = 'density'`
# - spectrum-based normalization by setting `spectro_normalization = 'spectrum'`
# In the cell below, you can also have access to the amplitude dynamics in dB throuh the parameters `dynamic_max` and `dynamic_min`, the colormap `spectro_colormap` to be used (see possible options in the [documentation](https://matplotlib.org/stable/tutorials/colors/colormaps.html)) and specify the frequency cut `HPfilter_freq_min` of a high-pass filter if needed.

spectrogram.spectro_normalization = "density"

spectrogram.dynamic_min = -140
spectrogram.dynamic_max = -12

#####################################################
# Check size of spectrogram

spectrogram.check_spectro_size()


#####################################################
# You can use the variable `file_list` in the cell below to adjust your spectrogram parameters on specific files; put their names in this list as follows, eg `file_list = ['2020_06_05T15_10_00.wav','2020_06_07T15_41_40.wav','2020_06_09T16_13_20.wav','2020_06_05T15_41_40.wav']`
# `dataset.number_adjustment_spectrograms` is the number of spectrogram examples used to adjust your parameters. If you are really not sure about your parameters, it is better to start with a small number, because each time you will have to wait for the generation of all your `dataset.number_adjustment_spectrograms` (x the different zoom levels) spectrograms before being able to re-generate spectrograms with another set of parameters. `dataset.batch_number` indicates the number of concurrent jobs. A higher number can speed things up until a certain point. It still does not work very well.

#####################################################
# Initialize everything needed for spectrogram computation ; in particular, this method will segment and/or resample audio files if needed

spectrogram.initialize()
wav_path = glob.glob(
    path_osmose_dataset
    + f"{dataset_name}/data/audio/{spectrogram.spectro_duration}_{spectrogram.dataset_sr}/*wav"
)
spectrogram.save_spectro_metadata(False)
spectrogram.process_all_files(list_wav_to_process=wav_path)

#####################################################
# Visualize an example of spectrogram

spectrogram_path = glob.glob(
    path_osmose_dataset
    + f"{dataset_name}/processed/spectrogram/{spectrogram.spectro_duration}_{spectrogram.dataset_sr}/{spectrogram.nfft}_{spectrogram.window_size}_{spectrogram.overlap}/image/*png"
)

from matplotlib import pyplot as plt
from matplotlib import image as mpimg

image = mpimg.imread(spectrogram_path[0])
plt.imshow(image, interpolation="nearest", aspect="auto")

ax = plt.gca()
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_xticks([])
ax.set_yticks([])

plt.show()
