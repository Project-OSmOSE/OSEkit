"""
==============================================
Segment and resample a dataset
==============================================

This code will show you how to segment and resample your original audio dataset into an analysis dataset.
"""

# %%

# sphinx_gallery_thumbnail_path = '_static/thumbnail_segment_resample.png'


from pathlib import Path
from OSmOSE import Spectrogram

#####################################################
# Define dataset path and name
path_osmose_dataset = "/home6/cazaudo/Bureau/osmose_sample_datasets/"
dataset_name = "SPM"
campaign_name = ""

dataset = Spectrogram(
    dataset_path=Path(path_osmose_dataset, campaign_name, dataset_name)
)

print(dataset)

#####################################################
# The two following parameters `spectro_duration` (in s) and `dataset_sr` (in Hz) will allow you to process your data using different file durations (ie segmentation) and/or sampling rate (ie resampling) parameters. `spectro_duration` is the maximal duration of the spectrogram display window. To process audio files from your original folder (ie without any segmentation and/or resampling operations), use the original audio file duration and sample rate parameters estimated at your dataset uploading (they are printed in the previous cell).
dataset.dataset_sr = 32000
dataset.spectro_duration = 60

#####################################################
# In case of audio segmentation, you can use the following variable `audio_file_overlap` (in seconds, default value = 0) to set an overlap in seconds between two consecutive segments.
dataset.audio_file_overlap = 0  # seconds

#####################################################
# The method :meth:`OSmOSE.Spectrogram.Spectrogram.initialize` allows you to segment and resample your original audio files based on `spectrogram.dataset_sr` and `spectrogram.spectro_duration`. This method will create another folder of audio files named `{spectro_duration}_{dataset_sr}`.
dataset.initialize()
