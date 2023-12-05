"""
Dealing with data anomalies
=====================================

This code will show you how to deal with anomalies in raw data or their metadata, from corrupted files to differences in sample frequency.
"""



# %%
# Preambule : what is called an anomaly ?
# --------------------------------------------
# For OSmOSE, audio metadata refers to any audio information collected both from audio file header or from audio content after file reading at our dataset uploading. Audio metadata read from file header (ie `read_header(audio_file)`):
#
# 1. filename: check filenames for consistency, especially they have to contain a unique timestamp template
# 2. timestamp
# 3. extension : check that are only .wav files are present (.WAV files represent an anomaly)
# 4. format
# 5. duration
# 6. sample rate
# 7. dutycyle (timedelta between current timestamp and previous one)
# 8. volume
# 9. sampwidth
# 10. number of channels (stereo or mono)
# 11. subtype (eg PCM16, see rumengol)
# Audio metadata extracted from audio content (ie after data = audio_f.read() )
# 12. min and max of sample amplitude

# %%
# Basic scan

# - this scan only look at audio metadata from headers (1 to 11)
# - run within the jupyter hub session
# - perform the tests 1-3 and 4

# %%
# Heavy scan

# - it will load all audio files and collect all 
# - run with pbs jobs
# - perform all tests

# %%
# Light anomaly tests

# 4. len(np.unique) > 1 sur duration (round to second) 
# 5. len(np.unique) > 1 sur sample_rate 
# 6. len(np.unique) > 1 sur inter_duration (round to second) 

# %%
# Strong anomaly tests

# 7. at least one file header is corrupted (ie output status of method read_header)
# 8. at least one wav file cannot be read (ie status of data = audio_f.read() )
# 9. at least one audio file contains data out of the range -1:1 (based on info 12)
# 10. at least one audio file extension is not supported (based on info 3)

# %%
# Strong anomaly test failing, the following operations are done:
    
# - interrupt upload
# - print the failing test(s) and the filename(s) concerned
# - block the use of force_upload
# - suggest to do a complete scan of the dataset
# - orient the user to the notebook «visualize and understand my audio metadata»
# - orient the user to the notebook «handle my badly shaped dataset »

# %%
# Light anomaly test failing, the following operations are done:

# - print the failing test(s) and the filename(s) concerned
# - suggest using force_upload
# - orient the user to the notebook «visualize and understand my audio metadata»




# %%
# Codes
# ------------------------

# sphinx_gallery_thumbnail_path = '_static/thumbnail_anomalies.png'

from pathlib import Path
from OSmOSE import Dataset

#####################################################
# Let's build the dataset following  :ref:`sphx_glr_gallery_basic_use_cases_Dataset_1_build_dataset.py`

path_osmose_dataset = "/home6/cazaudo/Bureau/osmose_sample_datasets/"
dataset_name = "Cetiroise"
campaign_name = ""

gps_coordinates = (48.44,-5.05)
depth = 20

dataset = Dataset(dataset_path = Path(path_osmose_dataset, campaign_name, dataset_name), gps_coordinates = gps_coordinates, depth = depth)

#####################################################
# When trying to build this dataset, it will not work directly.
dataset.build(date_template = "%Y_%m_%d_%H_%M_%S" )

#####################################################
# The variable `force_upload` allows you to upload your dataset on the platform despite detected anomalies.
force_upload = True

dataset.build(date_template = "%Y_%m_%d_%H_%M_%S" , force_upload = force_upload)


