"""
First time building a OSmOSE dataset
=====================================

This code will show you how to format your raw audio data into a OSmOSE dataset, in the case of a fixed hydrophone.
"""



# %%
# Prerequisites
# ------------------------
# Your raw data must be structured as detailed below. Besides, in this tutorial we only deal with the case of a fixed hydrophone ; for a mobile hydrophone you should pursue with the tutorial :ref:`sphx_glr_gallery_basic_use_cases_Dataset_2_mobile_hydrophone.py`


# %%
# Raw data preparation
# ------------------------
# Before you can build your dataset: 
#
# - choose a dataset name (should not contain any special character, including '-'⁾ ; 
# - create the folder ``{local_working_dir}/dataset/{dataset_name}``, or ``{local_working_dir}/dataset/{campaign_name}/{dataset_name}`` in case your dataset is part of a recording campaign; 
# - place in this folder your audio data, they can be individual files or contain within multiple sub-folders ; 

# %% 
# **About timestamps** 
# All timestamps from your original data (from your audio filenames or from your csv files) MUST follow the same timestamp template which should be given in ``date_template`` 


# %%
# Codes
# ------------------------

# sphinx_gallery_thumbnail_path = '_static/dataset_metadata.png'


from pathlib import Path
from OSmOSE import Dataset

#####################################################
# You first have to set the `path_osmose_dataset`, which is where your dataset named `dataset_name` should be ; unless it is part of a recording campaign named `campaign_name`, your dataset should be present in `{path_osmose_dataset}/{campaign_name}/{dataset_name}`.

path_osmose_dataset = "/home6/cazaudo/Bureau/osmose_sample_datasets/"
dataset_name = "MPSU"
campaign_name = ""

#####################################################
# In our dataset, we have made mandatory the setting of two metadata variables, namely `gps_coordinates` (tuple of (latitude , longitude) coordinates in decimal degree) and `depth` (positive integer in meter) of the hydrophone. 
gps_coordinates = (10,10)
depth = 10

#####################################################
# Lets' review now three optional parameters. You can set the `timezone` of your data if it happens to be different from UTC+00:00 (default value) ; its format MUST follow `"+02:00"` for UTC+02:00 for example.
timezone = "+00:00" 

#####################################################
# The variable `date_template` should be used to help us extracting the timestamp from your audio filenames ; it should be set in a strftime format.
date_template = "%Y%m%d_%H%M%S" 

#####################################################
# The variable `force_upload` allows you to upload your dataset on the platform despite detected anomalies.
force_upload = False

#####################################################
# Run the method :meth:`OSmOSE.Dataset.Dataset.build` of the class :class:`OSmOSE.Dataset.Dataset`
dataset = Dataset(dataset_path = Path(path_osmose_dataset, campaign_name, dataset_name), gps_coordinates = gps_coordinates, depth = depth, timezone=timezone)
dataset.build(date_template = date_template , force_upload=force_upload)




