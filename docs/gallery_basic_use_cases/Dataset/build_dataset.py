"""
First time building a OSmOSE dataset
=====================================

This code will format your raw data into a OSmOSE dataset
"""





# %%
# Raw data preparation
# ------------------------
# Before you can build your dataset: 
#
# - choose a dataset name (should not contain any special character, including '-'⁾ ; 
# - create the folder ``{local_working_dir}/dataset/{dataset_name}``, or ``{local_working_dir}/dataset/{campaign_name}/{dataset_name}`` in case your dataset is part of a recording campaign; 
# - place in this folder your audio data, they can be individual files or contain within multiple sub-folders ; 
# - if you have any csv files (either a ``timestamp.csv`` or ``*gps*.csv`` file) should also be placed in this folder.

# %% 
# **About timestamps** 
# All timestamps from your original data (from your audio filenames or from your csv files) MUST follow the same timestamp template which should be given in ``date_template`` 

# %% 
# **About auxiliary csv files** 
# The ``*gps*.csv`` file provides the GPS track (ie latitude and longitude coordinates) of a moving hydrophone. This file must contain the term *gps* in its filename. Auxiliary csv files : they must contain headers with the following standardized names : timestamp , depth , lat , lon

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
# In our dataset, we have made mandatory the setting of two metadata variables, namely `gps_coordinates` (in decimal degree) and `depth` (in m) of the hydrophone. The variable `gps_coordinates` is the tuple (latitude , longitude) and `depth` is a positive integer.
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
dataset.build(date_template = date_template , force_upload=force_upload, number_test_bad_files=1)




