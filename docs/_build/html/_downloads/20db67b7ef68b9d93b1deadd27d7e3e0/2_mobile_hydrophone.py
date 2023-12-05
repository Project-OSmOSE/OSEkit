"""
In case of a mobile hydrophone
=====================================

This code will show you how to build a OSmOSE dataset in case of a mobile hydrophone.
"""

# %%
# Prerequisites
# ------------------------
# The time-dependent coordinates of your mobile hydrophone should be stored in a csv file, put in the root folder where the audio files are, ie ``{path_osmose_dataset}/dataset/{dataset_name}``, or ``{path_osmose_dataset}/dataset/{campaign_name}/{dataset_name}``. 
# Its filename should also contain the term _gps_, and have at least the following standardized column names : timestamp, lat, lon and depth. In this tutorial we will use the file 'gps_depth.csv' of the dataset `SES`.


# %%
# Codes
# ------------------------
# Following code lines are similar to :ref:`sphx_glr_gallery_basic_use_cases_Dataset_1_build_dataset.py`. However the timestamps of dataset `SES` cannot be extracted from the filename so we had to prepare it manually and place it into ``{path_osmose_dataset}/dataset/{dataset_name}``.

# sphinx_gallery_thumbnail_path = '_static/thumbnail_mobile_hydrophone.png'

from pathlib import Path
from OSmOSE import Dataset

path_osmose_dataset = "/home6/cazaudo/Bureau/osmose_sample_datasets/"
dataset_name = "SES"
campaign_name = ""

# %%
# See how raw data preparation are organized
for x in Path(path_osmose_dataset, campaign_name, dataset_name).iterdir():
    print (x)
    
#####################################################
# The csv file containing the time-varying gps coordinates must be assigned to the variable `gps_coordinates`, same for the variable `depth`.
gps_coordinates = 'gps_depth.csv'
depth = 'gps_depth.csv'

#####################################################
# Run the method :meth:`OSmOSE.Dataset.Dataset.build` of the class :class:`OSmOSE.Dataset.Dataset` as in :ref:`sphx_glr_gallery_basic_use_cases_Dataset_1_build_dataset.py`. Note that we had to set `force_upload` to True to allows the building of the dataset despite detected anomalies; go to :ref:`sphx_glr_gallery_basic_use_cases_Dataset_4_dealwith_anomalies.py`
dataset = Dataset(dataset_path = Path(path_osmose_dataset, campaign_name, dataset_name), gps_coordinates = gps_coordinates, depth = depth)
dataset.build(force_upload=True)
