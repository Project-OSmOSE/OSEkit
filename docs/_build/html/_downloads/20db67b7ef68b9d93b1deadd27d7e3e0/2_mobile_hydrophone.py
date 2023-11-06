"""
In case of a mobile hydrophone
=====================================

This code will show you how to build a OSmOSE dataset in case of a mobile hydrophone.
"""

# %%
# Prerequisites
# ------------------------

# The time-dependent coordinates of your mobile hydrophone should be stored in a csv file, put in the root folder where the audio files are, ie ``{path_osmose_dataset}/dataset/{dataset_name}``, or ``{path_osmose_dataset}/dataset/{campaign_name}/{dataset_name}``. Its filename should also contain the term _gps_, and have at least the following standardized column names : timestamp, lat, lon and depth.


# %%
# Codes
# ------------------------
# First code lines are similar to :ref:`sphx_glr_gallery_basic_use_cases_Dataset_1_build_dataset.py`

from pathlib import Path
from OSmOSE import Dataset

path_osmose_dataset = "/home6/cazaudo/Bureau/osmose_sample_datasets/"
dataset_name = "SES1"
campaign_name = ""

#####################################################
# The csv file containing the time-varying gps coordinates must be assigned to the variable `gps_coordinates`, same for the variable `depth`.
gps_coordinates = 'gps_depth.csv'
depth = 'gps_depth.csv'

#####################################################
# Run the method :meth:`OSmOSE.Dataset.Dataset.build` of the class :class:`OSmOSE.Dataset.Dataset` as in :ref:`sphx_glr_gallery_basic_use_cases_Dataset_1_build_dataset.py`
dataset = Dataset(dataset_path = Path(path_osmose_dataset, campaign_name, dataset_name), gps_coordinates = gps_coordinates, depth = depth)
dataset.build(force_upload=True)



