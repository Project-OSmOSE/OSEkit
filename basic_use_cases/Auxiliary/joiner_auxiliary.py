"""
=====================================================
Join with csv files and ERA5 data
=====================================================

This code will join welch spectra with variables from ERA5 within a pandas dataframe
"""


# %%
# ERA5 downloading
# ------------------------
# To use this code you will first need to download and format some ERA5 data : to do so, please use this `notebook <./download_ERA5.html>`__. For OSmOSE members, this notebook can be directly executing
# on our `Google drive team  <https://drive.google.com/drive/folders/1QtNjUo1EaGEKSs4BY_E9iRUSWAlw4bOs>`_

# %%
# Requirement
# ------------------------
# This code uses latitude and longitude coordinates of the hydrophone (which can be time-dependent or not) to join welch spectra to ERA5. All other instrument auxiliary variables to be joined (eg depth, accelerometer) MUST be present in the same csv file where lat and lon are stored.


# %%
# Codes
# ------------------------

# sphinx_gallery_thumbnail_path = '_static/thumbnail_joiner_auxiliary.png'

#####################################################
# Define dataset path and name

from pathlib import Path
from OSmOSE.Auxiliary import Auxiliary


path_osmose_dataset = "/home6/cazaudo/Bureau/osmose_sample_datasets/"
dataset_name = "SES"
campaign_name = ""

date_template = "%Y%m%d_%H%M%S"


#####################################################
# Select your set of welch spectra through their time resolution and sampling rate

time_resolution_welch = 60
sample_rate_welch = 38400

#####################################################
# Instanciate the class :class:`OSmOSE.Auxiliary.Auxiliary`

joiner = Auxiliary(
    path_osmose_dataset, dataset_name, time_resolution_welch, sample_rate_welch
)

#####################################################
# Anytime you can print the joiner instance to visualize the dataframe being joined
print(joiner)


#####################################################
# The method :meth:`OSmOSE.Auxiliary.Auxiliary.join_welch` will perform a first spatio-temporal join
# between welch spectra and instrument auxiliary data

joiner.join_welch()

#####################################################
# Use the method :meth:`OSmOSE.Auxiliary.Auxiliary.join_other_csv_to_df` to perform any subsequent spatio-temporal joins
# with variables contained in a csv file given in input

# joiner.join_other_csv_to_df('environment/insitu_buoy.csv')

#####################################################
# Use the method :meth:`OSmOSE.Auxiliary.Auxiliary.join_era` to perform spatio-temporal join with ERA5 data

joiner.join_era()

#####################################################
# Use the method :meth:`OSmOSE.Auxiliary.Auxiliary.save_aux_data` to save your joined data into a csv file

joiner.save_aux_data()
