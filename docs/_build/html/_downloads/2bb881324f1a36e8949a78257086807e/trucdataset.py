"""
Format a dataset
==================

This code will format raw data into a OSmOSE dataset
"""



# %%
# Package importation
# ------------------------
from pathlib import Path
from OSmOSE import Dataset

# %%
# Define paths and names
# ------------------------

path_osmose_dataset = "/home6/cazaudo/Bureau/osmose_sample_datasets/"
dataset_name = "ohasisbio"
campaign_name = ""

# %%
# Define analysis parameters
# ----------------------------


save_matrix = False # Set to True if you want to generate the numpy matrices
save_image = True # Set to False if you don't want to generate the spectrogram images.

date_template = "%Y_%m_%dT%H_%M_%S" # strftime format, used to build the dataset from scratch (ignore if the dataset is already built)
depth = 10
gps_coordinates = (10,10)


# %%
# Run the class Dataset
# ------------------------

dataset = Dataset(dataset_path = Path(path_osmose_dataset, campaign_name, dataset_name), gps_coordinates = gps_coordinates, depth = depth, timezone='+00:00')
dataset.build(date_template = date_template , force_upload=False, number_test_bad_files=1)

print(dataset)


