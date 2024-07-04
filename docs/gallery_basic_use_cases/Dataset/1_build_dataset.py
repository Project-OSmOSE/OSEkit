"""
First time building a OSmOSE dataset
=====================================

This code will show you how to format your raw audio data into a OSmOSE dataset, in the case of a fixed hydrophone.
"""

# %%
# Preambule
# ------------------------
# In our dataset, only three metadata are mandatory for the moment: the timestamp of each audio file, and the gps location and depth of the hydrophone. In this tutorial we will how they can be set in the case of a fixed hydrophone ; for a mobile hydrophone you should pursue with the tutorial :ref:`sphx_glr_gallery_basic_use_cases_Dataset_2_mobile_hydrophone.py`.


# %%
# How should I prepare my raw data ?
# -------------------------------------
# Before you can build your dataset:
#
# - choose a dataset name (should not contain any special character, including '-'⁾ ;
# - create the folder ``{local_working_dir}/dataset/{dataset_name}``, or ``{local_working_dir}/dataset/{campaign_name}/{dataset_name}`` in case your dataset is part of a recording campaign;
# - place in this folder your audio data, they can be individual files or contain within multiple sub-folders ;

# %%
# How my timestamps are set ?
# --------------------------------------
# The two following solutions are possible depending on whether timestamps are contained in the audio filenames:
#
# - if this is the case, you just have to pass us the "timestamp signature" through the variable ``date_template`` (eg "%Y%m%d_%H%M%S")
# - if not, you have to create the timestamp.csv file yourself following this `template <example_timestamp.csv>`__ ; in this file your timestamps can follow any signature as long as it is provided in the ``date_template`` variable. See :ref:`sphx_glr_gallery_basic_use_cases_Dataset_2_mobile_hydrophone.py` for a code example on another dataset.


# %%
# Codes
# ------------------------

# sphinx_gallery_thumbnail_path = '_static/dataset_metadata.png'


from pathlib import Path
from OSmOSE import Dataset

#####################################################
# You first have to set the `path_osmose_dataset`, which is where your dataset named `dataset_name` should be ; unless it is part of a recording campaign named `campaign_name`, your dataset should then be placed in `{path_osmose_dataset}/{campaign_name}/{dataset_name}`.
path_osmose_dataset = "/home6/cazaudo/Bureau/osmose_sample_datasets/"
dataset_name = "SPM"
campaign_name = ""  # default value ; so no need to define it if your dataset is not part of a campaign

#####################################################
# In our dataset, we have made mandatory the setting of two metadata variables, namely `gps_coordinates` (tuple of (latitude , longitude) coordinates in decimal degree) and `depth` (positive integer in meter) of the hydrophone.
gps_coordinates = (46.89, -56.54)
depth = 20

#####################################################
# Before building your dataset, let's review two optional parameters. If the timezone of your data happens to be different from the different value UTC+00:00, use the input argument `timezone` of :class:`OSmOSE.Dataset.Dataset` to make your timestamps timezone-aware, following the str format `"+02:00"` for UTC+02:00 for example.
timezone = "-03:00"

#####################################################
# The variable `date_template` should be used to help us extracting the timestamp from your audio filenames. The default template is "%Y%m%d_%H%M%S", if you have a different one set its value in `date_template` with the same strftime format.
date_template = "%Y_%m_%dT%H:%M:%S"

#####################################################
# Run the method :meth:`OSmOSE.Dataset.Dataset.build` of the class :class:`OSmOSE.Dataset.Dataset`, and that's it your dataset is now OSmOSE compatible !
dataset = Dataset(
    dataset_path=Path(path_osmose_dataset, campaign_name, dataset_name),
    gps_coordinates=gps_coordinates,
    depth=depth,
    timezone=timezone,
)
dataset.build(date_template=date_template)
