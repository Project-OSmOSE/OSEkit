"""
EPD (Empirical Plot Distribution)
============================================================

This code will show you how to compute EPD from audio file-level welch spectra
"""


# %%
# Prerequisites
# ===============
# You first need to compute audio file-level spectrograms before computing LTAS ; see the use case :ref:`sphx_glr_gallery_basic_use_cases_Spectrogram_2_tune_parameters.py` to do this.

# %%
# Codes
# ===============


from pathlib import Path
from OSmOSE import Spectrogram
import glob

path_osmose_dataset = "/home6/cazaudo/Bureau/osmose_sample_datasets/"
dataset_name = "SPM"
campaign_name = ""

spectrogram = Spectrogram(
    dataset_path=Path(path_osmose_dataset, campaign_name, dataset_name)
)

# %%
# Parameters of LTAS
# ----------------------------
# Generate sequential LTAS : Sequential means that your welch spectra are processed over successive time period of fixed duration defined by the variable `time_scale` in the cell below (eg, this period can be set to one week, such that one soundscape figure will be generated for each successive week). `time_scale` can be set to the following values:
#
# - H for hours
# - D for days
# - M for months
# - Y for years
# - set `time_scale='all'` to generate a LTAS over your entire dataset.

# %%
# `time_resolution` and `sample_rate` allow us to identify your welch folder which sould be located in `processed/welch/` with a folder name following `{time_resolution}_{sample_rate}`.
# `Freq_min` (in Hz, default value 0)  and `Freq_max` (in Hz, default value fs/2) are respectively minimum and maximum frequencies to pass-band filter welch spectra (only available for SPL)

dataset_sr = 4000
time_res = 60

spectrogram.build_EPD(time_res, dataset_sr, show_fig=True)
