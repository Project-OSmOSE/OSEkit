import os
from pathlib import Path

import pytest
from OSmOSE import Spectrogram
from OSmOSE.config import OSMOSE_PATH

PARAMS = {
    "nfft": 512,
    "winsize": 512,
    "overlap": 97,
    "spectro_colormap": "viridis",
    "zoom_levels": 2,
    "number_adjustment_spectrograms": 2,
    "dynamic_min": 0,
    "dynamic_max": 150,
    "spectro_duration": 5,
    "data_normalization": "instrument",
    "HPfilter_min_freq": 0,
    "sensitivity_dB": -164,
    "peak_voltage": 2.5,
    "spectro_normalization": "density",
    "gain_dB": 14.7,
    "zscore_duration": "original",
}


def test_build_path(input_dataset):
    dataset = Spectrogram(
        dataset_path=input_dataset["main_dir"], sr_analysis=240, analysis_params=PARAMS
    )
    dataset.build()
    dataset._Spectrogram__build_path(adjust=True)

    assert dataset.audio_path == dataset.path.joinpath(OSMOSE_PATH.raw_audio, "5_240")
    assert dataset._Spectrogram__spectro_foldername == "adjustment_spectros"
    assert dataset._Spectrogram__path_summstats == dataset.path.joinpath(
        OSMOSE_PATH.spectrogram, "5_240", "normalization_parameters"
    )
    assert dataset._Spectrogram__path_output_spectrogram == dataset.path.joinpath(
        OSMOSE_PATH.spectrogram, "5_240", "adjustment_spectros", "image"
    )
    assert (
        dataset._Spectrogram__path_output_spectrogram_matrix
        == dataset.path.joinpath(
            OSMOSE_PATH.spectrogram, "5_240", "adjustment_spectros", "matrix"
        )
    )

    dataset._Spectrogram__build_path(adjust=False)
    assert dataset._Spectrogram__path_output_spectrogram == dataset.path.joinpath(
        OSMOSE_PATH.spectrogram, "5_240", "512_512_97", "image"
    )
    assert (
        dataset._Spectrogram__path_output_spectrogram_matrix
        == dataset.path.joinpath(
            OSMOSE_PATH.spectrogram, "5_240", "512_512_97", "matrix"
        )
    )
