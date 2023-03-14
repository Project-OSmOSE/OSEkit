import os
from pathlib import Path
import platform
import pandas as pd
import numpy as np

from OSmOSE import Spectrogram
from OSmOSE.config import OSMOSE_PATH
import soundfile as sf

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
        dataset_path=input_dataset["main_dir"],
        sr_analysis=240,
        analysis_params=PARAMS,
        local=True,
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


def test_initialize(input_dataset):
    dataset = Spectrogram(
        dataset_path=input_dataset["main_dir"],
        sr_analysis=240,
        analysis_params=PARAMS,
        local=True,
    )

    dataset.initialize(reshape_method="reshape")

    timestamp_path = dataset.path.joinpath(
        OSMOSE_PATH.raw_audio.joinpath("5_240", "timestamp.csv")
    )

    spectro_paths = [
        OSMOSE_PATH.spectrogram.joinpath("5_240", "512_512_97", "image"),
        OSMOSE_PATH.spectrogram.joinpath("5_240", "512_512_97", "matrix"),
        OSMOSE_PATH.spectrogram.joinpath("5_240", "normalization_parameters"),
        OSMOSE_PATH.spectrogram.joinpath("5_240", "512_512_97", "metadata.csv"),
        OSMOSE_PATH.raw_audio.joinpath("5_240"),
        OSMOSE_PATH.raw_audio.joinpath("5_240", "metadata.csv"),
        timestamp_path,
    ]

    for path in spectro_paths:
        assert dataset.path.joinpath(path).resolve().exists()

    all_audio_files = list(dataset.audio_path.glob("*wav"))

    print(all_audio_files)

    assert len(all_audio_files) == 7
    for file in all_audio_files:
        if not platform.system() == "Windows":
            assert sf.info(file).samplerate == 240
        assert sf.info(file).duration == 5.0

    full_input = np.concatenate(
        (
            tuple(
                [
                    sf.read(
                        dataset.path.joinpath(
                            OSMOSE_PATH.raw_audio, "3_44100", f"test_{i}.wav"
                        )
                    )[0]
                    for i in range(9)
                ]
            )
        )
    )

    csvFileArray = pd.read_csv(timestamp_path, header=None)
    print(csvFileArray)
    filename_csv = csvFileArray[0].values

    full_output = np.concatenate(
        ([sf.read(dataset.audio_path.joinpath(file))[0] for file in filename_csv[:-1]])
    )

    assert np.allclose(full_input == full_output)
