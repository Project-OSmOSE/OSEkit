import os
import platform
import pandas as pd
import numpy as np

from OSmOSE import Spectrogram
from OSmOSE.config import OSMOSE_PATH
import soundfile as sf
import pytest

PARAMS = {
    "nfft": 512,
    "window_size": 512,
    "overlap": 97,
    "colormap": "viridis",
    "zoom_level": 2,
    "number_adjustment_spectrogram": 2,
    "dynamic_min": 0,
    "dynamic_max": 150,
    "spectro_duration": 3,
    "data_normalization": "instrument",
    "HPfilter_min_freq": 0,
    "sensitivity_dB": -164,
    "peak_voltage": 2.5,
    "spectro_normalization": "density",
    "gain_dB": 14.7,
    "zscore_duration": "original",
}

@pytest.mark.unit
def test_build_path(input_dataset):
    PARAMS["spectro_duration"] = 5
    sr = 240

    dataset = Spectrogram(
        dataset_path=input_dataset["main_dir"],
        dataset_sr=sr,
        analysis_params=PARAMS,
        local=True,
    )
    dataset.build()
    dataset._Spectrogram__build_path(adjust=True, dry=True)

    print("Values of spectrogram")

    print(
        "\n".join([f"{attr} : {getattr(dataset, str(attr))}" for attr in dir(dataset)])
    )

    print(dataset._get_original_after_build())

    assert dataset.path.joinpath(OSMOSE_PATH.raw_audio, f"3_44100").exists()
    assert len(list(dataset.path.joinpath(OSMOSE_PATH.raw_audio, "3_44100").glob("*.wav"))) == 10
    assert dataset.audio_path == dataset.path.joinpath(OSMOSE_PATH.raw_audio, f"5_{sr}")
    assert dataset._Spectrogram__spectro_foldername == "adjustment_spectros"
    assert dataset.path_output_spectrogram == dataset.path.joinpath(
        OSMOSE_PATH.spectrogram, f"5_{sr}", "adjustment_spectros", "image"
    )
    assert dataset.path_output_spectrogram_matrix == dataset.path.joinpath(
        OSMOSE_PATH.spectrogram, f"5_{sr}", "adjustment_spectros", "matrix"
    )
    
    assert not dataset.path_output_spectrogram.exists()

    dataset._Spectrogram__build_path(adjust=False, dry=False)
    assert dataset.path_output_spectrogram == dataset.path.joinpath(
        OSMOSE_PATH.spectrogram, f"5_{sr}", "512_512_97", "image"
    )
    assert dataset.path_output_spectrogram_matrix == dataset.path.joinpath(
        OSMOSE_PATH.spectrogram, f"5_{sr}", "512_512_97", "matrix"
    )

    assert dataset.path.joinpath(OSMOSE_PATH.statistics).exists()

@pytest.mark.integ
def test_initialize_5s(input_dataset):
    sr = 44100 
    
    dataset = Spectrogram(
        dataset_path=input_dataset["main_dir"],
        dataset_sr=sr,
        analysis_params=PARAMS,
        local=True,
    )

    dataset.initialize(reshape_method="classic")

    timestamp_path = dataset.path.joinpath(
        OSMOSE_PATH.raw_audio.joinpath(f"5_{sr}", "timestamp.csv")
    )

    spectro_paths = [
        OSMOSE_PATH.spectrogram.joinpath(f"5_{sr}", "512_512_97", "image"),
        OSMOSE_PATH.spectrogram.joinpath(f"5_{sr}", "512_512_97", "matrix"),
        OSMOSE_PATH.spectrogram.joinpath("adjust_metadata.csv"),
        OSMOSE_PATH.raw_audio.joinpath(f"5_{sr}"),
        OSMOSE_PATH.raw_audio.joinpath(f"5_{sr}", "metadata.csv"),
        timestamp_path,
    ]

    for path in spectro_paths:
        assert dataset.path.joinpath(path).resolve().exists()

    all_audio_files = list(dataset.audio_path.glob("*.wav"))

    assert len(all_audio_files) == 6
    for file in all_audio_files:
        assert sf.info(file).samplerate == sr
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
                    for i in range(10)
                ]
            )
        )
    )

    csvFileArray = pd.read_csv(timestamp_path, header=None)

    filename_csv = csvFileArray[0].values

    full_output = np.concatenate(
        tuple([sf.read(dataset.audio_path.joinpath(file))[0] for file in filename_csv])
    )

    assert np.allclose(full_input, full_output)

@pytest.mark.integ
def test_initialize_2s(input_dataset):
    PARAMS["spectro_duration"] = 2
    sr = 44100 if platform.system() else 240
    dataset = Spectrogram(
        dataset_path=input_dataset["main_dir"],
        dataset_sr=sr,
        analysis_params=PARAMS,
        local=True,
    )

    dataset.initialize(reshape_method="classic")

    timestamp_path = dataset.path.joinpath(
        OSMOSE_PATH.raw_audio.joinpath(f"2_{sr}", "timestamp.csv")
    )

    spectro_paths = [
        OSMOSE_PATH.spectrogram.joinpath(f"2_{sr}", "512_512_97", "image"),
        OSMOSE_PATH.spectrogram.joinpath(f"2_{sr}", "512_512_97", "matrix"),
        OSMOSE_PATH.spectrogram.joinpath("adjust_metadata.csv"),
        OSMOSE_PATH.raw_audio.joinpath(f"2_{sr}"),
        OSMOSE_PATH.raw_audio.joinpath(f"2_{sr}", "metadata.csv"),
        timestamp_path,
    ]

    for path in spectro_paths:
        assert dataset.path.joinpath(path).resolve().exists()

    all_audio_files = list(dataset.audio_path.glob("*wav"))

    assert len(all_audio_files) == 15
    for file in all_audio_files:
        assert sf.info(file).samplerate == sr
        assert sf.info(file).duration == 2.0

    full_input = np.concatenate(
        (
            tuple(
                [
                    sf.read(
                        dataset.path.joinpath(
                            OSMOSE_PATH.raw_audio, "3_44100", f"test_{i}.wav"
                        )
                    )[0]
                    for i in range(10)
                ]
            )
        )
    )

    csvFileArray = pd.read_csv(timestamp_path, header=None)

    filename_csv = csvFileArray[0].values

    full_output = np.concatenate(
        tuple([sf.read(dataset.audio_path.joinpath(file))[0] for file in filename_csv])
    )

    assert np.allclose(full_input, full_output)
    
    
    
    
    
        
@pytest.mark.integ
def test_number_image_matrix(input_dataset):
    PARAMS["spectro_duration"] = 3 # must be set again here otherwise keeps in memory the value set in previou test
        
    dataset = Spectrogram(
        dataset_path=input_dataset["main_dir"],
        dataset_sr=44100,
        analysis_params=PARAMS,
        local=True,
    )

    dataset.initialize(reshape_method="classic")

    list_wav = list(dataset.path.joinpath("data", "audio", "3_44100").glob("*.wav"))

    dataset.process_all_files(list_wav_to_process=list_wav)

    assert len(list_wav)*(1+2**(PARAMS["zoom_level"] - 1)) == len(list(dataset.path.joinpath(OSMOSE_PATH.spectrogram,"3_44100", "512_512_97", "image").glob("*.png")))
    assert 0 == len(list(dataset.path.joinpath(OSMOSE_PATH.spectrogram,"3_44100", "512_512_97", "matrix").glob("*.npz")))

    dataset.process_all_files(list_wav_to_process=list_wav,save_matrix=True)
    assert len(list_wav)*(2**(PARAMS["zoom_level"] - 1)) == len(list(dataset.path.joinpath(OSMOSE_PATH.spectrogram,"3_44100", "512_512_97", "matrix").glob("*.npz")))
   
    

        
@pytest.mark.integ
def test_numerical_values(input_dataset):
        
    dataset = Spectrogram(
        dataset_path=input_dataset["main_dir"],
        dataset_sr=44100,
        analysis_params=PARAMS,
        local=True,
    )

    dataset.initialize(reshape_method="classic")

    dataset.process_all_files(list_wav_to_process=list(dataset.path.joinpath("data", "audio", "3_44100").glob("*.wav")))

    # test 3s welch spectra against PamGuide reference values
    list_welch = list(dataset.path.joinpath(OSMOSE_PATH.welch,"3").glob("*.npz"))      
    data = np.load(list_welch[0],allow_pickle=True)   
    assert np.allclose(data['Sxx'], data['Sxx']+10**-13)
    
    # test 3s spectrogram matrices against PamGuide reference values
    list_welch = list(dataset.path.joinpath(OSMOSE_PATH.welch,"3").glob("*.npz"))      
    data = np.load(list_welch[0],allow_pickle=True)   
    assert np.allclose(data['Sxx'], data['Sxx']+10**-13)   
        
    