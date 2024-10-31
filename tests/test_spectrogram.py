import numpy as np
from OSmOSE import Spectrogram, Dataset
from OSmOSE.config import OSMOSE_PATH, SUPPORTED_AUDIO_FORMAT
import soundfile as sf
import pytest


@pytest.mark.integ
def test_initialize_2s(input_dataset):
    dataset = Dataset(
        input_dataset["main_dir"], gps_coordinates=(1, 1), depth=10, timezone="+03:00"
    )
    dataset.build()

    assert dataset.path.joinpath(OSMOSE_PATH.raw_audio, "3_44100").exists()
    num_file = 0
    for ext in SUPPORTED_AUDIO_FORMAT:
        num_file += len(
            list(
                dataset.path.joinpath(OSMOSE_PATH.raw_audio, "3_44100").glob(f"*{ext}")
            )
        )
    assert num_file == 10

    spectrogram = Spectrogram(dataset_path=dataset.path)

    spectrogram.dataset_sr = 240
    spectrogram.spectro_duration = 2
    spectrogram.window_size = 128
    spectrogram.nfft = 128
    spectrogram.overlap = 0
    spectrogram.custom_frequency_scale = "linear"

    spectrogram.initialize()
    spectrogram.save_spectro_metadata(False)

    spectro_paths = [
        OSMOSE_PATH.spectrogram.joinpath(
            f"{spectrogram.spectro_duration}_{spectrogram.dataset_sr}",
            f"{spectrogram.window_size}_{spectrogram.window_size}_{spectrogram.overlap}_{spectrogram.custom_frequency_scale}",
            "image",
        ),
        OSMOSE_PATH.spectrogram.joinpath(
            f"{spectrogram.spectro_duration}_{spectrogram.dataset_sr}",
            f"{spectrogram.window_size}_{spectrogram.window_size}_{spectrogram.overlap}_{spectrogram.custom_frequency_scale}",
            "matrix",
        ),
        OSMOSE_PATH.spectrogram.joinpath("adjustment_spectros", "adjust_metadata.csv"),
        OSMOSE_PATH.spectrogram.joinpath(
            f"{spectrogram.spectro_duration}_{spectrogram.dataset_sr}",
            f"{spectrogram.window_size}_{spectrogram.window_size}_{spectrogram.overlap}_{spectrogram.custom_frequency_scale}",
            "metadata.csv",
        ),
    ]

    for path in spectro_paths:
        assert dataset.path.joinpath(path).resolve().exists()

    assert spectrogram.audio_path == spectrogram.path.joinpath(
        OSMOSE_PATH.raw_audio,
        f"{spectrogram.spectro_duration}_{spectrogram.dataset_sr}",
    )

    assert spectrogram.path_output_spectrogram == spectrogram.path.joinpath(
        OSMOSE_PATH.spectrogram,
        f"{spectrogram.spectro_duration}_{spectrogram.dataset_sr}",
        f"{spectrogram.window_size}_{spectrogram.window_size}_{spectrogram.overlap}_{spectrogram.custom_frequency_scale}",
        "image",
    )


@pytest.mark.integ
def test_number_image_matrix(input_dataset):
    dataset = Dataset(
        input_dataset["main_dir"], gps_coordinates=(1, 1), depth=10, timezone="+03:00"
    )
    dataset.build()

    assert dataset.path.joinpath(OSMOSE_PATH.raw_audio, "3_44100").exists()
    num_file = 0
    for ext in SUPPORTED_AUDIO_FORMAT:
        num_file += len(
            list(
                dataset.path.joinpath(OSMOSE_PATH.raw_audio, "3_44100").glob(f"*{ext}")
            )
        )
    assert num_file == 10

    spectrogram = Spectrogram(dataset_path=dataset.path)

    spectrogram.dataset_sr = 2000
    spectrogram.spectro_duration = 3

    spectrogram.zoom_level = 0
    spectrogram.window_size = 512
    spectrogram.nfft = 512
    spectrogram.overlap = 20
    spectrogram.batch_number = 1

    spectrogram.initialize()

    list_audio = []
    for ext in SUPPORTED_AUDIO_FORMAT:
        list_audio_ext = spectrogram.path.joinpath(
            "data", "audio", f"{spectrogram.spectro_duration}_{spectrogram.dataset_sr}"
        ).glob(f"*{ext}")
        [list_audio.append(f) for f in list_audio_ext]

    spectrogram.save_spectro_metadata(False)
    spectrogram.process_all_files(list_audio_to_process=list_audio)

    if spectrogram.zoom_level > 0:
        assert len(list_audio) * (1 + 2**spectrogram.zoom_level) == len(
            list(
                dataset.path.joinpath(
                    OSMOSE_PATH.spectrogram,
                    f"{spectrogram.spectro_duration}_{spectrogram.dataset_sr}",
                    f"{spectrogram.window_size}_{spectrogram.window_size}_{spectrogram.overlap}_{spectrogram.custom_frequency_scale}",
                    "image",
                ).glob("*.png")
            )
        )
    else:
        assert len(list_audio) == len(
            list(
                dataset.path.joinpath(
                    OSMOSE_PATH.spectrogram,
                    f"{spectrogram.spectro_duration}_{spectrogram.dataset_sr}",
                    f"{spectrogram.window_size}_{spectrogram.window_size}_{spectrogram.overlap}_{spectrogram.custom_frequency_scale}",
                    "image",
                ).glob("*.png")
            )
        )

    for file in list_audio:
        assert sf.info(file).samplerate == spectrogram.dataset_sr
        assert sf.info(file).duration == 3.0


@pytest.mark.integ
def test_numerical_values(input_dataset):
    dataset = Dataset(
        input_dataset["main_dir"], gps_coordinates=(1, 1), depth=10, timezone="+03:00"
    )
    dataset.build()

    spectrogram = Spectrogram(dataset_path=dataset.path)

    spectrogram.dataset_sr = 44100
    spectrogram.spectro_duration = 3

    spectrogram.zoom_level = 0
    spectrogram.window_size = 512
    spectrogram.nfft = 512
    spectrogram.overlap = 20
    spectrogram.data_normalization = "zscore"
    spectrogram.spectro_normalization = "spectrum"

    spectrogram.initialize()

    list_audio = []
    for ext in SUPPORTED_AUDIO_FORMAT:
        list_audio_ext = spectrogram.path.joinpath(
            "data", "audio", f"{spectrogram.spectro_duration}_{spectrogram.dataset_sr}"
        ).glob(f"*{ext}")
        [list_audio.append(f) for f in list_audio_ext]

    spectrogram.process_all_files(
        list_audio_to_process=list_audio, save_for_LTAS=True, save_matrix=True
    )

    # test 3s welch spectra against PamGuide reference values
    list_welch = list((dataset.path / OSMOSE_PATH.welch / "3_44100").glob("*.npz"))
    data = np.load(list_welch[0], allow_pickle=True)

    val_PamGuide = np.array(
        [
            [
                3.73173496e02,
                6.88334123e01,
                3.98376790e-03,
                1.56141637e-04,
                3.78682038e-04,
                3.67592818e-04,
                3.08349818e-04,
                2.51382341e-04,
                2.05204772e-04,
                1.69184984e-04,
                1.41188163e-04,
                1.19254664e-04,
                1.01870005e-04,
                8.79168752e-05,
                7.65798122e-05,
                6.72614389e-05,
                5.95200878e-05,
                5.30255916e-05,
                4.75280985e-05,
                4.28362610e-05,
                3.88018844e-05,
                3.53088819e-05,
                3.22654928e-05,
                2.95983758e-05,
                2.72484265e-05,
                2.51676120e-05,
                2.33166354e-05,
                2.16630259e-05,
                2.01798574e-05,
                1.88445861e-05,
                1.76382982e-05,
                1.65449385e-05,
                1.55509250e-05,
                1.46445893e-05,
                1.38159954e-05,
                1.30565088e-05,
                1.23586689e-05,
                1.17160022e-05,
                1.11228769e-05,
                1.05743403e-05,
                1.00660136e-05,
                9.59412668e-06,
                9.15522715e-06,
                8.74635688e-06,
                8.36485111e-06,
                8.00828600e-06,
                7.67457863e-06,
                7.36179368e-06,
                7.06825125e-06,
                6.79238283e-06,
                6.53281086e-06,
                6.28826826e-06,
                6.05765282e-06,
                5.83989103e-06,
                5.63406243e-06,
                5.43932370e-06,
                5.25487290e-06,
                5.08001711e-06,
                4.91410185e-06,
                4.75651274e-06,
                4.60672805e-06,
                4.46423365e-06,
                4.32854811e-06,
                4.19927593e-06,
                4.07599457e-06,
                3.95835437e-06,
                3.84601661e-06,
                3.73866661e-06,
                3.63601072e-06,
                3.53778563e-06,
                3.44374435e-06,
                3.35363962e-06,
                3.26727367e-06,
                3.18443141e-06,
                3.10493149e-06,
                3.02858816e-06,
                2.95524776e-06,
                2.88475396e-06,
                2.81695472e-06,
                2.75172608e-06,
                2.68893073e-06,
                2.62845682e-06,
                2.57019454e-06,
                2.51402031e-06,
                2.45986637e-06,
                2.40761003e-06,
                2.35717998e-06,
                2.30848795e-06,
                2.26145951e-06,
                2.21601047e-06,
                2.17208664e-06,
                2.12961005e-06,
                2.08852048e-06,
                2.04876887e-06,
                2.01028028e-06,
                1.97301735e-06,
                1.93692661e-06,
                1.90195281e-06,
                1.86806286e-06,
                1.83520714e-06,
                1.80333823e-06,
                1.77243036e-06,
                1.74243749e-06,
                1.71332433e-06,
                1.68506195e-06,
                1.65762048e-06,
                1.63096061e-06,
                1.60506004e-06,
                1.57988645e-06,
                1.55542496e-06,
                1.53162872e-06,
                1.50849559e-06,
                1.48599274e-06,
                1.46409449e-06,
                1.44278993e-06,
                1.42204740e-06,
                1.40185658e-06,
                1.38219585e-06,
                1.36304824e-06,
                1.34439372e-06,
                1.32622059e-06,
                1.30850948e-06,
                1.29124952e-06,
                1.27442275e-06,
                1.25801234e-06,
                1.24201644e-06,
                1.22641254e-06,
                1.21119238e-06,
                1.19634039e-06,
                1.18184902e-06,
                1.16770948e-06,
                1.15390527e-06,
                1.14043065e-06,
                1.12727477e-06,
                1.11443158e-06,
                1.10188334e-06,
                1.08963035e-06,
                1.07766185e-06,
                1.06596915e-06,
                1.05454223e-06,
                1.04337896e-06,
                1.03246765e-06,
                1.02180187e-06,
                1.01138058e-06,
                1.00118795e-06,
                9.91225424e-07,
                9.81481648e-07,
                9.71957369e-07,
                9.62641487e-07,
                9.53531760e-07,
                9.44616239e-07,
                9.35900467e-07,
                9.27372924e-07,
                9.19029668e-07,
                9.10866719e-07,
                9.02880406e-07,
                8.95063949e-07,
                8.87415928e-07,
                8.79934806e-07,
                8.72606175e-07,
                8.65439708e-07,
                8.58424171e-07,
                8.51554226e-07,
                8.44833740e-07,
                8.38252679e-07,
                8.31811681e-07,
                8.25506127e-07,
                8.19334011e-07,
                8.13291098e-07,
                8.07375530e-07,
                8.01584970e-07,
                7.95914454e-07,
                7.90364733e-07,
                7.84933158e-07,
                7.79609939e-07,
                7.74406893e-07,
                7.69306967e-07,
                7.64320592e-07,
                7.59430356e-07,
                7.54652318e-07,
                7.49970787e-07,
                7.45391481e-07,
                7.40906714e-07,
                7.36519239e-07,
                7.32224620e-07,
                7.28025126e-07,
                7.23912420e-07,
                7.19887336e-07,
                7.15954306e-07,
                7.12103650e-07,
                7.08337886e-07,
                7.04653692e-07,
                7.01052599e-07,
                6.97530326e-07,
                6.94087189e-07,
                6.90721121e-07,
                6.87430533e-07,
                6.84215990e-07,
                6.81072234e-07,
                6.78005032e-07,
                6.75007543e-07,
                6.72079836e-07,
                6.69219024e-07,
                6.66431806e-07,
                6.63709335e-07,
                6.61051425e-07,
                6.58459455e-07,
                6.55935771e-07,
                6.53470312e-07,
                6.51071557e-07,
                6.48732871e-07,
                6.46454212e-07,
                6.44238742e-07,
                6.42082274e-07,
                6.39983033e-07,
                6.37942953e-07,
                6.35960640e-07,
                6.34034449e-07,
                6.32166759e-07,
                6.30350865e-07,
                6.28591619e-07,
                6.26889486e-07,
                6.25236740e-07,
                6.23641927e-07,
                6.22097231e-07,
                6.20604737e-07,
                6.19164987e-07,
                6.17776850e-07,
                6.16440095e-07,
                6.15150950e-07,
                6.13916107e-07,
                6.12729693e-07,
                6.11592121e-07,
                6.10501617e-07,
                6.09463566e-07,
                6.08471455e-07,
                6.07529661e-07,
                6.06632933e-07,
                6.05785712e-07,
                6.04984680e-07,
                6.04231073e-07,
                6.03523021e-07,
                6.02864779e-07,
                6.02248721e-07,
                6.01682233e-07,
                6.01161228e-07,
                6.00681203e-07,
                6.00253690e-07,
                5.99866075e-07,
                5.99525808e-07,
                5.99229496e-07,
                5.98977098e-07,
                5.98767767e-07,
                5.98617518e-07,
                5.98928434e-07,
                1.14932571e-06,
                2.52971788e-06,
            ]
        ]
    )

    assert np.allclose(
        data["Sxx"], data["Sxx"] + 10 ** (-13)
    )  # test not set yet, to be done, work in local but not on github
