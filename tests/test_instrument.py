from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming

from osekit.config import TIMESTAMP_FORMATS_EXPORTED_FILES
from osekit.core_api.audio_data import AudioData
from osekit.core_api.audio_file import AudioFile
from osekit.core_api.instrument import Instrument
from osekit.core_api.spectro_data import SpectroData
from osekit.core_api.spectro_file import SpectroFile


@pytest.mark.parametrize(
    ("instrument", "expected_end_to_end"),
    [
        pytest.param(
            Instrument(),
            1.0,
            id="default_to_1",
        ),
        pytest.param(
            Instrument(sensitivity=2),
            0.5,
            id="2V_per_Pa_with_1V_Vmax",
        ),
        pytest.param(
            Instrument(gain_db=6.020599913279624),
            0.5,
            id="6dB_gain_doubles_voltage",
        ),
        pytest.param(
            Instrument(peak_voltage=2),
            2,
            id="2V_Vmax",
        ),
        pytest.param(
            Instrument(
                sensitivity=0.1,
                gain_db=20,
                peak_voltage=0.01,
            ),
            0.01,
            id="full_chain",
        ),
        pytest.param(
            Instrument(
                end_to_end_db=0.0,
            ),
            Instrument.P_REF,
            id="end_to_end_of_0dB_results_in_pref",
        ),
        pytest.param(
            Instrument(
                end_to_end_db=120.0,
            ),
            1e6 * Instrument.P_REF,
            id="end_to_end_of_120dBSPL_makes_1_Pa",
        ),
        pytest.param(
            Instrument(
                sensitivity=0.1,
                gain_db=20,
                peak_voltage=0.01,
                end_to_end_db=0.0,
            ),
            Instrument.P_REF,
            id="end_to_end_overwrites_the_rest",
        ),
    ],
)
def test_end_to_end(
    instrument: Instrument,
    expected_end_to_end: float,
) -> None:
    assert instrument.end_to_end == expected_end_to_end


@pytest.mark.parametrize(
    ("instrument", "n", "expected_p"),
    [
        pytest.param(
            Instrument(
                sensitivity=0.1,
                gain_db=20,
                peak_voltage=0.01,
            ),
            1.0,
            0.01,
            id="full_scale",
        ),
        pytest.param(
            Instrument(
                sensitivity=0.1,
                gain_db=20,
                peak_voltage=0.01,
            ),
            -1.0,
            -0.01,
            id="negative_full_scale",
        ),
        pytest.param(
            Instrument(
                sensitivity=0.1,
                gain_db=20,
                peak_voltage=0.01,
            ),
            0.0,
            0.0,
            id="zero",
        ),
        pytest.param(
            Instrument(
                sensitivity=0.1,
                gain_db=20,
                peak_voltage=0.01,
            ),
            0.3,
            0.003,
            id="within_scale",
        ),
        pytest.param(
            Instrument(
                end_to_end_db=-40,
            ),
            -0.3,
            -0.003 * Instrument.P_REF,
            id="specified_end_to_end",
        ),
    ],
)
def test_n_to_p(
    instrument: Instrument,
    n: float,
    expected_p: float,
) -> None:
    assert instrument.n_to_p(n) == expected_p


@pytest.mark.parametrize(
    ("natural", "decibel"),
    [
        pytest.param(
            1.0,
            0.0,
            id="no_gain",
        ),
        pytest.param(
            10.0,
            20.0,
            id="positive_gain",
        ),
        pytest.param(
            0.1,
            -20.0,
            id="negative_gain",
        ),
    ],
)
def test_db_conversion(natural: float, decibel: float) -> None:
    i = Instrument()

    i.gain = natural
    assert i.gain_db == decibel

    i.gain_db = decibel
    assert i.gain_db == decibel

    i.end_to_end = natural
    assert i.end_to_end_db == decibel - 20 * np.log10(Instrument.P_REF)

    i.end_to_end_db = decibel
    assert i.end_to_end == Instrument.P_REF * natural


@pytest.mark.parametrize(
    ("audio_files", "instrument", "sft", "expected_level"),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "series_type": "sine",
                "sine_frequency": 1_000,
                "magnitude": 1.0,
            },
            None,
            ShortTimeFFT(hamming(128), 10, 48_000, scale_to="magnitude"),
            0.0,
            id="0_db_fs",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "series_type": "sine",
                "sine_frequency": 1_000,
                "magnitude": 0.1,
            },
            None,
            ShortTimeFFT(hamming(128), 10, 48_000, scale_to="magnitude"),
            -20.0,
            id="negative_db_fs",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "series_type": "sine",
                "sine_frequency": 1_000,
                "magnitude": 1.0,
            },
            Instrument(
                end_to_end_db=0.0,
            ),
            ShortTimeFFT(hamming(128), 10, 48_000, scale_to="magnitude"),
            0.0,
            id="full_scale_0_dB_SPL",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "series_type": "sine",
                "sine_frequency": 1_000,
                "magnitude": 1.0,
            },
            Instrument(
                end_to_end_db=150.0,
            ),
            ShortTimeFFT(hamming(128), 10, 48_000, scale_to="magnitude"),
            150.0,
            id="full_scale_150_dB_SPL",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "series_type": "sine",
                "sine_frequency": 1_000,
                "magnitude": 0.1,
            },
            Instrument(
                end_to_end_db=150.0,
            ),
            ShortTimeFFT(hamming(128), 10, 48_000, scale_to="magnitude"),
            130.0,
            id="negative_dBFS_to_130_dB_SPL",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "series_type": "sine",
                "sine_frequency": 15_000,
                "magnitude": 0.1,
            },
            Instrument(
                end_to_end_db=150.0,
            ),
            ShortTimeFFT(hamming(128), 10, 48_000, scale_to="magnitude"),
            130.0,
            id="higher_signal_frequency",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "series_type": "sine",
                "sine_frequency": 15_000,
                "magnitude": 0.1,
            },
            Instrument(
                sensitivity=3.5e-6,
                gain_db=50.0,
                peak_voltage=0.1,
            ),
            ShortTimeFFT(hamming(128), 10, 48_000, scale_to="magnitude"),
            139.0,
            id="full_instrument_chain",
        ),
    ],
    indirect=["audio_files"],
)
def test_instrument_level_spectrum(
    tmp_path: pytest.fixture,
    audio_files: pytest.fixture,
    instrument: Instrument | None,
    sft: ShortTimeFFT,
    expected_level: float,
) -> None:
    afs, request = audio_files
    ad = AudioData.from_files(
        afs,
        instrument=instrument,
    )
    sd = SpectroData.from_audio_data(ad, sft)

    sine_frequency = request.param["sine_frequency"]

    # Get the bin index which center frequency is the closest to the signal frequency:
    bin_idx = min(enumerate(sft.f), key=lambda t: abs(t[1] - sine_frequency))[0]

    # Level in db FS if no instrument, dB SPL otherwise
    # We'll not land on exactly the expected level because energy
    # scatters around sine_frequency
    level_tolerance = 8

    equalized_sx = sd.to_db(sd.get_value())
    computed_level = equalized_sx[bin_idx, :].mean()

    # For the full chain, the expected level is:
    # L = 20*log10((M*peak_voltage)/(P_REF*S*10**(G/20))) with M being the signal
    # peak value, in raw wav data ([-1.;1])

    assert abs(computed_level - expected_level) < level_tolerance

    # Instrument calibration should be maintained when exporting/importing the sd

    # To a npz file:
    sd.write(tmp_path / "npz")
    sd_npz = SpectroData.from_files(
        [
            SpectroFile(
                next((tmp_path / "npz").glob("*.npz")),
                strptime_format=TIMESTAMP_FORMATS_EXPORTED_FILES,
            ),
        ],
    )

    equalized_sx_npz = sd_npz.to_db(sd_npz.get_value())
    computed_level_npz = equalized_sx_npz[bin_idx, :].mean()

    assert abs(computed_level_npz - expected_level) < level_tolerance

    # With a linked exported audio file:
    ad.sample_rate *= 2
    sft.fs = ad.sample_rate
    bin_idx = min(enumerate(sft.f), key=lambda t: abs(t[1] - sine_frequency))[0]

    ad.write(tmp_path / "audio")

    ad2 = AudioData.from_files(
        [
            AudioFile(
                next((tmp_path / "audio").glob("*.wav")),
                strptime_format=TIMESTAMP_FORMATS_EXPORTED_FILES,
            ),
        ],
        instrument=instrument,
    )

    sd2 = SpectroData.from_audio_data(ad2, sft)
    equalized_sx2 = sd2.to_db(sd2.get_value())
    computed_level_sx2 = equalized_sx2[bin_idx, :].mean()

    assert abs(computed_level_sx2 - expected_level) < level_tolerance
