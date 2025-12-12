from pathlib import Path

import numpy as np
import pytest
from pandas import Timestamp
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming

from osekit.core_api.audio_data import AudioData
from osekit.core_api.audio_dataset import AudioDataset
from osekit.core_api.instrument import Instrument
from osekit.core_api.spectro_data import SpectroData
from osekit.core_api.spectro_dataset import SpectroDataset


@pytest.mark.parametrize(
    ("audio_files", "sft", "instrument"),
    [
        pytest.param(
            {
                "duration": 2,
                "sample_rate": 48_000,
                "series_type": "noise",
            },
            ShortTimeFFT(
                win=hamming(2048),
                fs=48_000,
                hop=1024,
            ),
            None,
            id="no_instrument",
        ),
        pytest.param(
            {
                "duration": 2,
                "sample_rate": 48_000,
                "series_type": "noise",
            },
            ShortTimeFFT(
                win=hamming(2048),
                fs=48_000,
                hop=1024,
            ),
            Instrument(
                sensitivity=10**6 * 10 ** (-170 / 20),
                gain_db=0,
                peak_voltage=1,
            ),
            id="with_instrument",
        ),
    ],
    indirect=["audio_files"],
)
def test_welch_level(
    audio_files: pytest.fixture,
    sft: ShortTimeFFT,
    instrument: Instrument | None,
) -> None:
    afs, _ = audio_files
    ad = AudioData.from_files(files=afs, instrument=instrument)
    sd = SpectroData.from_audio_data(data=ad, fft=sft)

    ref = 1 if instrument is None else instrument.P_REF

    welch = sd.get_welch()
    welch_db = 10 * np.log10(welch / (ref**2))
    assert welch.shape == sft.f.shape

    th_psd_level = -10 * np.log10(ad.sample_rate / 2)
    if instrument is not None:
        th_psd_level += instrument.end_to_end_db

    db_threshold = 0.1  # 1 dB accuracy from the theoretical level
    assert abs(np.median(welch_db) - th_psd_level) < db_threshold


@pytest.mark.parametrize(
    ("audio_files", "sft", "instrument"),
    [
        pytest.param(
            {
                "duration": 10,
                "sample_rate": 48_000,
                "series_type": "noise",
                "nb_files": 10,
            },
            ShortTimeFFT(
                win=hamming(2048),
                fs=48_000,
                hop=1024,
            ),
            None,
            id="no_instrument",
        ),
        pytest.param(
            {
                "duration": 10,
                "sample_rate": 48_000,
                "series_type": "noise",
                "nb_files": 10,
            },
            ShortTimeFFT(
                win=hamming(2048),
                fs=48_000,
                hop=1024,
            ),
            Instrument(end_to_end_db=150.0),
            id="instrument",
        ),
    ],
    indirect=["audio_files"],
)
def test_welch_spectrodataset(
    audio_files: pytest.fixture,
    sft: ShortTimeFFT,
    instrument: Instrument | None,
) -> None:
    afs, _ = audio_files
    ads = AudioDataset.from_files(files=afs, instrument=instrument, mode="files")
    sds = SpectroDataset.from_audio_dataset(ads, fft=sft)

    folder = afs[0].path.parent / "output"

    sds.write_welch(folder=folder)

    output_files = list(folder.rglob("*.npz"))
    assert len(output_files) == 1

    npz_file = np.load(output_files[0])

    freq = npz_file["freq"]
    pxs = npz_file["pxs"]
    timestamps = npz_file["timestamps"]

    assert freq.shape == sft.f.shape
    assert len(timestamps) == len(afs)

    for i, t in enumerate(timestamps):
        t_timestamp = Timestamp(t.split("_")[0])
        sd = sds.data[i]
        assert sd.begin == t_timestamp
        assert np.array_equal(pxs[i], sd.get_welch())


def test_welch_provided_pxs(
    monkeypatch: pytest.MonkeyPatch,
    audio_files: pytest.fixture,
) -> None:
    afs, _ = audio_files
    ads = AudioDataset.from_files(files=afs, mode="files")
    sds = SpectroDataset.from_audio_dataset(
        ads,
        fft=ShortTimeFFT(hamming(512), 512, ads.sample_rate),
    )

    get_welch_method = SpectroDataset._get_welch
    pxs_computation_count = [0]

    def patch_get_welch(*args: list, **kwargs: dict) -> tuple[SpectroData, np.ndarray]:
        pxs_computation_count[0] += 1
        return get_welch_method(*args, **kwargs)

    monkeypatch.setattr(SpectroDataset, "_get_welch", patch_get_welch)

    pxs = sds.get_welch()

    assert pxs_computation_count[0] == 1

    def patch_none(*args: list, **kwargs: dict) -> None:
        return

    savez = {}

    def patch_savez(*args: list, **kwargs: dict) -> None:
        savez.update(kwargs)

    monkeypatch.setattr(np, "savez", patch_savez)
    monkeypatch.setattr(Path, "mkdir", patch_none)

    sds.write_welch(folder=afs[0].path, pxs=pxs)

    assert pxs_computation_count[0] == 1

    assert savez["timestamps"] == list(pxs.columns)
    assert np.array_equal(savez["pxs"], pxs.to_numpy().T)
