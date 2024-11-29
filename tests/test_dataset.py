import os

import pytest

from OSmOSE import Dataset


@pytest.mark.integ
def test_build(input_dataset):
    dataset = Dataset(
        input_dataset["main_dir"],
        gps_coordinates=(1, 1),
        depth=10,
        timezone="+03:00",
    )

    dataset.build(date_template="%Y%m%d_%H%M%S", original_folder="data/audio/original")

    new_expected_path = dataset.path.joinpath("data", "audio", "3_44100")

    assert not input_dataset["orig_audio_dir"].exists()
    assert new_expected_path.exists()
    assert sorted(os.listdir(new_expected_path)) == sorted(
        [
            "file_metadata.csv",
            "metadata.csv",
            "timestamp.csv",
        ]
        + [f"20220101_1200{str(3*i).zfill(2)}.wav" for i in range(5)]
        + [f"20220101_1200{str(3*i).zfill(2)}.flac" for i in range(5, 10)],
    )
