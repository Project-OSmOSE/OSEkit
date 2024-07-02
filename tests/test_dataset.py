import os
from pathlib import Path

import pytest
from OSmOSE import Dataset


@pytest.mark.unit
def test_find_or_create_original_folder(input_dataset):
    dataset = Dataset(input_dataset["main_dir"])
    folder = dataset._find_or_create_original_folder()

    assert folder == input_dataset["orig_audio_dir"]

    input_dataset["orig_audio_dir"].rename(
        input_dataset["orig_audio_dir"].with_name("unconventional_name")
    )

    folder2 = dataset._find_or_create_original_folder()

    assert folder2 == input_dataset["orig_audio_dir"].with_name("original")


@pytest.mark.integ
def test_build(input_dataset):
    dataset = Dataset(
        input_dataset["main_dir"], gps_coordinates=(1, 1), depth=10, timezone="+03:00"
    )

    dataset.build()

    new_expected_path = dataset.path.joinpath("data", "audio", "3_44100")

    assert not input_dataset["orig_audio_dir"].exists()
    assert new_expected_path.exists()
    assert sorted(os.listdir(new_expected_path)) == sorted(
        [
            "file_metadata.csv",
            "metadata.csv",
            "resume_test_anomalies.txt",
            "timestamp.csv",
        ]
        + [f"20220101_1200{str(3*i).zfill(2)}.wav" for i in range(5)]
        + [f"20220101_1200{str(3*i).zfill(2)}.flac" for i in range(5, 10)]
    )
