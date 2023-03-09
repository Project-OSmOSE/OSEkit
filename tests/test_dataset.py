import pytest
from OSmOSE import Dataset
from OSmOSE.config import OSMOSE_PATH


def test_init(input_dataset, capsys):
    dataset = Dataset(dataset_path=input_dataset["main_dir"])

    assert dataset.name == "sample_dataset"

    dataset.gps_coordinates
    captured = capsys.readouterr()
    assert "This dataset has no GPS coordinates." in captured.out


def test_get_original_folder(input_dataset):
    dataset = Dataset(input_dataset["main_dir"])
    folder = dataset._get_original_folder()

    assert folder == input_dataset["orig_audio_dir"]

    input_dataset["orig_audio_dir"].rename(
        input_dataset["orig_audio_dir"].with_name("unconventional_name")
    )

    folder2 = dataset._get_original_folder()

    assert folder2 == input_dataset["orig_audio_dir"].with_name("unconventional_name")


def test_build(input_dataset):
    pass
