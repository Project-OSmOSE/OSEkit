import os
import pytest
from OSmOSE import Dataset


def test_init(input_dataset, capsys):
    dataset = Dataset(dataset_path=input_dataset["main_dir"])

    assert dataset.name == "sample_dir"
    
    dataset.gps_coordinates
    captured = capsys.readouterr()
    assert "This dataset has no GPS coordinates." in captured.out
