import os
import pytest
from OSmOSE import Dataset


def test_init(input_dataset):
    path = os.path.join(os.path.dirname(__file__), "sample_dataset")
    print(input_dataset)
    assert True


def test_build(input_dataset):
    dataset = Dataset(input_dataset["main_dir"])

    dataset.build()
