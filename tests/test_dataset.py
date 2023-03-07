import os
import pytest


def test_init(input_dataset):
    path = os.path.join(os.path.dirname(__file__), "sample_dataset")
    print(input_dataset)
    assert True
