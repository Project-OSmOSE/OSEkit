import os
import pytest
import numpy as np
import soundfile as sf
import shutil


def capture_csv(monkeypatch):
    pass


@pytest.fixture
def input_dir(tmp_path):
    # Parameters
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    rate = 44100  # samples per second

    rng = np.random.default_rng()

    data = rng.standard_normal(1000)

    wav_file = input_dir / "test.wav"
    sf.write(wav_file, data, rate)

    yield input_dir


@pytest.fixture
def output_dir(tmp_path):
    output_dir = tmp_path / "output"
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    yield output_dir
