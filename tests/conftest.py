import os
import pytest
import numpy as np
import soundfile as sf
import shutil
import csv

def capture_csv(monkeypatch):
    pass

@pytest.fixture
def input_dataset(tmp_path):
    main_dir = tmp_path / "sample_dataset"
    main_audio_dir = main_dir / "raw" / "audio"
    orig_audio_dir = main_audio_dir / "original"
    process_dir = main_dir / "processed" / "spectrogram"

    folders_to_create = [main_dir, main_audio_dir, orig_audio_dir, process_dir]

    for folder in folders_to_create:
        os.makedirs(folder)

    rate = 44100  # samples per second
    duration = 3
    rng = np.random.default_rng()


    for i in range(3):
        data = rng.standard_normal(duration * rate)
        wav_file = orig_audio_dir / f"test{i}.wav"
        sf.write(wav_file, data, rate)
    
        with open(os.path.join(orig_audio_dir, "timestamp.csv"), "w", newline="") as timestampf:
            writer = csv.writer(timestampf)
            writer.writerow(
                    [
                        os.path.join(main_dir, f"test{i}.wav"),
                        f"2022-01-01T12:00:{str(3*i).zfill(2)}.000Z",
                        "UTC",
                    ]
            )

    yield dict(zip(["main_dir", "main_audio_dir", "orig_audio_dir", "process_dir"],folders_to_create))




@pytest.fixture
def input_dir(tmp_path):
    # Parameters
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    rate = 44100  # samples per second
    duration = 3
    rng = np.random.default_rng()

    data = rng.standard_normal(duration * rate)

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
