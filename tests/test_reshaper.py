import pandas as pd
from datetime import datetime, timedelta
from typing import List
from OSmOSE.cluster.audio_reshaper import *
import pytest
import csv


def test_substract_timestamps():
    # Create test data
    timestamp_data = {
        "filename": ["file1", "file2", "file3", "badfile"],
        "timestamp": [
            "2022-01-01T12:00:00.000Z",
            "2022-01-01T12:01:00.000Z",
            "2022-01-01T12:02:00.000Z",
            "20220101T12:03:00.000",
        ],
    }
    input_timestamp = pd.DataFrame(data=timestamp_data)

    files = ["file1", "file2", "file3", "badfile"]

    # Test the function for the first file
    result = substract_timestamps(input_timestamp, files, 0)
    expected_result = timedelta(seconds=0)
    assert result == expected_result

    # Test the function for the second file
    result = substract_timestamps(input_timestamp, files, 1)
    expected_result = timedelta(seconds=60)
    assert result == expected_result

    # Test the function for the third file
    result = substract_timestamps(input_timestamp, files, 2)
    expected_result = timedelta(seconds=60)
    assert result == expected_result

    # Test the function for the badly formatted  timestamp
    with pytest.raises(ValueError) as ts_error:
        substract_timestamps(input_timestamp, files, 3)

    assert (
        str(ts_error.value)
        == "time data '20220101T12:03:00.000' does not match format '%Y-%m-%dT%H:%M:%S.%fZ'"
    )


def test_reshape_errors(input_dir):
    with pytest.raises(ValueError) as e:
        reshape("/not/a/path", 15)

    assert (
        str(e.value)
        == "The input files must either be a valid folder path or a list of file path, not /not/a/path."
    )

    with pytest.raises(ValueError) as e:
        reshape(input_dir, 20, last_file_behavior="misbehave")

    assert (
        str(e.value)
        == "Bad value misbehave for last_file_behavior parameters. Must be one of truncate, pad or discard."
    )

    with pytest.raises(FileNotFoundError):
        reshape(input_dir, 20)  # Supposed to fail because there is no timestamp.csv


def test_reshape(input_dir, output_dir):
    for i in range(9):
        wav_file = os.path.join(input_dir, f"test{i}.wav")
        shutil.copyfile(os.path.join(input_dir, "test.wav"), wav_file)

    with open(os.path.join(input_dir, "timestamp.csv"), "w", newline="") as timestampf:
        writer = csv.writer(timestampf)
        writer.writerow(
            [os.path.join(input_dir, "test.wav"), "2022-01-01T11:59:57.000Z", "UTC"]
        )
        writer.writerows(
            [
                [
                    os.path.join(input_dir, f"test{i}.wav"),
                    f"2022-01-01T12:00:{str(3*i).zfill(2)}.000Z",
                    "UTC",
                ]
                for i in range(9)
            ]
        )

    with open(os.path.join(input_dir, "timestamp.csv"), "r") as f:
        print(f.readlines())

    reshape(input_files=input_dir, chunk_size=2, output_dir_path=output_dir)

    reshaped_files = os.listdir(output_dir)
    assert len(reshaped_files) == 15
    assert sf.info(reshaped_files[0]).duration == 2.0
    assert sf.info(reshaped_files[0]).samplerate == 44100
