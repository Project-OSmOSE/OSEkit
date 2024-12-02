from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pandas as pd
import pytest

from OSmOSE.utils.audio_utils import check_audio, is_supported_audio_format


@pytest.mark.unit
@pytest.mark.parametrize(
    ("filepath", "expected_output"),
    [
        pytest.param(Path("audio.wav"), True, id="simple_wav_file"),
        pytest.param(
            Path("audio_with_date_2024_02_14.wav"),
            True,
            id="complex_wav_file",
        ),
        pytest.param(Path("parent_folder/audio.wav"), True, id="file_in_parent_folder"),
        pytest.param(Path("audio.flac"), True, id="simple_flac_file"),
        pytest.param(Path("audio.WAV"), True, id="uppercase_wav_extension"),
        pytest.param(Path("audio.FLAC"), True, id="uppercase_flac_extension"),
        pytest.param(Path("audio.mp3"), False, id="unsupported_audio_extension"),
        pytest.param(
            Path("parent_folder/audio.MP3"),
            False,
            id="unsupported_in_parent_folder",
        ),
        pytest.param(Path("audio.pdf"), False, id="unsupported_extension"),
        pytest.param(Path("audio"), False, id="no_extension"),
    ],
)
def test_supported_audio_formats(filepath: Path, expected_output: bool) -> None:
    assert is_supported_audio_format(filepath) == expected_output


@pytest.mark.unit
@pytest.mark.parametrize(
    ("audio_metadata", "timestamps", "expectation"),
    [
        pytest.param(
            pd.DataFrame(
                [
                    ["file_1.wav", 128_000, 3_600],
                    ["file_2.wav", 128_000, 3_600],
                    ["file_3.wav", 128_000, 3_600],
                ],
                columns=["filename", "origin_sr", "duration"],
            ),
            pd.DataFrame(
                [
                    ["file_1.wav", pd.Timestamp("2024-01-01 12:12:00")],
                    ["file_2.wav", pd.Timestamp("2024-01-01 12:13:00")],
                    ["file_3.wav", pd.Timestamp("2024-01-01 12:14:00")],
                ],
                columns=["filename", "timestamp"],
            ),
            does_not_raise(),
            id="matching_dfs_do_not_raise",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    ["file_1.wav", 128_000, 3_600],
                    ["file_2.wav", 128_000, 3_600],
                    ["file_3.wav", 128_000, 3_600],
                ],
                columns=["filename", "origin_sr", "duration"],
            ),
            pd.DataFrame(
                [
                    ["file_1.wav", pd.Timestamp("2024-01-01 12:12:00")],
                    ["file_2.wav", pd.Timestamp("2024-01-01 12:13:00")],
                    ["file_4.wav", pd.Timestamp("2024-01-01 12:14:00")],
                ],
                columns=["filename", "timestamp"],
            ),
            pytest.raises(
                FileNotFoundError,
                match="file_3.wav has not been found in timestamp.csv",
            ),
            id="missing_file_in_timestamp_csv",
        ),
        pytest.param(
            pd.DataFrame(
                [["file_1.wav", 128_000, 3_600], ["file_2.wav", 128_000, 3_600]],
                columns=["filename", "origin_sr", "duration"],
            ),
            pd.DataFrame(
                [
                    ["file_1.wav", pd.Timestamp("2024-01-01 12:12:00")],
                    ["file_2.wav", pd.Timestamp("2024-01-01 12:13:00")],
                    ["file_3.wav", pd.Timestamp("2024-01-01 12:14:00")],
                ],
                columns=["filename", "timestamp"],
            ),
            pytest.raises(
                FileNotFoundError,
                match="file_3.wav is listed in timestamp.csv but hasn't be found.",
            ),
            id="missing_audio_file",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    ["file_1.wav", 128_000, 3_600],
                    ["file_2.wav", 128_000, 3_600],
                    ["file_3.wav", 128_001, 3_600],
                ],
                columns=["filename", "origin_sr", "duration"],
            ),
            pd.DataFrame(
                [
                    ["file_1.wav", pd.Timestamp("2024-01-01 12:12:00")],
                    ["file_2.wav", pd.Timestamp("2024-01-01 12:13:00")],
                    ["file_3.wav", pd.Timestamp("2024-01-01 12:14:00")],
                ],
                columns=["filename", "timestamp"],
            ),
            pytest.raises(
                ValueError, match="Your files do not have all the same sampling rate."
            ),
            id="mismatching_sr",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    ["file_1.wav", 128_000, 3_600],
                    ["file_2.wav", 128_000, 3_600],
                    ["file_3.wav", 128_000, 1_800],
                ],
                columns=["filename", "origin_sr", "duration"],
            ),
            pd.DataFrame(
                [
                    ["file_1.wav", pd.Timestamp("2024-01-01 12:12:00")],
                    ["file_2.wav", pd.Timestamp("2024-01-01 12:13:00")],
                    ["file_3.wav", pd.Timestamp("2024-01-01 12:14:00")],
                ],
                columns=["filename", "timestamp"],
            ),
            pytest.raises(
                ValueError, match="Your audio files have large duration discrepancies."
            ),
            id="mismatching_duration",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    ["file_1.wav", 128_000, 3_580],
                    ["file_2.wav", 128_000, 3_600],
                    ["file_3.wav", 128_000, 3_650],
                ],
                columns=["filename", "origin_sr", "duration"],
            ),
            pd.DataFrame(
                [
                    ["file_1.wav", pd.Timestamp("2024-01-01 12:12:00")],
                    ["file_2.wav", pd.Timestamp("2024-01-01 12:13:00")],
                    ["file_3.wav", pd.Timestamp("2024-01-01 12:14:00")],
                ],
                columns=["filename", "timestamp"],
            ),
            does_not_raise(),
            id="close_durations_should_not_raise",
        ),
    ],
)
def test_check_audio(
    audio_metadata: pd.DataFrame, timestamps: pd.DataFrame, expectation: None
) -> None:
    with expectation as e:
        assert check_audio(audio_metadata, timestamps) == e
