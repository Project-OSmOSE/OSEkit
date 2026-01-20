import builtins
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas import Timestamp

from osekit.config import TIMESTAMP_FORMATS_EXPORTED_FILES
from osekit.core_api.audio_data import AudioData
from osekit.core_api.audio_file import AudioFile

obspy = pytest.importorskip("obspy")


@pytest.mark.parametrize(
    ("streams", "files_begin", "begin", "end", "expected_data"),
    [
        pytest.param(
            [
                obspy.Stream(
                    obspy.Trace(
                        data=np.array(range(10), dtype=np.int32),
                        header={"sampling_rate": 10},
                    ),
                ),
            ],
            [Timestamp("2002-04-02 03:27:00")],
            None,
            None,
            np.array(range(10)),
            id="one_file_one_trace_full",
        ),
        pytest.param(
            [
                obspy.Stream(
                    obspy.Trace(
                        data=np.array(range(10), dtype=np.int32),
                        header={"sampling_rate": 10},
                    ),
                ),
                obspy.Stream(
                    obspy.Trace(
                        data=np.array(range(10, 20), dtype=np.int32),
                        header={"sampling_rate": 10},
                    ),
                ),
            ],
            [Timestamp("2002-04-02 03:27:00"), Timestamp("2002-04-02 03:27:01")],
            None,
            None,
            np.array(range(20)),
            id="multiple_files_one_trace_full",
        ),
        pytest.param(
            [
                obspy.Stream(
                    [
                        obspy.Trace(
                            data=np.array(range(10), dtype=np.int32),
                            header={"sampling_rate": 10},
                        ),
                        obspy.Trace(
                            data=np.array(range(10, 20), dtype=np.int32),
                            header={"sampling_rate": 10},
                        ),
                    ],
                ),
            ],
            [Timestamp("2002-04-02 03:27:00")],
            None,
            None,
            np.array(range(20)),
            id="one_file_two_traces_full",
        ),
        pytest.param(
            [
                obspy.Stream(
                    [
                        obspy.Trace(
                            data=np.array(range(10), dtype=np.int32),
                            header={"sampling_rate": 10},
                        ),
                        obspy.Trace(
                            data=np.array(range(10, 20), dtype=np.int32),
                            header={"sampling_rate": 10},
                        ),
                    ],
                ),
                obspy.Stream(
                    obspy.Trace(
                        data=np.array(range(20, 30), dtype=np.int32),
                        header={"sampling_rate": 10},
                    ),
                ),
            ],
            [Timestamp("2002-04-02 03:27:00"), Timestamp("2002-04-02 03:27:02")],
            None,
            None,
            np.array(range(30)),
            id="multiple_files_multiple_traces_full",
        ),
        pytest.param(
            [
                obspy.Stream(
                    obspy.Trace(
                        data=np.array(range(10), dtype=np.int32),
                        header={"sampling_rate": 10},
                    ),
                ),
            ],
            [Timestamp("2002-04-02 03:27:00")],
            Timestamp("2002-04-02 03:27:00.23"),
            Timestamp("2002-04-02 03:27:00.48"),
            np.array(range(2, 4)),
            id="one_file_one_trace_part",
        ),
        pytest.param(
            [
                obspy.Stream(
                    [
                        obspy.Trace(
                            data=np.array(range(10), dtype=np.int32),
                            header={"sampling_rate": 10},
                        ),
                        obspy.Trace(
                            data=np.array(range(10, 20), dtype=np.int32),
                            header={"sampling_rate": 10},
                        ),
                    ],
                ),
                obspy.Stream(
                    obspy.Trace(
                        data=np.array(range(20, 30), dtype=np.int32),
                        header={"sampling_rate": 10},
                    ),
                ),
            ],
            [Timestamp("2002-04-02 03:27:00"), Timestamp("2002-04-02 03:27:02")],
            Timestamp("2002-04-02 03:27:00.85"),
            Timestamp("2002-04-02 03:27:02.64"),
            np.array(range(8, 26)),
            id="multiple_files_multiple_traces_part",
        ),
    ],
)
def test_mseed_file_read(
    tmp_path: Path,
    streams: list[obspy.Stream],
    files_begin: list[pd.Timestamp],
    begin: pd.Timestamp | None,
    end: pd.Timestamp | None,
    expected_data: np.ndarray,
) -> None:
    # WRITE MSEED FILES
    for stream, file_begin in zip(streams, files_begin, strict=False):
        stream.write(
            tmp_path
            / f"{file_begin.strftime(TIMESTAMP_FORMATS_EXPORTED_FILES[1])}.mseed",
        )

    audio_files = sorted(
        (
            AudioFile(path, strptime_format=TIMESTAMP_FORMATS_EXPORTED_FILES[1])
            for path in tmp_path.glob("*.mseed")
        ),
        key=lambda af: af.begin,
    )

    assert all(
        audio_file.begin == file_begin
        for audio_file, file_begin in zip(audio_files, files_begin, strict=True)
    )
    assert all(
        audio_file.sample_rate == stream.traces[0].meta.sampling_rate
        for audio_file, stream in zip(audio_files, streams, strict=True)
    )

    audio_data = AudioData.from_files(audio_files, begin=begin, end=end)

    assert np.array_equal(audio_data.get_value()[:, 0], expected_data)


def test_inconsistent_mseed_sample_rate_error(tmp_path: Path) -> None:
    filename = (
        tmp_path
        / f"{Timestamp('2002-04-02 03:27:00').strftime(TIMESTAMP_FORMATS_EXPORTED_FILES[1])}.mseed"
    )

    obspy.Stream(
        [
            obspy.Trace(
                data=np.array(range(10), dtype=np.int32),
                header={"sampling_rate": 10},
            ),
            obspy.Trace(
                data=np.array(range(10, 20), dtype=np.int32),
                header={"sampling_rate": 20},
            ),
        ],
    ).write(filename)

    with pytest.raises(
        ValueError,
        match=r"Inconsistent sampling rates in MSEED file.",
    ) as e:
        assert (
            AudioFile(filename, strptime_format=TIMESTAMP_FORMATS_EXPORTED_FILES[1])
            == e
        )


def test_missing_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def raise_obspy_import_error(
        name: str,
        *args: list,
        **kwargs: dict,
    ) -> types.ModuleType:
        if name == "obspy":
            msg = "No module named 'obspy'"
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", raise_obspy_import_error)

    dummy_file = tmp_path / "dummy.mseed"
    dummy_file.touch()

    with pytest.raises(
        ImportError,
        match="MSEED support requires the optional dependency 'obspy'",
    ) as e:
        assert AudioFile(dummy_file, begin=Timestamp("2020-01-01 00:00:00")) == e
