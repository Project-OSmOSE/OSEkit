import typing
from pathlib import Path

import numpy as np
from pandas import Timedelta, Timestamp

from osekit.core.audio_data import AudioData
from osekit.core.audio_file import AudioFile


class MockedAudioFile(AudioFile):
    def __init__(
        self,
        mocked_value: np.ndarray,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        defaults = {
            "begin": Timestamp("2000-01-01 00:00:00"),
            "path": Path("foo"),
        }
        for key, value in defaults.items():
            if key not in kwargs:
                kwargs.update(**{key: value})

        if mocked_value.ndim == 1:
            mocked_value = mocked_value[:, None]

        self.mocked_value = mocked_value
        self.channels = self.mocked_value.shape[1]
        self.sample_rate = kwargs.get("sample_rate", 48000)
        self.begin = kwargs["begin"]
        self.end = self.begin + Timedelta(
            seconds=mocked_value.shape[0] / self.sample_rate
        )
        self.pointer = 0

    def read(self, start: Timestamp, stop: Timestamp) -> np.ndarray:
        start_sample, stop_sample = self.frames_indexes(start, stop)
        return self.mocked_value[start_sample:stop_sample]

    def stream(self, chunk_size: int) -> np.ndarray:
        values = self.mocked_value[self.pointer : self.pointer + chunk_size]
        self.pointer += chunk_size
        if values.ndim == 1:
            return values[:, None]  # 2D array to match the format of multichannel audio
        return values

    def seek(self, frame: int) -> None:
        self.pointer = frame


class MockedAudioData(AudioData):
    def __init__(
        self,
        mocked_value: list[float] | np.ndarray,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        defaults = {
            "begin": Timestamp("2000-01-01 00:00:00"),
            "end": Timestamp("2000-01-01 00:00:01"),
            "sample_rate": 48000,
        }
        for key, value in defaults.items():
            if key not in kwargs:
                kwargs.update(**{key: value})

        super().__init__(*args, **kwargs)
        if mocked_value is not None:
            self.mocked_value = mocked_value
            if type(mocked_value) is list or len(mocked_value.shape) == 1:
                self.mocked_value = np.array(self.mocked_value).reshape(
                    len(mocked_value),
                    1,
                )

    @property
    def length(self) -> int:
        return len(self.mocked_value)

    def get_raw_value(self) -> np.ndarray:
        return self.mocked_value
