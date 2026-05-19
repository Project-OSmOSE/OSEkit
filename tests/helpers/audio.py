import typing

import numpy as np
from pandas import Timestamp

from osekit.core.audio_data import AudioData


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
