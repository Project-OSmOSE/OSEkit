from os import PathLike
from typing import Protocol

import numpy as np


class AudioBackend(Protocol):
    def info(self, path: PathLike | str) -> tuple[int, int, int]:
        """Return the sample rate, number of frames and channels of the audio file.

        Parameters
        ----------
        path: PathLike | str
            Path to the audio file.

        Returns
        -------
        tuple[int,int,int]:
            Sample rate, number of frames and channels of the audio file.

        """
        ...

    def read(self, path: PathLike | str, start: int, stop: int) -> np.ndarray:
        """Read the content of an audio file.

        If the audio file is not the current opened file,
        the current opened file is switched.

        Parameters
        ----------
        path: PathLike | str
            Path to the audio file.
        start: int
            First frame to read.
        stop: int
            Frame after the last frame to read.

        Returns
        -------
        np.ndarray:
            A ``(channel * frames)`` array containing the audio data.

        """
        ...

    def close(self) -> None:
        """Close the currently opened file."""
        ...
