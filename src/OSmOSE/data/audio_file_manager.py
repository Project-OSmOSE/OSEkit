"""Audio File Manager which keeps an audio file open until a request in another file is made.

This workflow avoids closing/opening a same file repeatedly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import soundfile as sf

if TYPE_CHECKING:
    from os import PathLike

    import numpy as np


class AudioFileManager:
    """Audio File Manager which keeps an audio file open until a request in another file is made."""

    def __init__(self) -> None:
        """Initialize an audio file manager."""
        self.opened_file = None

    def _close(self) -> None:
        if self.opened_file is None:
            return
        self.opened_file.close()
        self.opened_file = None

    def _open(self, path: PathLike | str) -> None:
        self.opened_file = sf.SoundFile(path, "r")

    def _switch(self, path: PathLike | str) -> None:
        if self.opened_file is None:
            self._open(path)
        if self.opened_file.name == str(path):
            return
        self._close()
        self._open(path)

    def read(self, path: PathLike | str, start: int = 0, stop: int = -1) -> np.ndarray:
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
            A (channel * frames) array containing the audio data.

        """
        self._switch(path)
        self.opened_file.seek(start)
        return self.opened_file.read(stop - start)

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
        self._switch(path)
        return (
            self.opened_file.samplerate,
            self.opened_file.frames,
            self.opened_file.channels,
        )
