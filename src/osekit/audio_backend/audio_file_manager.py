"""Audio File Manager which keeps an audio file open until a request in another file is made.

This workflow avoids closing/opening a same file repeatedly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from osekit.audio_backend.mseed_backend import MSeedBackend
from osekit.audio_backend.soundfile_backend import SoundFileBackend

if TYPE_CHECKING:
    from os import PathLike

    import numpy as np


class AudioFileManager:
    """Audio File Manager which keeps an audio file open until a request in another file is made."""

    def __init__(self) -> None:
        """Initialize an audio file manager."""
        self._soundfile = SoundFileBackend()
        self._mseed: MSeedBackend | None = None

    def close(self) -> None:
        """Close the currently opened file."""
        self._soundfile.close()
        if self._mseed:
            self._mseed.close()

    def _backend(self, path: PathLike | str) -> SoundFileBackend | MSeedBackend:
        suffix = Path(path).suffix.lower()

        if suffix == ".mseed":
            if self._mseed is None:
                self._mseed = MSeedBackend()
            return self._mseed

        return self._soundfile

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
        return self._backend(path).info(path)

    def read(
        self,
        path: PathLike | str,
        start: int = 0,
        stop: int | None = None,
    ) -> np.ndarray:
        """Read the content of an audio file.

        If the audio file is not the current opened file,
        the current opened file is switched.

        Parameters
        ----------
        path: PathLike | str
            Path to the audio file.
        start: int
            First frame to read.
        stop: int | None
            Frame after the last frame to read.

        Returns
        -------
        np.ndarray:
            A ``(channel * frames)`` array containing the audio data.

        """
        _, frames, _ = self.info(path)

        if stop is None:
            stop = frames

        if stop is None:
            stop = frames

        if not 0 <= start < frames:
            msg = "Start should be between 0 and the last frame of the audio file."
            raise ValueError(msg)
        if not 0 <= stop <= frames:
            msg = "Stop should be between 0 and the last frame of the audio file."
            raise ValueError(msg)
        if start > stop:
            msg = "Start should be inferior to Stop."
            raise ValueError(msg)

        return self._backend(path).read(path=path, start=start, stop=stop)
