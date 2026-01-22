from os import PathLike

import numpy as np
import soundfile as sf


class SoundFileBackend:
    """Backend for reading conventional audio files (WAV, FLAC, MP3...)."""

    def __init__(self) -> None:
        """Instantiate a SoundFileBackend."""
        self._file: sf.SoundFile | None = None

    def close(self) -> None:
        """Close the currently opened file."""
        if self._file is None:
            return
        self._file.close()
        self._file = None

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
            self._file.samplerate,
            self._file.frames,
            self._file.channels,
        )

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
        stop: int
            Frame after the last frame to read.

        Returns
        -------
        np.ndarray:
            A ``(channel * frames)`` array containing the audio data.

        """
        self._switch(path)
        self._file.seek(start)
        return self._file.read(stop - start)

    def _close(self) -> None:
        if self._file is None:
            return
        self._file.close()
        self._file = None

    def _open(self, path: PathLike | str) -> None:
        self._file = sf.SoundFile(path, "r")

    def _switch(self, path: PathLike | str) -> None:
        if self._file is None or self._file.name != str(path):
            self._close()
            self._open(path)
