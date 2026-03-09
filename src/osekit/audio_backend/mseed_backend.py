from os import PathLike

import numpy as np


def _require_obspy() -> None:
    try:
        import obspy  # noqa: PLC0415, F401
    except ImportError as e:
        msg = "MSEED support requires the optional dependency 'obspy' "
        "Install with: ``pip install osekit[mseed]``. "
        "If you're on windows and don't use conda, may the force be with you."
        raise ImportError(msg) from e


class MSeedBackend:
    """Backend for reading sismology MSEED files."""

    def __init__(self) -> None:
        """Initialize the MSEED backend."""
        _require_obspy()
        self.seeked_frame = 0

    def close(self) -> None:
        """Close the currently opened file. No use in MSEED files."""

    def info(self, path: PathLike | str) -> tuple[int, int, int]:
        """Return the sample rate, number of frames and channels of the MSEED file.

        Parameters
        ----------
        path: PathLike | str
            Path to the audio file.

        Returns
        -------
        tuple[int,int,int]:
            Sample rate, number of frames and channels of the MSEED file.

        """
        _require_obspy()
        import obspy  # type: ignore[import-not-found]

        metadata = obspy.read(pathname_or_url=path, headonly=True)
        sample_rate = {trace.meta.sampling_rate for trace in metadata.traces}
        if len(sample_rate) != 1:
            msg = "Inconsistent sampling rates in MSEED file."
            raise ValueError(msg)

        frames = sum(trace.meta.npts for trace in metadata.traces)
        return (
            int(sample_rate.pop()),
            frames,
            1,
        )

    def read(
        self,
        path: PathLike | str,
        start: int = 0,
        stop: int | None = None,
    ) -> np.ndarray:
        """Read the content of a MSEED file.

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
            A ``(channel * frames)`` array containing the MSEED data.

        """
        _require_obspy()
        import obspy  # type: ignore[import-not-found]  # noqa: PLC0415

        file_content = obspy.read(path)

        data = np.concatenate([trace.data for trace in file_content])
        return data[start:stop]

    def seek(self, path: PathLike, frame: int) -> None:
        """Set the seeked_frame of the backend.

        Streamed data will be streamed from this frame.

        Parameters
        ----------
        path: PathLike | str
            No effect.
        frame: int
            Frame to seek.

        """
        self.seeked_frame = frame

    def stream(self, path: PathLike, chunk_size: int) -> np.ndarray:
        """Stream the content of the MSEED file from the seeked frame.

        Parameters
        ----------
        path: PathLike
            Path to the mseed file.
        chunk_size: int
            Number of frames to stream.

        Returns
        -------
        np.ndarray:
            Streamed data of length ``chunk_size`` from ``self.seeked_frame``.
        """
        return self.read(
            path=path, start=self.seeked_frame, stop=self.seeked_frame + chunk_size
        )
