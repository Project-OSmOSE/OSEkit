from __future__ import annotations

from enum import Flag, auto
from typing import TYPE_CHECKING, Literal

from osekit.utils.audio_utils import Normalization

if TYPE_CHECKING:
    from pandas import Timedelta, Timestamp
    from scipy.signal import ShortTimeFFT

    from osekit.core.frequency_scale import Scale


class OutputType(Flag):
    """Enum of flags that should be used to specify the type of transform to run.

    ``AUDIO``:
        Will add an ``AudioDataset`` to the output_datasets and write the reshaped audio files
        to disk.
        The new ``AudioDataset`` will be linked to the reshaped audio files rather
        than to the original files.
    ``SPECTRUM``:
        Will write the ``npz`` ``SpectroFiles`` to disk and link the ``SpectroDataset``
        to these files.
    ``SPECTROGRAM``:
        Will export the spectrogram ``png`` images.
    ``WELCH``:
        Will write the ``npz`` welches to disk.

    Multiple flags can be enabled thanks to the logical or ``|`` operator:
    ``OutputType.AUDIO | OutputType.SPECTROGRAM`` will export both audio files and
    spectrogram images.

    >>> # Exporting both the reshaped audio and the spectrograms
    >>> # (without the npz matrices):
    >>> export = OutputType.AUDIO | OutputType.SPECTROGRAM
    >>> OutputType.AUDIO in export
    True
    >>> OutputType.SPECTROGRAM in export
    True
    >>> OutputType.SPECTRUM in export
    False

    """

    AUDIO = auto()
    SPECTRUM = auto()
    SPECTROGRAM = auto()
    WELCH = auto()


class Transform:
    """Class that contains all parameter of a transform.

    Transform instances are passed to the public API project, which runs the transform.
    The ``Transform`` object contains all info on the transform to be done: the type(s) of
    core dataset(s) that will be created and added to the ``Project.output_datasets``
    property and which output files will be written to disk
    (reshaped audio files, ``npz`` spectra matrices, ``png`` spectrograms...) depend
    on the ``output_type`` parameter.
    The ``Transform`` instance also contains the technical parameters of the transforms
    (begin/end times, sft, sample rate...).
    """

    def __init__(
        self,
        output_type: OutputType,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        data_duration: Timedelta | None = None,
        mode: Literal["files", "timedelta_total", "timedelta_file"] = "timedelta_total",
        overlap: float = 0.0,
        sample_rate: float | None = None,
        normalization: Normalization = Normalization.RAW,
        name: str | None = None,
        subtype: str | None = None,
        fft: ShortTimeFFT | None = None,
        v_lim: tuple[float, float] | None = None,
        colormap: str | None = None,
        scale: Scale | None = None,
        nb_ltas_time_bins: int | None = None,
    ) -> None:
        """Initialize an ``Transform`` object.

        Parameters
        ----------
        output_type: OutputType
            The type of transform to run.
            See ``OutputType`` docstring for more info.
        begin: Timestamp | None
            The begin of the transform dataset.
            Defaulted to the begin of the original dataset.
        end: Timestamp | None
            The end of the transform dataset.
            Defaulted to the end of the original dataset.
        data_duration: Timedelta | None
            Duration of the data within the transform dataset.
            If provided, audio data will be evenly distributed between
            ``begin`` and ``end``.
            Else, one data object will cover the whole time period.
        mode: Literal["files", "timedelta_total", "timedelta_file"]
            Mode of creation of the dataset data from the original files.
            ``"files"``: one data will be created for each file.
            ``"timedelta_total"``: data objects of duration equal to ``data_duration``
            will be created from the begin timestamp to the end timestamp.
            ``"timedelta_file"``: data objects of duration equal to ``data_duration``
            will be created from the beginning of the first file that the begin
            timestamp is into, until it would resume in a data beginning between
            two files.
            Then, the next data object will be created from the
            beginning of the next original file and so on.
        overlap: float
            Overlap percentage between consecutive data.
        sample_rate: float | None
            Sample rate of the new transform data.
            Audio data will be resampled if provided, else the sample rate
            will be set to the one of the original dataset.
        normalization: Normalization
            The type of normalization to apply to the audio data.
        name: str | None
            Name of the transform dataset.
            Defaulted as the begin timestamp of the transform dataset.
            If both audio and spectro outputs are selected, the audio
            transform dataset name will be suffixed with ``"_audio"``.
        subtype: str | None
            Subtype of the written audio files as provided by the soundfile module.
            Defaulted as the default ``16-bit PCM`` for ``wav`` audio files.
            This parameter has no effect if ``Transform.AUDIO`` is not in transform.
        fft: ShortTimeFFT | None
            FFT to use for computing the spectra.
            This parameter is mandatory if either ``Transform.SPECTRUM``
            or ``Transform.SPECTROGRAM`` is in transform.
            This parameter has no effect if neither ``Transform.SPECTRUM``
            nor ``Transform.SPECTROGRAM`` is in the transform.
        v_lim: tuple[float, float] | None
            Limits (in ``dB``) of the colormap used for plotting the spectrogram.
            Has no effect if ``Transform.SPECTROGRAM`` is not in transform.
        colormap: str | None
            Colormap to use for plotting the spectrogram.
            Has no effect if ``Transform.SPECTROGRAM`` is not in transform.
        scale: osekit.core.frequecy_scale.Scale
            Custom frequency scale to use for plotting the spectrogram.
            Has no effect if ``Transform.SPECTROGRAM`` is not in transform.
        nb_ltas_time_bins: int | None
            If ``None``, the spectrogram will be computed regularly.
            If specified, the spectrogram will be computed as LTAS, with the value
            representing the maximum number of averaged time bins.

        """
        self._validate_sample_rate(sample_rate=sample_rate, fft=fft)

        self.output_type = output_type
        self.begin = begin
        self.end = end
        self.data_duration = data_duration
        self.mode = mode
        self.overlap = overlap
        self.fft = fft
        self.sample_rate = sample_rate
        self.name = name
        self.normalization = normalization
        self.subtype = subtype
        self.v_lim = v_lim
        self.colormap = colormap
        self.scale = scale
        self.nb_ltas_time_bins = nb_ltas_time_bins

        if self.is_spectro and fft is None:
            msg = "FFT parameter should be given if spectra outputs are selected."
            raise ValueError(msg)

    @property
    def is_spectro(self) -> bool:
        """Return ``True`` if the transform contains spectral computations, ``False`` otherwise."""
        return any(
            flag in self.output_type
            for flag in (
                OutputType.SPECTRUM,
                OutputType.SPECTROGRAM,
                OutputType.WELCH,
            )
        )

    @property
    def sample_rate(self) -> float | None:
        """Return the sample rate of the transform."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: float | None) -> None:
        """Set the sample rate of the transform."""
        if self.fft is not None and value is not None:
            self.fft.fs = value
        self._sample_rate = value

    @property
    def fft(self) -> ShortTimeFFT | None:
        """Return the FFT used in the transform."""
        return self._fft

    @fft.setter
    def fft(self, value: ShortTimeFFT | None) -> None:
        """Set the FFT used in the transform."""
        if hasattr(self, "_sample_rate"):
            self._validate_sample_rate(sample_rate=self.sample_rate, fft=value)
        self._fft = value

    @staticmethod
    def _validate_sample_rate(
        sample_rate: float | None,
        fft: ShortTimeFFT | None,
    ) -> None:
        if sample_rate is None:
            return
        if fft is None:
            return
        if fft.fs == sample_rate:
            return
        msg = (
            rf"The sample rate of the transform ({sample_rate} Hz) "
            rf"does not match the sampling frequency of the "
            rf"fft ({fft.fs} Hz)"
        )
        raise ValueError(msg)
