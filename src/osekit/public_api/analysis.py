from __future__ import annotations

from enum import Flag, auto
from typing import TYPE_CHECKING, Literal

from osekit.utils.audio_utils import Normalization

if TYPE_CHECKING:
    from pandas import Timedelta, Timestamp
    from scipy.signal import ShortTimeFFT

    from osekit.core_api.frequency_scale import Scale


class AnalysisType(Flag):
    """Enum of flags that should be used to specify the type of analysis to run.

    ``AUDIO``:
        Will add an ``AudioDataset`` to the datasets and write the reshaped audio files
        to disk.
        The new ``AudioDataset`` will be linked to the reshaped audio files rather
        than to the original files.
    ``MATRIX``:
        Will write the ``npz`` ``SpectroFiles`` to disk and link the ``SpectroDataset``
        to these files.
    ``SPECTROGRAM``:
        Will export the spectrogram ``png`` images.
    ``WELCH``:
        Will write the ``npz`` welches to disk.

    Multiple flags can be enabled thanks to the logical or ``|`` operator:
    ``AnalysisType.AUDIO | AnalysisType.SPECTROGRAM`` will export both audio files and
    spectrogram images.

    >>> # Exporting both the reshaped audio and the spectrograms
    >>> # (without the npz matrices):
    >>> export = AnalysisType.AUDIO | AnalysisType.SPECTROGRAM
    >>> AnalysisType.AUDIO in export
    True
    >>> AnalysisType.SPECTROGRAM in export
    True
    >>> AnalysisType.MATRIX in export
    False

    """

    AUDIO = auto()
    MATRIX = auto()
    SPECTROGRAM = auto()
    WELCH = auto()


class Analysis:
    """Class that contains all parameter of an analysis.

    Analysis instances are passed to the public API dataset, which runs the analysis.
    The ``Analysis`` object contains all info on the analysis to be done: the type(s) of
    core_api dataset(s) that will be created and added to the ``Dataset.datasets``
    property and which output files will be written to disk
    (reshaped audio files, ``npz`` spectra matrices, ``png`` spectrograms...) depend
    on the ``analysis_type`` parameter.
    The ``Analysis`` instance also contains the technical parameters of the analyses
    (begin/end times, sft, sample rate...).
    """

    def __init__(
        self,
        analysis_type: AnalysisType,
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
        """Initialize an ``Analysis`` object.

        Parameters
        ----------
        analysis_type: AnalysisType
            The type of analysis to run.
            See ``AnalysisType`` docstring for more info.
        begin: Timestamp | None
            The begin of the analysis dataset.
            Defaulted to the begin of the original dataset.
        end: Timestamp | None
            The end of the analysis dataset.
            Defaulted to the end of the original dataset.
        data_duration: Timedelta | None
            Duration of the data within the analysis dataset.
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
            Sample rate of the new analysis data.
            Audio data will be resampled if provided, else the sample rate
            will be set to the one of the original dataset.
        normalization: Normalization
            The type of normalization to apply to the audio data.
        name: str | None
            Name of the analysis dataset.
            Defaulted as the begin timestamp of the analysis dataset.
            If both audio and spectro analyses are selected, the audio
            analysis dataset name will be suffixed with ``"_audio"``.
        subtype: str | None
            Subtype of the written audio files as provided by the soundfile module.
            Defaulted as the default ``16-bit PCM`` for ``wav`` audio files.
            This parameter has no effect if ``Analysis.AUDIO`` is not in analysis.
        fft: ShortTimeFFT | None
            FFT to use for computing the spectra.
            This parameter is mandatory if either ``Analysis.MATRIX``
            or ``Analysis.SPECTROGRAM`` is in analysis.
            This parameter has no effect if neither ``Analysis.MATRIX``
            nor ``Analysis.SPECTROGRAM`` is in the analysis.
        v_lim: tuple[float, float] | None
            Limits (in ``dB``) of the colormap used for plotting the spectrogram.
            Has no effect if ``Analysis.SPECTROGRAM`` is not in analysis.
        colormap: str | None
            Colormap to use for plotting the spectrogram.
            Has no effect if ``Analysis.SPECTROGRAM`` is not in analysis.
        scale: osekit.core_api.frequecy_scale.Scale
            Custom frequency scale to use for plotting the spectrogram.
            Has no effect if ``Analysis.SPECTROGRAM`` is not in analysis.
        nb_ltas_time_bins: int | None
            If ``None``, the spectrogram will be computed regularly.
            If specified, the spectrogram will be computed as LTAS, with the value
            representing the maximum number of averaged time bins.

        """
        self.analysis_type = analysis_type
        self.begin = begin
        self.end = end
        self.data_duration = data_duration
        self.mode = mode
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.name = name
        self.normalization = normalization
        self.subtype = subtype
        self.fft = fft
        self.v_lim = v_lim
        self.colormap = colormap
        self.scale = scale
        self.nb_ltas_time_bins = nb_ltas_time_bins

        if self.is_spectro and fft is None:
            msg = "FFT parameter should be given if spectra outputs are selected."
            raise ValueError(msg)

    @property
    def is_spectro(self) -> bool:
        """Return ``True`` if the analysis contains spectral computations, ``False`` otherwise."""
        return any(
            flag in self.analysis_type
            for flag in (
                AnalysisType.MATRIX,
                AnalysisType.SPECTROGRAM,
                AnalysisType.WELCH,
            )
        )
