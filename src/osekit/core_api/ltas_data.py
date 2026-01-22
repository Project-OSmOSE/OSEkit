"""LTASData is a special form of SpectroData.

The Sx values from a ``LTASData`` object are computed recursively.
LTAS should be preferred in cases where the audio is really long.
In that case, the corresponding number of time bins (``scipy.ShortTimeFTT.p_nums``) is
too long for the whole Sx matrix to be computed once.

The LTAS are rather computed recursively. If the number of temporal bins is higher
than a target ``p_num`` value, the audio is split in ``p_num`` parts.
A separate sft is computed on each of these bits and averaged so that the end Sx
presents ``p_num`` temporal windows.

This averaging is performed recursively:
if the audio data is such that after a first split, the ``p_nums`` for each part
still is higher than ``p_num``, the parts are further split
and each part is replaced with an average of the stft performed within it.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import ShortTimeFFT

from osekit.core_api.spectro_data import SpectroData
from osekit.utils.multiprocess_utils import multiprocess

if TYPE_CHECKING:
    from pandas import Timestamp

    from osekit.core_api.audio_data import AudioData
    from osekit.core_api.spectro_item import SpectroItem


class LTASData(SpectroData):
    """``LTASData`` is a special form of ``SpectroData``.

    The Sx values from a ``LTASData`` object are computed recursively.
    LTAS should be preferred in cases where the audio is really long.
    In that case, the corresponding number of time bins (``scipy.ShortTimeFTT.p_nums``) is
    too long for the whole Sx matrix to be computed once.

    The LTAS are rather computed recursively. If the number of temporal bins is higher
    than a target ``p_num`` value, the audio is split in ``p_num`` parts.
    A separate sft is computed on each of these bits and averaged so that the end Sx
    presents ``p_num`` temporal windows.

    This averaging is performed recursively:
    if the audio data is such that after a first split, the ``p_nums`` for each part
    still is higher than ``p_num``, the parts are further split
    and each part is replaced with an average of the stft performed within it.

    """

    def __init__(
        self,
        items: list[SpectroItem] | None = None,
        audio_data: AudioData = None,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        fft: ShortTimeFFT | None = None,
        db_ref: float | None = None,
        v_lim: tuple[float, float] | None = None,
        colormap: str | None = None,
        nb_time_bins: int = 1920,
    ) -> None:
        """Initialize a ``LTASData`` from a list of ``SpectroItems``.

        Parameters
        ----------
        items: list[SpectroItem]
            List of the ``SpectroItem`` constituting the ``LTASData``.
        audio_data: AudioData
            The audio data from which to compute the spectrogram.
        begin: Timestamp | None
            Only effective if items is ``None``.
            Set the begin of the empty data.
        end: Timestamp | None
            Only effective if items is ``None``.
            Set the end of the empty data.
        fft: ShortTimeFFT
            The short time FFT used for computing the spectrogram.
        db_ref: float | None
            Reference value for computing sx values in decibel.
        v_lim: tuple[float,float]
            Lower and upper limits (in ``dB``) of the colormap used
            for plotting the spectrogram.
        colormap: str
            Colormap to use for plotting the spectrogram.
        nb_time_bins: int
            The maximum number of time bins of the LTAS.
            Given the audio data and the fft parameters,
            if the resulting spectrogram has a number of windows ``p_num
            <= nb_time_bins``, the LTAS is computed like a classic spectrogram.
            Otherwise, the audio data is split in ``nb_time_bins`` equal-duration
            audio data, and each bin of the LTAS consist in an average of the
            fft values obtained on each of these bins. The audio is split recursively
            until ``p_num <= nb_time_bins``.

        """
        ltas_fft = LTASData.get_ltas_fft(fft)
        super().__init__(
            items=items,
            audio_data=audio_data,
            begin=begin,
            end=end,
            fft=ltas_fft,
            db_ref=db_ref,
            v_lim=v_lim,
            colormap=colormap,
        )
        self.nb_time_bins = nb_time_bins
        self.sx_dtype = float

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the LTAS data."""
        return self.fft.f_pts, self.nb_time_bins

    @staticmethod
    def mean_value_part(sub_spectro: LTASData) -> np.ndarray:
        """Return the mean value of the LTAS part."""
        return np.mean(
            sub_spectro.get_value(depth=1),
            axis=1,
        )

    def get_value(self, depth: int = 0) -> np.ndarray:
        """Return the Sx matrix of the LTAS.

        The Sx matrix contains the absolute square of the STFT.
        """
        if not self.is_empty:
            return self._get_value_from_items(self.items)
        if super().shape[1] <= self.nb_time_bins:
            return super().get_value()
        sub_spectros = [
            LTASData.from_spectro_data(
                SpectroData.from_audio_data(ad, self.fft),
                nb_time_bins=self.nb_time_bins,
            )
            for ad in self.audio_data.split(self.nb_time_bins, pass_normalization=False)
        ]

        if depth != 0:
            return np.vstack(
                [self.mean_value_part(sub_spectro) for sub_spectro in sub_spectros],
            ).T

        return np.vstack(
            list(multiprocess(self.mean_value_part, sub_spectros)),
        ).T

    @classmethod
    def from_spectro_data(
        cls,
        spectro_data: SpectroData,
        nb_time_bins: int,
    ) -> LTASData:
        """Initialize a ``LTASData`` from a ``SpectroData``.

        Parameters
        ----------
        spectro_data: SpectroData
            The spectrogram to turn in a LTAS.
        nb_time_bins: int
            The maximum number of windows over which the audio will be split to perform
            a LTAS.

        Returns
        -------
        LTASData:
            The ``LTASData`` instance.

        """
        items = spectro_data.items
        audio_data = spectro_data.audio_data
        begin = spectro_data.begin
        end = spectro_data.end
        fft = spectro_data.fft
        db_ref = spectro_data.db_ref
        v_lim = spectro_data.v_lim
        colormap = spectro_data.colormap
        return cls(
            items=items,
            audio_data=audio_data,
            begin=begin,
            end=end,
            fft=fft,
            nb_time_bins=nb_time_bins,
            db_ref=db_ref,
            v_lim=v_lim,
            colormap=colormap,
        )

    @classmethod
    def from_audio_data(
        cls,
        data: AudioData,
        fft: ShortTimeFFT,
        v_lim: tuple[float, float] | None = None,
        colormap: str | None = None,
        nb_time_bins: int = 1920,
    ) -> SpectroData:
        """Instantiate a ``SpectroData`` object from an ``AudioData`` object.

        Parameters
        ----------
        data: AudioData
            Audio data from which the ``SpectroData`` should be computed.
        fft: ShortTimeFFT
            The ``ShortTimeFFT`` used to compute the spectrogram.
        v_lim: tuple[float,float]
            Lower and upper limits (in ``dB``) of the colormap used
            for plotting the spectrogram.
        colormap: str
            Colormap to use for plotting the spectrogram.
        nb_time_bins: int
            The maximum number of windows over which the audio will be split to perform
            Defaulted to ``1920``.

        Returns
        -------
        LTASData:
            The ``SpectroData`` object.

        """
        return cls(
            audio_data=data,
            fft=fft,
            begin=data.begin,
            end=data.end,
            v_lim=v_lim,
            colormap=colormap,
            nb_time_bins=nb_time_bins,
        )

    def to_dict(self, *, embed_sft: bool = True) -> dict:
        """Serialize a ``LTASData`` to a dictionary.

        Parameters
        ----------
        embed_sft: bool
            If ``True``, the SFT parameters will be included in the dictionary.
            In a case where multiple ``LTASData`` that share a same SFT are serialized,
            SFT parameters shouldn't be included in the dictionary, as the window
            values might lead to large redundant data.
            Rather, the SFT parameters should be serialized in
            a ``LTASDataset`` dictionary so that it can be only stored once
            for all ``LTASData`` instances.

        Returns
        -------
        dict:
            The serialized dictionary representing the ``LTASData``.

        """
        return super().to_dict(embed_sft=embed_sft) | {
            "nb_time_bins": self.nb_time_bins,
        }

    @classmethod
    def from_dict(cls, dictionary: dict, sft: ShortTimeFFT | None = None) -> LTASData:
        """Deserialize a ``LTASData`` from a dictionary.

        Parameters
        ----------
        dictionary: dict
            The serialized dictionary representing the ``LTASData``.
        sft: ShortTimeFFT | None
            The ``ShortTimeFFT`` used to compute the spectrogram.
            If not provided, the SFT parameters must be included in the dictionary.

        Returns
        -------
        LTASDataset
            The deserialized ``LTASData``.

        """
        return cls.from_spectro_data(
            SpectroData.from_dict(dictionary, sft=sft),
            nb_time_bins=dictionary["nb_time_bins"],
        )

    @staticmethod
    def get_ltas_fft(fft: ShortTimeFFT) -> ShortTimeFFT:
        """Return a ``ShortTimeFFT`` object optimized for computing LTAS.

        The overlap of the fft is forced set to ``0.``, as the value of consecutive
        windows will in the end be averaged.

        Parameters
        ----------
        fft: ShortTimeFFT
            The fft to optimize for LTAS computation.

        Returns
        -------
        ShortTimeFFT
            The optimized fft.

        """
        win = fft.win
        fs = fft.fs
        mfft = fft.mfft
        hop = win.shape[0]
        return ShortTimeFFT(win=win, hop=hop, fs=fs, mfft=mfft)
