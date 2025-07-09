"""LTASData is a special form of SpectroData.

The Sx values from a LTASData object are computed recursively.
LTAS should be preferred in cases where the audio is really long.
In that case, the corresponding number of time bins (scipy.ShortTimeFTT.p_nums) is
too long for the whole Sx matrix to be computed once.

The LTAS are rather computed recursively. If the number of temporal bins is higher
than a target p_num value, the audio is split in p_num parts.
A separate sft is computed on each of these bits and averaged so that the end Sx
presents p_num temporal windows.

This averaging is performed recursively:
if the audio data is such that after a first split, the p_nums for each part
still is higher than p_num, the parts are further split
and each part is replaced with an average of the stft performed within it.

"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import ShortTimeFFT
from tqdm import tqdm

from osekit.core_api.spectro_data import SpectroData

if TYPE_CHECKING:
    from pandas import Timestamp

    from osekit.core_api.audio_data import AudioData
    from osekit.core_api.spectro_item import SpectroItem


class LTASData(SpectroData):
    """LTASData is a special form of SpectroData.

    The Sx values from a LTASData object are computed recursively.
    LTAS should be preferred in cases where the audio is really long.
    In that case, the corresponding number of time bins (scipy.ShortTimeFTT.p_nums) is
    too long for the whole Sx matrix to be computed once.

    The LTAS are rather computed recursively. If the number of temporal bins is higher
    than a target p_num value, the audio is split in p_num parts.
    A separate sft is computed on each of these bits and averaged so that the end Sx
    presents p_num temporal windows.

    This averaging is performed recursively:
    if the audio data is such that after a first split, the p_nums for each part
    still is higher than p_num, the parts are further split
    and each part is replaced with an average of the stft performed within it.

    """

    def __init__(
        self,
        items: list[SpectroItem] | None = None,
        audio_data: AudioData = None,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        fft: ShortTimeFFT | None = None,
        nb_time_bins: int = 1920,
    ) -> None:
        """Initialize a SpectroData from a list of SpectroItems.

        Parameters
        ----------
        items: list[SpectroItem]
            List of the SpectroItem constituting the SpectroData.
        audio_data: AudioData
            The audio data from which to compute the spectrogram.
        begin: Timestamp | None
            Only effective if items is None.
            Set the begin of the empty data.
        end: Timestamp | None
            Only effective if items is None.
            Set the end of the empty data.
        fft: ShortTimeFFT
            The short time FFT used for computing the spectrogram.
        nb_time_bins: int
            The maximum number of time bins of the LTAS.
            Given the audio data and the fft parameters,
            if the resulting spectrogram has a number of windows p_num
            <= nb_time_bins, the LTAS is computed like a classic spectrogram.
            Otherwise, the audio data is split in nb_time_bins equal-duration
            audio data, and each bin of the LTAS consist in an average of the
            fft values obtained on each of these bins. The audio is split recursively
            until p_num <= nb_time_bins.

        """
        ltas_fft = LTASData.get_ltas_fft(fft)
        super().__init__(
            items=items,
            audio_data=audio_data,
            begin=begin,
            end=end,
            fft=ltas_fft,
        )
        self.nb_time_bins = nb_time_bins
        self.sx_dtype = float

    def get_value(self, depth: int = 0) -> np.ndarray:
        """Return the Sx matrix of the LTAS.

        The Sx matrix contains the absolute square of the STFT.
        """
        if self.shape[1] <= self.nb_time_bins:
            return super().get_value()
        sub_spectros = [
            LTASData.from_spectro_data(
                SpectroData.from_audio_data(ad, self.fft),
                nb_time_bins=self.nb_time_bins,
            )
            for ad in self.audio_data.split(self.nb_time_bins)
        ]

        return np.vstack(
            [
                np.mean(sub_spectro.get_value(depth + 1), axis=1)
                for sub_spectro in (
                    sub_spectros
                    if depth != 0
                    else tqdm(sub_spectros, disable=os.environ.get("DISABLE_TQDM", ""))
                )
            ],
        ).T

    @classmethod
    def from_spectro_data(
        cls,
        spectro_data: SpectroData,
        nb_time_bins: int,
    ) -> LTASData:
        """Initialize a LTASData from a SpectroData.

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
            The LTASData instance.

        """
        items = spectro_data.items
        audio_data = spectro_data.audio_data
        begin = spectro_data.begin
        end = spectro_data.end
        fft = spectro_data.fft
        return cls(
            items=items,
            audio_data=audio_data,
            begin=begin,
            end=end,
            fft=fft,
            nb_time_bins=nb_time_bins,
        )

    @classmethod
    def from_audio_data(
        cls,
        data: AudioData,
        fft: ShortTimeFFT,
        nb_time_bins: int = 1920,
    ) -> SpectroData:
        """Instantiate a SpectroData object from a AudioData object.

        Parameters
        ----------
        data: AudioData
            Audio data from which the SpectroData should be computed.
        fft: ShortTimeFFT
            The ShortTimeFFT used to compute the spectrogram.
        nb_time_bins: int
            The maximum number of windows over which the audio will be split to perform
            Defaulted to 1920.

        Returns
        -------
        LTASData:
            The SpectroData object.

        """
        return cls(
            audio_data=data,
            fft=fft,
            begin=data.begin,
            end=data.end,
            nb_time_bins=nb_time_bins,
        )

    @staticmethod
    def get_ltas_fft(fft: ShortTimeFFT) -> ShortTimeFFT:
        """Return a ShortTimeFFT object optimized for computing LTAS.

        The overlap of the fft is forced set to 0, as the value of consecutive
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
