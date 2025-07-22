"""LTASDataset is a collection of LTASData objects.

LTASDataset is a collection of LTASData, with methods
that simplify repeated operations on the LTAS data.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from scipy.signal import ShortTimeFFT

from osekit.core_api.frequency_scale import Scale
from osekit.core_api.ltas_data import LTASData
from osekit.core_api.spectro_dataset import SpectroDataset

if TYPE_CHECKING:
    from osekit.core_api.audio_dataset import AudioDataset


class LTASDataset(SpectroDataset):
    """LTASDataset is a collection of LTASData objects.

    LTASDataset is a collection of LTASData, with methods
    that simplify repeated operations on the LTAS data.

    """

    __sentinel_value = object()
    _use_multiprocessing_on_data = True

    def __init__(
        self,
        data: list[LTASData],
        name: str | None = None,
        suffix: str = "",
        folder: Path | None = None,
        scale: Scale | None = None,
        v_lim: tuple[float, float] | None | object = __sentinel_value,
    ) -> None:
        """Initialize a LTASDataset."""
        super().__init__(
            data=data,
            name=name,
            suffix=suffix,
            folder=folder,
            scale=scale,
            v_lim=v_lim,
        )

    @property
    def nb_time_bins(self) -> int:
        """Number of time bins used to compute the LTAS data.."""
        return max(
            {d.nb_time_bins for d in self.data},
            key=[d.nb_time_bins for d in self.data].count,
        )

    def to_dict(self) -> dict:
        """Serialize a LTASDataset to a dictionary.

        Returns
        -------
        dict:
            The serialized dictionary representing the LTASDataset.

        """
        return super().to_dict() | {"nb_time_bins": self.nb_time_bins}

    @classmethod
    def from_dict(cls, dictionary: dict) -> SpectroDataset:
        """Deserialize a LTASDataset from a dictionary.

        Parameters
        ----------
        dictionary: dict
            The serialized dictionary representing the LTASDataset.

        Returns
        -------
        LTASDataset
            The deserialized LTASDataset.

        """
        return cls.from_spectro_dataset(
            SpectroDataset.from_dict(dictionary),
            nb_time_bins=dictionary["nb_time_bins"],
        )

    @classmethod
    def from_spectro_dataset(
        cls,
        sds: SpectroDataset,
        nb_time_bins: int | None = None,
    ) -> SpectroDataset:
        """Instantiate a LTAS Dataset from a SpectroDataset."""
        return cls(
            [
                LTASData.from_spectro_data(sd, nb_time_bins=nb_time_bins)
                for sd in sds.data
            ],
            folder=sds.folder,
            name=sds.name,
            suffix=sds.suffix,
            scale=sds.scale,
            v_lim=sds.v_lim,
        )

    @classmethod
    def from_audio_dataset(
        cls,
        audio_dataset: AudioDataset,
        fft: ShortTimeFFT,
        name: str | None = None,
        colormap: str | None = None,
        v_lim: tuple[float, float] | None = __sentinel_value,
        scale: Scale | None = None,
    ) -> SpectroDataset:
        """Return a SpectroDataset object from an AudioDataset object.

        The SpectroData is computed from the AudioData using the given fft.
        """
        return cls.from_spectro_dataset(
            super().from_audio_dataset(
                audio_dataset=audio_dataset,
                fft=fft,
                name=name,
                colormap=colormap,
                v_lim=v_lim,
                scale=scale,
            ),
        )
