from pathlib import Path

from osekit.core_api.ltas_dataset import LTASDataset
from osekit.core_api.spectro_dataset import SpectroDataset


def deserialize_spectro_or_ltas_dataset(path: Path) -> SpectroDataset | LTASDataset:
    """Return a ``LTASDataset`` or a ``SpectroDataset`` from the specified json file.

    Parameters
    ----------
    path: Path
        Path to the json file.

    Returns
    -------
    SpectroDataset | LTASDataset
        The deserialized ``LTASDataset`` if ``nb_time_bins`` is set to an integer, else
        the deserialized ``SpectroDataset``.

    """
    try:
        return LTASDataset.from_json(file=path)
    except KeyError:
        return SpectroDataset.from_json(file=path)
