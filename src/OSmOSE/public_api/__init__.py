from enum import Flag, auto


class Analysis(Flag):
    """Enum of flags that should be use to specify the type of analysis to run.

    AUDIO:
        Will add an AudioDataset to the datasets and write the reshaped audio files
        to disk.
        The new AudioDataset will be linked to the reshaped audio files rather than to
        the original files.
    MATRIX:
        Will write the npz SpectroFiles to disk and link the SpectroDataset to
        these files.
    SPECTROGRAM:
        Will export the spectrogram png images.

    Multiple flags can be enabled thanks to the logical or | operator:
    Analysis.AUDIO | Analysis.SPECTROGRAM will export both audio files and
    spectrogram images.

    >>> # Exporting both the reshaped audio and the spectrograms
    >>> # (without the npz matrices):
    >>> export = Analysis.AUDIO | Analysis.SPECTROGRAM
    >>> Analysis.AUDIO in export
    True
    >>> Analysis.SPECTROGRAM in export
    True
    >>> Analysis.MATRIX in export
    False

    """

    AUDIO = auto()
    MATRIX = auto()
    SPECTROGRAM = auto()
