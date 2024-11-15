from pathlib import Path

from OSmOSE.config import AUDIO_METADATA, SUPPORTED_AUDIO_FORMAT


def is_supported_audio_format(filename: Path) -> bool:
    """Check if a given file is a supported audio file based on its extension.

    Parameters
    ----------
    filename : Path
        The path to the file to be checked.

    Returns
    -------
    bool
        True if the file has an extension that matches a supported audio format,
        False otherwise.

    """
    return filename.suffix.lower() in SUPPORTED_AUDIO_FORMAT


def get_all_audio_files(directory: Path) -> list[Path]:
    """Retrieve all supported audio files from a given directory.

    Parameters
    ----------
    file_path : Path
        The path to the directory to search for audio files

    Returns
    -------
    list[Path]
        A list of `Path` objects corresponding to the supported audio files
        found in the directory.

    """
    return sorted(
        file
        for extension in SUPPORTED_AUDIO_FORMAT
        for file in directory.glob(f"*{extension}")
    )


def get_audio_metadata(audio_file: Path) -> dict:
    """Read metadata from the audio file.

    Parameters
    ----------
    audio_file: pathlib.Path
        The path to the audio file.

    Returns
    -------
    dict:
        A dictionary containing the metadata of the audio file.
        The metadata list is grabbed from OSmOSE.config.AUDIO_METADATA.

    """
    return {key: f(audio_file) for key, f in AUDIO_METADATA.items()}
