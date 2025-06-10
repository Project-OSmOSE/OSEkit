Core API Usage
--------------

.. _coreapi_usage:

Audio File
^^^^^^^^^^

To turn an audio file into an **OSEkit** ``AudioFile`` (represented by the :class:`OSmOSE.core_api.audio_file.AudioFile` class), all you need is the path to the file and a begin timestamp.

Such begin timestamp can either be specified (as a :class:`pandas.Timestamp` instance), or parsed from the audio file name by specifing the `strptime format <https://strftime.org/>`_:

.. code-block:: python

    from OSmOSE.core_api.audio_file import AudioFile
    from pathlib import Path

    af = AudioFile(
        path = Path(r"foo/7189.230405154906.wav"),
        strptime_format = "%y%m%d%H%M%S",
        )

Audio Data
^^^^^^^^^^

The :class:`OSmOSE.core_api.audio_data.AudioData` class represent a chunk of audio taken between two specific timestamps from one or more `AudioFile` instances.

This class offers means of easily resample the data and access it on-demand (with an optimized I/O workflow to minimize the file openings/closings).

To create an ``AudioData`` from ``AudioFiles``, use the :meth:`OSmOSE.core_api.audio_data.AudioData.from_files` class method.

Example for an ``AudioData`` representing the first 2 seconds of the ``foo/7189.230405154906.wav`` audio file:

.. code-block:: python

    from pandas import Timestamp, Timedelta
    from OSmOSE.core_api.audio_data import AudioData

    ad = AudioData.from_files(
    files=[af],
    end=af.begin + Timedelta(seconds=2) # begin and end defaults to the files boundaries: this will take the 2 first seconds of the audio file.
    )

This will lead to the following *data* <-> *file* structure (see the :ref:`Data and Files <data_files>` section):

.. image:: _static/audio_data/ex1.svg

If the ``AudioData`` begin and end timestamps cover multiple ``AudioFile``, the corresponding ``AudioItem``.

For example, the `foo` folder here below contains 4 ``10 s``-long files, with a ``10 s`` gap between the 3rd and 4th files:

.. code-block::

    foo
    ├── 7189.230405154906.wav
    ├── 7189.230405154916.wav
    ├── 7189.230405154926.wav
    └── 7189.230405154946.wav

Let's create a single ``AudioData`` that covers the total duration of the 4 files:

.. code-block:: python

    afs = [
        AudioFile(f, strptime_format="%y%m%d%H%M%S")
        for f in foo.glob("*.wav")
        ]

    ad = AudioData.from_files(files=afs) # begin and end to default will cover the whole duration of the files

This will create one ``AudioItem`` per file, plus one **empty** ``AudioItem`` for the gap:

.. image:: _static/audio_data/ex2.svg

We can check that with code:

>>> print("\n".join("\n\t".join((f"Item {idx}", f"{f'Begin':<15}{str(item.begin):>20}", f"{f'End':<15}{str(item.end):>20}", f"{f'Is gap':<15}{"YES" if item.is_empty else "NO":>20}")) for idx,item in enumerate(sorted(ad.items, key=lambda i: i.begin))))
"""
Item 0
	Begin           2023-04-05 15:49:06
	End             2023-04-05 15:49:16
	Is gap                           NO
Item 1
	Begin           2023-04-05 15:49:16
	End             2023-04-05 15:49:26
	Is gap                           NO
Item 2
	Begin           2023-04-05 15:49:26
	End             2023-04-05 15:49:36
	Is gap                           NO
Item 3
	Begin           2023-04-05 15:49:36
	End             2023-04-05 15:49:46
	Is gap                          YES
Item 4
	Begin           2023-04-05 15:49:46
	End             2023-04-05 15:49:56
	Is gap                           NO
"""

