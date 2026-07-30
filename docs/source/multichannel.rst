.. _multichannel:

Working with multichannel audio files
-------------------------------------

The audio classes in **OSEkit** allow multichannel audio handling, e.g. the :meth:`osekit.core.audio_data.AudioData.get_value` method returns a ``samples x channels`` matrix.

The :attr:`osekit.core.audio_file.AudioFile.channels` property depicts the number of channels of a given ``AudioFile``:

.. code-block:: python

    from pathlib import Path
    from osekit.core.audio_file import AudioFile

    af = AudioFile(...)
    print(af.channels)

    >>> 3

The :attr:`osekit.core.audio_data.AudioData.channels` property relates to the list of channels concerned by this audio data:

.. code-block:: python

    from osekit.core.audio_data import AudioData

    ad = AudioData.from_files(files=[af])
    ad.channels = [0,2] # We want to keep only channels 0 and 2
    print(ad.get_value().shape[1])

    >>> 2 # One value array per channel

Finally, a ``SpectroData`` that has a linked ``AudioData`` targets a **specific channel** of this ``AudioData``:

.. code-block:: python

    from osekit.core.spectro_data import SpectroData
    from scipy.signal import ShortTimeFFT

    sd = SpectroData.from_audio_data(data=ad, ...)
    sd.audio_channel = 2 # The spectrum will be computed on the channel 2 of the file

.. important::

    The :attr:`osekit.core.spectro_data.SpectroData.audio_channel` value refers to the index of the channel of the **file**.

    If, as in the example above, the ``AudioFile`` has 3 channels ``[0,1,2]``, the ``AudioData`` targets channels ``[0,2]`` and the ``SpectroData`` targets the channel ``2``, the spectrum will be computed on the **third channel** of the file (with index ``2``).
