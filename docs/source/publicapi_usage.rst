.. _publicapi_usage:

Public API
----------

This API provides tools for working on large sets of audio data.

Basically, the whole point of **OSEkit**'s Public API is to export large amounts of spectrograms and/or reshaped audio files with no consideration of the original format of the audio files.

The :class:`osekit.public_api.dataset.Dataset` class is the cornerstone of **OSEkit**'s Public API.

.. _build:

Building a ``Dataset``
^^^^^^^^^^^^^^^^^^^^^^

At first, A ``Dataset`` is built from a raw folder containing the audio files to be processed.
For example, this folder containing 4 audio files plus some extra files:

.. code-block::

    7181.230205154906.wav
    7181.230205164906.wav
    7181.230205174906.wav
    7181.230205194906.wav
    foo
    ├── bar.zip
    └── 7181.wav
    bar.txt

Only the folder path and strptime format are required to initialize the ``Dataset``.

Extra parameters allow for e.g. localizing the files in a specific timezone or accounting for the measurement chain to link the raw wav data to the measured acoustic presure.
The complete list of extra parameters is provided in the :class:`osekit.public_api.dataset.Dataset` documentation.

.. code-block:: python

    from osekit.public_api.dataset import Dataset
    from pathlib import Path

    dataset = Dataset(
        folder=Path(r"...\dataset_folder"),
        strptime_format="%y%m%d%H%M%S" # Must match the strptime format of your audio files
    )

If you don't know (or don't care to parse) the start timestamps of your audio files, you can specify the ``first_file_begin`` parameter.
Then, the first valid audio file found in the folder will be considered as starting at this timestamp,
and each next valid audio file will be considered as starting directly after the end of the previous one:

.. code-block:: python

    from osekit.public_api.dataset import Dataset
    from pathlib import Path

    dataset = Dataset(
        folder=Path(r"...\dataset_folder"),
        strptime_format=None # Must match the strptime format of your audio files,
        first_file_begin=Timestamp("2020-01-01 00:00:00") # Will mark the start of your audio files
    )

Once this is done, the ``Dataset`` can be built using the :meth:`osekit.public_api.dataset.Dataset.build` method.

.. code-block:: python

    dataset.build()

The folder is now organized in the following fashion:

.. code-block::

    data
    └── audio
        └── original
            ├── 7181.230205154906.wav
            ├── 7181.230205164906.wav
            ├── 7181.230205174906.wav
            ├── 7181.230205194906.wav
            └── original.json
    other
    ├── foo
    │   ├── bar.zip
    │   └── 7181.wav
    └── bar.txt
    dataset.json

The **original audio files** have been turned into a :class:`osekit.core_api.audio_dataset.AudioDataset`.
In this ``AudioDataset``, one :class:`osekit.core_api.audio_data.AudioData` has been created per original audio file.
Additionally, both this Core API ``Audiodataset`` and the Public API ``Dataset`` have been serialized
into the ``original.json`` and ``dataset.json`` files, respectively.

Building a dataset from specific files
""""""""""""""""""""""""""""""""""""""

It is possible to pass a specific collection of files to build within the dataset.

This is done thanks to the :meth:`osekit.public_api.dataset.Dataset.build_from_files` method.
The collection of files passed as the ``files`` parameter will be either moved or copied (depending on
the ``move_files`` parameter) within ``Dataset.folder`` before the build is done. If ``Dataset.folder`` does
not exist, it will be created before the files are moved:

.. code-block:: python

    from pathlib import Path
    from osekit.public_api.dataset import Dataset

    # Pick the files you want to include to the dataset
    files = (
        r"cool\stuff\2007-11-05_00-01-00.wav",
        r"cool\things\2007-11-05_00-03-00.wav",
        r"cool\things\2007-11-05_00-05-00.wav"
    )

    # Set the DESTINATION folder of the dataset in the folder parameter
    dataset = Dataset(
        folder = Path(r"cool\odd_hours"),
        strptime_format="%Y-%m-%d_%H-%M-%S",
    )

    dataset.build_from_files(
        files=files,
        move_files=False, # Copy files rather than moving them
    )

Running an ``Analysis``
^^^^^^^^^^^^^^^^^^^^^^^

In **OSEkit**, **Analyses** are run with the :meth:`osekit.public_api.dataset.Dataset.run_analysis` method to process and export spectrogram images, spectrum matrices and audio files from original audio files.

.. note::

    **OSEkit** makes it easy to **reshape** the original audio: it is not bound to the original files, and can freely be reshaped in audio data of **any duration and sample rate**.

The analysis parameters are described by a :class:`osekit.public_api.analysis.Analysis` instance passed as a parameter to this method.

Analysis Type
"""""""""""""

The ``analysis_type`` parameter passed to the initializer is a :class:`osekit.public_api.analysis.AnalysisType` instance that defines the analysis output(s):

.. list-table:: Analysis Types
   :widths: 40 60
   :header-rows: 1

   * - Flag
     - Output
   * - ``AnalysisType.AUDIO``
     - Reshaped audio files
   * - ``AnalysisType.MATRIX``
     - `STFT <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.stft.html#scipy.signal.ShortTimeFFT.stft>`_ NPZ matrix files
   * - ``AnalysisType.WELCH``
     - `Welch <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html>`_ NPZ files
   * - ``AnalysisType.SPECTROGRAM``
     - PNG spectrogram images

Multiple outputs can be selected thanks to a logical or ``|`` separator.

For example, if an analysis aims at exporting both the reshaped audio files and the corresponding spectrograms:

.. code-block:: python

    from osekit.public_api.analysis import AnalysisType
    analysis_type = AnalysisType.AUDIO | AnalysisType.SPECTROGRAM


Analysis Parameters
"""""""""""""""""""

The remaining parameters of the analysis (begin and end **Timestamps**, duration and sample rate of the reshaped data...) are described in the :class:`osekit.public_api.analysis.Analysis` initializer docstring.

.. note::

   If the ``Analysis`` contains spectral computations (either ``AnalysisType.MATRIX``, ``AnalysisType.SPECTROGRAM`` or ``AnalysisType.WELCH`` is in ``analysis_type``), a `scipy ShortTimeFFT instance <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html#scipy.signal.ShortTimeFFT>`_ should be passed to the ``Analysis`` initializer.

.. _editing_analysis:

Checking/Editing the analysis
"""""""""""""""""""""""""""""

If you want to take a peek at what the analysis output will be before actually running it, the :meth:`osekit.public_api.dataset.Dataset.get_analysis_audiodataset` and :meth:`osekit.public_api.dataset.Dataset.get_analysis_spectrodataset` methods
return a :class:`osekit.core_api.audio_dataset.AudioDataset` and a :class:`osekit.core_api.spectro_dataset.SpectroDataset` instance, respectively.

The returned ``AudioDataset`` can be edited at will and passed as a parameter later on when the analysis is run:

.. code-block:: python

    ads = dataset.get_analysis_audiodataset(analysis=analysis)

    # Filtering out the AudioData that are not linked to any audio file:
    ads.data = [ad for ad in ads.data if not ad.is_empty]

The returned ``SpectroDataset`` can be used e.g. to plot sample spectrograms prior to the analysis:

.. code-block:: python

    import matplotlib.pyplot as plt

    sds = dataset.get_analysis_spectrodataset(analysis=analysis, audio_dataset=ads) # audio_dataset is optional: here, the sds will match the edited ads (with no empty data)

    # Computing/plotting the 100th SpectroData from the analysis
    sds.data[100].plot()
    plt.show()


Running the analysis
""""""""""""""""""""

To run the ``Analysis``, simply execute the :meth:`osekit.public_api.dataset.Dataset.run_analysis` method:

.. code-block:: python

    dataset.run_analysis(analysis=analysis) # And that's it!

If you edited the analysis ``AudioDataset`` as explained in the :ref:`Checking/Editing the analysis <editing_analysis>` section, you can specify the edited ``AudioDataset`` on which the analysis will be run:

.. code-block:: python

    dataset.run_analysis(analysis=analysis, audio_dataset=ads)


.. note::

    Any ``AnalysisType.Spectrogram`` can be computed as a :ref:`Long Term Average Spectrum <ltas>` by setting the ``nb_ltas_time_bins`` parameter to an integer value.

    When the field is at the default ``None`` value, spectrograms are computed regularly:

    .. code-block:: python

        # limit spectrograms to 3000 averaged time bins:
        dataset.run_analysis(analysis=analysis, audio_dataset=ads, nb_ltas_time_bins=3000)

Simple Example: Reshaping audio
"""""""""""""""""""""""""""""""

Regardless of the format(s) of the original audio files (as always in **OSEkit**), let's say we just want to resample our original audio data at ``48 kHz`` and export it as ``10 s``-long audio files.

The corresponding ``Analysis`` is the following:

.. code-block:: python

    from osekit.public_api.analysis import Analysis, AnalysisType
    from pandas import Timedelta

    analysis = Analysis(
        analysis_type = AnalysisType.AUDIO, # We just want to export the reshaped audio files
        data_duration=Timedelta("10s"), # Duration of the new audio files
        sample_rate=48_000, # Sample rate of the new audio files
        name="cool_reshape", # You can name the analysis, or keep the default name.
    )

    dataset.run_analysis(analysis=analysis) # And that's it!

.. _output_1:

Output 1
""""""""

Once the analysis is run, a :class:`osekit.core_api.audio_dataset.AudioDataset` instance named ``cool_reshape`` has been created and added to the dataset's :attr:`osekit.public_api.dataset.Dataset.datasets` field.

The dataset folder now looks like this:

.. code-block::

    data
    └── audio
        ├── original
        │   ├── 7181.230205154906.wav
        │   ├── 7181.230205164906.wav
        │   ├── 7181.230205174906.wav
        │   ├── 7181.230205194906.wav
        │   └── original.json
        └── cool_reshape
            ├── 2023_04_05_15_49_06_000000.wav
            ├── 2023_04_05_15_49_16_000000.wav
            ├── 2023_04_05_15_49_26_000000.wav
            ├── ...
            ├── 2023_04_05_20_48_46_000000.wav
            ├── 2023_04_05_20_48_56_000000.wav
            └── cool_reshape.json
    other
    ├── foo
    │   └── bar.zip
    └── bar.txt
    dataset.json

The ``cool_reshape`` folder has been created, containing the freshly created ``10 s``-long, ``48 kHz``-sampled audio files.

.. note::

    The ``cool_reshape`` folder also contains a ``cool_reshape.json`` serialized version of the ``cool_reshape`` ``AudioDataset``, which will be used for deserializing the ``dataset.json`` file in the dataset folder root.

Example: full analysis
""""""""""""""""""""""

Let's now say we want to export audio, spectrum matrices and spectrograms with the following parameters:

.. list-table:: Example analysis parameters
   :widths: 40 60
   :header-rows: 1

   * - Parameter
     - Value
   * - Begin
     - 00:30:00 after the begin of the original audio files
   * - End
     - 01:30:00 after the begin of the original audio files
   * - Data duration
     - ``10 s``
   * - Sample rate
     - ``48 kHz``
   * - Audio data normalization
     - ``dc_reject`` (removes the audio DC component)
   * - FFT
     - ``hamming window``, ``1024 points``, ``40% overlap``

Let's first instantiate the ``ShortTimeFFT`` since we want to run a spectral analysis:

.. code-block:: python

    from scipy.signal import ShortTimeFFT
    from scipy.signal.windows import hamming

    sft = ShortTimeFFT(
        win=hamming(1_024), # Window shape,
        hop=round(1_024*(1-.4)), # 40% overlap
        fs=48_000,
        scale_to="magnitude"
    )

Then we are all set for running the analysis:

.. code-block:: python

    from osekit.public_api.analysis import Analysis, AnalysisType
    from pandas import Timedelta

    analysis = Analysis(
        analysis_type = AnalysisType.AUDIO | AnalysisType.MATRIX | AnalysisType.WELCH | AnalysisType.SPECTROGRAM, # Full analysis : audio files, spectrum matrices and spectrograms will be exported.
        begin=dataset.origin_dataset.begin + Timedelta(minutes=30), # 30m after the begin of the original dataset
        end=dataset.origin_dataset.begin + Timedelta(hours=1.5), # 1h30 after the begin of the original dataset
        data_duration=Timedelta("10s"), # Duration of the output data
        sample_rate=48_000, # Sample rate of the output data
        normalization="dc_reject",
        name="full_analysis", # You can name the analysis, or keep the default name.
        fft=sft, # The FFT parameters
    )

    dataset.run_analysis(analysis=analysis) # And that's it!

Output 2
""""""""

Since the analysis contains both ``AnalysisType.AUDIO`` and spectral analysis types, two core API datasets were created and added to the dataset's :attr:`osekit.public_api.dataset.Dataset.datasets` field:

* A :class:`osekit.core_api.audio_dataset.AudioDataset` named ``full_analysis_audio`` (with the *_audio* suffix)
* A :class:`osekit.core_api.spectro_dataset.SpectroDataset` named ``full_analysis``

The dataset folder now looks like this (the output from the first example was removed for convenience):

.. code-block::

    data
    └── audio
        ├── original
        │   ├── 7181.230205154906.wav
        │   ├── 7181.230205164906.wav
        │   ├── 7181.230205174906.wav
        │   ├── 7181.230205194906.wav
        │   └── original.json
        └── full_analysis_audio
            ├── 2023_04_05_16_19_06_000000.wav
            ├── 2023_04_05_16_19_16_000000.wav
            ├── 2023_04_05_16_19_26_000000.wav
            ├── ...
            ├── 2023_04_05_17_18_46_000000.wav
            ├── 2023_04_05_17_18_56_000000.wav
            └── full_analysis_audio.json
    processed
    └── full_analysis
        ├── spectrogram
        │   ├── 2023_04_05_16_19_06_000000.png
        │   ├── 2023_04_05_16_19_16_000000.png
        │   ├── 2023_04_05_16_19_26_000000.png
        │   ├── ...
        │   ├── 2023_04_05_17_18_46_000000.png
        │   └── 2023_04_05_17_18_56_000000.png
        ├── matrix
        │   ├── 2023_04_05_16_19_06_000000.npz
        │   ├── 2023_04_05_16_19_16_000000.npz
        │   ├── 2023_04_05_16_19_26_000000.npz
        │   ├── ...
        │   ├── 2023_04_05_17_18_46_000000.npz
        │   └── 2023_04_05_17_18_56_000000.npz
        ├── welch
        │   └── 2023_04_05_16_19_06_000000.npz
        └── full_analysis.json
    other
    ├── foo
    │   └── bar.zip
    └── bar.txt
    dataset.json

As in :ref:`the output of example 1 <output_1>`, a ``full_analysis_audio`` folder was created, containing the reshaped audio files.

Additionally, the fresh ``processed`` folder contains the output spectrograms and NPZ matrices, along with the ``full_analysis.json`` serialized :class:`osekit.core_api.spectro_dataset.SpectroDataset`.


Recovering a ``Dataset``
^^^^^^^^^^^^^^^^^^^^^^^^

The ``dataset.json`` file in the root dataset folder can be used to deserialize a :class:`osekit.public_api.dataset.Dataset` object thanks to the :meth:`osekit.public_api.dataset.Dataset.from_json` method:

.. code-block:: python

    from pathlib import Path
    from osekit.public_api.dataset import Dataset

    json_file = Path(r"../dataset.json")
    dataset = Dataset.from_json(json_file) # That's it!


Resetting a ``Dataset``
^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

    Calling this method is irreversible

The :meth:`osekit.public_api.dataset.Dataset.reset` method **resets the dataset's folder** to its initial state.
All exported analyses ans json files will be removed, and the folder will be back to its state :ref:`before building the dataset <build>`.
