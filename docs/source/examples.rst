🐳 Examples
===========

.. _examples:

This section gathers **OSEkit** jupyter notebooks that complete typical tasks.

Example tasks will be completed with both the :ref:`Public <publicapi_usage>` and :ref:`Core API <coreapi_usage>` (see the :ref:`usage <usage>` section
for more info about the differences between the two APIs).

The examples use a small set of audio files that can be found in the **OSEkit** repository, under ``docs/source/_static/sample_audio``.
This sample dataset is made of 10 ``10 s``-long audio files sampled at ``48 kHz``. The 5 first and 5 last audio files are consecutive
(there is no recording gap between them), but both groups of 5 consecutive files are spaced by a ``30 s``-long recording gap.

===========

.. topic:: :doc:`Reshape one file <example_reshaping_one_file>`

    Extract a specific time period and/or resample an audio file of given duration and sample rate.

===========

.. topic:: :doc:`Reshape multiple files <example_reshaping_multiple_files>`

    Same example as the previous one, but at a larger scale: reshape and export multiples files from an audio folder.

===========

.. topic:: :doc:`Compute/plot a spectrogram <example_spectrogram>`

    | Compute the spectrum matrix and/or Power Spectral Density estimates of an ``AudioData``.
    | Export the matrices and/or plot a spectrogram.

===========

.. topic:: :doc:`Compute/plot multiple spectrograms <example_multiple_spectrograms>`

    Same example as the previous one, but at a larger scale: compute multiple spectrograms from an audio folder.

===========

.. topic:: :doc:`Compute/plot a LTAS <example_multiple_spectrograms>`

    Compute, plot and export a **L**\ ong-\ **T**\ erm **A**\ verage **S**\ pectrum (**LTAS**).

.. toctree ::
    :hidden:

    example_reshaping_one_file
    example_reshaping_multiple_files
    example_spectrogram
    example_multiple_spectrograms
    example_ltas