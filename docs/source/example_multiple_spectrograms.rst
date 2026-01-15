.. _example_multiple_spectrograms:

Computing multiple spectrograms (timestamped files)
===================================================

In this example, we want to export spectrograms drawn from the sample audio dataset with the following requirements:

* First spectrogram starts at ``2022-09-25 22:35:15``
* Last spectrogram ends at ``2022-09-25 22:36:25``
* Spectrograms represent ``5 s``-long audio data
* Audio data are downsampled sampled at ``24 kHz`` before spectrograms are computed
* The DC component of the audio data is rejected before spectrograms are computed
* Spectrograms that are in the gap between recordings should be skipped

The FFT used for computing the spectrograms will use a ``1024 samples``-long hamming window, with a ``128 samples``-long hop.

Moreoever, we will:

* Export the matrices that represent the spectrogram values in a NPZ file
* Compute a power spectrum density estimate of the audio data (welch)

.. toctree::
   :maxdepth: 1

   example_multiple_spectrograms_core
   example_multiple_spectrograms_public
