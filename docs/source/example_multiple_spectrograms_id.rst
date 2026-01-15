.. _example_multiple_spectrograms_id:

Computing multiple spectrograms (ID files)
==========================================

In this example, we want to export spectrograms drawn from the sample audio dataset with the following requirements:

* Start timestamps of the audio files are unknown
* One single ``8 s``-long spectrogram should be exported per audio file
* Audio data are downsampled sampled at ``24 kHz`` before spectrograms are computed
* The DC component of the audio data is rejected before spectrograms are computed
* Exported spectrogram images should be named after the audio file IDs

The FFT used for computing the spectrograms will use a ``1024 samples``-long hamming window, with a ``128 samples``-long hop.

.. toctree::
   :maxdepth: 1

   example_multiple_spectrograms_id_core
   example_multiple_spectrograms_id_public
