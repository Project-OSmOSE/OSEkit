.. _example_ltas:

Computing a Long-Term Average Spectrum
======================================

In this example, we want to compute a :ref:`LTAS <ltas>` of audio data over a long period of time.

For the sake of the example, the audio will not actually span over a long time period, but LTASes are meant
to be used in such case.

This LTAS will:

* Start at the begin of the first audio file
* End at the end of the last audio file
* Be downsampled at ``24 kHz``
* Have its DC component removed

| The FFT used for computing the spectrograms will use a ``1024 samples``-long hamming window.
| The ``hop`` of LTAS ``ShortTimeFFT`` objects is forced to the size of the window (no overlap).

.. toctree::
   :maxdepth: 1

   example_ltas_core
   example_ltas_public
