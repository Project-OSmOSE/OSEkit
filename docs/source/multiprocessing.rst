.. _multiprocessing:

Multiprocessing
---------------

**OSEkit** can be set up to distribute computations over multiple threads. Such computations are:

- All Core API ``BaseDataset`` child computations, e.g.:
    - :meth:`osekit.core_api.audio_dataset.AudioDataset.write`
    - :meth:`osekit.core_api.spectro_dataset.SpectroDataset.write`
    - :meth:`osekit.core_api.spectro_dataset.SpectroDataset.save_spectrogram`
    - :meth:`osekit.core_api.spectro_dataset.SpectroDataset.save_all`
- Long-Term Average Spectrum, computed thanks to the :meth:`osekit.core_api.ltas_data.LTASData.get_value` method.

To enable multiprocessing, simply set the ``osekit.config`` module up:

.. code-block:: python

    from osekit import config

    config.multiprocessing["is_active"] = True

Multiprocessing relies on `Python's multiprocessing process pools <https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool>`_.
The default number of worker processes that will be used if multiprocessing is active is defaulted to `os.process_cpu_count() <https://docs.python.org/3/library/os.html#os.process_cpu_count>`_.
This number can be adjusted thanks to the config module:

.. code-block:: python

    from osekit import config

    config.multiprocessing["is_active"] = True
    config.multiprocessing["nb_processes"] = 5 # Will limit the number of worker processes to 5.
