.. _usage:

🐳 Usage
========

In **OSEkit**, data is manipulated as **events** (which means something that has absolute **begin** and **end** timestamps).
All data classes from the API inherit from the :class:`osekit.core_api.event.Event` class, allowing for easy timestamp-based manipulations of different types of data.

The package combines two APIs:

====================

.. topic:: :ref:`Public API <publicapi_usage>`

    Designed as an application that contains high-level instructions for sorting datasets and processing/exporting spectrograms from large audio datasets.

====================

.. topic:: :ref:`Core API <coreapi_usage>`

    It is the backend of the :ref:`Public API <publicapi_usage>` but it can also be used on its own to precisely manipulate audio data and process it, notably to output spectrograms.

.. toctree::
    :hidden:

    publicapi_usage
    coreapi_home
    multiprocessing
    jobs
