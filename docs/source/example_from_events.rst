.. _example_from_events:

Accessing audio data between arbitrary timestamps
=================================================

In this example, we want to access the audio data on specific time regions specified in an external file.

For the sake of the example, we will use the ``start_datetime`` and ``end_datetime`` columns of the ``_static/detections/aplose_results.csv`` file.

We will see how to plot spectrograms between each start and stop datetime from the csv file, using either the ``Core`` or the ``Public`` API.

.. toctree::
   :maxdepth: 1

   example_from_events_core
   example_from_events_public
