.. _aplose:

Working with APLOSE results
---------------------------

`APLOSE <https://osmose.ifremer.fr/app//>`_ is **OSmOSE**'s web-based annotation platform.

**APLOSE** campaigns `results <https://project-osmose.github.io/APLOSE/user/annotation-campaign/phase-progress-result/>`_ are provided as csv files
that can be parsed in **OSEkit** as :class:`osekit.core.detection.Detection` instances.

Loading an APLOSE results file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Detections`` can be extracted from **APLOSE** results files thanks to the :meth:`osekit.core.detection.Detection.from_csv` method:

.. code-block:: python

    from pathlib import Path
    from osekit.core.detection import Detection

    detections = Detection.from_csv(csv=Path(r"_static/detections/aplose_results.csv"))

Detection / Audio interaction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`osekit.core.detection.Detection` class inherits from the :class:`osekit.core.event.Event` class: detections can easily be used to filter audio and spectro data:

.. code-block:: python

    from osekit.core.spectro_dataset import SpectroDataset

    detection = Detection(...) # Generally Detection.from_csv(...)[i]
    spectro_dataset = SpectroDataset(...)

    # Find all SpectroData in which detection appear:
    positive_spectrograms = SpectroDataset([sd for sd in spectro_dataset.data if sd.overlaps(detection)])

Plotting a detection
^^^^^^^^^^^^^^^^^^^^

Detection boxes can be plotted on spectrograms thanks to the :method:`osekit.core.detection.Detection.to_rectangle` method:

.. code-block:: python

    import matplotlib.pyplot as plt
    from osekit.core.spectro_data import SpectroData
    from osekit.core.detection import Detection

    sd = SpectroData(...)
    detection = Detection(...)

    fig, axs = plt.subplots()

    # Plot the spectrogram
    sd.plot(ax=ax)

    # Get a rectangle from the detection
    rectangle = detection.to_rectangle(fill = False)

    # Draw the detection
    ax.add_patch(rectangle)

    # Show the spectrogram
    plt.show()
