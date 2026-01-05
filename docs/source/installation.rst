.. _installation:

🐳 Installation
===============

To use **OSEkit**, you first need to install it using one of the following:

With pip
--------

To install **OSEkit** using pip, simply run ::

    pip install osekit

Alternatively, you can install it with a wheel file for any of our releases:

* Get the ``.whl`` wheel file from `our github repository <https://github.com/Project-OSmOSE/OSEkit/releases/>`_.
* Install it in a virtual environment using pip: ::

    pip install osmose-x.x.x.-py3-none-any.whl

.. _from-git:

From Git
--------

You can install **OSEkit** from `git <https://git-scm.com/>`_ if you want to have an editable version of it. This can be usefull for contributing to **OSEkit**, or to be able to easily update to the latest version.

The package will be installed with `uv <https://docs.astral.sh/uv/>`_, which is a Python package and project manager. Please refer to the `uv documentation <https://docs.astral.sh/uv/getting-started/installation/>`_ for installing uv.

To download the repository, simply clone it from git: ::

    git clone https://github.com/Project-OSmOSE/OSEkit.git

Then, you can pull the latest update: ::

    git pull origin main

You can now install the package using uv from the cloned repository: ::

    uv sync

This will create a virtual environment within your cloned repository and install all required dependencies for using and contributing to **OSEkit**.

MSEED (MiniSEED) support
------------------------

**OSEkit** is primarily designed for standard audio formats (WAV, FLAC, OGG, etc.) used in bioacoustics workflows.

Support for **MiniSEED** (.mseed) files is experimental and relies on the `ObsPy <https://docs.obspy.org/>`_ third-party library.

**MiniSEED** is a seismological data format and may contain:

* multiple traces
* gaps or overlaps in time
* inconsistent metadata

For these reasons, **MiniSEED** support in **OSEkit** is provided on a best-effort basis.

Installation with the mseed optional dependency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With pip
""""""""

::

    pip install osekit[mseed]


From Git
""""""""

Follow the same instructions as in the :ref:`From Git <from-git>` section above, but replace the uv command with:

::

    uv sync --extra mseed

.. warning::

    Installing ObsPy on Windows might be a stairway to hell.
    If installation fails, we recommand either:
    * Using Linux or `WSL <https://learn.microsoft.com/en-us/windows/wsl/install/>`_
    * Converting MiniSEED files to WAV/FLAC prior to processing
