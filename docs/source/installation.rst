🐳 Installation
===============

.. _installation:

To use **OSEkit**, you first need to install it using one of the following:

From the wheel
--------------

If you just want to use **OSEkit** in your project without editing it, the most straightforward way to do so is to:

* Get the ``.whl`` wheel file from our `latest release <https://github.com/Project-OSmOSE/OSEkit/releases/>`_.
* Install it in a virtual environment using pip: ::

    pip install osmose-0.2.5-py3-none-any.whl


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
