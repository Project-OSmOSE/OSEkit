🐳 Installation
===============

.. _installation:

To use **OSEkit**, you first need to install it using one of the following:

With pip
--------

To install **OSEkit** using pip, simply run ::

    pip install osekit

Alternatively, you can install it with a wheel file for any of our releases:

* Get the ``.whl`` wheel file from `our github repository <https://github.com/Project-OSmOSE/OSEkit/releases/>`_.
* Install it in a virtual environment using pip: ::

    pip install osmose-x.x.x.-py3-none-any.whl


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
