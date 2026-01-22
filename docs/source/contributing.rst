.. _contributing:

🐳 Contributing
===============

This guide will take you to the steps to follow in order to contribute to **OSEkit**.

There are many ways to contribute to this project, including:

- **Starring** the `OSEkit GitHub page <https://github.com/Project-OSmOSE/OSEkit>`_ to show the world that you use it
- **Referencing OSEkit** in your articles
- **Participating** in the `OSEkit GitHub <https://github.com/Project-OSmOSE/OSEkit>`_ by:
    - **Reporting** difficulties you encounter when using the package in new `issues <https://github.com/Project-OSmOSE/OSEkit/issues>`_
    - **Suggesting** functionalities that would come in handy in **OSEkit** in new `issues <https://github.com/Project-OSmOSE/OSEkit/issues>`_
    - **Reviewing** existing `pull requests <https://github.com/Project-OSmOSE/OSEkit/pulls>`_
    - **Authoring** new `pull requests <https://github.com/Project-OSmOSE/OSEkit/pulls>`_ to:
        - **Add** new cool functionalities
        - **Fix** things that don't work exactly the way they should
        - **Improve** the documentation

🐬 GitHub contributor workflow
------------------------------

Contributions to the **OSEkit** codebase are done with **GitHub**.

If you're new to this tool, we recommand taking a look at some resources to get you started,
such as the `Introduction to Github interactive course <https://github.com/skills/introduction-to-github>`_.

If you want to dig in **OSEkit**'s code to do anything you'd like (adding functionalities, fixing bugs, working on the documentation...),
you'll have to submit a new **pull request**.

There are lots of great tutorials out there that'll guide you in the process of submitting a pull request.
We recommand you follow one of those, e.g. `DigitalOcean's How To Create a Pull Request on GitHub <https://www.digitalocean.com/community/tutorials/how-to-create-a-pull-request-on-github>`_.

We use `uv <https://docs.astral.sh/uv/>`_ to manage the project, and suggest that you install the project
following the instructions in the :ref:`Install from git <from-git>` section of the documentation.

🐬 Specific OSEkit workflow
---------------------------

We use a `GitHub Action <https://github.com/Project-OSmOSE/OSEkit/blob/main/.github/workflows/github_ci.yml>`_ to
validate the code that is being pushed to our repo.

This action validates code that:
    * Has been formatted thanks to the `Ruff formatter <https://docs.astral.sh/ruff/formatter/>`_
    * Passes all the tests of the `pytest test suite <https://github.com/Project-OSmOSE/OSEkit/tree/main/tests>`_
    * Does not reduce the `coverage <https://coveralls.io/github/Project-OSmOSE/OSEkit?branch=version-bump>`_
      of the pytest test suite

Here's what you should do if any of these three checks isn't validated anymore:

😔 The code isn't properly formatted according to Ruff
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the ruff formatter in your repo (e.g. with ``uv`` if you used it to sync the project):

.. code-block:: bash

    uv run ruff format .

You might have to push the formatting in a new commit, and the CI should now pass! 🎉

😔 All tests don't pass
^^^^^^^^^^^^^^^^^^^^^^^

Run pytest in the ``tests`` module to take a look at the tests that don't pass anymore:

.. code-block:: bash

    uv run pytest .\tests\

This should point you to the part of the codebase that has been altered by your code modifications, and help
you fix it.

😔 The coverage has reduced and coverall isn't happy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This probably means that you have added lines to the codebase that are not tested in the pytest test suite.

The easiest way to locate these lines is to run the test suite with coverage
(example in `Pycharm <https://www.jetbrains.com/help/pycharm/running-test-with-coverage.html>`_
or in `VSCode <https://code.visualstudio.com/docs/python/testing#_run-tests-with-coverage>`_).
Then, you might have to write new tests that check that exhaustively test your new features! (Refer to
the `pytest documentation <https://docs.pytest.org/en/stable/>`_ if needed).
