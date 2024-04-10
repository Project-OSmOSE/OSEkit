
=================
Going further 
=================

Installation with github and poetry
===================================

.. note::

	We have made a :ref:`tutorial on Github <Github>` if you are starting with it


In this section we give you the instructions to install our toolkit from github using the `poetry <https://python-poetry.org/>`_ tool:


1. First clone our Github project repository

.. code:: bash

	git clone https://github.com/Project-OSmOSE/osmose-toolkit.git


2. Create and activate a Conda virtual environment in python 3.10 with the name you want (here osmose_dev)

.. code:: bash

	conda create --name osmose_dev python=3.10 -y
	conda activate osmose_dev

3. Install poetry and use it to install our toolkit

.. code:: bash

	conda install poetry
	poetry install

.. note::
	`poetry install` will use the file `pyproject.toml` so be sure to execute this command from its directory. Besides, if `poetry install` fails with the message ``ImportError: cannot import name 'PyProjectException'``, run this line instead:

	.. code:: bash

		conda install -c conda-forge poetry==1.3.2


The toolkit is now installed in editable mode! You can call it just like any regular python package as long as you are in its conda environment

.. code:: python

	import OSmOSE as osm
	dataset = osm.Dataset()


Note that on the contrary of the installation procedure given in :ref:`Getting started`, the OSmOSE toolkit is now installed in editable mode, meaning that any change made to the package's file will be reflected immediately on the environment, without needing to reload it. 






Contribute
=================


Here is the general workflow for contributing to our project:


1. Install our toolkit in local following instructions in :ref:`Installation with github and poetry` ;

2. To be able to push/pull codes to/from github, add remote to your origin folder

.. code:: bash
		
	git remote add origin git@github.com:Project-OSmOSE/osmose-toolkit.git

3. Review our Github `issues <https://github.com/Project-OSmOSE/osmose-toolkit/issues>`_ : find an open issue to tackle and /or ask if you can help write a new feature (follow standard procedures such as: short description, assignees, status, label...) ;


4. Develop and validate your contribution in local using a branch checkout from our main branch and dedicated to your issue. As is often the case, no pushes are accepted on our main branch. Note that after creating your issue, under Development in the right panel, you can directly "Create a branch for this issue or link a pull request." that willl provide you a command of the form

.. code:: bash
			
	git fetch origin
	git checkout 112-create-a-double-fs-spectrogram-concatenated-horizontally


5. (Optional) If intended to be deployed on Datarmor, you will have to also validate your contribution on it (see section :ref:`On Datarmor` for details) ;

6. You can suggest changing in the version of the toolkit through the parameter `version` in `pyproject.toml`, depending on the importance of the changes: 
	- small patch : change patch version (x.x.X)
	- non breaking changes : change minor version (x.X.x)
	- breaking changes : change major version (X.x.x)

7. Push your branch to our Github project and create a pull-request so it can be properly peer-reviewed ;

8. Use `poetry build` and upload generated files (present locally in the folder `~/dist`) in the corresponding package version on github `here <https://github.com/Project-OSmOSE/osmose-toolkit/releases/tag/v0.1.0>`_ 


.. note::

	Beyond this workflow, you can find in this `guide <https://opensource.guide/how-to-contribute/>`_ a more exhaustive list of possible contributions for an open source project, do not hesite to make us original propositions. This `document <https://www.dataschool.io/how-to-contribute-on-github/>`_ provides a more detailed step-by-step guide on how to make contributinons on github, and this `one <https://developer.mozilla.org/en-US/docs/MDN/Community/Issues>`_ is a guideline of best practices on github.



.. include:: gallery_advanced_use_cases/index.rst




