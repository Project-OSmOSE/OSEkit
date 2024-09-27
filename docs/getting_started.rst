
=================
Getting started
=================

 

Quick installation
===================

The quickest and safest way to start using our toolkit is to install its latest stable version locally in a virtual environment. To do so follow these steps:

1. Download the `wheel file <https://github.com/Project-OSmOSE/osmose-toolkit/releases/tag/v0.1.0>`_ (.whl extension) of the latest version available in your local working folder

2. Create and activate a Conda virtual environment in python 3.10, named for example osmose_stable

.. code-block:: bash	

	conda create --name osmose python=3.10 -y
	conda activate osmose

.. note::

	We have made a :ref:`tutorial on Conda <Conda>` if you are starting with it


3. Install the pip tool in your conda environment, go to the folder `folder_wheel` where the wheel file is located and use it to install the toolkit from the wheel file 

.. code-block:: bash

	conda install pip
        cd <folder_wheel>
	pip install <toolkit_version.whl>

.. note:: 
	If it happens that the toolkit version has not changed, use instead

	.. code-block:: bash

		pip install --force-reinstall <toolkit_version.whl>



.. warning::
	We discourage users to install the toolkit from our git repository, as it is under permanent development and thus potentially unstable and unfinished.

.. note::
	If for any reason you get an unexpected error message or an incorrect result, please open a new issue in the `issue tracker <https://github.com/Project-OSmOSE/osmose-toolkit/issues>`_ 



Sample datasets
=============================

Our toolkit comes with a diversified set of sample datasets containing both audio recordings and auxiliary data, described in the table below.

======================= ========== ========== ========================= =======================================================
Code name               Date       Time       Location                  Sample frequency (Hz)
======================= ========== ========== ========================= =======================================================
SPM2011                 2011-09-19 11:00      Saint Pierre et Miquelon  48000
BallenyIslands2015      2011-09-19 11:00      Balleny Islands 2015      1000
======================= ========== ========== ========================= =======================================================




Most of the examples found in modules and in the use case gallery use these audio recordings. To test and run the scripts, we recommend downloading these files to your local hard drive. The files can be downloaded from the GitHub `repository <https://github.com/project-OSmOSE>`_


 

.. include:: tutorials.rst



Joining datasets
=============================

The Auxiliary class automatically joins environmental and instrument data to acoustic data. By default, the processed acoustic dataset serves as the reference dataset and other data will be joined to corresponding time and localization. So far, only csv files and ERA5 re-analysis netCDF files can be joined to your acoustic data. The API functions to download ERA data can be found in the module but an internet access is necessary. 

.. image:: figures/trajectory_wind_speed.png
  :width: 400
  :align: center
  :alt: Elephant seal trajectory with wind speed (m/s) as a color map
