.. _example_reshaping_multiple_files:

Reshaping multiple files
========================

In this example, we want to export reshaped files from the sample audio dataset with the following requirements:

* First file starts at ``2022-09-25 22:35:15``
* Last file ends at ``2022-09-25 22:36:25``
* Files are ``5 s``-long
* Files are sampled at ``24 kHz``
* Files are DC-filtered
* Files that are in the gap between recordings should be skipped

.. toctree::
   :maxdepth: 1

   example_reshaping_multiple_files_core
   example_reshaping_multiple_files_public
