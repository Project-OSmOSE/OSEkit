Working with jobs
-----------------

**OSEkit** can be set to send analyses instructions to be computed on a remote server
through the PBS queuing system.

This feature has mainly be thought for the Public API, but it can nonetheless be used for
any Core API operation.

The job module is located at :mod:`osekit.utils.job`.

Public API
^^^^^^^^^^

Running Public API Analyses through PBS jobs only requires adding a :class:`osekit.utils.job.JobBuilder`
instance to the :attr:`osekit.public_api.dataset.Dataset.job_builder` attribute:

.. code-block:: python

    from osekit.utils.job import JobConfig, JobBuilder
    from osekit.public_api.dataset import Dataset

    dataset = Dataset(...) # See the Dataset documentation

    job_config = JobConfig(
        nb_nodes=1, # Number of nodes on which the job runs
        ncpus=28, # Number of total cores used per node
        mem="60gb", # Maximum amount of physical memory used by the job
        walltime=Timedelta(hours=5), # Maximum amount of real itime during which the job can be running
        venv_name=os.environ["CONDA_DEFAULT_ENV"], # Works only for conda venvs
        queue="omp" # Queue in which the job will be submitted
    )

    dataset.job_builder = JobBuilder(
        config=job_config,
    )

    # Now the dataset has a non-None job_builder attribute,
    # running an analysis will write a PBS file in the logs directory
    # and submit it to the requested queue.

    dataset.run_analysis(...) # See the Analysis documentation


Core API
^^^^^^^^

Exporting Core API datasets with jobs is doable by explicitly instantiating a :class:`osekit.utils.job.Job` object.

The export parameters are specified in the ``script_args`` parameter of the ``Job`` constructor,
and follow the console arguments of the :mod:`osekit.public_api.export_analysis` script.

.. code-block:: python

    import os

    from osekit.core_api.spectro_dataset import SpectroDataset
    from osekit.core_api.audio_dataset import AudioDataset
    from osekit.utils.job import JobConfig, Job

    # Some Public API imports are required
    from osekit.public_api.analysis import AnalysisType
    from osekit.public_api import export_analysis

    ads = AudioDataset(...) # See the AudioDataset doc
    sds = SpectroDataset(...) # See the SpectroDataset doc

    # We must specify the folder in which the files will be exported
    # This is an example with both audio and spectro exports.
    ads.folder = Path(...)
    sds.folder = Path(...)

    # Datasets must be serialized
    ads.write_json(ads.folder/"output")
    sds.write_json(sds.folder/"output")

    # Export specifications
    # All parameters are listed in this example, but all parameters other than analysis have default values
    args = {
        "analysis": (AnalysisType.AUDIO|AnalysisType.SPECTROGRAM).value,
        "ads-json": ads.foler/"output"/f"{ads.name}.json",
        "sds-json": sds.foler/"output"/f"{sds.name}.json",
        "subtype": "FLOAT",
        "matrix-folder-path": "None", # Folder in which npz matrices are exported
        "spectrogram-folder-path": sds.folder/"output", # Folder in which png spectrograms are exported
        "welch-folder-path": "None",  # Folder in which npz welch matrices are exported
        "first": 0, # First data of the dataset to be exported
        "last": len(ads.data), # Last data of the dataset to be exported, up to the last one if not included
        "downsampling-quality": "HQ",
        "upsampling-quality": "VHQ",
        "umask": 0o022,
        "tqdm-disable": "False", # Disable TQDM progress bars
        "multiprocessing": "True",
        "nb-processes": "None",  # Should be a string. "None" uses the max number of processes, otherwise e.g. "3" will use 3.
        "use-logging-setup": "True", # Call osekit.setup_logging() before exporting the dataset.
    }

    # Job and server configuration
    job_config = JobConfig(
        nb_nodes=1,
        ncpus=28,
        mem="60gb",
        walltime=Timedelta(hours=1),
        venv_name=os.environ["CONDA_DEFAULT_ENV"],
        queue="omp"
    )

    job = Job(
        script_path = Path(export_analysis.__file__),
        script_args=args,
        config=job_config,
        name="test_job_core",
        output_folder=Path(...), # Path in which the .out and .err files are written
    )

    # Write the PBS file and submit the job
    job.write_pbs(Path(...) / f"{job.name}.pbs")
    job.submit_pbs()

You can then follow the status of the submitted job:

.. code-block:: python

    job.update_status()
