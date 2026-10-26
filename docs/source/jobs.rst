Working with jobs
-----------------

**OSEkit** can be set to send transform instructions to be computed on a remote server
through queuing systems.

This feature has mainly be thought for the Public API, but it can nonetheless be used for
any Core API operation.

This is done thanks to the job package, located in :mod:`osekit.job`.

Public API
^^^^^^^^^^

Running Public API Analyses through jobs only requires adding a :class:`osekit.job.builder.JobBuilder`
instance to the :attr:`osekit.public.project.Project.job_builder` attribute.

The :class:`osekit.job.builder.JobBuilder`

Here is an example for running a transform on a PBS queue:

.. code-block:: python

    import os

    from pandas import Timedelta

    from osekit.job.builder import JobBuilder
    from osekit.job.config import JobConfig
    from osekit.job.scheduler.pbs import Pbs
    from osekit.public.project import Project

    project = Project(...)  # See the Project documentation

    job_config = JobConfig(
        nb_nodes=1,  # Number of nodes on which the job runs
        ncpus=28,  # Number of total cores used per node
        ngpus=1,  # Number of total GPU used per node
        mem="60gb",  # Maximum amount of physical memory used by the job
        walltime=Timedelta(
            hours=5
        ),  # Maximum amount of real itime during which the job can be running
        venv_name=os.environ["CONDA_DEFAULT_ENV"],  # Works only for conda venvs
    )

    scheduler = Pbs(queue="omp")  # Scheduler in which the job is submitted

    project.job_builder = JobBuilder(
        config=job_config,
        scheduler=scheduler,
    )

    # Now the dataset has a non-None job_builder attribute,
    # running a transform will write a PBS file in the logs directory
    # and submit it through the selected scheduler.

    project.run(...)  # See the Transform documentation


Core API
^^^^^^^^

Exporting Core API datasets with jobs is doable by explicitly instantiating a :class:`osekit.job.job.Job` object.

The export parameters are specified in the ``script_args`` parameter of the ``Job`` constructor,
and follow the console arguments of the :mod:`osekit.public.export` script.

.. code-block:: python

    import os
    from pathlib import Path

    from pandas import Timedelta

    from osekit.core.audio_dataset import AudioDataset
    from osekit.core.spectro_dataset import SpectroDataset
    from osekit.job.config import JobConfig
    from osekit.job.job import Job
    from osekit.job.scheduler.pbs import Pbs
    from osekit.public import export_transform

    # Some Public API imports are required
    from osekit.public.transform import OutputType

    ads = AudioDataset(...)  # See the AudioDataset doc
    sds = SpectroDataset(...)  # See the SpectroDataset doc

    # We must specify the folder in which the files will be exported
    # This is an example with both audio and spectro exports.
    ads.folder = Path(...)
    sds.folder = Path(...)

    # Datasets must be serialized
    ads.write_json(ads.folder / "output")
    sds.write_json(sds.folder / "output")

    # Export specifications
    # All parameters are listed in this example, but all parameters other than transform have default values
    args = {
        "output_type": (OutputType.AUDIO | OutputType.SPECTROGRAM).value,
        "ads-json": ads.foler / "output" / f"{ads.name}.json",
        "sds-json": sds.foler / "output" / f"{sds.name}.json",
        "subtype": "FLOAT",
        "spectrum-folder-path": "None",  # Folder in which npz matrices are exported
        "spectrogram-folder-path": sds.folder
        / "output",  # Folder in which png spectrograms are exported
        "welch-folder-path": "None",  # Folder in which npz welch matrices are exported
        "first": 0,  # First data of the dataset to be exported
        "last": len(
            ads.data,
        ),  # Last data of the dataset to be exported, up to the last one if not included
        "downsampling-quality": "HQ",
        "upsampling-quality": "VHQ",
        "umask": 0o022,
        "tqdm-disable": "False",  # Disable TQDM progress bars
        "multiprocessing": "True",
        "nb-processes": "None",  # Should be a string. "None" uses the max number of processes, otherwise e.g. "3" will use 3.
        "use-logging-setup": "True",  # Call osekit.setup_logging() before exporting the dataset.
    }

    # Job and server configuration
    job_config = JobConfig(
        nb_nodes=1,
        ncpus=28,
        mem="60gb",
        walltime=Timedelta(hours=1),
        venv_name=os.environ["CONDA_DEFAULT_ENV"],
    )

    # Scheduler configuration
    scheduler = Pbs(queue="omp")

    job = Job(
        script_path=Path(export_transform.__file__),
        script_args=args,
        config=job_config,
        name="test_job_core",
        output_folder=Path(...),  # Path in which the .out and .err files are written
    )

    # Write the job  file and submit it through the scheduler
    scheduler.write(job=job, path=Path(...) / f"{job.name}.pbs")
    scheduler.submit(job=job)

You can then follow the status of the submitted job through the scheduler:

.. code-block:: python

    scheduler.update_status(job=job)
