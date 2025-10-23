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

    job_builder = JobBuilder(
        config=job_config,
    )

    dataset.job_builder = job_builder

    # Now the dataset has a non-None job_builder attribute,
    # running an analysis will write a PBS file in the logs directory
    # and submit it to the requested queue.

    dataset.run_analysis(...) # See the Analysis documentation
