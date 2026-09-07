from pathlib import Path

from osekit.job.config import JobConfig
from osekit.job.job import Job, JobStatus
from osekit.job.scheduler.pbs import Pbs
from osekit.job.scheduler.scheduler import Scheduler
from osekit.utils.core import file_indexes_per_batch


class JobBuilder:
    """Class that should be attached to a Public API ``Project`` for working with jobs.

    If a ``Project`` has a ``JobBuilder``, it will run its transforms through jobs
    using the specified scheduler.

    """

    def __init__(
        self,
        config: JobConfig | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        """Initialize a ``JobBuilder`` instance.

        Parameters
        ----------
        config: JobConfig
            Config of the jobs built by this job builder.
        scheduler: Scheduler
            Scheduler used to format, write and submit jobs.

        """
        self.config = config or JobConfig()
        self.scheduler = scheduler or Pbs()
        self.jobs = []

    @property
    def jobs(self) -> list[Job]:
        """Return the jobs created by this job builder."""
        return self._jobs

    @jobs.setter
    def jobs(self, jobs: list[Job]) -> None:
        self._jobs = jobs

    def create_jobs(
        self,
        nb_tasks: int,
        script_path: Path,
        script_args: dict,
        output_folder: Path,
        job_name: str = "osekit_transform",
        nb_jobs: int = 1,
    ) -> None:
        """Create the jobs corresponding to each batch.

        Parameters
        ----------
        nb_tasks:
            The number of tasks that are distributed across ``nb_jobs`` jobs.
        script_path: Path
            Path to the export script.
        script_args: dict
            Arguments passed to the export script.
        job_name: str
            Name of the job.
            If there are multiple batches, each batch will be suffixed
            with "_{index}".
        output_folder: Path
            Folder in which the job output log files are saved.
        nb_jobs: int
            Number of batches used to run the transform.
            Each batch will run in a separate job.

        """
        batch_indexes = file_indexes_per_batch(
            total_nb_files=nb_tasks,
            nb_batches=nb_jobs,
        )
        for index, (start, stop) in enumerate(batch_indexes):
            self.create_job(
                script_path=script_path,
                script_args=script_args | {"first": start, "last": stop},
                name=job_name + (f"_{index}" if len(batch_indexes) > 1 else ""),
                output_folder=output_folder,
            )

    def create_job(
        self,
        script_path: Path,
        script_args: dict | None = None,
        name: str = "osekit_transform",
        output_folder: Path | None = None,
    ) -> None:
        """Create a new ``Job`` instance.

        Parameters
        ----------
        script_path: Path
            Path to the script file the job must run.
        script_args: dict | None
            Additional arguments to pass to the script file.
        name: str
            Name of the job.
        output_folder: Path | None
            Folder in which the output files (``.out`` and ``.err``) will be written.

        """
        job = Job(
            script_path=script_path,
            script_args=script_args,
            name=name,
            output_folder=output_folder,
            config=self.config,
        )
        self.scheduler.write(
            job=job,
            path=output_folder / f"{name}.{self.scheduler.JOB_FILE_EXTENSION}",
        )
        self.jobs.append(job)

    def submit(
        self,
        dependencies: dict[Job, dict[str, str | Job | list[str | Job]]] | None = None,
    ) -> None:
        """Submit all prepared jobs to the scheduler system.

        Parameters
        ----------
        dependencies: dict[Job, dict[str, str | Job | list[str | Job]]] | None
            Optional mapping of the jobs dependencies.

            For each key (which is a job), the value is a
            dictionary explaining its dependencies.

            Such dictionary follows the following format:
            The keys of the dictionary are the dependency types,
            that are proper to the scheduler.
            The values are the  other jobs (or their ID) ``job`` depends on
            with the given dependency type.

            If ``None``, the jobs are submitted without any dependency.

        """
        dependencies = dependencies or {}
        for job in self.jobs:
            if self.scheduler.update_status(job=job) is not JobStatus.PREPARED:
                continue

            # Check if this job has dependencies
            job_dependencies = dependencies.get(job, None)

            self.scheduler.submit(job=job, dependencies=job_dependencies)
