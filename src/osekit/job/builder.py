from pathlib import Path

from osekit.job.config import JobConfig
from osekit.job.job import Job, JobStatus


class JobBuilder:
    """Class that should be attached to a Public API ``Project`` for working with jobs.

    If a ``Project`` has a ``JobBuilder``, it will use it to run transforms through jobs.

    """

    def __init__(self, config: JobConfig = JobConfig) -> None:
        """Initialize a ``JobBuilder`` instance.

        Parameters
        ----------
        config: JobConfig
            Config of the jobs built by this job builder.

        """
        self.config = config
        self.jobs = []

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
        job.write_pbs(output_folder / f"{name}.pbs")
        self.jobs.append(job)

    def submit_pbs(
        self,
        dependencies: dict[str, Job | list[Job]] | None = None,
    ) -> None:
        """Submit all prepared jobs to the ``pbs`` queueing system.

        Parameters
        ----------
        dependencies: dict[str, Job | list[Job]] | None
            Optional dictionary mapping job names to their dependencies.
            Example: ``{"job2": job1, "job3": [job1, job2]}``

        """
        for job in self.jobs:
            if job.update_status() is not JobStatus.PREPARED:
                continue

            # Check if this job has dependencies
            depend_on = None
            if dependencies and job.name in dependencies:
                depend_on = dependencies[job.name]

            job.submit_pbs(dependency=depend_on)
