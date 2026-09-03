from typing import Literal

from osekit.job.job import Job, JobStatus
from osekit.job.scheduler.scheduler import Scheduler


class Slurm(Scheduler):
    """Abstract class representing a job scheduler."""

    JOB_FILE_EXTENSION = "slurm"
    SUBMIT_CMD = "sbatch"

    def __init__(self, partition: Literal["cpu", "gpu", "ops"] = "cpu") -> None:
        """Initialize the SLURM scheduler."""
        self.partition = partition

    @property
    def partition(self) -> str:
        """Partition in which the job will be submitted."""
        return self._partition

    @partition.setter
    def partition(self, partition: Literal["omp", "mpi"]) -> None:
        self._partition = partition

    def _build_job_specification(self, job: Job) -> str:
        """Build the job specification string.

        Parameters
        ----------
        job: Job
            The job for which to build the specifications.

        Returns
        -------
        str:
            Job specification string.
            It includes the name of the job, the requested resources,
            output log directories, etc.

        """
        specifications = {
            "nodes": job.nb_nodes,
            "cpus-per-task": job.ncpus,
            "mem": job.mem,
            "job-name": job.name,
            "partition": self.partition,
            "time": job.walltime_str,
            "output": f"{job.output_folder / job.name}.out"
            if job.output_folder
            else None,
            "error": f"{job.output_folder / job.name}.err"
            if job.output_folder
            else None,
        }

        if job.ngpus is not None:
            specifications["gpus"] = job.ngpus

        return "\n".join(
            f"#SBATCH --{key}={value}" for key, value in specifications.items() if value
        )

    def submit(
        self,
        job: Job,
        dependencies: dict[str, Job | str | list[Job | str]] | None = None,
    ) -> None:
        """Submit the job to the scheduler.

        Parameters
        ----------
        job: Job
            Job to submit to the scheduler.
        dependencies: dict[str, Job | str | list[Job|str]]
            The dependencies of the submitted job.
            The keys of the dictionary are the dependency types,
            that are proper to the scheduler.
            The values are the  other jobs (or their ID) ``job`` depends on
            with the given dependency type.
            If ``None``, the job is submitted without any dependency.

        """

    def update_info(self, job: Job) -> None:
        """Request info about the job and update it."""

    def update_status(self, job: Job) -> JobStatus:
        """Request info about the job and update its status.

        Returns
        -------
        JobStatus:
            The updated status of the job.

        """

    @staticmethod
    def _build_venv_string(job: Job) -> str:
        """Bash script used for activating the conda virtual environment."""
        return f"module load conda\nconda activate {job.venv_name}"

    @classmethod
    def _validate_dependency_type(cls, dependency_type: str) -> None:
        pass

    @staticmethod
    def _validate_dependency(dependency: list[str] | list[Job]) -> list[str]:
        pass

    @classmethod
    def _build_dependency_string(
        cls,
        dependencies: dict[str, Job | str | list[Job | str]],
    ) -> str:
        """Build a job dependency string.

        Parameters
        ----------
        dependencies: dict[str, Job | str | list[Job|str]]
            The dependencies of the submitted job.
            The keys of the dictionary are the dependency types,
            that are proper to the scheduler.
            The values are the  other jobs (or their ID) ``job`` depends on
            with the given dependency type.
            If ``None``, the job is submitted without any dependency.

        Returns
        -------
        str
            Job dependency string.

        """
