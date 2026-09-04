import typing
from typing import Literal

from osekit.job.job import Job, JobStatus
from osekit.job.scheduler.scheduler import Scheduler


class Slurm(Scheduler):
    """Abstract class representing a job scheduler."""

    JOB_FILE_EXTENSION: typing.ClassVar = "slurm"
    INFO_CMD: typing.ClassVar = ["squeue", "--jobs"]
    SUBMIT_CMD: typing.ClassVar = "sbatch"
    JOB_STATUS_CODES: typing.ClassVar = {
        "PD": JobStatus.QUEUED,
        "R": JobStatus.RUNNING,
        "S": JobStatus.SUSPENDED,
        "CG": JobStatus.COMPLETED,
        "CD": JobStatus.COMPLETED,
    }

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

    @classmethod
    def _parse_info_str(cls, job: Job, info: str) -> None:
        """Parse the info from the requested squeue info string."""
        keys, values = info.splitlines()

        # Get keys order in the string
        known_keys = [
            "JOBID",
            "PARTITION",
            "NAME",
            "USER",
            "ST",
            "TIME",
            "NODES",
            "NODELIST(REASON)",
        ]
        keys = sorted(known_keys, key=keys.index)

        # Get the associated values
        kvp = dict(zip(keys, values.split(), strict=True))

        job.info["user"] = kvp["USER"]
        job.info["time"] = kvp["TIME"]
        job.info["partition"] = kvp["PARTITION"]
        job.info["node_list"] = kvp["NODELIST(REASON)"]

        if status := cls.JOB_STATUS_CODES.get(kvp["ST"], False):
            job.status = status

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
