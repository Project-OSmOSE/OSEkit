from abc import ABC, abstractmethod
from pathlib import Path

from osekit.job.job import Job, JobStatus


class Scheduler(ABC):
    """Abstract class representing a job scheduler."""

    JOB_FILE_EXTENSION = "job"

    @abstractmethod
    def write(self, job: Job, path: Path) -> None:
        """Write a job script to file.

        Parameters
        ----------
        job: Job
            Job of which to write the script.
        path: Path
            Path of the file in which the job script is written.

        """
        ...

    @abstractmethod
    def submit(
        self, job: Job, dependency: Job | list[Job] | str | list[str] | None = None
    ) -> None:
        """Submit the job to the scheduler.

        Parameters
        ----------
        job: Job
            Job to submit to the scheduler.
        dependency: Job | list[Job] | str | None
            Job dependency. Can be:
            - A ``Job`` instance: will wait for that job to complete successfully
            - A ``list[Job]``: will wait for all jobs to complete successfully
            - A ``str``: job ID (e.g., ``"12345.datarmor"``) or dependency specification
            - ``None``: no dependency

        """
        ...

    @abstractmethod
    def update_info(self, job: Job) -> None:
        """Request info about the job and update it."""
        ...

    @abstractmethod
    def update_status(self, job: Job) -> JobStatus:
        """Request info about the job and update its status.

        Returns
        -------
        JobStatus:
            The updated status of the job.

        """
        ...

    @staticmethod
    @abstractmethod
    def _build_venv_string(job: Job) -> str: ...

    @classmethod
    @abstractmethod
    def _validate_dependency_type(cls, dependency_type: str) -> None: ...

    @staticmethod
    @abstractmethod
    def _validate_dependency(dependency: list[str] | list[Job]) -> list[str]: ...

    @classmethod
    @abstractmethod
    def _build_dependency_string(
        cls,
        dependency: str | Job | list[str] | list[Job],
        dependency_type: str = "",
    ) -> str:
        """Build a job dependency string.

        Parameters
        ----------
        dependency: Job | str
            ``Job`` or job ID to depend on.
        dependency_type: str
            Type of dependency.

        Returns
        -------
        str
            Job dependency string.
        """
        ...
