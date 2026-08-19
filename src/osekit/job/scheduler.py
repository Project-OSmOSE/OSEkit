from abc import ABC, abstractmethod
from pathlib import Path

from osekit.job.job import Job


class Scheduler(ABC):
    """Abstract class representing a job scheduler."""

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

    @abstractmethod
    def update_info(self, job: Job) -> None:
        """Request info about the job and update it."""
        ...

    @abstractmethod
    def update_status(self, job: Job) -> None:
        """Request info about the job and update its status.

        Returns
        -------
        JobStatus:
            The updated status of the job.

        """
