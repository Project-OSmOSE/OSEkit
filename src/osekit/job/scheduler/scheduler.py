import subprocess
import typing
from abc import ABC, abstractmethod
from pathlib import Path

from osekit.job.job import Job, JobStatus


class Scheduler(ABC):
    """Abstract class representing a job scheduler."""

    _VALID_DEPENDENCY_TYPES: typing.ClassVar = frozenset()
    JOB_FILE_EXTENSION: typing.ClassVar = "job"
    SUBMIT_CMD: typing.ClassVar = ""
    INFO_CMD: typing.ClassVar = []

    def write(self, job: Job, path: Path) -> None:
        """Write a job script to file.

        Parameters
        ----------
        job: Job
            Job of which to write the script.
        path: Path
            Path of the file in which the job script is written.

        """
        preamble = "#!/bin/bash"

        request_str = self._build_job_specification(job=job)
        venv_str = self._build_venv_string(job=job)
        python_script = f"python {job.script_path} {job.get_arg_string()}"

        script = f"{preamble}\n\n{request_str}\n\n{venv_str}\n\n{python_script}"

        with path.open("w") as file:
            file.write(script)

        job.path = path
        job.status = JobStatus.PREPARED

    @abstractmethod
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
        ...

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
        if self.update_status(job=job) is not JobStatus.PREPARED:
            msg = "Job should be written before being submitted."
            raise ValueError(msg)

        cmd = [self.SUBMIT_CMD]

        if dependencies:
            cmd.extend(self._build_dependency_string(dependencies=dependencies).split())

        cmd.append(str(job.path))

        try:
            request = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except subprocess.CalledProcessError as e:
            msg = f"Submission failed with exit code {e.returncode}"
            raise RuntimeError(msg) from e

        job.job_id = request.stdout.split(".", maxsplit=1)[0].strip()
        self.update_status(job=job)

    def update_info(self, job: Job) -> None:
        """Request info about the job and update it.

        Parameters
        ----------
        job: Job
            Job for which to update the info.

        """
        if job.job_id is None:
            return

        try:
            request = subprocess.run(
                [*self.INFO_CMD, str(job.job_id)],
                capture_output=True,
                text=True,
                check=False,
            )
            stdout = request.stdout
        except subprocess.CalledProcessError as e:
            msg = f"{self.INFO_CMD[0]} failed with exit code {e.returncode}"
            raise RuntimeError(msg) from e

        if not stdout:
            err = request.stderr
            if err:
                msg = f"{job.job_id}: {err}"
                raise ValueError(msg)
            return

        self._parse_info_str(job=job, info=stdout)

    @classmethod
    @abstractmethod
    def _parse_info_str(cls, job: Job, info: str) -> None:
        """Parse the info string to update a job info.

        Parameters
        ----------
        job: Job
            Job of which the info is updated
        info: str
            Info string as returned by a request.

        """
        ...

    def update_status(self, job: Job) -> JobStatus:
        """Request info about the job and update its status.

        Returns
        -------
        JobStatus:
            The updated status of the job.

        """
        if job.job_id is None:
            job.status = (
                JobStatus.PREPARED
                if job.path and job.path.exists()
                else JobStatus.UNPREPARED
            )
            return job.status

        self.update_info(job=job)
        return job.status

    @staticmethod
    @abstractmethod
    def _build_venv_string(job: Job) -> str: ...

    @classmethod
    def _validate_dependency_type(cls, dependency_type: str) -> None:
        """Validate dependency types in the dependencies instruction."""
        if dependency_type not in cls._VALID_DEPENDENCY_TYPES:
            msg = (
                f"Unsupported dependency type '{dependency_type}'.\n"
                f"Expected one of:\n\t{'\n\t'.join(sorted(cls._VALID_DEPENDENCY_TYPES))}."
            )
            raise ValueError(msg)

    @staticmethod
    def _parse_job_ids(
        dependencies: dict[str, Job | str | list[Job | str]],
    ) -> dict[str, list[str]]:
        """Replace all ``Job`` instances by their ID string."""
        parsed_dependencies = {}
        for key, value in dependencies.items():
            parsed_values = value if isinstance(value, list) else [value]
            parsed_values = [
                parsed_value.job_id if isinstance(parsed_value, Job) else parsed_value
                for parsed_value in parsed_values
            ]
            parsed_dependencies[key] = parsed_values

        return parsed_dependencies

    @classmethod
    @abstractmethod
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
        ...
