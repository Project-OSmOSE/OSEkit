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

    def __init__(self, cmd_prefix: list[str] | None = None) -> None:
        """Initialize the scheduler.

        Parameters
        ----------
        cmd_prefix: list[str]|None
            Commands that are prefixed to commands run through
            the ``Scheduler.run_subprocess()`` method.

        """
        self.cmd_prefix = cmd_prefix or []
        self.shebang = "#!/bin/bash"

    @property
    def cmd_prefix(self) -> list[str]:
        """Return the list of command prefixes.

        These commands are prefixed to commands run through
        the ``Scheduler.run_subprocess()`` method.

        Returns
        -------
        list[str]
            The prefix commands.

        """
        return self._cmd_prefix

    @cmd_prefix.setter
    def cmd_prefix(self, cmd_prefix: list[str]) -> None:
        self._cmd_prefix = cmd_prefix

    @property
    def shebang(self) -> str:
        """Return the shebang string of the scheduler job files.

        Defaulted to ``"#!/bin/bash"``.

        Returns
        -------
        str:
            Shebang of the scheduler job files.

        """
        return self._shebang

    @shebang.setter
    def shebang(self, value: str) -> None:
        self._shebang = value

    def write(self, job: Job, path: Path) -> None:
        """Write a job script to file.

        Parameters
        ----------
        job: Job
            Job of which to write the script.
        path: Path
            Path of the file in which the job script is written.

        """
        request_str = self._build_job_specification(job=job)
        venv_str = self._build_venv_string(job=job)
        python_script = f"python {job.script_path} {job.get_arg_string()}"

        script = f"{self.shebang}\n\n{request_str}\n\n{venv_str}\n\n{python_script}"

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
            request = self.run_subprocess(cmd=cmd)
        except subprocess.CalledProcessError as e:
            msg = f"Submission failed with exit code {e.returncode}"
            raise RuntimeError(msg) from e

        job.job_id = self._parse_job_id(submit_output=request.stdout)
        self.update_status(job=job)

    def run_subprocess(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        """Run a command through the subprocess package.

        Parameters
        ----------
        cmd: list[str]
            Command to run.

        Returns
        -------
        subprocess.CompletedProcess[str]
            Process output.

        """
        return subprocess.run(
            self._cmd_prefix + cmd,
            capture_output=True,
            text=True,
            check=False,
        )

    @staticmethod
    @abstractmethod
    def _parse_job_id(submit_output: str) -> str:
        """Parse the job id from the submit cmd output."""
        ...

    def update_info(self, job: Job) -> None:
        """Request info about the job and update it.

        Parameters
        ----------
        job: Job
            Job for which to update the info.

        """
        if job.job_id is None:
            return

        info = self._get_info(job=job)
        if not info:
            return

        self._parse_info_str(job=job, info=info)

    @abstractmethod
    def _get_info(self, job: Job) -> str:
        """Request information about a job.

        Parameters
        ----------
        job: Job
            Job for which the information is requested.

        Returns
        -------
        str:
            The information string.
            Depends on the scheduler.

        """
        ...

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
