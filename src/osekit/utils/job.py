"""The job module provides classes that run analyses on a remote server.

If a ``JobBuilder`` is attached to a Public API ``Dataset``,
the analyses will run through jobs, with writting/submitting of ``pbs`` files.

"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

from pandas import Timedelta

if TYPE_CHECKING:
    from pathlib import Path


class JobStatus(Enum):
    """Status of the job.

    ``UNPREPARED``: The job file hasn't been written yet.
    ``PREPARED``: The job file has been written but not submitted.
    ``QUEUED``: The job has been queued.
    ``RUNNING``: The job is currently running.
    ``COMPLETED``: The job has been completed.

    """

    UNPREPARED = 1
    PREPARED = 2
    QUEUED = 3
    RUNNING = 4
    COMPLETED = 5


@dataclass
class JobConfig:
    """Config used for creating a job.

    Parameters
    ----------
    nb_nodes: int
        Number of nodes on which the job runs.
    ncpus: int
        Number of total cores used per node.
    mem: str
        Maximum amount of physical memory used by the job.
    walltime: str | Timedelta
        Maximum amount of real time during which the job can be running.
    venv_name: str
        Name (or path) of the conda virtual environment in which the job is running.
    queue: Literal["omp", "mpi"]
        Queue in which the job will be submitted.

    """

    nb_nodes: int = 1
    ncpus: int = 2
    mem: str = "8gb"
    walltime: str | Timedelta = "01:00:00"
    venv_name: str = "osmose"
    queue: Literal["omp", "mpi"] = "omp"


class Job:
    """Job that concerns a specific analysis."""

    def __init__(
        self,
        script_path: Path,
        script_args: dict | None = None,
        config: JobConfig | None = None,
        name: str = "osekit_analysis",
        output_folder: Path | None = None,
    ) -> None:
        """Initialize a Job.

        Parameters
        ----------
        script_path: Path
            Path to the script file the job must run.
        script_args: dict | None
            Additional arguments to pass to the script file.
        config: JobConfig | None
            Optional configuration to pass to the server request.
        name: str
            Name of the job.
        output_folder: Path | None
            Folder in which the output files (``.out`` and ``.err``) will be written.

        """
        config = JobConfig() if config is None else config
        self.script_path = script_path
        self.script_args = script_args if script_args else {}
        self.nb_nodes = config.nb_nodes
        self.ncpus = config.ncpus
        self.mem = config.mem
        self.walltime = config.walltime
        self.venv_name = config.venv_name
        self.queue = config.queue
        self.name = name
        self.output_folder = output_folder
        self._status = JobStatus.UNPREPARED
        self._path = None
        self._id = None
        self._info = None

    @property
    def script_path(self) -> Path:
        """Path to the script file the job must run."""
        return self._script_path

    @script_path.setter
    def script_path(self, path: Path) -> None:
        self._script_path = path

    @property
    def script_args(self) -> dict:
        """Additional arguments to pass to the script file."""
        return self._script_args

    @script_args.setter
    def script_args(self, args: dict) -> None:
        self._script_args = args

    @property
    def nb_nodes(self) -> int:
        """Number of nodes on which the job runs."""
        return self._chunks

    @nb_nodes.setter
    def nb_nodes(self, chunks: int) -> None:
        self._chunks = chunks

    @property
    def ncpus(self) -> int:
        """Number of total cores used per node."""
        return self._ncpus

    @ncpus.setter
    def ncpus(self, ncpus: int) -> None:
        self._ncpus = ncpus

    @property
    def mem(self) -> str:
        """Maximum amount of physical memory used by the job."""
        return self._mem

    @mem.setter
    def mem(self, mem: str) -> None:
        self._mem = mem

    @property
    def walltime(self) -> Timedelta:
        """Maximum amount of real time during which the job can be running."""
        return self._walltime

    @property
    def walltime_str(self) -> str:
        """String representation of the ``walltime``."""
        return str(self.walltime).split("days")[-1].strip()

    @walltime.setter
    def walltime(self, walltime: str | Timedelta) -> None:
        self._walltime = (
            walltime if type(walltime) is Timedelta else Timedelta(walltime)
        )

    @property
    def venv_name(self) -> str:
        """Name of the conda virtual environment in which the job is running."""
        return self._venv_name

    @venv_name.setter
    def venv_name(self, venv_name: str) -> None:
        self._venv_name = venv_name

    @property
    def venv_activate_script(self) -> str:
        """Bash script used for activating the conda virtual environment."""
        return f". /appli/anaconda/latest/etc/profile.d/conda.sh; conda activate {self.venv_name}"

    @property
    def queue(self) -> Literal["omp", "mpi"]:
        """Queue in which the job will be submitted."""
        return self._queue

    @queue.setter
    def queue(self, queue: Literal["omp", "mpi"]) -> None:
        self._queue = queue

    @property
    def name(self) -> str:
        """Name of the job."""
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def status(self) -> JobStatus:
        """Status of the job.

        ``UNPREPARED``: The job file hasn't been written yet.
        ``PREPARED``: The job file has been written but not submitted.
        ``QUEUED``: The job has been queued.
        ``RUNNING``: The job is currently running.
        ``COMPLETED``: The job has been completed.

        """
        return self._status

    @status.setter
    def status(self, status: JobStatus) -> None:
        self._status = status

    @property
    def path(self) -> Path | None:
        """Path of the job file."""
        return self._path

    @path.setter
    def path(self, path: Path) -> None:
        self._path = path

    @property
    def output_folder(self) -> Path | None:
        """Folder in which the output files (``.out`` and ``.err``) will be written."""
        return self._output_folder

    @output_folder.setter
    def output_folder(self, output_folder: Path | None) -> None:
        self._output_folder = output_folder

    @property
    def job_id(self) -> str:
        """Job ID."""
        return self._id

    @job_id.setter
    def job_id(self, job_id: str) -> None:
        self._id = job_id

    @property
    def job_info(self) -> dict | None:
        """Information about the job as returned by a qstat request."""
        return self._info

    @job_info.setter
    def job_info(self, info: dict) -> None:
        self._info = info

    def progress(self) -> None:
        """Bring the job to the next state."""
        if self.status == JobStatus.COMPLETED:
            return
        self._status = JobStatus(self._status.value + 1)

    def _build_arg_string(self) -> str:
        """Build a string representation of the job's arguments."""
        arg_list = []
        for key, value in self.script_args.items():
            if isinstance(value, bool):
                arg_list.append(f"--{'no-' if not value else ''}{key}")
            else:
                arg_list.append(f"--{key} {value}")
        return " ".join(arg_list)

    def write_pbs(self, path: Path) -> None:
        """Write a ``pbs`` file matching the job.

        Parameters
        ----------
        path: Path
            Path of the ``pbs`` file to write.

        """
        preamble = "#!/bin/bash"
        request = {
            "-N": self.name,
            "-q": self.queue,
            "-l": [
                f"select={self.nb_nodes}:ncpus={self.ncpus}:mem={self.mem}",
                f"walltime={self.walltime_str}",
            ],
            "-o": f"{self.output_folder}/{self.name}.out"
            if self.output_folder
            else None,
            "-e": f"{self.output_folder}/{self.name}.err"
            if self.output_folder
            else None,
        }
        request_str = "\n".join(
            f"#PBS {key} {value}"
            if type(value) is not list
            else "\n".join(f"#PBS {key} {value_part}" for value_part in value)
            for key, value in request.items()
            if value
        )

        script = f"python {self.script_path} {self._build_arg_string()}"

        pbs = f"{preamble}\n{request_str}\n{self.venv_activate_script}\n{script}"
        with path.open("w") as file:
            file.write(pbs)

        self.path = path
        self.progress()

    def submit_pbs(
        self,
        dependency: Job | list[Job] | str | list[str] | None = None,
    ) -> None:
        """Submit the ``pbs`` file of the job to a ``pbs`` queueing system.

        Parameters
        ----------
        dependency: Job | list[Job] | str | None
            Job dependency. Can be:
            - A ``Job`` instance: will wait for that job to complete successfully
            - A ``list[Job]``: will wait for all jobs to complete successfully
            - A ``str``: job ID (e.g., ``"12345.datarmor"``) or dependency specification
            - ``None``: no dependency

        """
        if self.update_status() is not JobStatus.PREPARED:
            msg = "Job should be written before being submitted."
            raise ValueError(msg)

        cmd = ["qsub"]

        if dependency is not None:
            dependency_str = self._build_dependency_string(dependency)
            if dependency_str:
                cmd.extend(["-W", f"depend={dependency_str}"])

        cmd.append(str(self.path))

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

        self.job_id = request.stdout.split(".", maxsplit=1)[0].strip()
        self.progress()

    _VALID_DEPENDENCY_TYPES = {"afterok", "afterany", "afternotok", "after"}

    @staticmethod
    def _validate_dependency_type(dependency_type: str) -> None:
        if dependency_type not in Job._VALID_DEPENDENCY_TYPES:
            msg = (
                f"Unsupported dependency type '{dependency_type}'. "
                f"Expected one of {sorted(Job._VALID_DEPENDENCY_TYPES)}."
            )
            raise ValueError(msg)

    @staticmethod
    def _validate_dependency(dependency: list[str] | list[Job]) -> list[str]:
        job_ids = [dep.job_id if isinstance(dep, Job) else dep for dep in dependency]
        job_id_length = 7
        for job_id in job_ids:
            if not job_id.isdigit() or len(job_id) != job_id_length:
                msg = (
                    f"Invalid job ID '{job_id}'. "
                    f"Job IDs must be {job_id_length} digits long."
                )
                raise ValueError(msg)
        return job_ids

    @staticmethod
    def _build_dependency_string(
        dependency: str | Job | list[str] | list[Job],
        dependency_type: str = "afterok",
    ) -> str:
        """Build a PBS dependency string.

        Parameters
        ----------
        dependency: Job | str
            ``Job`` or job ID to depend on.
        dependency_type: str
            Type of dependency (``afterok``, ``afterany``, ``afternotok``, ``after``).

        Returns
        -------
        str
            PBS dependency string.

        Examples
        --------
        >>> Job._build_dependency_string("1234567")
        'afterok:1234567'
        >>> Job._build_dependency_string(["1234567", "4567891"])
        'afterok:1234567:4567891'
        >>> Job._build_dependency_string("7894561", dependency_type="afterany")
        'afterany:7894651'

        """
        dependency = dependency if isinstance(dependency, list) else [dependency]
        id_str = Job._validate_dependency(dependency)
        Job._validate_dependency_type(dependency_type)

        if unsubmitted_job := next(
            (
                j
                for j in dependency
                if isinstance(j, Job) and j.status.value < JobStatus.QUEUED.value
            ),
            None,
        ):
            msg = f"Job '{unsubmitted_job.name}' has not been submitted yet."
            raise ValueError(msg)

        return f"{dependency_type}:{':'.join(id_str)}"

    def update_info(self) -> None:
        """Request info about the job and update it."""
        if self.job_id is None:
            return

        try:
            request = subprocess.run(
                ["qstat", "-f", self.job_id],
                capture_output=True,
                text=True,
                check=False,
            )
            stdout = request.stdout
        except subprocess.CalledProcessError as e:
            msg = f"Qstat failed with exit code {e.returncode}"
            raise RuntimeError(msg) from e

        if not stdout:
            err = request.stderr
            if "Job has finished" in err:
                self.status = JobStatus.COMPLETED
                self.job_info["job_state"] = "C"
            if "Unknown Job Id" in err:
                msg = f"Unknown Job Id {self.job_id}"
                raise ValueError(msg)
            return

        info = {}
        for line in stdout.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            info[key.strip()] = value.strip()
        self.job_info = info

    def update_status(self) -> JobStatus:
        """Request info about the job and update its status.

        Returns
        -------
        JobStatus:
            The updated status of the job.

        """
        if self.job_id is None:
            self.status = (
                JobStatus.PREPARED
                if self.path and self.path.exists()
                else JobStatus.UNPREPARED
            )
            return self.status

        self.update_info()

        if self.status == JobStatus.COMPLETED:
            return self.status

        job_state = {
            "Q": JobStatus.QUEUED,
            "R": JobStatus.RUNNING,
        }
        if self.job_info["job_state"] in job_state:
            self.status = job_state[self.job_info["job_state"]]
        return self.status


class JobBuilder:
    """Class that should be attached to a Public API ``Dataset`` for working with jobs.

    If a ``Dataset`` has a ``JobBuilder``, it will use it to run analyses through jobs.

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
        name: str = "osekit_analysis",
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
