"""The job module provides classes that run transforms on a remote server.

If a ``JobBuilder`` is attached to a Public API ``Project``,
the transforms will run through jobs, with writting/submitting of ``pbs`` files.

"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pandas import Timedelta

from osekit.job.config import JobConfig

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


class Job:
    """Job that concerns a specific transform."""

    def __init__(
        self,
        script_path: Path,
        script_args: dict | None = None,
        config: JobConfig | None = None,
        name: str = "osekit_transform",
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
        self.ngpus = config.ngpus
        self.mem = config.mem
        self.walltime = config.walltime
        self.venv_name = config.venv_name
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
    def ngpus(self) -> int | None:
        """Number of total GPU used per node."""
        return self._ngpus

    @ngpus.setter
    def ngpus(self, ngpus: int) -> None:
        self._ngpus = ngpus

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
        total_seconds = self.walltime.total_seconds()
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return ":".join(f"{t:02}" for t in map(int, (hours, minutes, seconds)))

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
    def job_id(self) -> str | None:
        """Job ID."""
        return self._id

    @job_id.setter
    def job_id(self, job_id: str | None) -> None:
        self._id = job_id

    @property
    def job_info(self) -> dict | None:
        """Information about the job."""
        return self._info

    @job_info.setter
    def job_info(self, info: dict) -> None:
        self._info = info

    def progress(self) -> None:
        """Bring the job to the next state."""
        if self.status == JobStatus.COMPLETED:
            return
        self._status = JobStatus(self._status.value + 1)

    def get_arg_string(self) -> str:
        """Build a string representation of the job's arguments."""
        arg_list = []
        for key, value in self.script_args.items():
            if isinstance(value, bool):
                arg_list.append(f"--{'no-' if not value else ''}{key}")
            else:
                arg_list.append(f"--{key} {value}")
        return " ".join(arg_list)
