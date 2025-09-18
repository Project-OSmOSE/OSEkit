from enum import Enum
from pathlib import Path
from typing import Literal

from pandas import Timedelta


class JobStatus(Enum):
    PREPARED = 1
    QUEUED = 2
    RUNNING = 3
    COMPLETED = 4


class Job:
    def __init__(
        self,
        script_path: Path,
        script_args: list[str] | None = None,
        chunks: int = 1,
        ncpus: int = 1,
        mem: str = "8gb",
        walltime: str | Timedelta = "01:00:00",
        venv_name: str = "osmose",
        queue: Literal["omp", "mpi"] = "omp",
        name: str = "osekit_analysis",
    ) -> None:
        self.script_path = script_path
        self.script_args = script_args if script_args else []
        self.chunks = chunks
        self.ncpus = ncpus
        self.mem = mem
        self.walltime = walltime
        self.venv_name = venv_name
        self.queue = queue
        self.name = name
        self._status = JobStatus.PREPARED

    @property
    def script_path(self) -> Path:
        return self._script_path

    @script_path.setter
    def script_path(self, path: Path) -> None:
        self._script_path = path

    @property
    def script_args(self) -> list[str]:
        return self._script_args

    @script_args.setter
    def script_args(self, args: list[str]) -> None:
        self._script_args = args

    @property
    def chunks(self) -> int:
        return self._chunks

    @chunks.setter
    def chunks(self, chunks: int) -> None:
        self._chunks = chunks

    @property
    def ncpus(self) -> int:
        return self._ncpus

    @ncpus.setter
    def ncpus(self, ncpus: int) -> None:
        self._ncpus = ncpus

    @property
    def mem(self) -> str:
        return self._mem

    @mem.setter
    def mem(self, mem: str) -> None:
        self._mem = mem

    @property
    def walltime(self) -> Timedelta:
        return self._walltime

    @walltime.setter
    def walltime(self, walltime: str | Timedelta) -> None:
        self._walltime = (
            walltime if type(walltime) is Timedelta else Timedelta(walltime)
        )

    @property
    def venv_name(self) -> str:
        return self._venv_name

    @venv_name.setter
    def venv_name(self, venv_name: str) -> None:
        self._venv_name = venv_name

    @property
    def queue(self) -> Literal["omp", "mpi"]:
        return self._queue

    @queue.setter
    def queue(self, queue: Literal["omp", "mpi"]) -> None:
        self._queue = queue

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def status(self) -> JobStatus:
        return self._status

    def progress(self) -> None:
        if self.status == JobStatus.COMPLETED:
            return
        self._status = JobStatus(self._status.value + 1)
