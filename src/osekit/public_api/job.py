from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from pandas import Timedelta


class JobStatus(Enum):
    UNPREPARED = 1
    PREPARED = 2
    QUEUED = 3
    RUNNING = 4
    COMPLETED = 5


@dataclass
class JobConfig:
    chunks: int = (1,)
    ncpus: int = (1,)
    mem: str = ("8gb",)
    walltime: str | Timedelta = ("01:00:00",)
    venv_name: str = ("osmose",)
    queue: Literal["omp", "mpi"] = "omp"


class Job:
    def __init__(
        self,
        script_path: Path,
        script_args: list[str] | None = None,
        config: JobConfig | None = None,
        name: str = "osekit_analysis",
    ) -> None:
        config = JobConfig() if config is None else config
        self.script_path = script_path
        self.script_args = script_args if script_args else []
        self.chunks = config.chunks
        self.ncpus = config.ncpus
        self.mem = config.mem
        self.walltime = config.walltime
        self.venv_name = config.venv_name
        self.queue = config.queue
        self.name = name
        self._status = JobStatus.UNPREPARED

    @property
    def script_path(self) -> Path:
        return self._script_path

    @script_path.setter
    def script_path(self, path: Path) -> None:
        self._script_path = path

    @property
    def output_folder(self) -> Path:
        return self.script_path.parent

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

    @property
    def walltime_str(self) -> str:
        return str(self.walltime).split("days")[-1].strip()

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
    def venv_activate_script(self) -> str:
        return f". /appli/anaconda/latest/etc/profile.d/conda.sh; conda activate {self.venv_name}"

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

    def write_pbs(self, path: Path) -> None:
        with path.open("w") as file:
            file.write(f"""#!/bin/bash
#PBS -N {self.name}
#PBS -q {self.queue}
#PBS -l select={self.chunks}:ncpus={self.ncpus}:mem={self.mem}
#PBS -l walltime={self.walltime_str}
#PBS -o {self.output_folder / self.name!s}.out
#PBS -e {self.output_folder / self.name!s}.err

{self.venv_activate_script}
python {self.script_path!s} {" ".join(self.script_args)}
""")
        self.progress()
