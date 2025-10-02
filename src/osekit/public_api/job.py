import subprocess
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
    chunks: int = 1
    ncpus: int = 2
    mem: str = "8gb"
    walltime: str | Timedelta = "01:00:00"
    venv_name: str = "osmose"
    queue: Literal["omp", "mpi"] = "omp"


class Job:
    def __init__(
        self,
        script_path: Path,
        script_args: dict | None = None,
        config: JobConfig | None = None,
        name: str = "osekit_analysis",
        output_folder: Path | None = None,
    ) -> None:
        config = JobConfig() if config is None else config
        self.script_path = script_path
        self.script_args = script_args if script_args else {}
        self.chunks = config.chunks
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
        return self._script_path

    @script_path.setter
    def script_path(self, path: Path) -> None:
        self._script_path = path

    @property
    def script_args(self) -> dict:
        return self._script_args

    @script_args.setter
    def script_args(self, args: dict) -> None:
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

    @status.setter
    def status(self, status: JobStatus) -> None:
        self._status = status

    @property
    def path(self) -> Path | None:
        return self._path

    @path.setter
    def path(self, path: Path) -> None:
        self._path = path

    @property
    def output_folder(self) -> Path | None:
        return self._output_folder

    @output_folder.setter
    def output_folder(self, output_folder: Path | None) -> None:
        self._output_folder = output_folder

    @property
    def job_id(self) -> str:
        return self._id

    @job_id.setter
    def job_id(self, job_id: str) -> None:
        self._id = job_id

    @property
    def job_info(self) -> dict | None:
        return self._info

    @job_info.setter
    def job_info(self, info: dict) -> None:
        self._info = info

    def progress(self) -> None:
        if self.status == JobStatus.COMPLETED:
            return
        self._status = JobStatus(self._status.value + 1)

    def write_pbs(self, path: Path) -> None:
        preamble = "#!/bin/bash"
        request = {
            "-N": self.name,
            "-q": self.queue,
            "-l": [
                f"select={self.chunks}:ncpus={self.ncpus}:mem={self.mem}",
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
        script = f"python {self.script_path} {' '.join(f'--{key} {value}' for key, value in self.script_args.items())}"

        pbs = "\n".join((preamble, request_str, self.venv_activate_script, script))
        with path.open("w") as file:
            file.write(pbs)

        self.path = path
        self.progress()

    def submit_pbs(self) -> None:
        request = subprocess.run(
            ["qsub", self.path], capture_output=True, text=True, check=False
        )
        self.job_id = request.stdout.split(".", maxsplit=1)[0].strip()
        self.progress()

    def update_info(self) -> None:
        if self.job_id is None:
            return

        request = subprocess.run(
            ["qstat", "-f", self.job_id], capture_output=True, text=True, check=True
        )
        stdout = request.stdout

        if not stdout:
            err = request.stderr
            if "Job has finished" in err:
                self.status = JobStatus.COMPLETED
            if "Unknown Job Id" in err:
                raise ValueError(f"Unknown Job Id {self.job_id}")
            return

        info = {}
        for line in stdout.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            info[key.strip()] = value.strip()
        self.job_info = info
