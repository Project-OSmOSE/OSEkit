import subprocess
from pathlib import Path
from typing import Literal

from osekit.job.job import Job, JobStatus
from osekit.job.scheduler.scheduler import Scheduler


class Pbs(Scheduler):
    """Abstract class representing a job scheduler."""

    _VALID_DEPENDENCY_TYPES = frozenset({"afterok", "afterany", "afternotok", "after"})
    JOB_FILE_EXTENSION = "pbs"

    def __init__(self, queue: Literal["omp", "mpi"] = "omp") -> None:
        """Initialize the PBS scheduler."""
        self.queue = queue

    @property
    def queue(self) -> str:
        """Queue in which the job will be submitted."""
        return self._queue

    @queue.setter
    def queue(self, queue: Literal["omp", "mpi"]) -> None:
        self._queue = queue

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

        select_parts = {
            "select": job.nb_nodes,
            "ncpus": job.ncpus,
            "mem": job.mem,
        }
        if job.ngpus is not None:
            select_parts["ngpus"] = job.ngpus
        select_str = ":".join(f"{k}={v}" for k, v in select_parts.items())

        request = {
            "-N": job.name,
            "-q": self.queue,
            "-l": [
                select_str,
                f"walltime={job.walltime_str}",
            ],
            "-o": f"{job.output_folder}/{job.name}.out" if job.output_folder else None,
            "-e": f"{job.output_folder}/{job.name}.err" if job.output_folder else None,
        }
        request_str = "\n".join(
            f"#PBS {key} {value}"
            if type(value) is not list
            else "\n".join(f"#PBS {key} {value_part}" for value_part in value)
            for key, value in request.items()
            if value
        )

        script = f"python {job.script_path} {job.get_arg_string()}"

        pbs = f"{preamble}\n{request_str}\n{self._build_venv_string(job=job)}\n{script}"
        with path.open("w") as file:
            file.write(pbs)

        job.path = path
        job.progress()

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
        if self.update_status(job=job) is not JobStatus.PREPARED:
            msg = "Job should be written before being submitted."
            raise ValueError(msg)

        cmd = ["qsub"]

        if dependency is not None:
            dependency_str = self._build_dependency_string(dependency)
            if dependency_str:
                cmd.extend(["-W", f"depend={dependency_str}"])

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
        """Request info about the job and update it."""
        if job.job_id is None:
            return

        try:
            request = subprocess.run(
                ["qstat", "-f", str(job.job_id)],
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
                job.status = JobStatus.COMPLETED
                job.job_info["job_state"] = "C"
            if "Unknown Job Id" in err:
                msg = f"Unknown Job Id {job.job_id}"
                raise ValueError(msg)
            return

        info = {}
        for line in stdout.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            info[key.strip()] = value.strip()
        job.job_info = info

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

        if job.status == JobStatus.COMPLETED:
            return job.status

        job_state = {
            "Q": JobStatus.QUEUED,
            "R": JobStatus.RUNNING,
        }
        if job.job_info["job_state"] in job_state:
            job.status = job_state[job.job_info["job_state"]]
        return job.status

    @staticmethod
    def _build_venv_string(job: Job) -> str:
        """Bash script used for activating the conda virtual environment."""
        return f". /appli/anaconda/latest/etc/profile.d/conda.sh; conda activate {job.venv_name}"

    @classmethod
    def _validate_dependency_type(cls, dependency_type: str) -> None:
        if dependency_type not in cls._VALID_DEPENDENCY_TYPES:
            msg = (
                f"Unsupported dependency type '{dependency_type}'. "
                f"Expected one of {cls._VALID_DEPENDENCY_TYPES}."
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

    @classmethod
    def _build_dependency_string(
        cls,
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
        >>> Pbs._build_dependency_string("1234567")
        'afterok:1234567'
        >>> Pbs._build_dependency_string(["1234567", "4567891"])
        'afterok:1234567:4567891'
        >>> Pbs._build_dependency_string("7894561", dependency_type="afterany")
        'afterany:7894651'

        """
        dependency = dependency if isinstance(dependency, list) else [dependency]
        id_str = cls._validate_dependency(dependency=dependency)
        cls._validate_dependency_type(dependency_type=dependency_type)

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
