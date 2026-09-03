import subprocess
import typing
from typing import Literal

from osekit.job.job import Job, JobStatus
from osekit.job.scheduler.scheduler import Scheduler


class Pbs(Scheduler):
    """Abstract class representing a PBS job scheduler."""

    _VALID_DEPENDENCY_TYPES: typing.ClassVar = frozenset(
        {
            "after",
            "afterok",
            "afternotok",
            "afterany",
            "before",
            "beforeok",
            "beforenotok",
            "beforeany",
            "on",
            "runone",
        },
    )
    JOB_FILE_EXTENSION: typing.ClassVar = "pbs"

    JOB_STATUS_CODES: typing.ClassVar = {
        "Q": JobStatus.QUEUED,
        "R": JobStatus.RUNNING,
        "S": JobStatus.SUSPENDED,
        "H": JobStatus.SUSPENDED,
        "E": JobStatus.COMPLETED,
        "F": JobStatus.COMPLETED,
    }

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
            "-o": f"{job.output_folder / job.name}.out" if job.output_folder else None,
            "-e": f"{job.output_folder / job.name}.err" if job.output_folder else None,
        }
        return "\n".join(
            f"#PBS {key} {value}"
            if type(value) is not list
            else "\n".join(f"#PBS {key} {value_part}" for value_part in value)
            for key, value in request.items()
            if value
        )

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
            see https://help.altair.com/2022.1.0/PBS%20Professional/PBSReferenceGuide2022.1.pdf#page=151
            for the list of supported values.
            The values are the  other jobs (or their ID) ``job`` depends on
            with the given dependency type.
            If ``None``, the job is submitted without any dependency.

        """
        if self.update_status(job=job) is not JobStatus.PREPARED:
            msg = "Job should be written before being submitted."
            raise ValueError(msg)

        cmd = ["qsub"]

        if dependencies is not None:
            dependency_str = self._build_dependency_string(dependencies)
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
                ["qstat", "-x", str(job.job_id)],
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
            if "Unknown Job Id" in err:
                msg = f"Unknown Job Id {job.job_id}"
                raise ValueError(msg)
            return

        self._parse_info_str(job=job, info=stdout)

    @classmethod
    def _parse_info_str(cls, job: Job, info: str) -> None:
        """Parse the info from the requested qstat info string."""
        keys, _, values = info.splitlines()

        # Get keys order in the string
        known_keys = ["Job id", "Name", "User", "Time Use", "S", "Queue"]
        keys = sorted(known_keys, key=keys.index)

        # Get the associated values
        kvp = dict(zip(keys, values.split(), strict=True))

        job.info["user"] = kvp["User"]
        job.info["time"] = kvp["Time Use"]
        job.info["queue"] = kvp["Queue"]

        if status := cls.JOB_STATUS_CODES.get(kvp["S"], False):
            job.status = status

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
        if job.info["job_state"] in job_state:
            job.status = job_state[job.info["job_state"]]
        return job.status

    @staticmethod
    def _build_venv_string(job: Job) -> str:
        """Bash script used for activating the conda virtual environment."""
        return (
            f". /appli/anaconda/latest/etc/profile.d/conda.sh\n"
            f"conda activate {job.venv_name}"
        )

    @classmethod
    def _validate_dependency_type(cls, dependency_type: str) -> None:
        if dependency_type not in cls._VALID_DEPENDENCY_TYPES:
            msg = (
                f"Unsupported dependency type '{dependency_type}'.\n"
                f"Expected one of:\n\t{'\n\t'.join(sorted(cls._VALID_DEPENDENCY_TYPES))}."
            )
            raise ValueError(msg)

    @classmethod
    def _build_dependency_string(
        cls,
        dependencies: dict[str, Job | str | list[Job | str]],
    ) -> str:
        """Build a PBS dependency string.

        Parameters
        ----------
        dependencies: dict[str, Job | str | list[Job|str]]
            The dependencies of the submitted job.
            The keys of the dictionary are the dependency types,
            see https://help.altair.com/2022.1.0/PBS%20Professional/PBSReferenceGuide2022.1.pdf#page=151
            for the list of supported values.
            The values are the  other jobs (or their ID) ``job`` depends on
            with the given dependency type.

        Returns
        -------
        str
            PBS dependency string.

        Examples
        --------
        >>> Pbs._build_dependency_string({"afterok": "1234567"})
        'afterok:1234567'
        >>> Pbs._build_dependency_string({"afterok": ["1234567","4567891"]})
        'afterok:1234567:4567891'
        >>> from pathlib import Path
        >>> job = Job(Path())
        >>> job._id = "7894561"
        >>> Pbs._build_dependency_string({"afterany":job})
        'afterany:7894561'
        >>> from pathlib import Path
        >>> job1 = Job(Path())
        >>> job1._id = "7894561"
        >>> job2 = Job(Path())
        >>> job2._id = "4839572"
        >>> Pbs._build_dependency_string({"afterany":[job1,job2]})
        'afterany:7894561:4839572'

        """
        # Check that types are valid before submitting
        for dependency_type in dependencies:
            cls._validate_dependency_type(dependency_type=dependency_type)

        id_str = cls._parse_job_ids(dependencies=dependencies)

        return ",".join(
            f"{dependency_type}:{':'.join(ids)}"
            for dependency_type, ids in id_str.items()
        )
