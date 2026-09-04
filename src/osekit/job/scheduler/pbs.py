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

    SUBMIT_CMD: typing.ClassVar = "qsub"
    INFO_CMD: typing.ClassVar = ["qstat", "-x"]

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

    @staticmethod
    def _build_venv_string(job: Job) -> str:
        """Bash script used for activating the conda virtual environment."""
        return (
            f". /appli/anaconda/latest/etc/profile.d/conda.sh\n"
            f"conda activate {job.venv_name}"
        )

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
        '-W depend=afterok:1234567'
        >>> Pbs._build_dependency_string({"afterok": ["1234567","4567891"]})
        '-W depend=afterok:1234567:4567891'
        >>> from pathlib import Path
        >>> job = Job(Path())
        >>> job._id = "7894561"
        >>> Pbs._build_dependency_string({"afterany":job})
        '-W depend=afterany:7894561'
        >>> from pathlib import Path
        >>> job1 = Job(Path())
        >>> job1._id = "7894561"
        >>> job2 = Job(Path())
        >>> job2._id = "4839572"
        >>> Pbs._build_dependency_string({"afterany":[job1,job2]})
        '-W depend=afterany:7894561:4839572'

        """
        # Check that types are valid before submitting
        for dependency_type in dependencies:
            cls._validate_dependency_type(dependency_type=dependency_type)

        id_str = cls._parse_job_ids(dependencies=dependencies)

        return "-W depend=" + ",".join(
            f"{dependency_type}:{':'.join(ids)}"
            for dependency_type, ids in id_str.items()
        )
