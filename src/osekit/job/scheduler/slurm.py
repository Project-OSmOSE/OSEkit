import subprocess
import typing
from typing import Literal

from osekit.job.job import Job, JobStatus
from osekit.job.scheduler.scheduler import Scheduler


class Slurm(Scheduler):
    """Abstract class representing a job scheduler."""

    _VALID_DEPENDENCY_TYPES: typing.ClassVar = frozenset(
        {
            "after",
            "afterany",
            "afterburstbuffer",
            "aftercorr",
            "afternotok",
            "afterok",
            "singleton",
        },
    )
    JOB_FILE_EXTENSION: typing.ClassVar = "slurm"
    SUBMIT_CMD: typing.ClassVar = "sbatch"
    JOB_STATUS_CODES: typing.ClassVar = {
        "PENDING": JobStatus.QUEUED,
        "RUNNING": JobStatus.RUNNING,
        "SUSPENDED": JobStatus.SUSPENDED,
        "COMPLETING": JobStatus.RUNNING,
        "COMPLETED": JobStatus.COMPLETED,
    }

    def __init__(self, partition: Literal["cpu", "gpu", "ops"] = "cpu") -> None:
        """Initialize the SLURM scheduler."""
        self.partition = partition

    @property
    def partition(self) -> str:
        """Partition in which the job will be submitted."""
        return self._partition

    @partition.setter
    def partition(self, partition: Literal["omp", "mpi"]) -> None:
        self._partition = partition

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
        specifications = {
            "nodes": job.nb_nodes,
            "cpus-per-task": job.ncpus,
            "mem": job.mem,
            "job-name": job.name,
            "partition": self.partition,
            "time": job.walltime_str,
            "output": f"{job.output_folder / job.name}.out"
            if job.output_folder
            else None,
            "error": f"{job.output_folder / job.name}.err"
            if job.output_folder
            else None,
        }

        if job.ngpus is not None:
            specifications["gpus"] = job.ngpus

        return "\n".join(
            f"#SBATCH --{key}={value}" for key, value in specifications.items() if value
        )

    @staticmethod
    def _parse_job_id(submit_output: str) -> str:
        r"""Parse the output of the submit command.

        Parameters
        ----------
        submit_output: str
            stdout after a successful sbatch cmd.

        Returns
        -------
        str:
            ID of the submitted job.

        Examples
        --------
        >>> Slurm._parse_job_id(submit_output="Submitted batch job 3647090\n")
        '3647090'

        """
        return submit_output.removeprefix("Submitted batch job ").strip("\n")

    @classmethod
    def _get_info(cls, job: Job) -> str:
        """Request information about a job.

        Parameters
        ----------
        job: Job
            Job for which the information is requested.

        Returns
        -------
        str:
            The information string, as returned by the ``squeue`` command.
            If the job is complete and doesn't appear in the queue,
            the info is requested through the ``sacct`` command, parsed
            by a specific parser, and an empty string is returned.

        """
        request = subprocess.run(
            [
                "/usr/bin/squeue",
                "--jobs",
                str(job.job_id),
                "--noheader",
                (
                    "--format="
                    '"%i|'  # ID
                    "%P|"  # Partition
                    "%j|"  # Job name
                    "%u|"  # User name
                    "%T|"  # Status
                    "%M|"  # Time
                    "%D"  # Nodes nb
                    '|%R"'
                ),  # Nodes List (Reason)
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if info := request.stdout.strip():
            return info

        # The job doesn't appear in squeue anymore
        request = subprocess.run(
            [
                "/usr/bin/sacct",
                "--jobs",
                str(job.job_id),
                "--allocations",
                "--noheader",
                "--parsable2",
                "--format=JobID,Partition,JobName,User,State,Elapsed,NNodes,NodeList,Reason",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if acct_info := request.stdout.strip():
            return acct_info

        if error := request.stderr.strip():
            msg = f"{job.job_id}: {error}"
            raise ValueError(msg)

        return ""

    @classmethod
    def _parse_info_str(cls, job: Job, info: str) -> None:
        """Parse the info from the requested squeue info string."""
        values = info.strip().split("|", maxsplit=7)

        (
            _job_id,
            partition,
            _name,
            user,
            status,
            time,
            nodes,
            node_list,
        ) = values

        job.info["user"] = user
        job.info["time"] = time
        job.info["partition"] = partition

        # If info is fetched by sacct, node_list could contain
        # both the node name and the reason, sill separated by a "|"
        if "|" in node_list:
            node, reason = node_list.split("|", maxsplit=1)
            node_list = f"{node} ({reason})"
        job.info["node_list"] = node_list

        if status := cls.JOB_STATUS_CODES.get(status, False):
            job.status = status

    @staticmethod
    def _build_venv_string(job: Job) -> str:
        """Bash script used for activating the conda virtual environment."""
        return f"module load conda\nconda activate {job.venv_name}"

    @classmethod
    def _build_dependency_string(
        cls,
        dependencies: dict[str, Job | str | list[Job | str]],
        *,
        instructions_or: bool = False,
    ) -> str:
        """Build a job dependency string.

        Parameters
        ----------
        dependencies: dict[str, Job | str | list[Job|str]]
            The dependencies of the submitted job.
            The keys of the dictionary are the dependency types,
            see https://slurm.schedmd.com/sbatch.html
            The values are the  other jobs (or their ID) ``job`` depends on
            with the given dependency type.
            If ``None``, the job is submitted without any dependency.
        instructions_or: bool, optional
            If ``True``, the instructions in the ``dependencies`` list
            are joined with a logical OR (``?`` character in Slurm).
            If ``False``, the instructions are joined with a logical
            AND (``,`` character in Slurm).

        Returns
        -------
        str
            Job dependency string.

        Examples
        --------
        >>> Slurm._build_dependency_string({"afterok": "1234567"})
        '-d afterok:1234567'
        >>> Slurm._build_dependency_string({"afterok": ["1234567","4567891"]})
        '-d afterok:1234567:4567891'
        >>> Slurm._build_dependency_string({"afterok": ["1234567","4567891"], "afterany":"7654321"}, instructions_or=True)
        '-d afterok:1234567:4567891?afterany:7654321'
        >>> from pathlib import Path
        >>> job = Job(Path())
        >>> job._id = "7894561"
        >>> Slurm._build_dependency_string({"afterany":job})
        '-d afterany:7894561'
        >>> from pathlib import Path
        >>> job1 = Job(Path())
        >>> job1._id = "7894561"
        >>> job2 = Job(Path())
        >>> job2._id = "4839572"
        >>> Slurm._build_dependency_string({"afterany":[job1,job2]})
        '-d afterany:7894561:4839572'

        """  # noqa: E501
        # Check that types are valid before submitting
        for dependency_type in dependencies:
            cls._validate_dependency_type(dependency_type=dependency_type)

        id_str = cls._parse_job_ids(dependencies=dependencies)

        logical_join_character = "?" if instructions_or else ","

        return "-d " + logical_join_character.join(
            f"{dependency_type}:{':'.join(ids)}"
            for dependency_type, ids in id_str.items()
        )
