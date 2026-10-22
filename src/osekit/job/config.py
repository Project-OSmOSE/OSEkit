from dataclasses import dataclass

from pandas import Timedelta


@dataclass
class JobConfig:
    """Configuration of the computing resources allowed for a job.

    Parameters
    ----------
    nb_nodes: int
        Number of nodes on which the job runs.
    ncpus: int
        Number of total cores used per node.
    ngpus: int | None
        Number of total GPU used per node.
    mem: str
        Maximum amount of physical memory used by the job.
    walltime: str | Timedelta
        Maximum amount of real time during which the job can be running.
    venv_name: str
        Name (or path) of the conda virtual environment in which the job is running.

    """

    nb_nodes: int = 1
    ncpus: int = 2
    ngpus: int | None = None
    mem: str = "8gb"
    walltime: str | Timedelta = "01:00:00"
    venv_name: str = "osekit"
