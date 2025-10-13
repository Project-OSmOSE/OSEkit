from pathlib import Path

import pytest
from pandas import Timedelta

from osekit.public_api.job import Job, JobConfig, JobStatus


@pytest.mark.parametrize(
    ("initial_status", "expected_status"),
    [
        pytest.param(
            JobStatus.UNPREPARED,
            JobStatus.PREPARED,
            id="unprepared_becomes_prepared",
        ),
        pytest.param(
            JobStatus.PREPARED,
            JobStatus.QUEUED,
            id="prepared_becomes_queued",
        ),
        pytest.param(JobStatus.QUEUED, JobStatus.RUNNING, id="queued_becomes_running"),
        pytest.param(
            JobStatus.RUNNING,
            JobStatus.COMPLETED,
            id="running_becomes_completed",
        ),
        pytest.param(
            JobStatus.COMPLETED,
            JobStatus.COMPLETED,
            id="completed_remains_completed",
        ),
    ],
)
def test_job_progress(initial_status: JobStatus, expected_status: JobStatus) -> None:
    job = Job(script_path=Path())
    assert job.status == JobStatus.UNPREPARED
    job._status = initial_status
    job.progress()
    assert job.status == expected_status


def test_properties_and_venv_activation() -> None:
    script = Path("myscript.py")
    job_config = JobConfig(
        nb_nodes=2,
        ncpus=28,
        mem="16gb",
        walltime=Timedelta(hours=2),
        venv_name="merriweather",
        queue="mpi",
    )
    job = Job(
        script_path=script,
        script_args={"purple": "bottle"},
        config=job_config,
        name="post_pavillion",
        output_folder=Path("output"),
    )

    assert job.script_path == script
    assert job.script_args == {"purple": "bottle"}
    assert job.nb_nodes == 2
    assert job.ncpus == 28
    assert job.mem == "16gb"
    assert job.walltime == Timedelta(hours=2)
    assert job.venv_name == "merriweather"
    assert job.name == "post_pavillion"
    assert job.queue == "mpi"
    assert job.output_folder == Path("output")

    # venv activation
    expected = (
        ". /appli/anaconda/latest/etc/profile.d/conda.sh; conda activate merriweather"
    )
    assert job.venv_activate_script == expected


def test_progress_transitions() -> None:
    job = Job(Path("strawberry.py"))
    assert job.status == JobStatus.UNPREPARED
    for expected in (
        JobStatus.PREPARED,
        JobStatus.QUEUED,
        JobStatus.RUNNING,
        JobStatus.COMPLETED,
    ):
        job.progress()
        assert job.status == expected
    job.progress()
    assert job.status == JobStatus.COMPLETED


def test_walltime_str_and_setter() -> None:
    job = Job(Path("bossanova.py"))
    for walltime in ("13:08:09", Timedelta(hours=13, minutes=8, seconds=9)):
        job.walltime = walltime
        assert job.walltime == Timedelta("13:08:09")
        assert job.walltime_str == "13:08:09"


def test_write_pbs(tmp_path: Path) -> None:
    script = tmp_path / "shpouik_shpouik.py"
    script.write_text("print('edgar')")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    job = Job(
        script_path=script,
        script_args={"vieille": "face", "de": "rat"},
        name="berlioz",
        output_folder=output_dir,
    )
    pbs_path = tmp_path / "lafayette.pbs"
    job.write_pbs(pbs_path)

    content = pbs_path.read_text().splitlines()
    assert content[0] == "#!/bin/bash"
    assert any(line.startswith(f"#PBS -N {job.name}") for line in content)
    assert any(line.startswith("#PBS -q omp") for line in content)
    assert any("select=1:ncpus=2:mem=8gb" in line for line in content)
    assert any("walltime=01:00:00" in line for line in content)
    assert any(
        line.startswith(f"#PBS -o {job.output_folder}/{job.name}.out")
        for line in content
    )
    assert any(
        line.startswith(f"#PBS -e {job.output_folder}/{job.name}.err")
        for line in content
    )

    assert (
        ". /appli/anaconda/latest/etc/profile.d/conda.sh; conda activate osmose"
        in content
    )
    last = content[-1]
    assert last.startswith(f"python {script}")
    assert "--vieille face" in last
    assert "--de rat" in last

    assert job.path == pbs_path
    assert job.status == JobStatus.PREPARED
