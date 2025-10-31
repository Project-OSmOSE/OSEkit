import subprocess
from pathlib import Path

import pytest
from pandas import Timedelta

import osekit.utils.job as job_module
from osekit.utils.job import Job, JobBuilder, JobConfig, JobStatus


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
    nb_nodes = 2
    ncpus = 28
    job_config = JobConfig(
        nb_nodes=nb_nodes,
        ncpus=ncpus,
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
    assert job.nb_nodes == nb_nodes
    assert job.ncpus == ncpus
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


def test_submit_pbs_without_write_raises() -> None:
    job = Job(Path("script.py"))
    with pytest.raises(
        ValueError,
        match="Job should be written before being submitted.",
    ) as e:
        assert job.submit_pbs() == e


def test_submit_pbs_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "boc.py"
    script.write_text("")
    outdir = tmp_path
    job = Job(script, name="amobishoproden", output_folder=outdir)
    pbs_path = tmp_path / "amobishoproden.pbs"
    job.write_pbs(pbs_path)

    class Dummy:
        def __init__(self) -> None:
            """Dummy subprocess.run output."""
            self.stdout = "35173.server\n"
            self.stderr = ""

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: Dummy(),
    )

    assert job.status == JobStatus.PREPARED
    job.submit_pbs()

    assert job.job_id == "35173"
    assert job.status == JobStatus.QUEUED


def test_submit_pbs_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "boc.py"
    script.write_text("")
    outdir = tmp_path
    job = Job(script, name="amobishoproden", output_folder=outdir)
    pbs_path = tmp_path / "amobishoproden.pbs"
    job.write_pbs(pbs_path)

    class Dummy:
        def __init__(self) -> None:
            """Dummy subprocess.run output."""
            raise subprocess.CalledProcessError(5, "err")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: Dummy(),
    )

    assert job.status == JobStatus.PREPARED
    with pytest.raises(RuntimeError, match="Submission failed with exit code 5") as e:
        assert job.submit_pbs() == e

    assert job.status == JobStatus.PREPARED


def test_update_info_no_job_id() -> None:
    job = Job(Path("pixies.py"))
    job.job_id = None
    job.update_info()
    assert job.job_info is None


def test_update_info_parse_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    job = Job(Path("fontaines.py"))
    job.job_id = "43"
    job.status = JobStatus.RUNNING
    raw = " frankie = cosmos \navey=tare\nattic= abasement\nthis will be ignored"

    class Dummy:
        stdout = raw
        stderr = ""

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: Dummy(),
    )
    job.update_info()
    assert job.job_info == {"frankie": "cosmos", "avey": "tare", "attic": "abasement"}


def test_update_info_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    job = Job(Path("amok.py"))
    job.job_id = "25022013"
    job.job_info = {}

    class Dummy:
        stdout = ""
        stderr = "Atoms\nJob has finished\nFor peace"

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: Dummy(),
    )

    job.update_info()
    assert job.status == JobStatus.COMPLETED
    assert job.job_info["job_state"] == "C"


def test_update_info_unknown_job_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    job = Job(Path("pompom.py"))
    job.job_id = "17112014"

    class Dummy:
        stdout = ""
        stderr = "Error: Unknown Job Id 17112014"

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: Dummy(),
    )

    with pytest.raises(ValueError, match="Unknown Job Id 17112014") as e:
        assert job.update_info() == e


def test_update_info_error(monkeypatch: pytest.MonkeyPatch) -> None:
    job = Job(Path("pompom.py"))
    job.job_id = "17112014"

    class Dummy:
        def __init__(self) -> None:
            raise subprocess.CalledProcessError(5, "err")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: Dummy(),
    )

    with pytest.raises(RuntimeError, match="Qstat failed with exit code 5") as e:
        assert job.update_info() == e


def test_update_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = Job(Path("porticoquartet.py"))
    job.path = tmp_path / "pompidou.pbs"

    assert job.update_status() == JobStatus.UNPREPARED

    job.path.write_text("prickly pear")
    assert job.update_status() == JobStatus.PREPARED

    monkeypatch.setattr(
        job,
        "update_info",
        lambda: None,
    )

    job.job_info = {"job_state": "Q"}
    job.job_id = "5129195"
    assert job.update_status() == JobStatus.QUEUED
    assert job.status == JobStatus.QUEUED

    job.job_info = {"job_state": "R"}
    assert job.update_status() == JobStatus.RUNNING
    assert job.status == JobStatus.RUNNING

    job.status = JobStatus.COMPLETED
    assert job.update_status() == JobStatus.COMPLETED
    assert job.status == JobStatus.COMPLETED


def test_job_builder_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}

    class DummyJob:
        def __init__(self, **kwargs: dict) -> None:
            called["init_job"] = kwargs
            self.path = None
            self.status = JobStatus.UNPREPARED

        def write_pbs(self, path: Path) -> None:
            called["write_pbs"] = path
            self.path = path
            self.status = JobStatus.PREPARED

    monkeypatch.setattr(job_module, "Job", DummyJob)

    job_config = JobConfig(
        nb_nodes=2,
        ncpus=16,
        mem="60gb",
        walltime=Timedelta(hours=2),
        venv_name="abyssinie",
        queue="mpi",
    )

    job_builder = JobBuilder(config=job_config)

    assert job_builder.jobs == []

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    script = tmp_path / "script.py"
    script.write_text("")

    job_builder.create_job(
        script_path=script,
        script_args={"les": "fantômes", "de": "baleines"},
        name="idylle_des_abysses",
        output_folder=output_dir,
    )

    keywords = called["init_job"]
    assert keywords["script_path"] == script
    assert keywords["script_args"] == {"les": "fantômes", "de": "baleines"}
    assert keywords["name"] == "idylle_des_abysses"
    assert keywords["output_folder"] == output_dir

    assert len(job_builder.jobs) == 1

    assert called["write_pbs"] == output_dir / "idylle_des_abysses.pbs"

    assert job_builder.jobs[0].status == JobStatus.PREPARED


def test_job_builder_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    submitted_jobs = []

    class DummyJob:
        def __init__(self, name: str, status: JobStatus) -> None:
            self.name = name
            self.status = status

        def submit_pbs(self) -> None:
            submitted_jobs.append(self.name)

        def update_status(self) -> JobStatus:
            return self.status

    monkeypatch.setattr(job_module, "Job", DummyJob)

    jobs = [
        DummyJob(name="unprepared", status=JobStatus.UNPREPARED),
        DummyJob(name="prepared", status=JobStatus.PREPARED),
        DummyJob(name="queued", status=JobStatus.QUEUED),
        DummyJob(name="running", status=JobStatus.RUNNING),
        DummyJob(name="completed", status=JobStatus.COMPLETED),
    ]

    job_builder = JobBuilder()
    job_builder.jobs = jobs

    job_builder.submit_pbs()

    assert submitted_jobs == ["prepared"]
