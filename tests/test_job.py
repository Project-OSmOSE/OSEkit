from __future__ import annotations

import subprocess
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
from pandas import Timedelta

from osekit.job.builder import JobBuilder
from osekit.job.config import JobConfig
from osekit.job.job import Job, JobStatus
from osekit.job.scheduler.pbs import Pbs
from osekit.job.scheduler.scheduler import Scheduler


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


def test_properties() -> None:
    script = Path("myscript.py")
    nb_nodes = 2
    ncpus = 28
    job_config = JobConfig(
        nb_nodes=nb_nodes,
        ncpus=ncpus,
        mem="16gb",
        walltime=Timedelta(hours=2),
        venv_name="merriweather",
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
    assert job.ngpus is None
    assert job.mem == "16gb"
    assert job.walltime == Timedelta(hours=2)
    assert job.venv_name == "merriweather"
    assert job.name == "post_pavillion"
    assert job.output_folder == Path("output")


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
        script_args={"vieille": "face", "de": "rat", "cool": False, "fun": True},
        name="berlioz",
        output_folder=output_dir,
    )
    pbs_path = tmp_path / "lafayette.pbs"

    pbs_scheduler = Pbs(queue="omp")
    pbs_scheduler.write(job=job, path=pbs_path)

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
        ". /appli/anaconda/latest/etc/profile.d/conda.sh; conda activate osekit"
        in content
    )
    last = content[-1]
    assert last.startswith(f"python {script}")
    assert "--vieille face" in last
    assert "--de rat" in last
    assert "--no-cool" in last
    assert "--fun" in last

    assert job.path == pbs_path
    assert job.status == JobStatus.PREPARED


def test_write_pbs_job_with_gpu(tmp_path: Path) -> None:
    script = tmp_path / "deville.py"
    script.write_text("print('cruella')")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    job = Job(
        script_path=script,
        script_args={"cruelle": "diablesse"},
        name="penny",
        config=JobConfig(ngpus=2),
        output_folder=output_dir,
    )
    pbs_path = tmp_path / "patch.pbs"
    pbs_scheduler = Pbs(queue="omp")
    pbs_scheduler.write(job=job, path=pbs_path)

    content = pbs_path.read_text().splitlines()
    assert any("select=1:ncpus=2:mem=8gb:ngpus=2" in line for line in content)
    last = content[-1]
    assert last.startswith(f"python {script}")
    assert "--cruelle diablesse" in last


def test_submit_pbs_without_write_raises() -> None:
    job = Job(Path("script.py"))
    pbs = Pbs(queue="omp")
    with pytest.raises(
        ValueError,
        match=r"Job should be written before being submitted.",
    ):
        pbs.submit(job=job)


def test_submit_pbs_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "boc.py"
    script.write_text("")
    outdir = tmp_path
    job = Job(script, name="amobishoproden", output_folder=outdir)
    pbs_path = tmp_path / "amobishoproden.pbs"
    pbs = Pbs()
    pbs.write(job=job, path=pbs_path)

    class Dummy:
        def __init__(self) -> None:
            """Dummy subprocess.run."""
            self.stdout = "35173.server\n"
            self.stderr = ""

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: Dummy(),
    )

    updated_jobs = []

    def mock_update_status(self, job: Job) -> JobStatus:
        updated_jobs.append(job)
        return JobStatus.PREPARED

    monkeypatch.setattr(Pbs, "update_status", mock_update_status)

    assert job.status == JobStatus.PREPARED
    pbs.submit(job=job)

    assert job.job_id == "35173"
    assert np.array_equal(
        updated_jobs,
        [job] * 2,
    )  # Call before and after submitting the job


def test_submit_pbs_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "boc.py"
    script.write_text("")
    outdir = tmp_path
    job = Job(script, name="amobishoproden", output_folder=outdir)
    pbs_path = tmp_path / "amobishoproden.pbs"
    pbs_scheduler = Pbs(queue="omp")
    pbs_scheduler.write(job=job, path=pbs_path)

    class Dummy:
        def __init__(self) -> None:
            """Dummy subprocess.run."""
            raise subprocess.CalledProcessError(5, "err")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: Dummy(),
    )

    assert job.status == JobStatus.PREPARED
    with pytest.raises(RuntimeError, match="Submission failed with exit code 5"):
        pbs_scheduler.submit(job=job)

    assert job.status == JobStatus.PREPARED


def test_update_info_no_job_id() -> None:
    job = Job(Path("pixies.py"))
    pbs_scheduler = Pbs()
    job.job_id = None
    pbs_scheduler.update_info(job=job)
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
    scheduler = Pbs()
    scheduler.update_info(job=job)
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

    scheduler = Pbs()
    scheduler.update_info(job=job)
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

    scheduler = Pbs()
    with pytest.raises(ValueError, match="Unknown Job Id 17112014"):
        scheduler.update_info(job=job)


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

    scheduler = Pbs()
    with pytest.raises(RuntimeError, match="Qstat failed with exit code 5"):
        scheduler.update_info(job=job)


def test_update_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = Job(Path("porticoquartet.py"))
    job.path = tmp_path / "pompidou.pbs"

    scheduler = Pbs()

    assert scheduler.update_status(job=job) == JobStatus.UNPREPARED

    job.path.write_text("prickly pear")
    assert scheduler.update_status(job=job) == JobStatus.PREPARED

    monkeypatch.setattr(
        scheduler,
        "update_info",
        lambda job: None,
    )

    job.job_info = {"job_state": "Q"}
    job.job_id = "5129195"
    assert scheduler.update_status(job=job) == JobStatus.QUEUED
    assert job.status == JobStatus.QUEUED

    job.job_info = {"job_state": "R"}
    assert scheduler.update_status(job=job) == JobStatus.RUNNING
    assert job.status == JobStatus.RUNNING

    job.status = JobStatus.COMPLETED
    assert scheduler.update_status(job=job) == JobStatus.COMPLETED
    assert job.status == JobStatus.COMPLETED


def test_pbs_job_builder_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}

    class DummyJob:
        def __init__(self, **kwargs: dict) -> None:
            called["init_job"] = kwargs
            self.path = None
            self.status = JobStatus.UNPREPARED

    def mock_write(self: Pbs, job: Job, path: Path) -> None:
        called["write_pbs"] = path
        self.path = path
        job.status = JobStatus.PREPARED

    monkeypatch.setattr("osekit.job.builder.Job", DummyJob)
    monkeypatch.setattr(Pbs, "write", mock_write)

    job_config = JobConfig(
        nb_nodes=2,
        ncpus=16,
        mem="60gb",
        walltime=Timedelta(hours=2),
        venv_name="abyssinie",
    )

    job_builder = JobBuilder(scheduler=Pbs(), config=job_config)

    assert job_builder.jobs == []

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    script = tmp_path / "script.py"
    script.write_text("")

    job_builder.create_job(
        script_path=script,
        script_args={"les": "fantômes", "de": "baleines", "bool": False},
        name="idylle_des_abysses",
        output_folder=output_dir,
    )

    keywords = called["init_job"]
    assert keywords["script_path"] == script
    assert keywords["script_args"] == {
        "les": "fantômes",
        "de": "baleines",
        "bool": False,
    }
    assert keywords["name"] == "idylle_des_abysses"
    assert keywords["output_folder"] == output_dir

    assert len(job_builder.jobs) == 1

    assert called["write_pbs"] == output_dir / "idylle_des_abysses.pbs"

    assert job_builder.jobs[0].status == JobStatus.PREPARED


def test_build_arg_string_booleans(tmp_path: Path):
    job_builder = JobBuilder()
    assert job_builder.jobs == []

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    script = tmp_path / "script.py"
    script.write_text("")

    job_builder.create_job(
        script_path=script,
        script_args={
            "danser": False,
            "avec": True,
            "le": 0.3,
            "vent": "test",
        },
        name="danser_avec_le_vent",
        output_folder=output_dir,
    )

    job = next(iter(job_builder.jobs))
    arg_str = job.get_arg_string()

    assert arg_str == "--no-danser --avec --le 0.3 --vent test"


def test_job_builder_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    submitted_jobs = []

    class DummyJob:
        def __init__(self, name: str, status: JobStatus) -> None:
            self.name = name
            self.status = status

    def mock_submit(
        self: Scheduler, job: Job, dependency: Job | str | None = None
    ) -> None:
        submitted_jobs.append((job.name, dependency))

    def mock_update_status(self: Scheduler, job: Job) -> JobStatus:
        return job.status

    monkeypatch.setattr("osekit.job.job.Job", DummyJob)
    monkeypatch.setattr(Pbs, "submit", mock_submit)
    monkeypatch.setattr(Pbs, "update_status", mock_update_status)

    jobs = [
        DummyJob(name="unprepared", status=JobStatus.UNPREPARED),
        DummyJob(name="prepared", status=JobStatus.PREPARED),
        DummyJob(name="queued", status=JobStatus.QUEUED),
        DummyJob(name="running", status=JobStatus.RUNNING),
        DummyJob(name="completed", status=JobStatus.COMPLETED),
    ]

    job_builder = JobBuilder()
    job_builder.jobs = jobs

    dependencies = {"prepared": jobs[0]}

    job_builder.submit(dependencies=dependencies)

    assert submitted_jobs == [("prepared", jobs[0])]


@pytest.mark.parametrize(
    ("dependency", "ids", "status", "expected"),
    [
        pytest.param(
            ["1234567"],
            [None],
            [None],
            nullcontext("afterok:1234567"),
            id="single_job_id",
        ),
        pytest.param(
            ["1234567", "4567891", "7891234"],
            [None] * 3,
            [None] * 3,
            nullcontext("afterok:1234567:4567891:7891234"),
            id="multiple_job_ids",
        ),
        pytest.param(
            ["123"],
            [None],
            [None],
            pytest.raises(
                ValueError,
                match=r"Invalid job ID '123'\. Job IDs must be 7 digits long\.",
            ),
            id="invalid_job_id_too_short",
        ),
        pytest.param(
            [Job(script_path=Path("test.py"), name="job_1")],
            ["12345678"],
            [JobStatus.QUEUED],
            pytest.raises(
                ValueError,
                match=r"Invalid job ID '12345678'\. Job IDs must be 7 digits long\.",
            ),
            id="invalid_job_id_too_long",
        ),
        pytest.param(
            ["abcdefg"],
            [None],
            [None],
            pytest.raises(
                ValueError,
                match=r"Invalid job ID 'abcdefg'\. Job IDs must be 7 digits long\.",
            ),
            id="invalid_job_id_non_numeric",
        ),
        pytest.param(
            ["1234567", "not_a_job_id"],
            [None] * 2,
            [None] * 2,
            pytest.raises(
                ValueError,
                match=r"Invalid job ID 'not_a_job_id'\. Job IDs must be 7 digits long\.",
            ),
            id="multiple_job_id_one_invalid",
        ),
        pytest.param(
            [Job(script_path=Path("test.py"), name="job_1")],
            ["1234567"],
            [JobStatus.QUEUED],
            nullcontext("afterok:1234567"),
            id="single_job_instance",
        ),
        pytest.param(
            [
                Job(script_path=Path("horse_with.py"), name="job_1"),
                Job(script_path=Path("no_name.py"), name="job_2"),
            ],
            ["1234567", "4567891"],
            [JobStatus.QUEUED, JobStatus.QUEUED],
            nullcontext("afterok:1234567:4567891"),
            id="multiple_job_instance",
        ),
        pytest.param(
            [
                Job(script_path=Path("king_crimson.py"), name="job_1"),
                Job(script_path=Path("crimson_king.py"), name="job_2"),
            ],
            ["1234567", "not_an_id"],
            [JobStatus.QUEUED, JobStatus.QUEUED],
            pytest.raises(
                ValueError,
                match=r"Invalid job ID 'not_an_id'\. Job IDs must be 7 digits long\.",
            ),
            id="multiple_job_instance_invalid_one",
        ),
        pytest.param(
            [
                Job(script_path=Path("king_crimson.py"), name="job_1"),
                "9876543",
            ],
            ["1234567", None],
            [JobStatus.QUEUED, None],
            nullcontext("afterok:1234567:9876543"),
            id="job_and_string_input",
        ),
        pytest.param(
            [Job(script_path=Path("test.py"), name="tornero")],
            ["1234567"],
            [JobStatus.UNPREPARED],
            pytest.raises(
                ValueError,
                match="Job 'tornero' has not been submitted yet.",
            ),
            id="unprepared_job_instance",
        ),
        pytest.param(
            [
                Job(script_path=Path("script.py"), name="dalida"),
                Job(script_path=Path("script.py"), name="mourir_sur_scene"),
            ],
            ["1234567", "4567896"],
            [JobStatus.QUEUED, JobStatus.PREPARED],
            pytest.raises(
                ValueError,
                match="Job 'mourir_sur_scene' has not been submitted yet.",
            ),
            id="multiple_job_instance_one_not_submitted",
        ),
    ],
)
def test_pbs_build_dependency_string_with_string_input(
    dependency: list[str] | list[Job],
    ids: list[str] | None,
    status: list[JobStatus],
    expected: str | None,
) -> None:
    """Test building PBS dependency string from string and Job inputs."""
    scheduler = Pbs()
    for dep, id, st in zip(dependency, ids, status, strict=True):
        if isinstance(dep, Job):
            dep.status = st
            dep.job_id = id

    with expected as e:
        assert scheduler._build_dependency_string(dependency=dependency) == e


def test_submit_pbs_adds_dependency_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "script.py"
    script.write_text("")
    scheduler = Pbs()
    job = Job(script, name="crazy_diamond", output_folder=tmp_path)
    scheduler.write(job=job, path=tmp_path / "wywh.pbs")

    captured_cmd = {}

    class Dummy:
        stdout = "1234567.server\n"
        stderr = ""

    def fake_run(cmd: list[str], *args: None, **kwargs: None) -> Dummy:
        captured_cmd["cmd"] = cmd
        return Dummy()

    def mock_update_status(self: Pbs, job: Job) -> JobStatus:
        return JobStatus.PREPARED

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(Pbs, "update_status", mock_update_status)

    scheduler.submit(job=job, dependency="1234567")

    assert "-W" in captured_cmd["cmd"]
    assert "depend=afterok:1234567" in captured_cmd["cmd"]


@pytest.mark.parametrize(
    ("dependency_type", "expected"),
    [
        pytest.param("afterok", nullcontext("afterok:1234567"), id="afterok"),
        pytest.param("afterany", nullcontext("afterany:1234567"), id="afterany"),
        pytest.param("afternotok", nullcontext("afternotok:1234567"), id="afternotok"),
        pytest.param("after", nullcontext("after:1234567"), id="after"),
        pytest.param(
            "not_a_supported_type",
            pytest.raises(
                ValueError,
                match=r"Unsupported dependency type 'not_a_supported_type'",
            ),
            id="invalid_dependency_type",
        ),
    ],
)
def test_pbs_build_dependency_string_with_different_types(
    dependency_type: str,
    expected: type[Exception],
) -> None:
    """Test building dependency strings with different dependency types."""
    scheduler = Pbs()
    with expected as e:
        assert (
            scheduler._build_dependency_string(
                dependency="1234567", dependency_type=dependency_type
            )
            == e
        )


@pytest.mark.parametrize(
    "walltime",
    [
        pytest.param(
            "01:00:00",
            id="hours_only_str",
        ),
        pytest.param(
            "01:24:32",
            id="hours_minutes_seconds_str",
        ),
        pytest.param(
            "30:12:10",
            id="more_than_a_day_str",
        ),
        pytest.param(
            Timedelta(hours=1, minutes=0, seconds=0),
            id="hours_only_timedelta",
        ),
        pytest.param(
            Timedelta(hours=1, minutes=24, seconds=32),
            id="hours_minutes_seconds_timedelta",
        ),
        pytest.param(
            Timedelta(hours=30, minutes=12, seconds=10),
            id="more_than_a_day_timedelta",
        ),
    ],
)
def test_job_walltime(walltime: str | Timedelta) -> None:
    job = Job(Path(), config=JobConfig(walltime=walltime))
    assert Timedelta(job.walltime_str) == Timedelta(walltime)
