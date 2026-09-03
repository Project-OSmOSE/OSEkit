from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from pandas import Timedelta

from osekit.job.builder import JobBuilder
from osekit.job.config import JobConfig
from osekit.job.job import Job, JobStatus
from osekit.job.scheduler.pbs import Pbs
from osekit.job.scheduler.scheduler import Scheduler


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


def test_walltime_str_and_setter() -> None:
    job = Job(Path("bossanova.py"))
    for walltime in ("13:08:09", Timedelta(hours=13, minutes=8, seconds=9)):
        job.walltime = walltime
        assert job.walltime == Timedelta("13:08:09")
        assert job.walltime_str == "13:08:09"


def test_pbs_build_job_specifications() -> None:
    job = Job(
        script_path=Path(),
        config=JobConfig(
            nb_nodes=2,
            ncpus=3,
            ngpus=1,
            mem="16gb",
            walltime=Timedelta(hours=2),
            venv_name="cool_env",
        ),
        output_folder=Path(r"cool/folder"),
        name="cool_job",
    )

    specifications = Pbs(queue="mpi")._build_job_specification(job=job).splitlines()

    for expected_specification in (
        "#PBS -N cool_job",
        "#PBS -q mpi",
        "#PBS -l select=2:ncpus=3:mem=16gb:ngpus=1",
        "#PBS -l walltime=02:00:00",
        f"#PBS -o {Path('cool/folder') / 'cool_job.out'}",
        f"#PBS -e {Path('cool/folder') / 'cool_job.err'}",
    ):
        assert expected_specification in specifications


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
        line.startswith(f"#PBS -o {job.output_folder / job.name}.out")
        for line in content
    )
    assert any(
        line.startswith(f"#PBS -e {job.output_folder / job.name}.err")
        for line in content
    )

    assert ". /appli/anaconda/latest/etc/profile.d/conda.sh" in content
    assert "conda activate osekit" in content

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
    job = Job(Path())
    pbs_scheduler = Pbs(queue="omp")
    job.status = JobStatus.PREPARED

    class Dummy:
        def __init__(self) -> None:
            """Dummy subprocess.run."""
            raise subprocess.CalledProcessError(5, "err")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: Dummy(),
    )

    def mock_update_status(self: Pbs, job: Job) -> JobStatus:
        return JobStatus.PREPARED

    monkeypatch.setattr(Pbs, "update_status", mock_update_status)

    # Submit error should leave the job prepared:
    with pytest.raises(RuntimeError, match="Submission failed with exit code 5"):
        pbs_scheduler.submit(job=job)

    assert job.status == JobStatus.PREPARED


def test_pbs_update_info_no_job_id() -> None:
    job = Job(Path("pixies.py"))
    pbs_scheduler = Pbs()
    job.job_id = None
    pbs_scheduler.update_info(job=job)
    assert not job.info


def test_pbs_update_info_parse_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    job = Job(script_path=Path("fontaines.py"), name="SwissArmyMan")
    job.job_id = "7137005"

    class Dummy:
        stdout = (
            Path(__file__).parent / "_static/job_status_request_results/pbs.txt"
        ).read_text()
        stderr = ""

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: Dummy(),
    )
    scheduler = Pbs()
    scheduler.update_info(job=job)
    assert job.job_id == "7137005"
    assert job.name == "SwissArmyMan"
    assert job.status == JobStatus.RUNNING
    assert job.info == {
        "user": "daniels",
        "time": "00:10:37",
        "queue": "jetski",
    }


def test_pbs_update_info_unknown_job_raises(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_pbs_update_info_error(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_pbs_update_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = Job(Path("porticoquartet.py"))
    job.path = tmp_path / "pompidou.pbs"

    scheduler = Pbs()

    assert scheduler.update_status(job=job) == JobStatus.UNPREPARED

    job.path.write_text("prickly pear")
    assert scheduler.update_status(job=job) == JobStatus.PREPARED

    def mock_update_info(
        job: Job,
        status: JobStatus,
        *args: list,
        **kwargs: dict,
    ) -> None:
        job.status = status

    monkeypatch.setattr(
        scheduler,
        "update_info",
        lambda job: mock_update_info(job=job, status=JobStatus.QUEUED),
    )

    job.job_id = "5129195"
    assert scheduler.update_status(job=job) == JobStatus.QUEUED
    assert job.status == JobStatus.QUEUED


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
        self: Scheduler,
        job: Job,
        dependencies: Job | str | None = None,
    ) -> None:
        submitted_jobs.append((job, dependencies))

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

    unprepared_job = job_builder.jobs[0]
    prepared_job = job_builder.jobs[1]

    dependencies = {
        prepared_job: {"beforeok": unprepared_job},
        unprepared_job: {"afterany": prepared_job},
    }

    job_builder.submit(dependencies=dependencies)

    # Only the prepared job should be submitted
    assert len(submitted_jobs) == 1
    submitted_job = submitted_jobs[0]
    assert submitted_job[0] == prepared_job

    # Only the submitted job dependencies should be injected
    assert submitted_job[1] == dependencies[prepared_job]


def test_pbs_build_dependencies_string_validates_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_calls = []

    def mock_validate(dependency_type: str) -> None:
        validate_calls.append(dependency_type)

    monkeypatch.setattr(Pbs, "_validate_dependency_type", mock_validate)

    dependencies = {"afterok": "1234567", "afterany": ["2345678", "3456789"]}
    Pbs()._build_dependency_string(
        dependencies=dependencies,
    )

    assert all(dependency_type in validate_calls for dependency_type in dependencies)


def test_pbs_validate_dependency_type() -> None:
    pbs = Pbs()

    # Supported dependency type shouldn't raise
    pbs._validate_dependency_type("afterok")

    # Unsupported dependency type should raise
    with pytest.raises(ValueError) as e:
        pbs._validate_dependency_type("afterdummy")

    assert e.match("Unsupported dependency type 'afterdummy'")
    for supported in Pbs._VALID_DEPENDENCY_TYPES:
        assert e.match(supported)


@pytest.mark.parametrize(
    ("dependencies", "expected"),
    [
        pytest.param(
            {"afterok": "1234567"},
            "afterok:1234567",
            id="one_type_one_job",
        ),
        pytest.param(
            {"afterok": ["1234567", "2345678"]},
            "afterok:1234567:2345678",
            id="one_type_multiple_jobs",
        ),
        pytest.param(
            {"afterok": "1234567", "afterany": "2345678"},
            "afterok:1234567,afterany:2345678",
            id="multiple_types_one_job",
        ),
        pytest.param(
            {"afterok": ["1234567", "2345678"], "afterany": ["3456789", "4567890"]},
            "afterok:1234567:2345678,afterany:3456789:4567890",
            id="multiple_types_multiple_jobs",
        ),
    ],
)
def test_pbs_build_dependencies_string(
    dependencies: dict[str, str | list[str]],
    expected: str,
) -> None:
    # %% Dependencies string from job IDs
    assert Pbs()._build_dependency_string(dependencies=dependencies) == expected

    # %% Dependencies string from Job instances
    def id_to_job(job_id: str | list[str]) -> Job | list[Job]:
        """Convert a Job ID ``job_id`` to a Job object with an ID of ``job_id``

        If ``job_id`` is a list, converts the list of job_ids to a list of jobs
        with the given IDs."""
        if isinstance(job_id, str):
            job = Job(Path())
            job._id = job_id
            job.status = JobStatus.QUEUED
            return job
        output = []
        for j_id in job_id:
            job = Job(Path())
            job._id = j_id
            job.status = JobStatus.QUEUED
            output.append(job)
        return output

    dependencies = {key: id_to_job(value) for key, value in dependencies.items()}
    assert Pbs()._build_dependency_string(dependencies=dependencies) == expected


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

    scheduler.submit(job=job, dependencies={"afterok": "1234567"})

    assert "-W" in captured_cmd["cmd"]
    assert "depend=afterok:1234567" in captured_cmd["cmd"]


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


@pytest.mark.parametrize(
    (
        "nb_tasks",
        "script_path",
        "script_args",
        "output_folder",
        "job_name",
        "nb_jobs",
        "expected_task_indexes",
    ),
    [
        pytest.param(
            10,
            Path("path/to/script.py"),
            {"int_arg": 1, "str_arg": "cool"},
            Path("path/to/output"),
            "cool_name",
            1,
            [(0, 10)],
            id="one_job_covers_all_tasks",
        ),
        pytest.param(
            10,
            Path("path/to/script.py"),
            {"int_arg": 1, "str_arg": "cool"},
            Path("path/to/output"),
            "cool_name",
            5,
            [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)],
            id="tasks_are_equally_distributed",
        ),
    ],
)
def test_create_jobs(  # noqa: PLR0917
    monkeypatch: pytest.MonkeyPatch,
    nb_tasks: int,
    script_path: Path,
    script_args: dict,
    output_folder: Path,
    job_name: str,
    nb_jobs: int,
    expected_task_indexes: list[tuple[int, int]],
) -> None:
    created_jobs = {}

    def patch_create_job(self: JobBuilder, **kwargs: str) -> None:
        job_name = kwargs.pop("name")
        created_jobs[job_name] = kwargs

    monkeypatch.setattr(JobBuilder, "create_job", patch_create_job)

    JobBuilder().create_jobs(
        nb_tasks=nb_tasks,
        script_path=script_path,
        script_args=script_args,
        output_folder=output_folder,
        job_name=job_name,
        nb_jobs=nb_jobs,
    )

    # Correct number of jobs
    assert len(created_jobs) == nb_jobs

    # Correct distribution across jobs
    for job in created_jobs.values():
        assert (
            job["script_args"]["first"],
            job["script_args"]["last"],
        ) in expected_task_indexes

    # Script path
    assert all(job["script_path"] == script_path for job in created_jobs.values())

    # Script args
    for job in created_jobs.values():
        for arg in script_args:
            assert arg in job["script_args"]

    # Output folder
    assert all(job["output_folder"] == output_folder for job in created_jobs.values())

    # Job names
    if nb_jobs == 1:
        assert np.array_equal(list(created_jobs.keys()), [job_name])
    else:
        for idx, job in enumerate(
            sorted(
                created_jobs.items(),
                key=lambda kvp: kvp[1]["script_args"]["first"],
            ),
        ):
            assert job[0] == f"{job_name}_{idx}"
