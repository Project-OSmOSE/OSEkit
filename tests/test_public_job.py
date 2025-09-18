from osekit.public_api.job import Job, JobStatus
import pytest
from pathlib import Path


@pytest.mark.parametrize(
    ("initial_status", "expected_status"),
    [
        pytest.param(
            JobStatus.PREPARED, JobStatus.QUEUED, id="prepared_becomes_queued"
        ),
        pytest.param(JobStatus.QUEUED, JobStatus.RUNNING, id="queued_becomes_running"),
        pytest.param(
            JobStatus.RUNNING, JobStatus.COMPLETED, id="running_becomes_completed"
        ),
        pytest.param(
            JobStatus.COMPLETED, JobStatus.COMPLETED, id="completed_remains_completed"
        ),
    ],
)
def test_job_progress(initial_status: JobStatus, expected_status: JobStatus) -> None:
    job = Job(script_path=Path(r""))
    assert job.status == JobStatus.PREPARED
    job._status = initial_status
    job.progress()
    assert job.status == expected_status
