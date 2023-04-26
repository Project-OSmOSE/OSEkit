import os
from pathlib import Path

import pytest
from OSmOSE import Job_builder
from OSmOSE.utils import read_config, convert
from importlib import resources

default_config = read_config(resources.files("OSmOSE")).joinpath("config.toml")
custom_config = convert({
    "job_scheduler":"Torque",
    "env_script":"conda activate",
    "env_name":"osmose",
    "queue":"normal",
    "nodes":2,
    "walltime":"12:00:00",
    "ncpus":16,
    "mem":"42G",
    "outfile":"%j.out",
    "errfile":"%j.err"
})

def test_build_job_file():
    script_path = "/path/to/script"
    script_args = "--arg1 value1 --arg2 value2"
    jobname = "test_job"

    