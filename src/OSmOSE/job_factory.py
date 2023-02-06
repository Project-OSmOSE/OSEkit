from typing import Literal
import os
from utils import read_config

def build_job_file(jobname, job_scheduler: Literal["Torque","Slurm"], job_args, script_path, script_args, env_name):
    """Build a job file corresponding to your job scheduler."""

    pwd = os.path.dirname(__file__)
    jobdir = os.path.join(pwd, "Ongoing_jobs")

    if not os.path.exists(jobdir):
        os.makedirs(jobdir)

    job_file = ["#!/bin/bash"]

    #region Build header

    #endregion

    #TODO config
    env_script = "/appli/hibd/rdma-hadoop-2.x-1.3.5-x86/sbin/quick-hadoop-get-env.sh"

class Job_Config():
    def __init__(self, config_file: str = None):
        if config_file is None:
            self.cwd = os.path.dirname(__file__)
            self.config = read_config(os.path.join(self.cwd, "job_config.toml"))
        else:
            self.cwd = os.path.dirname(config_file)
            self.config = read_config(config_file)

        required_properties = ["job_scheduler","env_script","queue","walltime","ncpus","mem","outfile","errfile"]

        if not all(prop in self.config._fields for prop in required_properties):
            raise ValueError(f"The provided configuration file is missing the following attributes: {'; '.join(list(set(required_properties).difference(set(self.config._fields))))}")