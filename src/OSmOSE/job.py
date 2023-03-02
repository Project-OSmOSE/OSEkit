import os
import json
import tomlkit
import subprocess
from string import Template
from typing import NamedTuple, List, Literal
from warnings import warn
from datetime import datetime
from importlib import resources
from OSmOSE.utils import read_config


class Job_builder:
    def __init__(self, config_file: str = None):
        if config_file is None:
            self.__configfile = "job_config.toml"
            self.__config: NamedTuple = read_config(
                resources.files("OSmOSE").joinpath(self.__configfile)
            )
            print(self.__config)
        else:
            self.__configfile = config_file
            self.__config: NamedTuple = read_config(config_file)

        self.__prepared_jobs = []
        self.__ongoing_jobs = []
        self.__finished_jobs = []

        required_properties = [
            "job_scheduler",
            "env_script",
            "queue",
            "nodes",
            "walltime",
            "ncpus",
            "mem",
            "outfile",
            "errfile",
        ]

        if not all(prop in self.__config._fields for prop in required_properties):
            raise ValueError(
                f"The provided configuration file is missing the following attributes: {'; '.join(list(set(required_properties).difference(set(self.__config._fields))))}"
            )

    def edit(self, attribute: str, value: any):
        self.__config._replace(attribute=value)

    # region Properties
    # READ-ONLY properties
    @property
    def config(self) -> dict:
        return self.__config

    @property
    def prepared_jobs(self):
        return self.__prepared_jobs

    @property
    def ongoing_jobs(self):
        self.update_job_status()
        return self.__ongoing_jobs

    @property
    def finished_jobs(self):
        self.update_job_status()
        return self.__finished_jobs

    # READ-WRITE properties
    @property
    def job_scheduler(self):
        return self.__config.job_scheduler

    @job_scheduler.setter
    def job_Scheduler(self, value: Literal["Torque", "Slurm"]):
        self.edit("job_scheduler", value)

    @property
    def env_script(self):
        return self.__config.env_script

    @env_script.setter
    def env_script(self, value):
        self.edit("env_script", value)

    @property
    def queue(self):
        return self.__config.queue

    @queue.setter
    def queue(self, value):
        self.edit("queue", value)

    @property
    def nodes(self):
        return self.__config.nodes

    @nodes.setter
    def nodes(self, value):
        self.edit("nodes", value)

    @property
    def walltime(self):
        return self.__config.walltime

    @walltime.setter
    def walltime(self, value):
        self.edit("walltime", value)

    @property
    def ncpus(self):
        return self.__config.ncpus

    @ncpus.setter
    def ncpus(self, value):
        self.edit("ncpus", value)

    @property
    def mem(self):
        return self.__config.mem

    @mem.setter
    def mem(self, value):
        self.edit("mem", value)

    @property
    def outfile(self):
        return self.__config.outfile

    @outfile.setter
    def outfile(self, value):
        self.edit("outfile", value)

    @property
    def errfile(self):
        return self.__config.errfile

    @errfile.setter
    def errfile(self, value):
        self.edit("errfile", value)

    # endregion

    # TODO: Move to utils
    def write_configuration(self, output_file: str = None):
        """Writes the configuration to the original configuration file, or a new file if specified.

        Parameter
        ---------
            `output_file`: the path to the new configuration file (in TOML or JSON format).
        """

        output_format = (
            os.path.splitext(output_file)[1]
            if output_file
            else os.path.splitext(self.__configfile)[1]
        )

        match output_format:
            case "toml":
                tomlkit.dumps(self.__config)
            case "json":
                json.dumps(self.__config)

    def build_job_file(
        self,
        *,
        script_path: str,
        script_args: str,
        jobname: str = None,
        preset: Literal["low", "medium", "high"] = None,
        job_scheduler: Literal["Torque", "Slurm"] = None,
        env_script: str = None,
        env_name: str = None,
        queue: str = None,
        nodes: int = None,
        walltime: str = None,
        ncpus: int = None,
        mem: str = None,
        outfile: str = None,
        errfile: str = None,
    ) -> str:
        """Build a job file corresponding to your job scheduler.

        When a job file is created, it becomes a `prepared_job` and can be launched automatically using the `submit_job()` method
        with no arguments.

        This method can use the configuration file to set default job parameters. Use presets to quickly adjust the resource
        consumption or fill the parameters. If a parameter is left empty, it will fall back to the value in the configuration file.
        The default preset is `low` in order to save as much resource as possible.

        Parameters
        ----------
        script_path : `str`, keyword-only
            The path to the script that will be executed in the cluster job
        script_args : `str`, keyword-only
            All the arguments required by the script, as one string.
        jobname : `str`, optional, keyword-only
            The name of the job as seen on the job list (qstat, squeue, ...). The default is the script name.
        preset : `{"low", "medium", "high"}`, optional, keyword-only
            Resource preset for the job. It is best to set up the presets in the configuration file before using them,
            as the default might not correspond to your structure. The default is `low`.
        job_scheduler : `{"Torque", "Slurm"}`, optional, keyword-only
            The job scheduler to which the jobs will be submitted. As of now, only Torque (PBS) and Slurm (Sbatch) are supported.
        env_script : `str`, optional, keyword-only
            The script used to activate the conda environment. If there is no particular script, it can just be `source` or `conda activate`
        env_name : `str`, optional, keyword-only
            The name of the conda environment this job should run into.
        queue : `str`, optional, keyword-only
            The partition/queue/qol in which the job should run. It is very dependant on the cluster.
        nodes : `int`, optional, keyword-only
            The number of nodes (i.e. computers) this job will use. If it is not using a protocol like MPI or an adapted job scheduler (like PEGASUS),
            using more than one node will have no effect.
        walltime : `str`, optional, keyword-only
            The time limit of the job. If exceeded, the job will be killed by the job scheduler. It is best to view large and ask for 1.5 or 2 times
            more than the anticipated time of the job, especially for tasks with a high degree ofo variability in processing time.
        ncpus : `int`, optional, keyword-only
            The number of CPUs required per node.
        mem : `str`, optional, keyword-only
            The quantity of RAM required per CPU. Most of the time it is an integer followed by a G for Gigabytes or Mb for Megabytes.
        outfile : `str`, optional, keyword-only
            The name of the main log file which will record the standard output from the job. If only the name is provided, it will be created in the current
            directory.
        errfile : `str`, optional, keyword-only
            The name of the error log file which will record the error output from the job. If only the name is provided, it will be created in the
            current directory. If there is no error output, it will not be created.

        Returns
        -------
        job_path : `str`
            The path to the created job file.
        """

        pwd = os.path.dirname(__file__)
        jobdir = os.path.join(pwd, "ongoing_jobs")

        if not os.path.exists(jobdir):
            os.makedirs(jobdir)

        job_file = ["#!/bin/bash"]

        # region Build header
        #! HEADER
        # Getting all header variables from the parameters or the configuration defaults
        if not job_scheduler:
            job_scheduler = self.job_scheduler
        if not jobname:
            jobname = os.path.basename(script_path)
        if not env_name:
            env_name = self.env_name
        if not queue:
            queue = self.queue
        if not nodes:
            nodes = self.nodes
        if not walltime:
            walltime = self.walltime
        if not ncpus:
            ncpus = self.ncpus
        if not mem:
            mem = self.mem
        if not outfile:
            outfile = self.outfile
        if not errfile:
            errfile = self.errfile

        match job_scheduler:
            case "Torque":
                prefix = "#PBS"
                name_param = "-N "
                queue_param = "-q "
                node_param = "-l nodes="
                cpu_param = "-l ncpus="
                time_param = "-l walltime="
                mem_param = "-l mem="
                outfile_param = "-o "
                errfile_param = "-e "
                outfile = outfile.replace("_%j", "").format(jobname)
                errfile = errfile.replace("_%j", "").format(jobname)
            case "Slurm":
                prefix = "#SBATCH"
                name_param = "-J "
                queue_param = "-p "
                node_param = "-n "
                cpu_param = "-c "
                time_param = "-time "
                mem_param = "-mem "
                outfile_param = "-o "
                errfile_param = "-e "
                outfile = outfile.format(jobname)
                errfile = errfile.format(jobname)
            case _:
                prefix = "#"
                name_param = ""
                queue_param = ""
                node_param = ""
                cpu_param = ""
                time_param = ""
                mem_param = ""
                outfile_param = ""
                errfile_param = ""
                warn(
                    "Unrecognized job scheduler. Please review the PBS file and or specify a job scheduler between Torque or Slurm"
                )

        job_file.append(f"{prefix} {name_param}{jobname}")
        job_file.append(f"{prefix} {queue_param}{queue}")
        job_file.append(f"{prefix} {node_param}{nodes}")
        job_file.append(f"{prefix} {cpu_param}{ncpus}")
        job_file.append(f"{prefix} {time_param}{walltime}")
        job_file.append(f"{prefix} {mem_param}{mem}")
        job_file.append(f"{prefix} {outfile_param}{outfile}")
        job_file.append(f"{prefix} {errfile_param}{errfile}")
        # endregion

        #! ENV
        # The env_script should contain all the command(s) needed to load the script, with the $env_name template where the environment name should be.
        job_file.append(
            Template(f"\n{env_script if env_script else self.env_script}").substitute(
                env_name=env_name
            )
        )

        #! SCRIPT
        job_file.append(f"python {script_path} {script_args}")

        #! FOOTER
        outfilename = f"{jobname}_{datetime.now().strftime('%H-%M-%S')}_{job_scheduler}_{len(os.listdir(jobdir))}.pbs"

        job_file.append(f"\nchmod 770 {outfile} {errfile}")
        job_file.append(f"\nrm {os.path.join(jobdir, outfilename)}")

        #! BUILD DONE => WRITING
        with open(os.path.join(jobdir, outfilename), "w") as jobfile:
            jobfile.write("\n".join(job_file))

        self.__prepared_jobs.append(os.path.join(jobdir, outfilename))

        return os.path.join(jobdir, outfilename)

    # TODO support multiple dependencies and job chaining (?)
    def submit_job(
        self, jobfile: str = None, dependency: str | List[str] = None
    ) -> List[str]:
        """Submits the job file to the cluster using the job scheduler written in the file name or in the configuration file.

        Parameters:
        -----------
            jobfile: `str`
                The absolute path of the job file to be submitted. If none is provided, all the prepared job files will be submitted.

            dependency: `str` or `list(str)`
                The job ID this job is dependent on. The afterok:dependency will be added to the command for all jobs submitted.

        Returns:
        --------
            A list containing the job ids of the submitted jobs.
        """

        jobfile_list = [jobfile] if jobfile else self.prepared_jobs

        jobid_list = []

        # TODO: Think about the issue when a job have too many dependencies and workaround.

        for jobfile in jobfile_list:
            if "torque" in jobfile.lower():
                dep = f" -W depend=afterok:{dependency}" if dependency else ""
                jobid = (
                    subprocess.run([f"qsub{dep}", jobfile], stdout=subprocess.PIPE)
                    .stdout.decode("utf-8")
                    .rstrip("\n")
                )
                jobid_list.append(jobid)
            elif "slurm" in jobfile.lower():
                dep = f"-d afterok:{dependency}" if dependency else ""
                jobid = (
                    subprocess.run(["sbatch", dep, jobfile], stdout=subprocess.PIPE)
                    .stdout.decode("utf-8")
                    .rstrip("\n")
                )
                jobid_list.append(jobid)

            if jobfile in self.prepared_jobs:
                self.__prepared_jobs.remove(jobfile)
            self.__ongoing_jobs.append(jobfile)

        return jobid_list

    def update_job_status(self):
        """Iterates over the list of ongoing jobs and mark them as finished if the job file does not exist."""
        for file in self.__ongoing_jobs:
            if not os.path.exists(file):
                self.__ongoing_jobs.remove(file)
                self.__finished_jobs.append(file)