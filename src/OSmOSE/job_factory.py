import os
import json
import tomlkit
import subprocess
from typing import NamedTuple, List, Literal
from warnings import warn
from datetime import datetime
from importlib import resources
from OSmOSE.utils import read_config

class Job_builder():
    def __init__(self, config_file: str = None):
        if config_file is None:
            self.__configfile = "job_config.toml"
            self.__config: NamedTuple = read_config(resources.files("OSmOSE").joinpath(self.__configfile))
            print(self.__config)
        else:
            self.__configfile = config_file
            self.__config: NamedTuple = read_config(config_file)

        self.__prepared_jobs = []
        self.__ongoing_jobs = []
        self.__finished_jobs = []

        required_properties = ["job_scheduler","env_script","queue","nodes","walltime","ncpus","mem","outfile","errfile"]

        if not all(prop in self.__config._fields for prop in required_properties):
            raise ValueError(f"The provided configuration file is missing the following attributes: {'; '.join(list(set(required_properties).difference(set(self.__config._fields))))}")

    def edit(self, attribute: str, value: any):
        self.__config._replace(attribute = value)

    #region Properties
    # READ-ONLY properties
    @property
    def Config(self) -> dict:
        return self.__config

    @property
    def Prepared_jobs(self):
        return self.__prepared_jobs
    
    @property
    def Ongoing_jobs(self):
        self.update_job_status()
        return self.__ongoing_jobs

    @property
    def Finished_jobs(self):
        self.update_job_status()
        return self.__finished_jobs


    # READ-WRITE properties
    @property
    def Job_scheduler(self):
        return self.__config.job_scheduler
    
    @Job_scheduler.setter
    def Job_Scheduler(self, value: Literal["Torque","Slurm"]):
        self.edit("job_scheduler",value)

    @property
    def Env_script(self):
        return self.__config.env_script
    
    @Env_script.setter
    def Env_script(self, value):
        self.edit("env_script", value)

    @property
    def Queue(self):
        return self.__config.queue
    
    @Queue.setter
    def Queue(self, value):
        self.edit("queue", value)

    @property
    def Nodes(self):
        return self.__config.nodes
    
    @Nodes.setter
    def Nodes(self, value):
        self.edit("nodes",value)
        
    @property
    def Walltime(self):
        return self.__config.walltime
    
    @Walltime.setter
    def Walltime(self, value):
        self.edit("walltime", value)
        
    @property
    def Ncpus(self):
        return self.__config.ncpus
    
    @Ncpus.setter
    def Ncpus(self, value):
        self.edit("ncpus", value)
        
    @property
    def Mem(self):
        return self.__config.mem
    
    @Mem.setter
    def Mem(self, value):
        self.edit("mem", value)
        
    @property
    def Outfile(self):
        return self.__config.outfile
    
    @Outfile.setter
    def Outfile(self, value):
        self.edit("outfile", value)
        
    @property
    def Errfile(self):
        return self.__config.errfile
    
    @Errfile.setter
    def Errfile(self, value):
        self.edit("errfile", value)
    #endregion

    def write_configuration(self, output_file:str = None):
        """Writes the configuration to the original configuration file, or a new file if specified.
        
        Parameter:
        ----------
            `output_file`: the path to the new configuration file (in TOML or JSON format)."""

        output_format = os.path.splitext(output_file)[1] if output_file else os.path.splitext(self.__configfile)[1]
        
        match output_format:
            case "toml":
                tomlkit.dumps(self.__config)
            case "json":
                json.dumps(self.__config)
        
        
    def build_job_file(self, *, script_path: str, script_args:str, jobname: str = None, preset: Literal["low","medium","high"]= None, job_scheduler: Literal["Torque","Slurm"] = None,
     env_name = None, env_script = None, queue = None, nodes = None, walltime = None, ncpus = None, mem = None, outfile = None, errfile = None) -> str:
        """Build a job file corresponding to your job scheduler."""

        pwd = os.path.dirname(__file__)
        jobdir = os.path.join(pwd, "Ongoing_jobs")

        if not os.path.exists(jobdir):
            os.makedirs(jobdir)

        job_file = ["#!/bin/bash"]

        #region Build header
        #Getting all header variables from the parameters or the configuration defaults
        if not job_scheduler: job_scheduler = self.Job_scheduler
        if not jobname: jobname = self.Job_name
        if not env_name: env_name = self.env_name
        if not queue: queue = self.Queue
        if not nodes: nodes = self.Nodes
        if not walltime: walltime = self.Walltime
        if not ncpus: ncpus = self.Ncpus
        if not mem: mem = self.Mem
        if not outfile: outfile = self.Outfile
        if not errfile: errfile = self.Errfile

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
                outfile = outfile.replace("_%j","").format(jobname)
                errfile = errfile.replace("_%j","").format(jobname)
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
                warn("Unrecognized job scheduler. Please review the PBS file and or specify a job scheduler between Torque or Slurm")
        
        job_file.append(f"{prefix} {name_param}{jobname}")
        job_file.append(f"{prefix} {queue_param}{queue}")
        job_file.append(f"{prefix} {node_param}{nodes}")
        job_file.append(f"{prefix} {cpu_param}{ncpus}")
        job_file.append(f"{prefix} {time_param}{walltime}")
        job_file.append(f"{prefix} {mem_param}{mem}")
        job_file.append(f"{prefix} {outfile_param}{outfile}")
        job_file.append(f"{prefix} {errfile_param}{errfile}")
        #endregion

        #TODO: use conda activate to load env
        job_file.append(f"\nsource {env_script if env_script else self.Env_script } --conda-env {env_name}")
        job_file.append(f"python {script_path} {script_args}")


        outfilename = f"{jobname}_{datetime.now().strftime('%H-%M-%S')}_{job_scheduler}_{len(os.listdir(jobdir))}.pbs"

        job_file.append(f"\nrm {os.path.join(jobdir, outfilename)}")

        with open(os.path.join(jobdir, outfilename), "w") as jobfile:
            jobfile.write("\n".join(job_file))

        self.__prepared_jobs.append(os.path.join(jobdir, outfilename))

        return os.path.join(jobdir, outfilename)

    # TODO support multiple dependecies and job chaining (?)
    def submit_job(self, jobfile: str = None, dependency: str | List[str] = None) -> List[str]:
        """Submits the job file to the cluster using the job scheduler written in the file name or in the configuration file.
        
        Parameters:
        -----------
            jobfile: the absolute path of the job file to be submitted. If none is provided, all the prepared job files will be submitted.
            
            dependency: the job ID this job is dependent on. The afterok:dependency will be added to the command for all jobs submitted.

        Returns:
        --------
            A list containing the job ids of the submitted jobs.
            """

        jobfile_list = [jobfile] if jobfile else self.Prepared_jobs
        
        jobid_list = []

        # TODO: Think about the issue when a job have too many dependencies and workaround.

        for jobfile in jobfile_list:
            if "torque" in jobfile.lower():
                dep = f" -W depend=afterok:{dependency}" if dependency else ""
                jobid = subprocess.run([f"qsub{dep}", jobfile],stdout=subprocess.PIPE).stdout.decode('utf-8').rstrip('\n')
                jobid_list.append(jobid)
            elif "slurm" in jobfile.lower():
                dep = f"-d afterok:{dependency}" if dependency else ""
                jobid = subprocess.run(["sbatch", dep, jobfile],stdout=subprocess.PIPE).stdout.decode('utf-8').rstrip('\n')
                jobid_list.append(jobid)

            if jobfile in self.Prepared_jobs:
                self.__prepared_jobs.remove(jobfile)
            self.__ongoing_jobs.append(jobfile)

        return jobid_list

    def update_job_status(self):
        for file in self.__ongoing_jobs:
            if not os.path.exists(file):
                self.__ongoing_jobs.remove(file)
                self.__finished_jobs.append(file)