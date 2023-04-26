import os
from pathlib import Path

import pytest
from OSmOSE import Job_builder
from OSmOSE.utils import read_config
from importlib import resources

default_config = read_config(resources.files("OSmOSE")).joinpath("config.toml")
custom_config = {
    
}