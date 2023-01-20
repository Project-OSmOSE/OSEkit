from distutils.errors import UnknownFileError
from typing import NamedTuple, Union
from collections import namedtuple
import os
import tomllib
import json

def read_config(raw_config: Union[str, dict]) -> NamedTuple:
    """Read the given configuration file or dict and converts it to a namedtuple. Only TOML and JSON formats are accepted for now.
    
        Parameter:
            raw_config: the path of the configuration file, or the dict object containing the configuration.
            
        Returns:
            The configuration as a NamedTuple object."""
            
    if isinstance(raw_config, str):
        if not os.path.isfile(raw_config):
            raise FileNotFoundError(f"The configuration file {raw_config} does not exist.")
        
        with open(raw_config, "rb") as input_config:
            match os.path.splitext(raw_config)[1]:
                case "toml":
                    pre_config = tomllib.load(input_config)
                case "json":
                    pre_config = json.load(input_config)
                case "yaml":
                    raise NotImplementedError("YAML support will eventually get there (unfortunately)")
                case _:
                    raise UnknownFileError(f"The provided configuration file extension (.{os.path.splitext(raw_config)[1]} is not a valid extension. Please use .toml or .json files.")

    return namedtuple('GenericDict', pre_config.keys())(**pre_config)