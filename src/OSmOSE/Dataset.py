import os
import stat
from pathlib import Path
from typing import Union, Tuple, List
from datetime import datetime
from warnings import warn
from statistics import fmean as mean
import shutil
import glob
from os import PathLike
import sys
import re
import soundfile as sf
   
try:
    import grp
    skip_perms = False
except ModuleNotFoundError:
    skip_perms = True

import pandas as pd
import numpy as np
from tqdm import tqdm
from OSmOSE.utils import read_header, check_n_files, make_path, set_umask
from OSmOSE.timestamps import write_timestamp
from OSmOSE.config import *


class Dataset:
    """Super class used to create dataset compatible with the rest of the package.

    A dataset is a set of audio files located in a folder whose name is the dataset name.
    The files must be in the `raw/audio/original`subfolder. and be alongside a `timestamp.csv` file, which includes
    the name of the file and the associated timestamp, in the `%Y-%m-%dT%H:%M:%S.%fZ` strftime format.

    This file can be created using the `OSmOSE.write_timestamp` function.
    """

    def __init__(
        self,
        dataset_path: str,
        *,
        gps_coordinates: Union[str, list, Tuple] = None,
        depth: Union[str, int] = None,  
        timezone: str = None,
        owner_group: str = None,
        original_folder: str = None,
        local: bool = True,
    ) -> None:
        """Instanciate the dataset with at least its path.

        Parameters
        ----------
        dataset_path : `str`
            The absolute path to the dataset folder. The last folder in the path will be considered as the name of the dataset.

        gps_coordinates : `str` or `list` or `Tuple`, optional, keyword-only
            The GPS coordinates of the listening location. If it is of type `str`, it must be the name of a csv file located in `data/auxiliary`,
            otherwise a list or a tuple with the first element being the latitude coordinates and second the longitude coordinates.

        owner_group : `str`, optional, keyword-only
            The name of the group using the OsmOSE package. All files created using this dataset will be accessible by the osmose group.
            Will not work on Windows.

        original_folder : `str`, optional, keyword-only
            The path to the folder containing the original audio files. It can be set right away, passed in the build() function or automatically detected.

        Example
        -------
        >>> from pathlib import Path
        >>> from OSmOSE import Dataset
        >>> dataset = Dataset(Path("home","user","my_dataset"), coordinates = [49.2, -5], owner_group = "gosmose")
        """
        self.__path = Path(dataset_path)
        self.__name = self.__path.stem
        self.owner_group = owner_group
        self.__gps_coordinates = []
        self.__local = local

        if gps_coordinates is not None:
            self.gps_coordinates = gps_coordinates          
            
        if depth is not None:
            self.depth = depth
            
        if timezone is not None:
            self.timezone = timezone            
            
        self.__original_folder = original_folder

        if skip_perms:
            print(
                "It seems you are on a non-Unix operating system (probably Windows). The build() method will not work as intended and permission might be incorrectly set."
            )

        pd.set_option("display.float_format", lambda x: "%.0f" % x)

    # region Properties
    @property
    def name(self):
        """str: The Dataset name. It is readonly."""
        return self.__name

    @property
    def path(self):
        """Path: The Dataset path. It is readonly."""
        return self.__path

    @property
    def original_folder(self):
        """Path: The folder containing the original audio file."""
        return (
            self.__original_folder
            if self.__original_folder
            else self._get_original_after_build()
        )

    @property
    def gps_coordinates(
        self,
    ) -> Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]]:
        """The GPS coordinates of the listening location. First element is latitude, second is longitude.

        GPS coordinates are used to localize the dataset and required for some utilities, like the
        weather and environment utility.

        Parameter
        ---------
        coordinates: `str` or `list` or `tuple`
            If the coordinates are a string, it must be the name of a csv file located in `data/auxiliary/instrument/`, containing two columns: 'lat' and 'lon'
            Else, they can be either a list or a tuple of two float, the first being the latitude and second the longitude; or a
            list or a tuple containing two lists or tuples respectively of floats. In this case, the coordinates are not treated as a point but
            as an area.

        Returns
        -------
        The GPS coordinates as a tuple.
        """
        return self.__gps_coordinates

    @gps_coordinates.setter
    def gps_coordinates(
        self,
        new_coordinates: Union[
            str,
            List[float],
            List[List[float]],
            Tuple[float, float],
            Tuple[Tuple[float, float], Tuple[float, float]],
        ],
    ):

        match new_coordinates:
            case str():
                
                aux_data_path = next(
                    self.path.rglob(new_coordinates), False
                )
                
                if aux_data_path:
                    csvFileArray = pd.read_csv(aux_data_path)
                    self.__gps_coordinates = [np.mean(csvFileArray["lat"]), np.mean(csvFileArray["lon"])]
                else:
                    raise FileNotFoundError(f"The {new_coordinates} has been found no where within {self.path}")                   
                
            case tuple():
                self.__gps_coordinates = new_coordinates
            case _:
                raise TypeError(
                    f"GPS coordinates must be either a list of coordinates or the name of csv containing the coordinates, but {type(new_coordinates)} found."
                )


    @property
    def depth(
        self,
    ) -> int:
        """The depth of the hydrophone, in meter.

        Parameter
        ---------
        depth: `str` or `int`
            If the depth is a string, it must be the name of a csv file located in `data/auxiliary/instrument/`, containing at least a column 'depth'

        Returns
        -------
        The depth as an int.
        """
        return self.__depth
    
    @depth.setter
    def depth(
        self,
        new_depth: Union[
            str,
            int,
        ],
    ):

        match new_depth:
            case str():

                aux_data_path = next(
                    self.path.rglob(new_depth), False
                )                
                if aux_data_path:
                    csvFileArray = pd.read_csv(aux_data_path)
                    self.__depth = int(np.mean(csvFileArray["depth"]))                
                else:
                    raise FileNotFoundError(f"The {new_coordinates} has been found no where within {self.path}")
                                   
            case int():
                self.__depth = new_depth
            case _:
                raise TypeError(
                    f"Variable depth must be either an int value for fixed hydrophone or a csv filename for moving hydrophone"
                )
                
                
    @property
    def owner_group(self):
        """str: The Unix group able to interact with the dataset."""
        if self.__group is None:
            print(
                "The OSmOSE group name is not defined. Please specify the group name before trying to build the dataset."
            )
        return self.__group

    @owner_group.setter
    def owner_group(self, value):
        if skip_perms:
            print("Cannot set osmose group on a non-Unix operating system.")
            self.__group = None
            return
        if value:
            try:
              gid = grp.getgrnam(value).gr_gid
            except KeyError as e:
              raise KeyError(
                f"The group {value} does not exist on the system. Full error trace: {e}"
            )

        self.__group = value


    def build(
        self,
        *,
        original_folder: str = None,
        owner_group: str = None,
        date_template: str = None,
        bare_check: bool = False,
        auto_normalization: bool = False,
        force_upload: bool = False,
        number_test_bad_files: int = 1,
        dico_aux_substring:dict = {'instrument':['depth','gps'],'environment':['insitu']}
    ) -> Path:
        """
        Set up the architecture of the dataset.

        The following operations will be performed on the dataset. None of them are destructive:
            - open and read the header of audio files located in `raw/audio/original/`.
            - rename files containing illegal characters.
            - generate some stastics regarding the files and dataset durations.
            - write the raw/metadata.csv file.
            - Identify and record files with anomalies (short duration, unreadable header...).
            - Set the permission of the dataset to the osmose group.

        Parameters
        ----------
            original_folder: `str`, optional, keyword-only
                The name of the folder containing the original audio file. It is named "original" by convention.
                If none is passed, the program expects to find either only one folder in the dataset audio folder, or
                a folder named `original`
            owner_group: `str`, optional, keyword_only
                The name of the group using the osmose dataset. It will have all permissions over the dataset.
            date_template: `str`, optional, keyword_only
                the date template in strftime format. For example, `2017/02/24` has the template `%Y/%m/%d`.
                It is used to generate automatically the timestamp.csv file. Alternatively, you can call the script to create the timestamp file first.
                If no template is provided, will assume that the file already exists. In future versions, the template will be guessed automatically.
                For more information on strftime template, see https://strftime.org/.
            bare_check : `bool`, optional, keyword_only
                Only do the checks and build steps that requires low resource usage. If you build the dataset on a login node or
                if you are sure it is already good to use, set to True. Otherwise, it should be inside a job. Default is False.
            auto_normalization: `bool`, optional, keyword_only
                If true, automatically normalize audio files if the data would cause issues downstream. The default is False.
            force_upload: `bool`, optional, keyword_only
                If true, ignore the file anomalies and build the dataset anyway. The default is False.

        Returns
        -------
            dataset: `Dataset`
                The dataset object.

        Example
        -------
            >>> from pathlib import Path
            >>> from OSmOSE import Dataset
            >>> dataset = Dataset(Path("home","user","my_dataset"))
            >>> dataset.build()

            DONE ! your dataset is on OSmOSE platform !
        """
        metadata_path = next(
            self.path.joinpath(OSMOSE_PATH.raw_audio).rglob("metadata.csv"), False
        )
        if metadata_path and metadata_path.exists() and pd.read_csv(metadata_path)["is_built"][0] and not force_upload:
            print("This dataset has already been built. To run the build() method on an already built dataset, you have to use the force_upload parameter.")
            sys.exit()

        if self.gps_coordinates is None:
            raise ValueError(
                f"GPS coordinates must be defined !"
            )  
        if self.depth is None:
            raise ValueError(
                f"Depth must be defined !"
            )  
            
        self.dico_aux_substring = dico_aux_substring


        if not self.__local:
            set_umask()
            if owner_group is None:
                owner_group = self.owner_group
    
            if not skip_perms:
                    print("\nSetting OSmOSE permission to the dataset...\n")
                    if owner_group:
                        gid = grp.getgrnam(owner_group).gr_gid
                        try:
                            os.chown(self.path, -1, gid)
                        except PermissionError:
                            print(f"You have not the permission to change the owner of the {self.path} folder. This might be because you are trying to rebuild an existing dataset. The group owner has not been changed.")
    
                    # Add the setgid bid to the folder's permissions, in order for subsequent created files to be created by the same user group.
                    os.chmod(self.path, DPDEFAULT)

        path_raw_audio = original_folder if original_folder is not None else self._find_or_create_original_folder()
        path_timestamp_formatted = path_raw_audio.joinpath("timestamp.csv")
    
        resume_test_anomalies = path_raw_audio.joinpath("resume_test_anomalies.txt")

        if not path_timestamp_formatted.exists():
            user_timestamp = False
            if not date_template:
                raise FileNotFoundError(f"The timestamp.csv file has not been found in {path_raw_audio}. You can create it automatically but to do so you have to set the date template as argument.")
            else:
                write_timestamp(audio_path=path_raw_audio, date_template=date_template, timezone=self.timezone, verbose=False)
        else:
            user_timestamp = True
        
        # read the timestamp.csv file
        timestamp_csv = pd.read_csv(path_timestamp_formatted)["timestamp"].values
        filename_csv = pd.read_csv(path_timestamp_formatted)["filename"].values
        
        # intialize the dataframe to collect audio metadata from header
        audio_metadata = pd.DataFrame(columns = ["filename", "timestamp","duration",
                                      "origin_sr","duration_inter_file","size","sampwidth","channel_count","status_read_header"])  
        audio_metadata["status_read_header"]=audio_metadata["status_read_header"].astype(bool)
        
        audio_file_list = [Path(path_raw_audio, indiv) for indiv in filename_csv]

        if not True:
            number_bad_files = check_n_files(
                audio_file_list,
                number_test_bad_files,
                auto_normalization=auto_normalization,
            )
        else: 
            number_bad_files = 0

        already_printed_1 = False
        for ind_dt in tqdm(range(len(timestamp_csv)), desc='Scanning audio files'):
            audio_file = audio_file_list[ind_dt]
            
            cur_timestamp, _ = self._format_timestamp(timestamp_csv[ind_dt],date_template,already_printed_1)

            # define final audio filename, especially we remove the sign '-' in filenames (because of our qsub_resample.sh)
            if "-" in audio_file.name:
                cur_filename = audio_file.name.replace("-", "_")
                path_raw_audio.joinpath(audio_file.name).rename(
                    path_raw_audio.joinpath(cur_filename)
                )
                if ind_dt==0: 
                    print(f"\n We do not accept the sign '-' in our filenames, we transformed them into '_'. In case you have to rebuild your dataset be careful to change your timestamp template accordingly.. \n")
            else:
                cur_filename = audio_file.name  
                
            try:
                origin_sr, frames, sampwidth, channel_count,size = read_header(path_raw_audio.joinpath(cur_filename))
                sf_meta = sf.info(path_raw_audio.joinpath(cur_filename))

            except Exception as e:
                print(f"error message making status read header False : \n {e}")
                # append audio metadata read from header for files with corrupted headers
                audio_metadata=pd.concat([audio_metadata , 
                                          pd.DataFrame({"filename":cur_filename,
                                                        "timestamp":cur_timestamp,
                                                        "duration":np.nan,
                                                        "origin_sr":np.nan,
                                                        "sampwidth":None,
                                                        "size":None,
                                                        "duration_inter_file":None,
                                                        "channel_count":None,
                                                        "status_read_header":False}, index=[0]) ],axis=0)
                continue
                
            # # define duration_inter_file; does not have a value for the last timestamp
            # if ind_dt > 0:
            #     duration_inter_file = (datetime.strptime(
            #         timestamp_csv[ind_dt], '%Y-%m-%dT%H:%M:%S.%f%z'
            #     ) - datetime.strptime(timestamp_csv[ind_dt-1], '%Y-%m-%dT%H:%M:%S.%f%z')).total_seconds()
            # else:
            #     duration_inter_file = None        

            # append audio metadata read from header in the dataframe audio_metadata
            audio_metadata=pd.concat([audio_metadata , 
                                      pd.DataFrame({"filename":cur_filename,
                                                    "timestamp":cur_timestamp,
                                                    "duration": sf_meta.duration ,#frames / float(origin_sr),
                                                    "origin_sr":int(sf_meta.samplerate),
                                                    "sampwidth":sampwidth,
                                                    "size":size / 1e6,
                                                    "duration_inter_file":None,
                                                    "channel_count":channel_count,
                                                    "status_read_header":True}, index=[0]) ],axis=0)
                
        audio_metadata['duration_inter_file'] = audio_metadata['duration'].diff()
                    
        # write file_metadata.csv
        audio_metadata.to_csv(
            path_raw_audio.joinpath("file_metadata.csv"),
            index=False
        )
        os.chmod(path_raw_audio.joinpath("file_metadata.csv"), mode=FPDEFAULT)

        # define anomaly tests of level 0 and 1
        test_level0_1 = (len(np.unique(audio_metadata["origin_sr"].values[~pd.isna(audio_metadata["origin_sr"].values)]))==1)
        test_level0_2 = (number_bad_files==0)        
        test_level0_3 = (sum(audio_metadata["status_read_header"].values)==len(timestamp_csv)) 
        test_level1_1 = (len(np.unique(audio_metadata["duration"].values[~pd.isna(audio_metadata["duration"].values)]))==1)        
        list_tests_level0 =  [test_level0_1 , test_level0_2 , test_level0_3]
        list_tests_level1 =  [test_level1_1]

        # write resume_test_anomalies.txt
        if resume_test_anomalies.exists():
            status_text='w'
        else:
            status_text='a'                
        lines = ["Anomalies of level 0", f"- Test 1 : {test_level0_1}", f"- Test 2 : {test_level0_2}", f"- Test 3 : {test_level0_3}","---------------------","Anomalies of level 1", f"- Test 1 : {test_level1_1}"]
        lines = [ll.replace('False','FAILED').replace('True','PASSED') for ll in lines]
        
        with open(resume_test_anomalies, status_text) as f:
            f.write('\n'.join(lines))                  
        
        # write messages in prompt for user
        if (len(list_tests_level0)-sum(list_tests_level0)>0):# if presence of anomalies of level 0
            print(f"\n\n Your dataset failed {len(list_tests_level0)-sum(list_tests_level0)} anomaly test of level 0 (over {len(list_tests_level0)}); see details below. \n Anomalies of level 0 block dataset uploading as long as they are present. Please correct your anomalies first, and try uploading it again after. \n You can inspect your metadata saved here {path_raw_audio.joinpath('file_metadata.csv')} using the notebook /home/datawork-osmose/osmose-datarmor/notebooks/metadata_analyzer.ipynb.")                   

            if len(list_tests_level1)-sum(list_tests_level1)>0:# if also presence of anomalies of level 1
                print(f"\n Your dataset also failed {len(list_tests_level1)-sum(list_tests_level1)} anomaly test of level 1 (over {len(list_tests_level1)}).")

            with open(resume_test_anomalies) as f: 
                print(f.read())
            
            # we remove timestamp.csv here to force its recreation as we may have changed the filenames during a first pass (eg - transformed into _)
            if not user_timestamp:# in case where the user did not bring its own timestamp.csv file
                os.remove(path_raw_audio.joinpath("timestamp.csv"))
                
        elif (len(list_tests_level1)-sum(list_tests_level1)>0) and not force_upload:# if presence of anomalies of level 1
            print(f"\n\n Your dataset failed {len(list_tests_level1)-sum(list_tests_level1)} anomaly test of level 1 (over {len(list_tests_level1)}); see details below. \n  Anomalies of level 1 block dataset uploading, but anyone can force it by setting the variable `force_upload` to True. \n You can inspect your metadata saved here {path_raw_audio.joinpath('file_metadata.csv')} using the notebook  /home/datawork-osmose/osmose-datarmor/notebooks/metadata_analyzer.ipynb. \n")              

            with open(resume_test_anomalies) as f: 
                print(f.read())

            if not user_timestamp:
                os.remove(path_raw_audio.joinpath("timestamp.csv"))
            
        else:# no anomalies

            # rebuild the timestamp.csv file (necessary as we might have changed filenames) and set permissions
            df = pd.DataFrame({"filename": audio_metadata["filename"].values, "timestamp": audio_metadata["timestamp"].values})
            df.sort_values(by=["timestamp"], inplace=True)
            df.to_csv(
                path_raw_audio.joinpath("timestamp.csv"),
                index=False
            )

            os.chmod(path_raw_audio.joinpath("timestamp.csv"), mode=FPDEFAULT)
    
            # change name of the original wav folder
            new_folder_name = path_raw_audio.parent.joinpath(
                str(int(mean(audio_metadata["duration"].values))) + "_" + str(int(mean(audio_metadata["origin_sr"].values)))
            )
            
            if new_folder_name.exists():
                new_folder_name.rmdir()
    
            path_raw_audio = path_raw_audio.rename(new_folder_name)
            self.__original_folder = path_raw_audio
    
            for subpath in OSMOSE_PATH:
                if "data" in str(subpath):
                    make_path(self.path.joinpath(subpath), mode=DPDEFAULT)

            # rename filenames in the subset_files.csv if any to replace -' by '_'
            subset_path = OSMOSE_PATH.processed.joinpath("subset_files.csv")
            if subset_path.is_file():
                xx = pd.read_csv(subset_path, header=None).values
                pd.DataFrame([ff[0].replace("-", "_") for ff in xx]).to_csv(
                    subset_path,
                    index=False,
                    header=None
                )
                os.chmod(subset_path, mode=FPDEFAULT)
    
            # write summary metadata.csv
            data = {
                "origin_sr": int(mean(audio_metadata["origin_sr"].values)),
                "sample_bits": int(8 * mean(audio_metadata["sampwidth"].values)),
                "channel_count": int(mean(audio_metadata["channel_count"].values)),
                "audio_file_count": len(audio_metadata["filename"].values),
                "start_date": timestamp_csv[0],
                "end_date": timestamp_csv[-1],
                # "duty_cycle": dutyCycle_percent,
                "audio_file_origin_duration": int(mean(audio_metadata["duration"].values)),
                "audio_file_origin_volume": round(mean(audio_metadata["size"].values),1),
                "dataset_origin_volume": max(1,round(sum(audio_metadata["size"].values)/ 1000)),# cannot be inferior to 1 GB
                "dataset_origin_duration": round(sum(audio_metadata["duration"].values)),
                "is_built": True,
                "audio_file_dataset_overlap": 0,
            }
            df = pd.DataFrame.from_records([data])
            df["lat"] = self.gps_coordinates[0]
            df["lon"] = self.gps_coordinates[1]
            df["depth"] = self.depth
            df["dataset_sr"] = int(mean(audio_metadata["origin_sr"].values))
            df["audio_file_dataset_duration"] = int(mean(audio_metadata["duration"].values))                       
            df.to_csv(
                path_raw_audio.joinpath("metadata.csv"),
                index=False
            )
            os.chmod(path_raw_audio.joinpath("metadata.csv"), mode=FPDEFAULT)

            for path, _, files in os.walk(self.path.joinpath(OSMOSE_PATH.auxiliary)):
                for f in files:
                    if f.endswith((".csv")):       
                        print(f"\n Checking your timestamp format in {Path(path,f).name}")
                        self._format_timestamp(Path(path,f),date_template,False)
                                                
            print("\n DONE ! your dataset is on OSmOSE platform !")


    def _format_timestamp(self,cur_timestamp_not_formatted:str,date_template:str,already_printed_1:int):

        format_OK = False

        if isinstance(cur_timestamp_not_formatted, PathLike):
            
            already_printed_1 = False            
            list_cur_timestamp_formatted = []
            dataF = pd.read_csv(cur_timestamp_not_formatted)
            for val_timestamp_not_formatted in dataF["timestamp"].values:
                cur_timestamp_formatted, format_OK = self._format_timestamp(val_timestamp_not_formatted,date_template,True)
                if format_OK:
                    print(f"-> Format OK \n")
                    return None
                else:
                    list_cur_timestamp_formatted.append(cur_timestamp_formatted)
            
            print(f"We reformatted timestamps in your file {cur_timestamp_not_formatted.name} \n")
            dataF["timestamp"] = list_cur_timestamp_formatted 

            dataF.to_csv(
                cur_timestamp_not_formatted,
                index=False
            )
            os.chmod(cur_timestamp_not_formatted, mode=FPDEFAULT)
            
            return None
            
                             
        # trick, we need to remove an ending Z so this code works properly, will need to be cleaned
        if cur_timestamp_not_formatted.endswith('Z'):
            cur_timestamp_not_formatted = cur_timestamp_not_formatted[:-1]
            if date_template and date_template.endswith('Z'):
                date_template = date_template[:-1]            
        
        try:
            check_right_format = datetime.strptime(cur_timestamp_not_formatted, '%Y-%m-%dT%H:%M:%S.%f%z')
            cur_timestamp_formatted = cur_timestamp_not_formatted
            format_OK = True
            
        except Exception as e:
                        
            if not already_printed_1:
                already_printed_1 = True
                print(f"Timestamp format {cur_timestamp_not_formatted} does not fit our template '%Y-%m-%dT%H:%M:%S.%f%z' let's reformat it")    
            if not date_template:
                raise FileNotFoundError(f"You have to define a date_template please.")
            else:
                date_obj = datetime.strptime(cur_timestamp_not_formatted+self.timezone, date_template+'%z')
                cur_timestamp_formatted = datetime.strftime(date_obj,'%Y-%m-%dT%H:%M:%S.%f%z')
                
        return cur_timestamp_formatted, format_OK                
                


    def _find_or_create_original_folder(self) -> Path:
        """Search for the original folder or create it from existing files.

        This function does in order:
    - If there is any audio file in the top directory, consider them all original and create the data/audio/original/ directory before
        moving all audio files in it.
    - If there is only one folder in the top directory, move it to /data/audio/original.
    - If there is a folder named "original" in the raw audio path, then return it
    - If there is only one folder in the raw audio path, then return it as the original.
    If none of the above is true, then a ValueError is raised as the original folder could not be found nor created. 

        Returns
        -------
            original_folder: `Path`
                The path to the folder containing the original files.

        Raises
        ------
            ValueError
                If the original folder is not found and could not be created.
        """
        path_raw_audio = self.path.joinpath(OSMOSE_PATH.raw_audio)
        audio_files = []
        parent_dir_list=[]
        timestamp_files = []

        make_path(path_raw_audio.joinpath("original"), mode=DPDEFAULT)
        make_path(self.path.joinpath(OSMOSE_PATH.other), mode=DPDEFAULT)
        make_path(self.path.joinpath(OSMOSE_PATH.instrument), mode=DPDEFAULT)
        make_path(self.path.joinpath(OSMOSE_PATH.environment), mode=DPDEFAULT)

        for path, _, files in os.walk(self.path):
            for f in files:
                if not Path(path,f).parent.name == "original" and not Path(path,f).parent.name == "auxiliary":
                    if f.endswith((".wav",".WAV","*.mp3",".*flac")):
                        audio_files.append(Path(path,f))
                        if str(Path(path,f).parent) != str(self.path):
                            parent_dir_list.append(Path(path,f).parent) if Path(path,f).parent not in parent_dir_list else parent_dir_list                        
                    elif f=="timestamp.csv":
                        Path(path,f).rename(path_raw_audio.joinpath("original","timestamp.csv"))
                    else:
                        for key_dico in self.dico_aux_substring:
                            if re.search('|'.join(self.dico_aux_substring[key_dico]), f):
                                Path(path,f).rename(self.path.joinpath(OSMOSE_PATH.auxiliary,key_dico,f))                    
    

        # if len(timestamp_files) > 1:
        #     res = "-1"
        #     choice = ""
        #     for i, ts in enumerate(timestamp_files):
        #         choice += f"{i+1}: {ts}\n"
        #     while int(res) not in range(1,len(timestamp_files) +1):
        #         res = input(f"Multiple timestamp.csv detected. Choose which one should be considered the original:\n{choice}")

        #         timestamp_files[int(res)-1].rename(path_raw_audio.joinpath("original","timestamp.csv"))
        # elif len(timestamp_files) == 1:
        #     timestamp_files[0].rename(path_raw_audio.joinpath("original","timestamp.csv"))

        for audio in audio_files:
            audio.rename(path_raw_audio.joinpath("original",audio.name))
            #os.chmod(path_raw_audio.joinpath("original",audio.name), mode=FPDEFAULT)
            
        for parent_dir in parent_dir_list:
            if len(os.listdir(parent_dir))>0:
                print(f'- Removing your subfolder: {parent_dir}, but be aware that you had the following non-wav data in your subfolder {parent_dir}: {os.listdir(parent_dir)} \n')
            else:
                print(f'- Removing your subfolder: {parent_dir}\n')
                
            shutil.rmtree(parent_dir)
            
        return path_raw_audio.joinpath("original")
        # if any(
        #     file.endswith(".wav") for file in os.listdir(self.path)
        # ):  # If there are audio files in the dataset folder
        #     make_path(path_raw_audio.joinpath("original"), mode=DPDEFAULT)

        #     for audiofile in os.listdir(self.path):
        #         if audiofile.endswith(".wav"):
        #             self.path.joinpath(audiofile).rename(
        #                 path_raw_audio.joinpath("original", audiofile)
        #             )
        #     return path_raw_audio.joinpath("original")
        # elif path_raw_audio.exists():
        #     if path_raw_audio.joinpath("original").is_dir():
        #         return path_raw_audio.joinpath("original")
        #     elif len(list(path_raw_audio.iterdir())) == 1:
        #         return path_raw_audio.joinpath(next(path_raw_audio.iterdir()))
        # elif (
        #     len(next(os.walk(self.path))[1]) == 1
        # ):  # If there is exactly one folder in the dataset folder
        #     make_path(path_raw_audio, mode=DPDEFAULT)
        #     orig_folder = self.path.joinpath(next(os.walk(self.path))[1][0])
        #     new_path = orig_folder.rename(path_raw_audio.joinpath(orig_folder.name))
        #     return new_path


        # else:
        #     raise ValueError(
        #         f"No folder has been found in {path_raw_audio}. Please create the raw audio file folder and try again."
        #     )

    def _get_original_after_build(self) -> Path:
        """Find the original folder path after the dataset has been built.

        Returns
        -------
            original_folder: `Path`
                The path to the folder containing the original audio file.

        Raises
        ------
            ValueError
                If no metadata.csv has been found and the original folder is not able to be found.
        """
        # First, grab any metadata.csv
        all_datasets = self.path.joinpath(OSMOSE_PATH.raw_audio).iterdir()
        while True:
            audio_dir = next(all_datasets)
            try:
                metadata_path = audio_dir.resolve().joinpath("metadata.csv")
            except StopIteration:
                # If we get to the end of the generator, it means that no metadata file has been found, so we raise a more explicit error.
                raise ValueError(
                    f"No metadata file found in {self.path.joinpath(OSMOSE_PATH.raw_audio, audio_dir)}. Impossible to find the original data file."
                )
            if metadata_path.exists():
                break

        metadata = pd.read_csv(metadata_path)
        # Catch the parameters inscribed in the original folder name
        audio_file_origin_duration = int(metadata["audio_file_origin_duration"][0])
        origin_sr = int(metadata["origin_sr"][0])

        self.__original_folder = self.path.joinpath(
            OSMOSE_PATH.raw_audio, f"{audio_file_origin_duration}_{origin_sr}"
        )

        return self.original_folder

    def __str__(self):
        metadata = pd.read_csv(self.original_folder.joinpath("metadata.csv"))
        list_display_metadata = [
            "audio_file_origin_duration",
            "origin_sr",
            "start_date",
            "end_date",            
            "audio_file_count",
            "audio_file_origin_volume",
            "dataset_origin_volume"
        ]  # restrain metadata to a shorter list of variables to be displayed
        ending_charac = ["(s)","(Hz)","","","","(MB)","(GB)"] # assign units to variables
        joined_str = ""
        print(f"Metadata of {self.name} :")
        ct=0
        for var in list_display_metadata:
            joined_str += f"- {var} : {metadata[var][0]} {ending_charac[ct]} \n"
            ct+=1
        return joined_str
