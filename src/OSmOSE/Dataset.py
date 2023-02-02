from typing import Union, List, Tuple
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import wave
import grp
from warnings import warn
from utils import read_config

class Dataset():
    def __init__(self, config: Union[str, dict]) -> None:
        self.__config = read_config(config)
        
        self.__name = config.dataset_name
        self.__path = os.path.join(config.dataset_folder_path, self.__name)
        self.__group = config.osmose_group_name


        """gps: The GPS coordinates of the listening location. It can be a list of 2 elements [latitude, longitude], or the 
                name of a csv file located in the `raw/auxiliary/` folder containing two columns: `lat` and `lon` with those informations."""
        if isinstance(self.__config.gps, str):
            csvFileArray = pd.read_csv(os.path.join(self.Path,'raw' ,'auxiliary' ,self.__config.gps))
            self.__coords = [(np.min(csvFileArray['lat']) , np.max(csvFileArray['lat'])) , (np.min(csvFileArray['lon']) , np.max(csvFileArray['lon']))]
        elif not isinstance(self.__config.gps, list):
            raise TypeError(f"GPS coordinates must be either a list of coordinates or the name of csv containing the coordinates, but {type(self.__config.gps)} found.")

        pd.set_option('display.float_format', lambda x: '%.0f' % x)

    #region Properties
    @property
    def Name(self):
        """The Dataset name."""
        return self.__name

    @property
    def Path(self):
        """The Dataset path."""
        return os.path.join(self.__path, self.Name)
    
    @property
    def Coords(self) -> Union[Tuple[float,float], Tuple[Tuple[float,float],Tuple[float,float]]] :
        """The GPS coordinates of the dataset. First element is latitude, second is longitude."""
        return self.__coords

    @property
    def Owner_Group(self):
        """The Unix group able to interact with the dataset."""
        return self.__group
    
    @property
    def Info_dict(self):
        """The information of configuration of the Dataset as a dict"""
        return self.__config

    @property
    def is_built(self):
        """Checks if self.Path/raw/audio contains at least one folder and none called "original"."""
        return len(os.listdir(os.path.join(self.Path, "raw","audio"))) > 0 and not os.path.exists(os.path.join(self.Path, "raw","audio","original"))
    #endregion

    def build(self, *, osmose_group_name:str = None, force_upload: bool = False) -> Tuple[list, list]:
        """
        
        Parameters:
        -----------
            dataset_ID: the name of the dataset folder
                
            osmose_group_name: The name of the group using the osmose dataset. It will have all permissions over the dataset.
            
            force_upload: If true, ignore the file anomalies and build the dataset anyway.
            
        Returns:
        --------
            A tuple containing a list of the abnormal filenames and the duration of good files."""

        
        if osmose_group_name is None:
            osmose_group_name = self.Owner_Group        

        path_timestamp_formatted = os.path.join(self.Path,'raw' ,'audio' ,'original','timestamp.csv')
        path_raw_audio = os.path.join(self.Path,'raw' ,'audio','original')
        
        
        csvFileArray = pd.read_csv(path_timestamp_formatted, header=None)

        timestamp_csv = csvFileArray[1].values
        filename_csv= csvFileArray[0].values
        
        list_filename_abnormal_duration=[]
        
        list_file_problem = []
        timestamp = []
        filename_rawaudio = []
        list_duration = []
        list_samplingRate = []
        list_interWavInterval = []
        list_size = []
        list_sampwidth = []
        list_filename = []
            
        for ind_dt in tqdm(range(len(timestamp_csv))):

            if ind_dt < len(timestamp_csv) - 1:
                diff = datetime.strptime(timestamp_csv[ind_dt +1], '%Y-%m-%dT%H:%M:%S.%fZ') - datetime.strptime \
                    (timestamp_csv[ind_dt], '%Y-%m-%dT%H:%M:%S.%fZ')
                list_interWavInterval.append(diff.total_seconds())

            filewav = os.path.join(path_raw_audio ,filename_csv[ind_dt])
            
            list_filename.append(filename_csv[ind_dt])

            try:
                with wave.open(filewav, "rb") as wave_file:
                    params = wave_file.getparams()
                    sr = params.framerate
                    frames = params.nframes
                    sampwidth = params.sampwidth
            
            except Exception as e:
                list_file_problem.append(filewav)
                print(f'The audio file {filewav} could not be loaded, its importation has been canceled.\nDescription of the error: {e}')
                list_filename_abnormal_duration.append(filewav)

            list_size.append(os.path.getsize(filewav) / 1e6)

            list_duration.append(frames / float(sr))
            #     list_volumeFile.append( np.round(sr * params.nchannels * (sampwidth) * frames / float(sr) /1024 /1000))
            list_samplingRate.append( float(sr) )
            list_sampwidth.append(sampwidth)
            
            # reformat timestamp.csv
            date_obj=datetime.strptime(timestamp_csv[ind_dt], '%Y-%m-%dT%H:%M:%S.%fZ')
            dates = datetime.strftime(date_obj, '%Y-%m-%dT%H:%M:%S.%f')
            # simply chopping !
            dates_final = dates[:-3] + 'Z'
            timestamp.append(dates_final) 
            
            # we remove the sign '-' in filenames (because of our qsub_resample.sh)        
            if '-' in filename_csv[ind_dt]:
                cur_filename = filename_csv[ind_dt].replace('-','_')
                os.rename(os.path.join(path_raw_audio ,filename_csv[ind_dt]), os.path.join(path_raw_audio ,cur_filename))
            else:
                cur_filename = filename_csv[ind_dt]
            filename_rawaudio.append(cur_filename)
        
        
        if list_filename_abnormal_duration:
            print('Please see list of audio files above that canceled your dataset importation (maybe corrupted files with OkB volume ?). You can also find it in the list list_filename_abnormal_duration, and execute following cell to directly delete them. Those filenames have been written in the file ./raw/audio/files_not_loaded.csv')
            
            with open(os.path.join(self.Path,'raw','audio','files_not_loaded.csv'), 'w') as fp:
                fp.write('\n'.join(list_filename_abnormal_duration))
            
            return list_filename_abnormal_duration
        

        
        dd = pd.DataFrame(list_interWavInterval).describe()
        print('Summary statistics on your INTER-FILE DURATION')
        print(dd[0].to_string())
        if dd[0]['std'] <1e-10:
            dutyCycle_percent =  round(100 *pd.DataFrame(list_duration).values.flatten().mean( ) /pd.DataFrame
                (list_interWavInterval).values.flatten().mean() ,1)
        else:
            dutyCycle_percent = np.nan


        # write raw/metadata.csv
        data = {'orig_fs' :float(pd.DataFrame(list_samplingRate).values.flatten().mean())
                ,'sound_sample_size_in_bits' :int( 8 *pd.DataFrame(list_sampwidth).values.flatten().mean())
                ,'nchannels' :int(params.nchannels) ,'nberWavFiles': len(filename_csv) ,'start_date' :timestamp_csv[0]
                ,'end_date' :timestamp_csv[-1] ,'dutyCycle_percent' :dutyCycle_percent
                ,'orig_fileDuration' :round(pd.DataFrame(list_duration).values.flatten().mean() ,2)
                ,'orig_fileVolume' :pd.DataFrame(list_size).values.flatten().mean()
                ,'orig_totalVolume' :round(pd.DataFrame(list_size).values.flatten().mean() * len(filename_csv) /1000, 1),
                'orig_totalDurationMins': round(pd.DataFrame(list_duration).values.flatten().mean() * len(filename_csv) / 60, 2),
                'lat':self.Coords[0],'lon':self.Coords[1]}
        df = pd.DataFrame.from_records([data])
        df.to_csv( os.path.join(self.Path,'raw' ,'metadata.csv') , index=False)  
        
        # write raw/audio/original/metadata.csv
        df['dataset_fs'] = float(pd.DataFrame(list_samplingRate).values.flatten().mean()) 
        df['dataset_fileDuration']=round(pd.DataFrame(list_duration).values.flatten().mean() ,2)
        df.to_csv( os.path.join(self.Path,'raw' ,'audio','original', 'metadata.csv') , index=False)  
        

        # get files with too small duration
        nominalVal_duration= int(np.percentile(list_duration, 10))
        print('\n Summary statistics on your file DURATION')
        dd_duration = pd.DataFrame(list_duration).describe()
        print(dd_duration[0].to_string())
        # go through the duration and check whether abnormal files
        ct_abnormal_duration=0
        self.list_abnormalFilename_name = []
        list_abnormalFilename_duration = []
        for name,duration in zip(list_filename,list_duration):
            if int(duration) < int(nominalVal_duration):
                ct_abnormal_duration+=1
                self.list_abnormalFilename_name.append(name)            
                list_abnormalFilename_duration.append(duration)            

            
        
        if ct_abnormal_duration > 0 and not force_upload:
            print('\n \n SORRY but your dataset contains files with different durations, especially',str(len(self.list_abnormalFilename_name)),'files that have durations smaller than the 10th percentile of all your file durations.. \n')
            
            print('Here are their summary stats:',pd.DataFrame(list_abnormalFilename_duration).describe()[0].to_string(),'\n')
            
            print('So YOUR DATASET HAS NOT BEEN IMPORTED ON OSMOSE PLATFORM, but you have the choice now : either 1) you can force the upload using the variable force_upbload , or 2) you can first delete those files with small durations, they have been put into the variable list_abnormalFilename_name and can be removed from your dataset using the cell below')
                    
        else:

            df = pd.DataFrame({'filename':filename_rawaudio,'timestamp':timestamp})
            df.sort_values(by=['timestamp'], inplace=True)
            df.to_csv(os.path.join(self.Path,'raw' ,'audio','original','timestamp.csv'), index=False,na_rep='NaN',header=None)        
            
            # change name of the original wav folder
            new_folder_name = os.path.join(self.Path,'raw' ,'audio',str(int(pd.DataFrame(list_duration).values.flatten().mean()))+'_'+str(int(float(pd.DataFrame(list_samplingRate).values.flatten().mean()))))
            os.rename( os.path.join(self.Path,'raw' ,'audio','original') , new_folder_name)
            
            # rename filenames in the subset_files.csv if any to replace -' by '_'
            if os.path.isfile( os.path.join(self.Path , 'analysis/subset_files.csv') ):
                xx=pd.read_csv(os.path.join(self.Path , 'analysis/subset_files.csv'),header=None).values
                pd.DataFrame([ff[0].replace('-','_') for ff in xx]).to_csv(os.path.join(self.Path , 'analysis/subset_files.csv'),index=False,header=None)   
                
            # save lists of metadata in metadata_file
            f = open(os.path.join(new_folder_name,"metadata_file.csv"), "w")
            for i in range(len(list_duration)):
                f.write(f"{filename_rawaudio[i]} {list_duration[i]} {list_samplingRate[i]}\n")
            f.close()        

            # change permission on the dataset
            if force_upload:
                print('\n Well you have anomalies but you choose to FORCE UPLOAD')
            print('\n Now setting OSmOSE permissions ; wait a bit ...')
            gid = grp.getgrnam(osmose_group_name).gr_gid

            os.chown(self.Path, -1, gid)
            os.chmod(self.Path, 0o770)
            for dirpath, dirnames, filenames in os.walk(self.Path):
                for filename in filenames:
                    os.chown(os.path.join(dirpath, filename), -1, gid)
                    os.chmod(os.path.join(dirpath, filename), 0o770)
            print('\n DONE ! your dataset is on OSmOSE platform !')


        return list_duration

    def delete_abnormal_files(self) -> None:
        """Delete all files with abnormal durations in the dataset, and rewrite the timestamps.csv file to reflect the changes."""
        
        if not self.list_abnormalFilename_name:
            warn("No abnormal file detected. You need to run the Dataset.build() method in order to detect abnormal files before using this method.")
            return

        path_raw_audio = os.path.join(self.Path,'raw' ,'audio','original')

        csvFileArray = pd.read_csv(os.path.join(path_raw_audio,'timestamp.csv'), header=None)

        for abnormal_file in self.list_abnormalFilename_name:

            filewav = os.path.join(path_raw_audio, abnormal_file)

            csvFileArray=csvFileArray.drop(csvFileArray[csvFileArray[0].values == os.path.basename(abnormal_file)].index)    

            print(f'removing : {os.path.basename(abnormal_file)}')
            os.remove(filewav)

        csvFileArray.sort_values(by=[1], inplace=True)
        csvFileArray.to_csv(os.path.join(path_raw_audio,'timestamp.csv'), index=False,na_rep='NaN',header=None)

        print('\n ALL AbNORMAL FILES REMOVED ! you can now re-run the previous file to finish importing it on OSmOSE platform')
