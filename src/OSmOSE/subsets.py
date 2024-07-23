from os.path import join, isfile
from glob import glob
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from random import sample
from shutil import copytree, copy2
from pathlib import Path
from alive_progress import alive_bar


from OSmOSE.utils.pylogger import logger as log

@dataclass
class SubsetGenerator():
    path_to_list_of_files : str = ''
    n_background_noise : int = 2
    remove_ext : bool = True

    def __post_init__(self):

        log.info('Initializing SubsetGenerator....')
        self.path_to_list_of_files = Path(self.path_to_list_of_files)
        log.info(f'Scanning paths and gathering infos from metadata files and {self.path_to_list_of_files.stem}.txt....')

        if isfile(self.path_to_list_of_files):
            self.subset_config, self.all_audio_file = self.scan_directory(self.path_to_list_of_files)

            for key, value in self.subset_config.items():
                 log.info(f"{key} = {value}")

            audio_metadata_df = pd.read_csv(self.subset_config['audio_metadata'])
            timestamp_metadata_df = pd.read_csv(self.subset_config['timestamp_metadata'])

            #Get positive samples
            with open(self.path_to_list_of_files) as f:
                positive_sample_list = f.read().splitlines()

            if self.remove_ext:
                positive_sample_list_wav = [file[:-4]+'.wav' for file in positive_sample_list]
            else:
                positive_sample_list_wav = [file+'.wav' for file in positive_sample_list]

            #Add n_background negative samples
            negative_samples = [ f for f in self.all_audio_file if f not in positive_sample_list_wav]
            chosen_background_files = sample(population=negative_samples, k = self.n_background_noise)
            #Concat both
            self.subset_wav_files = positive_sample_list_wav + chosen_background_files
            #write new timestamps for subset
            timestamp_subset_df = timestamp_metadata_df[timestamp_metadata_df.filename.isin(self.subset_wav_files)]            
            #write new audio metadata for subset
            audio_subset_metadata_df = audio_metadata_df.copy()
            audio_subset_metadata_df.loc[0,'audio_file_count'] = len(timestamp_subset_df)
            audio_subset_metadata_df.loc[0,'start_date'] = pd.to_datetime(timestamp_subset_df.loc[:,'timestamp']).min()
            audio_subset_metadata_df.loc[0,'end_date'] = pd.to_datetime(timestamp_subset_df.loc[:,'timestamp']).max()
            #write new spectro_metadata for subset
            # No need, it doesn't change ? double check with ENSTA.
            audio_subset_metadata_df.to_csv(Path(self.subset_config['audio_metadata']).parent / 'subset_metadata.csv', index=False)
            timestamp_subset_df.to_csv(Path(self.subset_config['timestamp_metadata']).parent / 'subset_timestamp.csv', index=False)
            log.info('Writing list of negative files in subset_chosen_background.txt')
            with open(Path(self.subset_config['timestamp_metadata']).parent /'subset_chosen_background.txt', 'w') as f: f.write('\n'.join(chosen_background_files))
            log.info(f'Number of positive samples selected : {len(positive_sample_list_wav)}')
            log.info(f'Number of negative samples selected : {len(chosen_background_files)}')
            log.info(f'Number of samples in subset : {len(self.subset_wav_files)}')
            
        else:
            print('NOT A CORRECT FILE PATH')
        return
    
    @staticmethod
    def ignore_files(dir, files):
        # https://stackoverflow.com/questions/15663695/shutil-copytree-without-files
        return [f for f in files if isfile(join(dir, f))]
    
    def copy_subset(self, output_path="", replace_original_metadata = True, remote_copy = False):
        log.info('Creating an empty tree structure...')
        copytree(src=self.path_to_list_of_files.parents[4], dst=output_path, ignore=self.ignore_files, dirs_exist_ok=True)

        spectro_path = Path(output_path) / 'processed' / 'spectrogram' / f"{self.subset_config['spectro_duration']}_{self.subset_config['sr']}" / f"{self.subset_config['nfft']}_{self.subset_config['window_size']}_{self.subset_config['overlap']}_{self.subset_config['custom_frequency_scale']}"
        audio_path = Path(output_path) / 'data' / 'audio' / f"{str(self.subset_config['spectro_duration'])}_{str(self.subset_config['sr'])}"

        log.info('Copying subset audio & image files to new directory...')
        with alive_bar(len(self.subset_wav_files)) as bar:
            for f in self.subset_wav_files:
                #audio
                src = Path(self.subset_config['audio_metadata']).parent / f 
                dest = audio_path / f
                copy2(src,dest)
                #images
                src = Path(self.subset_config['spectro_metadata']).parent / 'image' / (str(f)[:-4] + '_1_0.png')
                dest = spectro_path / 'image' / (str(f)[:-4]+'_1_0.png')
                copy2(src,dest)
                bar()
        log.info('Copying subset audio & image files to new directory...DONE')

        log.info('Writing subset metadata files...')

        if replace_original_metadata:
            metadata_file = 'metadata.csv'
            timestamp_metadata = 'timestamp.csv'
        else:
            metadata_file = 'subset_metadata.csv'
            timestamp_metadata = 'subset_timestamp.csv'

        #audio metadata
        src = Path(self.subset_config['audio_metadata']).parent / 'subset_metadata.csv' 
        dest = audio_path / metadata_file
        copy2(src,dest)
        #txt files
        src = Path(self.subset_config['spectro_metadata']).parent / 'list_of_positive_samples.txt' 
        dest = spectro_path / 'list_of_positive_samples.txt' 
        copy2(src,dest)

        src = Path(self.subset_config['audio_metadata']).parent / 'subset_chosen_background.txt' 
        dest = audio_path / 'subset_chosen_background.txt' 
        copy2(src,dest)

        #spectro metadata
        src = Path(self.subset_config['spectro_metadata']).parent / 'metadata.csv' 
        dest = spectro_path / metadata_file
        copy2(src,dest)

        #timestamps
        src = Path(self.subset_config['timestamp_metadata']).parent / 'subset_timestamp.csv' 
        dest = audio_path / timestamp_metadata
        copy2(src,dest)
    
        log.info('Writing subset metadata files...DONE')
        
    @staticmethod
    def scan_directory(input_path):
    # Convert input_path to a Path object if it's not already one
        path = Path(input_path) if not isinstance(input_path, Path) else input_path

        # Extract relevant components
        parts = path.parts
        dataset_path = Path(*parts[:-5])  # Up to 'ACQUISITION_20201120_20201215'
        spectrogram_info = parts[-2].split('_')
        audio_info = parts[-3].split('_')
        
        # Assign extracted values to variables
        spectro_duration = audio_info[0]
        sr = audio_info[1]
        nfft = spectrogram_info[0]
        window_size = spectrogram_info[1]
        overlap = spectrogram_info[2]
        custom_frequency_scale = spectrogram_info[3]

        # Create metadata paths
        # dataset_metadata = dataset_path / 'metadata.csv'
        spectro_metadata = path.parent / 'metadata.csv'
        timestamp_metadata = dataset_path / 'data' / 'audio' / f'{spectro_duration}_{sr}' / 'timestamp.csv'
        audio_metadata = dataset_path / 'data' / 'audio' / f'{spectro_duration}_{sr}' / 'metadata.csv'

        all_audio_files = glob(join(dataset_path,'data','audio',f'{spectro_duration}_{sr}','*.wav'))
        all_audio_files = [Path(path).stem+'.wav' for path in all_audio_files]


        return {
            'dataset_path': str(dataset_path),
            'spectro_duration': spectro_duration,
            'sr': sr,
            'nfft': nfft,
            'window_size': window_size,
            'overlap': overlap,
            'custom_frequency_scale': custom_frequency_scale,
            'audio_metadata': str(audio_metadata),
            'spectro_metadata': str(spectro_metadata),
            'timestamp_metadata': str(timestamp_metadata)
        } , all_audio_files
    
