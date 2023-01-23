import os
import sys
import soundfile as sf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List

def substract_timestamps(df: pd.DataFrame, files: List[str], index: int) -> timedelta:
    """Substracts two timestamps from the "timestamp" column of a dataframe at the indexes of files[i] and files[i-1] and returns the time delta between them
    
        Parameters:
        -----------
            df: the pandas DataFrame containing at least two columns: filename and timestamp

            files: the list of file names corresponding to the filename column of the dataframe

            index: the index of the file whose timestamp will be substracted
            
        Returns:
        --------
            The time between the two timestamps as a datetime.timedelta object"""

    cur_timestamp: str = df[df["filename"] == files[index]]["timestamp"].values[0]
    cur_timestamp: datetime = datetime.strptime(cur_timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
    prev_timestamp: str = df[df["filename"] == files[index -1]]["timestamp"].values[0]
    prev_timestamp: datetime = datetime.strptime(prev_timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")

    return cur_timestamp - prev_timestamp

def reshape(chunk_size: int, input_dir_path: str, output_dir_path: str = None, verbose : bool = False) -> List[str]:
    """Reshape all audio files in the folder to be of the specified duration. If chunk_size is superior to the base duration of the files, they will be fused according to their order in the timestamp.csv file in the same folder.
    
        Parameters:
        -----------
            `chunk_size`: the target duration for all the files.
            
            `input_dir_path`: the directory containing the audio files and the timestamp.csv file.
            
            `output_dir_path`: the directory where the newly created audio files will be created. If none is provided it will be the same as the input directory. This is not recommended.
            
        Returns:
        --------
            The list of the path of newly created audio files.
            """

    if not os.path.exists(os.path.join(input_dir_path, "timestamp.csv")):
        raise FileNotFoundError(f"The timestamp.csv file must be present in the directory {input_dir_path} and correspond to the audio files in the same location.")

    if not output_dir_path:
        print("Not output directory provided. Will use the input directory instead.")
        output_dir_path = input_dir_path

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)


    df = pd.read_csv(os.path.join(input_dir_path, "timestamp.csv"), header=None, names=["filename", "timestamp"])
    
    files = list(df["filename"])

    if verbose: 
        print(f"Files to be reshaped: {','.join(files)}")
        
    result = []
    timestamps = []
    timestamp: datetime = None
    prevdata = np.empty(1)
    sample_rate = 0
    i = 0
    t = 0
    proceed = False

    while i < len(files) -1:
        data, sample_rate = sf.read(os.path.join(input_dir_path, files[i]))

        if i == 0:
            timestamp = df[df["filename"] == files[i]]["timestamp"].values[0]
            timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
        elif substract_timestamps(df, files, i).seconds != len(data):
                print(f"Warning: You are trying to merge two audio files that are not chronologically consecutive.\n{files[i-1]} starts at {df[df['filename'] == files[i-1]]['timestamp'].values[0]} and {files[i]} starts at {df[df['filename'] == files[i]]['timestamp'].values[0]}.")
                if not proceed:
                    res = input("If you proceed, some timestamps will be lost in the reshaping. Proceed anyway? This message won't show up again if you choose to proceed. ([yes]/no)")
                    if "yes" in res.lower() or res == "":
                        proceed = True
                    else:
                        sys.exit()

        if prevdata.size > 0:
            data = np.concatenate((prevdata,data))
        # If the audio duration is longer than the chunk size, we cut it and add the rest for the next turn
        if len(data) > chunk_size * sample_rate:
            output = data[:chunk_size * sample_rate]
            prevdata = data[chunk_size * sample_rate:]
        
        elif len(data) == chunk_size * sample_rate:
            output = data
            prevdata = []

        # Else it is shorter, then while the duration is shorter than the desired chunk,
        # we read the next file and append it to the current one.
        else:
            while len(data) < chunk_size * sample_rate:
                nextdata, next_sample_rate = sf.read(os.path.join(input_dir_path, files[i+1]))
                rest = (chunk_size * next_sample_rate) - len(data)
                data = np.concatenate((data, nextdata[:rest] if rest <= len(nextdata) else nextdata))
                i+=1
            output = data
            prevdata = nextdata[rest:]

        outfilename = os.path.join(output_dir_path, f"reshaped_from_{t * chunk_size}_to_{(t +1) * chunk_size}_sec.wav")
        result.append(os.path.basename(outfilename))

        timestamps.append(datetime.strftime(timestamp, '%Y-%m-%dT%H:%M:%S.%f')[:-3] + "Z")
        timestamp += timedelta(seconds=chunk_size)

        sf.write(outfilename, output, sample_rate)

        if verbose:
            print(f"{outfilename} written! File is {(len(output)//sample_rate)//60} minutes long. {(len(prevdata)//sample_rate)//(60)} minutes left from slicing.")
        i +=1
        t += 1

    while len(prevdata) >= chunk_size * sample_rate:
        output = prevdata[:chunk_size * sample_rate]
        prevdata = prevdata[chunk_size * sample_rate:]

        outfilename = os.path.join(output_dir_path, f"from_{i * chunk_size}_to_{(i + 1) * chunk_size}_sec.wav")
        result.append(os.path.basename(outfilename))

        timestamps.append(datetime.strftime(timestamp, '%Y-%m-%dT%H:%M:%S.%f')[:-3] + "Z")
        timestamp += timedelta(seconds=chunk_size)

        sf.write(outfilename, output, sample_rate)
        
        print(f"{outfilename} written! File is {((len(output)//sample_rate)//60)} minutes long. {(len(prevdata)//sample_rate)//(60)} minutes left from slicing.")
        i += 1


    df = pd.DataFrame({'filename':result,'timestamp':timestamps})
    df.sort_values(by=['timestamp'], inplace=True)
    df.to_csv(os.path.join(output_dir_path,'timestamp.csv'), index=False,na_rep='NaN',header=None)


    return [os.path.join(output_dir_path, res) for res in result]

