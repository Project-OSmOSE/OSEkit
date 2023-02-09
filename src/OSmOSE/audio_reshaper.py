import os
import shutil
import sys
import soundfile as sf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Union
from argparse import ArgumentParser

def substract_timestamps(input_timestamp: pd.DataFrame, files: List[str], index: int) -> timedelta:
    """Substracts two timestamps from the "timestamp" column of a dataframe at the indexes of files[i] and files[i-1] and returns the time delta between them
    
        Parameters:
        -----------
            input_timestamp: the pandas DataFrame containing at least two columns: filename and timestamp

            files: the list of file names corresponding to the filename column of the dataframe

            index: the index of the file whose timestamp will be substracted
            
        Returns:
        --------
            The time between the two timestamps as a datetime.timedelta object"""

    cur_timestamp: str = input_timestamp[input_timestamp["filename"] == files[index]]["timestamp"].values[0]
    cur_timestamp: datetime = datetime.strptime(cur_timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
    prev_timestamp: str = input_timestamp[input_timestamp["filename"] == files[index -1]]["timestamp"].values[0]
    prev_timestamp: datetime = datetime.strptime(prev_timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")

    return cur_timestamp - prev_timestamp

# TODO: what to do with overlapping last file ? Del or incomplete ?
def reshape(chunk_size: int, input_files: Union[str, list], *, output_dir_path: str = None, ind_min: int = 0, ind_max: int = -1, 
        offset_beginning: int = 0, offset_end: int = 0, max_delta_interval: int = 5, verbose : bool = False, overwrite : bool = False,
        force_reshape: bool = False) -> List[str]:
    """Reshape all audio files in the folder to be of the specified duration. If chunk_size is superior to the base duration of the files, they will be fused according to their order in the timestamp.csv file in the same folder.
    
        Parameters:
        -----------            
            `input_files`: Either the directory containing the audio files and the timestamp.csv file, in which case all audio files will be considered,
                            OR a list of audio files all located in the same directory alongside a timestamp.csv, in which case only they will be used.
            
            `chunk_size`: The target duration for all the files.

            `output_dir_path`: The directory where the newly created audio files will be created. If none is provided it will be the same as the input directory. This is not recommended.
            
            `ind_min`: The first file of the list to be processed. Default is 0.

            `ind_max`: The last file of the list to be processed. Default is -1, meaning the entire list is processed.

            `offset_beginning`: The number of seconds that should be skipped in the first input file. When parallelising the reshaping,
             it would mean that the beginning of the file is being processed by another job. Default is 0.

            `offset_end`: The number of seconds that should be ignored in the last input file. When parallelising the reshaping, it would mean that the end of this file is processed by another job.
            Default is 0, meaning that nothing is ignored.

            `max_delta_interval`: The maximum number of second allowed for a delta between two timestamps to still be considered the same. Default is 5s up and down.

            `verbose`: Display informative messages or not

            `overwrite`: Deletes the content of `output_dir_path` before writing the results. If it is implicitly the `input_files` directory,
             nothing happens. WARNING: If `output_dir_path` is explicitly set to be the same as `input_files`, then it will be overwritten! 

             `force_reshape`: Ignore all warnings and non-fatal errors while reshaping.
        Returns:
        --------
            The list of the path of newly created audio files.
            """
    files = []

    if isinstance(input_files, list):
        input_dir_path = os.path.dirname(input_files[0])
        files = [os.path.basename(file) for file in input_files]
        if verbose: print(f"Input directory detected as {input_dir_path}")

    elif not os.path.isdir(input_files):
        raise ValueError("The input files must either be a folder path or a list of file path.")

    else:
        input_dir_path = input_files

    if not os.path.exists(os.path.join(input_dir_path, "timestamp.csv")):
        raise FileNotFoundError(f"The timestamp.csv file must be present in the directory {input_dir_path} and correspond to the audio files in the same location.")

    if overwrite and output_dir_path:
        shutil.rmtree(output_dir_path)

    if not output_dir_path:
        print("No output directory provided. Will use the input directory instead.")
        output_dir_path = input_dir_path
        if overwrite:
            print("Cannot overwrite input directory when the output directory is implicit! Choose a different output directory instead.")

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)


    input_timestamp = pd.read_csv(os.path.join(input_dir_path, "timestamp.csv"), header=None, names=["filename", "timestamp", "timezone"])

    # When automatically reshaping, will populate the files list
    if not files: files = list(input_timestamp["filename"][ind_min:ind_max if ind_max > 0 else input_timestamp.size])

    if verbose: 
        print(f"Files to be reshaped: {','.join(files)}")
        
    result = []
    timestamps = []
    timestamp: datetime = None
    prevdata = np.empty(1)
    sample_rate = 0
    i = 0
    t = 0
    proceed = force_reshape #Default is False

    while i < len(files):
        print("read file ", i)
        data, sample_rate = sf.read(os.path.join(input_dir_path, files[i]))


        if i == 0:
            timestamp = input_timestamp[input_timestamp["filename"] == files[i]]["timestamp"].values[0]
            print(f"First timestamp: {timestamp}")
            timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
            data = data[offset_beginning * sample_rate:]
        elif i == len(files) - 1 and offset_end != 0:
            data = data[:offset_end * sample_rate]

        # Need to check if size > 1 because numpy arrays are never empty urgh
        if prevdata.size > 1:
            data = np.concatenate((prevdata,data))
            prevdata = np.empty(1)

        # While the duration of the audio is longer than the target chunk, we segment it into small files
        # This means to account for the creation of 10s long files from big one and not overload data.
        if len(data) > chunk_size * sample_rate:
            print("data is longer")
            while len(data) > chunk_size * sample_rate:
                print(f"{len(data) > chunk_size * sample_rate}. Preuve : data is {len(data)}, chunk is {chunk_size * sample_rate}")
                output = data[:chunk_size * sample_rate]
                prevdata = data[chunk_size * sample_rate:]

                end_time = (t + 1) * chunk_size if chunk_size * sample_rate <= len(output) else t * chunk_size + len(output)//sample_rate

                outfilename = os.path.join(output_dir_path, f"reshaped_from_{t * chunk_size}_to_{end_time}_sec.wav")
                result.append(os.path.basename(outfilename))

                timestamps.append(datetime.strftime(timestamp, '%Y-%m-%dT%H:%M:%S.%f')[:-3] + "Z")
                timestamp += timedelta(seconds=chunk_size)

                sf.write(outfilename, output, sample_rate)

                if verbose:
                    print(f"{outfilename} written! File is {(len(output)/sample_rate)/60} minutes long. {(len(prevdata)/sample_rate)/(60)} minutes left from slicing.")

                t += 1
                data = prevdata

            # If after we get out of the previous while loop we don't have any data left, then we look at the next file.
            if (len(data)/sample_rate)/(60) == 0.0:
                i += 1
                continue

        # Else if data is already in the desired duration, output it
        if len(data) == chunk_size * sample_rate:
            output = data
            prevdata = np.empty(1)

        # Else it is shorter, then while the duration is shorter than the desired chunk,
        # we read the next file and append it to the current one.
        elif len(data) < chunk_size * sample_rate and i+1 < len(files):
            print("data is shorter than the desired chunk, let's read the next file !")

            # Check if the timestamps can safely be merged
            if not (len(data) - max_delta_interval < substract_timestamps(input_timestamp, files, i).seconds < len(data) + max_delta_interval):
                print(f"Warning: You are trying to merge two audio files that are not chronologically consecutive.\n{files[i-1]} starts at {input_timestamp[input_timestamp['filename'] == files[i-1]]['timestamp'].values[0]} and {files[i]} starts at {input_timestamp[input_timestamp['filename'] == files[i]]['timestamp'].values[0]}.")
                if not proceed and sys.__stdin__.isatty(): #check if the script runs in an interactive shell. Otherwise will fail if proceed = False
                    res = input("If you proceed, some timestamps will be lost in the reshaping. Proceed anyway? This message won't show up again if you choose to proceed. ([yes]/no)")
                    if "yes" in res.lower() or res == "":
                        proceed = True
                    else:
                        sys.exit()
                elif not proceed and not sys.__stdin__.isatty():
                    print("Error: Cannot merge non-continuous audio files if force_reshape is false.")
                    sys.exit(1)

            while len(data) < chunk_size * sample_rate and i+1 < len(files):
                nextdata, next_sample_rate = sf.read(os.path.join(input_dir_path, files[i+1]))
                rest = (chunk_size * next_sample_rate) - len(data)
                data = np.concatenate((data, nextdata[:rest] if rest <= len(nextdata) else nextdata))
                i+=1
            output = data
            prevdata = nextdata[rest:]

        end_time = (t + 1) * chunk_size if chunk_size * sample_rate <= len(output) else t * chunk_size + len(output)//sample_rate

        outfilename = os.path.join(output_dir_path, f"reshaped_from_{t * chunk_size}_to_{end_time}_sec.wav")
        result.append(os.path.basename(outfilename))

        timestamps.append(datetime.strftime(timestamp, '%Y-%m-%dT%H:%M:%S.%f')[:-3] + "Z")
        timestamp += timedelta(seconds=chunk_size)

        sf.write(outfilename, output, sample_rate)

        if verbose:
            print(f"{outfilename} written! File is {(len(output)//sample_rate)/60} minutes long. {(len(prevdata)//sample_rate)/(60)} minutes left from slicing.")
        i +=1
        t += 1

    while len(prevdata) >= chunk_size * sample_rate:
        output = prevdata[:chunk_size * sample_rate]
        prevdata = prevdata[chunk_size * sample_rate:]

        end_time = (i + 1) * chunk_size if chunk_size * sample_rate <= len(output) else i * chunk_size + len(output)//sample_rate

        outfilename = os.path.join(output_dir_path, f"reshaped_from_{i * chunk_size}_to_{end_time}_sec.wav")
        result.append(os.path.basename(outfilename))

        timestamps.append(datetime.strftime(timestamp, '%Y-%m-%dT%H:%M:%S.%f')[:-3] + "Z")
        timestamp += timedelta(seconds=chunk_size)

        sf.write(outfilename, output, sample_rate)
        
        print(f"{outfilename} written! File is {((len(output)//sample_rate)/60)} minutes long. {(len(prevdata)//sample_rate)/(60)} minutes left from slicing.")
        i += 1


    input_timestamp = pd.DataFrame({'filename':result,'timestamp':timestamps})
    input_timestamp.sort_values(by=['timestamp'], inplace=True)
    input_timestamp.to_csv(os.path.join(output_dir_path,'timestamp.csv'), index=False,na_rep='NaN',header=None)


    return [os.path.join(output_dir_path, res) for res in result]

if __name__ == "__main__":
    parser = ArgumentParser()
    required = parser.add_argument_group('required arguments')
    required.add_argument("--input-files", "-i", help="The files to be reshaped, as either the path to a directory containing audio files and a timestamp.csv or a list of filenames all in the same directory alongside a timestamp.csv.")
    required.add_argument("--chunk-size", "-s", type=int, help="The time in seconds of the reshaped files.")
    parser.add_argument("--output-dir", "-o", help="The path to the directory to write reshaped files. Default is same as --input-files directory.")
    parser.add_argument("--ind-min","-min", type=int, default=0, help="The first file of the list to be processed. Default is 0.")
    parser.add_argument("--ind-max", "-max", type=int, default=-1, help="The last file of the list to be processed. Default is -1, meaning the entire list is processed.")
    parser.add_argument("--offset-beginning",type=int, default=0, help="number of seconds that should be skipped in the first input file. When parallelising the reshaping, it would mean that the beginning of the file is being processed by another job. Default is 0.")
    parser.add_argument("--offset-end", type=int, default=0, help="The number of seconds that should be ignored in the last input file. When parallelising the reshaping, it would mean that the end of this file is processed by another job. Default is 0, meaning that nothing is ignored.")
    parser.add_argument("--max-delta-interval", type=int, default=5, help="The maximum number of second allowed for a delta between two timestamps to still be considered the same. Default is 5s up and down.")
    parser.add_argument("--verbose", "-v", action="store_true", default=True, help="Whether the script prints informative messages. Default is true.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="If set, deletes all content in --output-dir before writing the output. Default false, deactivated if the --output-dir is the same as --input-file dir.")
    parser.add_argument("--force", "-f", action="store_true", default=False, help="Ignore all warnings and non-fatal errors while reshaping.")

    args = parser.parse_args()

    input_files = args.input_files.split(" ") if not os.path.isdir(args.input_files) else args.input_files

    files = reshape(chunk_size=args.chunk_size, input_files=input_files, output_dir_path=args.output_dir, ind_min=args.ind_min, \
    ind_max=args.ind_max, offset_beginning=args.offset_beginning, offset_end=args.offset_end, max_delta_interval=args.max_delta_interval, \
    verbose=args.verbose, overwrite=args.overwrite, force_reshape=args.force)

    if args.verbose:
        print(f"All {len(files)} reshaped audio files written in {os.path.dirname(files[0])}.")