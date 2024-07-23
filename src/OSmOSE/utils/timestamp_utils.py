from datetime import datetime, timedelta
import pandas as pd
from typing import List
import os


def check_epoch(df):
    "Function that adds epoch column to dataframe"
    df['timestamp'] = df.timestamp.apply(to_timestamp).dt.tz_localize(None)
    if "epoch" in df.columns:
        return df
    try :
        df['epoch'] = df.timestamp.apply(lambda x : x.timestamp())
    except ValueError:
        print("Please check that you have either a timestamp column (format ISO 8601 Micro s) or an epoch column")
    return df


def substract_timestamps(
    input_timestamp: pd.DataFrame, files: List[str], index: int
) -> timedelta:
    """Substracts two timestamp_list from the "timestamp" column of a dataframe at the indexes of files[i] and files[i-1] and returns the time delta between them

    Parameters:
    -----------
        input_timestamp: the pandas DataFrame containing at least two columns: filename and timestamp

        files: the list of file names corresponding to the filename column of the dataframe

        index: the index of the file whose timestamp will be substracted

    Returns:
    --------
        The time between the two timestamp_list as a datetime.timedelta object"""

    if index == 0:
        return timedelta(seconds=0)

    cur_timestamp: str = input_timestamp[input_timestamp["filename"] == files[index]][
        "timestamp"
    ].values[0]
    cur_timestamp: datetime = to_timestamp(cur_timestamp)
    next_timestamp: str = input_timestamp[
        input_timestamp["filename"] == files[index + 1]
    ]["timestamp"].values[0]
    next_timestamp: datetime = to_timestamp(next_timestamp)

    return next_timestamp - cur_timestamp


def to_timestamp(string: str) -> datetime:
    if isinstance(string, datetime):
        return string
    try:
        return datetime.strptime(string, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        try:
            return datetime.strptime(string, "%Y-%m-%dT%H-%M-%S_%fZ")
        except ValueError:
            try :
                return datetime.strptime(string, "%Y-%m-%dT%H:%M:%S.%f%z")
            except ValueError:
                raise ValueError(
                    f"The timestamp '{string}' must match either format %Y-%m-%dT%H:%M:%S.%fZ or %Y-%m-%dT%H-%M-%S_%fZ"
                )


def from_timestamp(date: datetime) -> str:
    return datetime.strftime(date, "%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def get_timestamps(
    path_osmose_dataset: str, campaign_name: str, dataset_name: str, resolution: str
) -> pd.DataFrame:
    """Read infos from APLOSE timestamp csv file
    Parameters
    -------
        path_osmose_dataset: 'str'
            usually '/home/datawork-osmose/dataset/'

        campaign_name: 'str'
            Name of the campaign
        dataset_name: 'str'
            Name of the dataset
        resolution: 'str'
            Resolution of the dataset
    Returns
    -------
        df: pd.DataFrame
            The timestamp file is read and returned as a DataFrame
    """

    csv = os.path.join(
        path_osmose_dataset,
        campaign_name,
        dataset_name,
        "data",
        "audio",
        resolution,
        "timestamp.csv",
    )

    if os.path.exists(csv):
        df = pd.read_csv(csv)
        return df
    else:
        raise ValueError(f"{csv} does not exist")
