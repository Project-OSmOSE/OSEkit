from pathlib import Path
import pandas as pd

from export2Raven import df_Raven


def APLOSE2Raven(df: pd.DataFrame) -> pd.DataFrame:
    """Export an APLOSE formatted result file to Raven formatted DataFrame

    Parameters
    ----------
    df: APLOSE formatted result DataFrame

    Returns
    -------
    df_PG2Raven: Raven formatted DataFrame
    """
    start_time = [
        (st - df["start_datetime"][0]).total_seconds() for st in df["start_datetime"]
    ]
    end_time = [st + dur for st, dur in zip(start_time, df["end_time"])]

    df2Raven = pd.DataFrame()
    df2Raven["Selection"] = list(range(1, len(df) + 1))
    df2Raven["View"], df2Raven["Channel"] = [1] * len(df), [1] * len(df)
    df2Raven["Begin Time (s)"] = start_time
    df2Raven["End Time (s)"] = end_time
    df2Raven["Low Freq (Hz)"] = df["start_frequency"]
    df2Raven["High Freq (Hz)"] = df["end_frequency"]

    return df2Raven


# %% export to Raven format

APLOSE_file = Path("path/to/APLOSE/result/file")
df = (
    pd.read_csv(APLOSE_file, parse_dates=["start_datetime", "end_datetime"])
    .sort_values("start_datetime")
    .reset_index(drop=True)
)

df_Raven = APLOSE2Raven(df)


# %% export to json format
df.to_json("output/path.json")

# %% import json format
df_from_json = pd.read_json("path/to/json/file")