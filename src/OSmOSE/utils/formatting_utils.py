import pandas as pd


def aplose2raven(df: pd.DataFrame) -> pd.DataFrame:
    """Export an APLOSE formatted result file to Raven formatted DataFrame

    Parameters
    ----------
    df: APLOSE formatted result DataFrame

    Returns
    -------
    df2raven: Raven formatted DataFrame

    Example of use
    --------------
    aplose_file = Path("path/to/aplose/result/file")
    df = (
        pd.read_csv(aplose_file, parse_dates=["start_datetime", "end_datetime"])
        .sort_values("start_datetime")
        .reset_index(drop=True)
    )

    df_raven = aplose2raven(df)

    # export to json format
    df.to_json("output/path.json")

    # import json format
    df_from_json = pd.read_json("path/to/json/file")
    """
    start_time = [
        (st - df["start_datetime"][0]).total_seconds() for st in df["start_datetime"]
    ]
    end_time = [st + dur for st, dur in zip(start_time, df["end_time"])]

    df2raven = pd.DataFrame()
    df2raven["Selection"] = list(range(1, len(df) + 1))
    df2raven["View"], df2raven["Channel"] = [1] * len(df), [1] * len(df)
    df2raven["Begin Time (s)"] = start_time
    df2raven["End Time (s)"] = end_time
    df2raven["Low Freq (Hz)"] = df["start_frequency"]
    df2raven["High Freq (Hz)"] = df["end_frequency"]

    return df2raven
