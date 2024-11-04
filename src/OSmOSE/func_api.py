import os
from datetime import date, datetime

import cdsapi
import numpy as np
import pandas as pd
from netCDF4 import Dataset


def make_cds_file(key, udi, path_to_api):
    os.chdir(os.path.expanduser("~"))
    try:
        os.remove(".cdsapirc")
    except FileNotFoundError:
        pass

    cmd1 = "echo url: https://cds.climate.copernicus.eu/api/v2 >> .cdsapirc"
    cmd2 = f"echo key: {udi}:{key} >> .cdsapirc"
    os.system(cmd1)
    os.system(cmd2)

    try:
        os.mkdir("Documents/api")
    except FileExistsError:
        pass

    if path_to_api is None:
        path_to_api = os.path.join(os.path.expanduser("~"), "Documents/api/")

    os.chdir(path_to_api)
    os.getcwd()


def return_cdsapi(filename, key, variable, year, month, day, time, area):
    filename = filename + ".nc"

    c = cdsapi.Client()
    if month == "all":
        month = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    r = c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": variable,
            "year": year,
            "month": month,
            "day": day,
            "time": time,
            "area": area,
            "format": "netcdf",
            "grid": [0.25, 0.25],
        },
        filename,
    )
    r.download(filename)
    print("\n[===>--------]\n")


def format_nc(filename):
    downloaded_cds = filename + ".nc"
    fh = Dataset(downloaded_cds, mode="r")

    variables = list(fh.variables.keys())[3:]
    single_levels = np.zeros(
        (
            len(variables),
            fh.dimensions.get("time").size,
            fh.dimensions.get("latitude").size,
            fh.dimensions.get("longitude").size,
        ),
    )

    lon = fh.variables["longitude"][:]
    lat = fh.variables["latitude"][:]
    time_bis = fh.variables["time"][:]
    for i in range(len(variables)):
        single_levels[i] = fh.variables[variables[i]][:]

    def transf_temps(time):
        time_str = float(time) / 24 + date.toordinal(date(1900, 1, 1))
        return date.fromordinal(int(time_str))

    time_ = list(time_bis)

    data = pd.DataFrame()
    data["time"] = time_
    data["time"] = data["time"].apply(lambda x: transf_temps(x))
    hours = np.array(time_) % 24
    dates = np.array(
        [
            datetime(elem.year, elem.month, elem.day, hour)
            for elem, hour in list(zip(data["time"], hours))
        ],
    )

    print("\n[======>-----]\n")

    return dates, lon, lat, single_levels, variables


def save_results(dates, lat, lon, single_levels, variables, filename):
    for i in range(len(variables)):
        np.save(
            variables[i] + "_" + filename,
            np.ma.filled(single_levels[i], fill_value=float("nan")),
            allow_pickle=True,
        )
    stamps = np.zeros((len(dates), len(lat), len(lon), 3), dtype=object)
    for i in range(len(dates)):
        for j in range(len(lat)):
            for k in range(len(lon)):
                stamps[i, j, k] = [dates[i], lat[j], lon[k]]
    np.save("stamps_" + filename, stamps, allow_pickle=True)
    print("\n[============]\nDone")


def final_creation(
    df1,
    filename,
    key,
    variable,
    year,
    month,
    day,
    time,
    area,
    type_crea="complexe",
):
    return_cdsapi(filename, key, variable, year, month, day, time, area)
    dates, lon, lat, single_levels, variables = format_nc(filename)
    save_results(dates, lat, lon, single_levels, variables, filename)
    return variables


###################################################################################################################################


###################################################################################################################################


def mise_en_forme_old(filename):
    downloaded_cds = filename + ".nc"
    fh = Dataset(downloaded_cds, mode="r")

    variables = list(fh.variables.keys()[3:])
    single_levels = np.zeros(
        (
            len(variables),
            fh.dimensions.get("longitude").size,
            fh.dimensions.get("longitude").size,
            fh.dimensions.get("longitude").size,
        ),
    )

    lons = fh.variables["longitude"][:]
    lats = fh.variables["latitude"][:]
    time_bis = fh.variables["time"][:]
    for i in range(len(variables)):
        single_levels[i] = fh.variables[variables[i]][:]

    def transf_temps(time):
        time_str = float(time) / 24 + date.toordinal(date(1900, 1, 1))
        return date.fromordinal(int(time_str))

    time_ = list(time_bis)

    data = pd.DataFrame()
    data["time"] = time_
    data["time"] = data["time"].apply(lambda x: transf_temps(x))
    data["year"] = [elem.year for elem in data["time"]]
    data["month"] = [elem.month for elem in data["time"]]
    data["day"] = [elem.day for elem in data["time"]]
    data["hour"] = np.array(time_) % 24

    data["year"] = data["year"].apply(int)
    data["month"] = data["month"].apply(int)
    data["day"] = data["day"].apply(int)
    data["hour"] = data["hour"].apply(int)

    print("\n[======>-----]\n")

    return data, lons, lats


# df_spams_fin : df1, data : df2
def creation_complexe_old(df1, df2, u10, v10, tp, lats, lons):
    u10_ = []
    v10_ = []
    tp_ = []

    for i in range(len(df1)):
        j = 0
        n = len(df2)
        test_time = date(df2["year"].loc[j], df2["month"].loc[j], df2["day"].loc[j])
        test_hour = int(df2["hour"].loc[j])

        test_time_ = date(df1["year"].iloc[i], df1["month"].iloc[i], df1["day"].iloc[i])
        test_hour_ = int(df1["hour"].iloc[i])

        condition1 = test_time != test_time_
        condition2 = test_hour != test_hour_
        condition = condition1 or condition2

        while condition:
            if j == n - 1:
                break
            test_time = date(df2["year"].loc[j], df2["month"].loc[j], df2["day"].loc[j])
            test_hour = int(df2["hour"].loc[j])
            condition1 = test_time != test_time_
            condition2 = test_hour != test_hour_
            condition = condition1 or condition2
            j += 1

        if j == n - 1:
            None

        else:
            lat_test = df1["lat"].loc[i]
            long_test = df1["long"].loc[i]

            diff_lat = abs(lats - lat_test)
            idx_lat = np.argmin(diff_lat)

            diff_long = abs(lons - long_test)
            idx_long = np.argmin(diff_long)

            u10_.append(u10[j][idx_lat][idx_long])
            v10_.append(v10[j][idx_lat][idx_long])
            tp_.append(tp[j][idx_lat][idx_long])

    data_date = df2.merge(df1, on=["year", "month", "day", "hour"])
    data_fin = pd.DataFrame(
        data={
            "year": data_date["year"],
            "month": data_date["month"],
            "day": data_date["day"],
            "hour": data_date["hour"],
            "u10": u10_,
            "v10": v10_,
            "tp": tp_,
        },
    )
    print("\n[============]\n")
    return data_fin


def prepare_coord_target_old(filename):
    # dir = filename + '.csv'
    # test_ = pd.read_csv(dir)
    test_ = pd.DataFrame(columns=["time", "long", "lat", "profondeur"])
    # test_.columns = ['time', 'long', 'lat', 'profondeur']
    test_["time"] = test_["time"].apply(
        lambda x: pd.Timestamp(
            datetime.utcfromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )

    test_["year"] = test_["time"]
    test_["month"] = test_["time"]
    test_["day"] = test_["time"]
    test_["hour"] = test_["time"]

    for i in range(len(test_)):
        test_["year"].loc[i] = test_["time"].loc[i].year
        test_["month"].loc[i] = test_["time"].loc[i].month
        test_["day"].loc[i] = test_["time"].loc[i].day
        test_["hour"].loc[i] = test_["time"].loc[i].hour

    group_gps = test_.groupby(by=["year", "month", "day", "hour"])
    i = 0
    moy_long = []
    moy_lat = []

    for key, item in group_gps:
        a_group = group_gps.get_group(key)
        moy_long.append(np.mean(a_group["long"]))
        moy_lat.append(np.mean(a_group["lat"]))

    test_fin = test_[["year", "month", "day", "hour"]].drop_duplicates()
    test_fin["long"] = moy_long
    test_fin["lat"] = moy_lat
    test_fin["year"] = test_fin["year"].apply(int)
    test_fin["month"] = test_fin["month"].apply(int)
    test_fin["day"] = test_fin["day"].apply(int)
    test_fin["hour"] = test_fin["hour"].apply(int)

    print("\n[=========>--]\n")

    return test_fin.reset_index().drop(["index"], axis=1)
