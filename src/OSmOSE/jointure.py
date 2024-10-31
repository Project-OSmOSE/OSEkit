import numpy as np
import time
import calendar
import datetime
import scipy.interpolate as inter
from pykdtree.kdtree import KDTree

#########################################################################################

# JOINTURE SPATIALE & TEMPORELLE POUR ERA5

########################################################################################


def get_era_time(x):
    return calendar.timegm(x.timetuple()) if isinstance(x, datetime.datetime) else x

def g(x):
    return calendar.timegm(time.strptime(str(x)[:-11], "%Y-%m-%dT%H"))


def rect_interpolation_era(stamps, var, method="linear"):
    # Enter the stamps array, the desired single level to interpolate
    # “linear”, “nearest”, “slinear”, “cubic”, “quintic” and “pchip”.
    t, x, y = (
        list(map(get_era_time, np.unique(stamps[:, :, :, 0]))),
        list(reversed(np.unique(stamps[:, :, :, 1]).astype("float64"))),
        np.unique(stamps[:, :, :, 2]).astype("float64"),
    )
    try:
        return inter.RegularGridInterpolator((t, x, y), var, method=method)
    except ValueError:
        return inter.RegularGridInterpolator((t, x, y), var, method="linear")


def time_interpolation_era(stamps, var, method="linear"):
    t = list(map(get_era_time, np.unique(stamps[:, :, :, 0])))
    interp = inter.interp1d(t, var, kind=method)
    return interp


def time_interpolation(time, var, method="linear"):
    interp = inter.interp1d(time, var, kind=method)
    return interp


def interpolation_gps(time, latitude, longitude, depth=None, method="linear"):
    interp_lat = inter.interp1d(time, latitude, kind=method)
    interp_lon = inter.interp1d(time, longitude, kind=method)
    if isinstance(depth, type(None)):
        return interp_lat, interp_lon
    else:
        interp_depth = inter.interp1d(time, depth, kind=method)
        return interp_lat, interp_lon, interp_depth


def apply_interp(interp, date, latitude=None, longitude=None, gps=None):
    if isinstance(interp, inter.interp1d):
        return interp(date)
    elif isinstance(interp, inter.RegularGridInterpolator):
        return interp(np.stack((date, latitude, longitude), axis=1))


def nearest_point(data, var):
    tree = KDTree(var)
    neighbor_dists, neighbor_indices = tree.query(data)
    return neighbor_indices, neighbor_dists
