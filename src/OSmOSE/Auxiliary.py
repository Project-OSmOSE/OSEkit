import cdsapi
import numpy as np
import pandas as pd
import os
import calendar
from glob import glob
from tqdm import tqdm
import netCDF4 as nc
from typing import Union, Tuple, List
from datetime import datetime, date
from OSmOSE.config import *
from OSmOSE.Spectrogram import Spectrogram
from scipy import interpolate 



def check_epoch(df):
	"Function that adds epoch column to dataframe"
	if 'epoch' in df.columns :
		return df
	else :
		try :
			df['epoch'] = df.timestamp.apply(lambda x : datetime.strptime(x[:26], '%Y-%m-%dT%H:%M:%S.%f').timestamp())
			return df
		except ValueError :
			print('Please check that you have either a timestamp column (format ISO 8601 Micro s) or an epoch column')
			return df

class Auxiliary(Spectrogram):
	'''
	This class joins environmental and instrument data to acoustic data. 
	The acoustic data is first fetched using the dataset path, the data's samplerate and the analysis parameters.
	If no analysis parameters are provided then data will be joined to corresponding raw audio files.
	'''

	# CHECK THAT ALL TIMEZONES ARE THE SAME PLEASE (UTC 00)

	def __init__(
		self,
		dataset_path: str,
		*,
		gps_coordinates: Union[str, List, Tuple, bool] = True,
		depth: Union[str, int, bool] = True,
		dataset_sr: int = None,
		owner_group: str = None,
		analysis_params: dict = None,
		batch_number: int = 5,
		local: bool = True,
		
		era : Union[str, bool] = False,
		annotation : Union[dict, bool] = False,
		other: dict = None
		):
		
		"""		
		Parameters:
		       dataset_path (str): The path to the dataset.
		       dataset_sr (int, optional): The dataset sampling rate. Default is None.
		       analysis_params (dict, optional): Additional analysis parameters. Default is None.
		       gps_coordinates (str, list, tuple, bool, optional): Whether GPS data is included. Default is True. If string, enter the filename (csv) where gps data is stored.
		       depth (str, int, bool, optional): Whether depth data is included. Default is True. If string, enter the filename (csv) where depth data is stored.
		       era (bool, optional): Whether era data is included. Default is False. If string, enter the filename (Network Common Data Form) where era data is stored.
		       annotation (bool, optional): Annotation data is included. Dictionary containing key (column name of annotation data) and absolute path of csv file where annotation data is stored. Default is False. 
		       other (dict, optional): Additional data (csv format) to join to acoustic data. Key is name of data (column name) to join to acoustic dataset, value is the absolute path where to find the csv. Default is None.
		Notes:
		       The parameters `gps`, `depth`, `era`, `annotation`, and `other` are used as flags
		       to indicate the presence of data to join to the corresponding spectrogram generation. When set to True, the respective
		       data will be processed and included.
		"""
		
		super().__init__(dataset_path, gps_coordinates=gps_coordinates, depth=depth, dataset_sr=dataset_sr, owner_group=owner_group, analysis_params=analysis_params, batch_number=batch_number, local=local)
				
		# Load reference data that will be used to join all other data
		self.df = pd.read_csv(self.audio_path.joinpath('timestamp.csv'))
		self.metadata = pd.read_csv(self._get_original_after_build().joinpath("metadata.csv"), header=0)
		self._depth, self._gps_coordinates = depth, gps_coordinates
		if self._depth :
			self.depth = 'depth.csv' if depth==True else depth
		if self._gps_coordinates :
			self.gps_coordinates = 'gps.csv' if gps_coordinates==True else gps_coordinates
		self.era = era
		if self.era == True:
			fns = glob.glob(self.path.joinpath(OSMOSE_PATH.era, '*nc'))
			assert len(fns) == 1, "Make sur there is one (and only one) nc file in your era folder"
			self.era = fns[0]		
		self.other = other if other is not None else {}	
		if annotation : 
			self.other = {**self.other, **annotation}

	
	def __str__(self):
		print(f'For the {self.name} dataset')
		elems = []
		if self._gps_coordinates :
			elems.append('gps')
		if self._depth :
			elems.append('depth')
		if self.era :
			elems.append('era')
		if self.other:
			elems.extend(self.other.keys())
		if len(elems) != 0:
			aux_str = 'This class will join : \n' 
			for elem in elems :
				aux_str += f'   - {elem}\n'
			return aux_str
		else: 
			return 'You have not selected any data to join to your dataset ! '
	
	def join_depth(self):
		"""
		Code to join depth data to reference dataframe.
		The depth files should contain times in both Datetime and epoch format and a depth column
		"""
		
		if isinstance(self.depth, pd.DataFrame) :
			assert ('epoch' in self.depth.columns) and ('depth' in self.depth.columns), "Make sure the depth file has both an 'epoch' and 'depth' column."
			# Join data using a 1D interpolation
			self.df['depth'] = interpolate.interp1d(self.depth.epoch, self.depth.depth, bounds_error = False)(self.df.epoch)
			
		elif type(self.depth) == int :
			self.df['depth'] = self.depth
			
			
	def join_gps(self):
		"""
		Code to join gps data to reference dataframe.
		The gps files should contain times in both Datetime and epoch format and a depth column
		"""
		
		if isinstance(self.gps_coordinates, pd.DataFrame):
			assert ('epoch' in self.gps_coordinates.columns) and ('lat' in self.gps_coordinates.columns) and ('lon' in self.gps_coordinates.columns), "Make sure the depth file has both an 'epoch' and 'lat'/'lon' columns."
			# Join data using a 1D interpolation
			self.df['lat'] = interpolate.interp1d(self.gps_coordinates.epoch, self.gps_coordinates.lat, bounds_error = False)(self.df.epoch)
			self.df['lon'] = interpolate.interp1d(self.gps_coordinates.epoch, self.gps_coordinates.lon, bounds_error = False)(self.df.epoch)
			
		elif type(self.gps_coordinates) in [list, tuple] :
			self.df['lat'] = self.gps_coordinates[0]
			self.df['lon'] = self.gps_coordinates[1]

	def join_other(self, csv_path: str = None, variable_name : Union[str, List, Tuple] = None):
		'''
		Method to join data using solely temporal interpolation.
		Parameters:
		       csv_path (str): Absolute path to new csv file (containing epoch column) you want to join to auxiliary dataframe 
		       variable_name (str, List, Tuple) : Variable names (and future column names) you want to join to auxiliary dataframe
		'''
		
		if variable_name and csv_path:
			variable_name = list(variable_name) if variable_name is not str else [variable_name]
			self.other = {**self.other, **{key: csv_path for key in variable_name}}
		
		for key in self.other.keys():
			_csv = pd.read_csv(self.other[key])
			self.df[key] = interpolate.interp1d(_csv.epoch, _csv[key], bounds_error=False)(self.df.epoch)


	def interpolation_era(self):
		"""Computes interpolated variables values from ERA5 mesh points.
		ERA5 is suited for a scipy's 3D grid interpolation in time, latitude and longitude.
		As this method is quick, it is computed for all available variables.
		Results are saved in self.df as 'interp_{variable}'
		"""
		ds = nc.Dataset(self.path.joinpath(OSMOSE_PATH.era, self.era))
		variables = list(ds.variables.keys())[3:]
		
		#Handle ERA time
		era_time = ds.variables['time'][:].data
		def transf_temps(time) :
			time_str = float(time)/24 + date.toordinal(date(1900,1,1))
			return date.fromordinal(int(time_str))
		hours = np.array(era_time)%24
		days = list(map(transf_temps, era_time))
		timestamps = np.array([datetime(elem.year, elem.month, elem.day, hour).timestamp() for elem, hour in list(zip(days, hours))])
		
		pbar = tqdm(total=len(variables), position=0, leave=True)
		for variable in variables:
			pbar.update(1); pbar.set_description("Loading and formatting %s" % variable)
			self.df[variable] = interpolate.RegularGridInterpolator((timestamps, ds['lat'][:], ds['lon'][:]), ds[variable][:])((self.df.epoch, self.df.lat, self.df.lon))

	
	def automatic_join(self):
		''' Automatically join all the available data'''
		if self._depth :
			self.join_depth()
		if self._gps_coordinates :
			self.join_gps()
		if self.era:
			self.interpolation_era()
		if self.other :
			self.join_other()


def make_cds_file(key, udi, path):
        os.chdir(os.path.expanduser("~"))
        try :
           os.remove('.cdsapirc')
        except FileNotFoundError :
            pass

        cmd1 = "echo url: https://cds.climate.copernicus.eu/api/v2 >> .cdsapirc"
        cmd2 = "echo key: {}:{} >> .cdsapirc".format(udi, key)
        os.system(cmd1)
        os.system(cmd2)

        if path == None:
                try :
                   os.mkdir('api')
                except FileExistsError:
                    pass
                path_to_api = os.path.join(os.path.expanduser("~"), "api/")
        else :
                try :
                   os.mkdir(os.path.join(path+'api'))
                except FileExistsError:
                    pass
                path_to_api = os.path.join(path, 'api')

        os.chdir(path_to_api)
        os.getcwd()



def return_cdsapi(filename, key, variables, years, months, days, hours, area):

        print('You have selected : \n')
        sel = [print(variables) for data in variables]
        print('\nfor the following times')
        print(f'Years : {years} \n Months : {months} \n Days : {days} \n Hours : {hours}')

        print('\nYour boundaries are : North {}°, South {}°, East {}°, West {}°'.format(area[0], area[2],
                                                                                 area[3], area[1]))

        filename = filename + '.nc'

        c = cdsapi.Client()

        if days == 'all':
                days = ['01', '02', '03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
        if months == 'all':
                months = ['01','02','03','04','05','06','07','08','09','10','11','12']
        if hours == 'all':
                hours = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']

        r = c.retrieve('reanalysis-era5-single-levels',
            {
              'product_type' : 'reanalysis',
              'variable' : variables,
              'year' : years,
              'month' : months,
              'day' : days,
              'time' : hours,
              'area' : area,
              'format' : 'netcdf',
              'grid':[0.25, 0.25],
            },
            filename,
            )
        r.download(filename)
                                    


'''def nearest_era(
elf, *, time_off=600, lat_off=0.5, lon_off=0.5, variables=["u10", "v10"]
):
"""joins nearest mesh point values to dataset.

The variable values from the nearest mesh point at a given time is joined to the timestamp.
This occurs only if the mesh point is within the given time, latitude and longitude offsets.
Latitude, longitude and time distance have the same weight.

Parameter
---------
time_off: 'int' or 'float'
Maximum time offset allowed for mesh point to be taken into consideration, in seconds.
lat_off : 'int' or 'float'
Maximum latitude offset allowed for mesh point to be taken into consideration, in degrees.
lon_off : 'int' or 'float'
Maximum longitude offset allowed for mesh point to be taken into consideration, in degrees.

Returns
-------
None. Results are saved in self.df as 'interp_{variable}'
"""
self.load_era(method="nearest")
data = self.df[["time", "lat", "lon"]].dropna().to_numpy()
self.stamps[:, :, :, 0] = np.array(
list(map(get_era_time, self.stamps[:, :, :, 0].ravel()))
).reshape(self.stamps[:, :, :, 0].shape)
pos_ind, pos_dist = j.nearest_point(data, self.stamps.reshape(-1, 3))
if variables == "None":
variables = self.variables
pbar = tqdm(total=len(variables), position=0, leave=True)
for variable in variables:
pbar.update(1)
pbar.set_description("Loading and formatting %s" % variable)
temp_variable = np.full([len(self.timestamps)], np.nan)
var = np.load(
os.path.join(self.era_path, variable + "_" + self.dataset + ".npy")
)
var = var.flatten()[pos_ind]
mask = np.any(
abs(data - self.stamps.reshape(-1, 3)[pos_ind])
> [time_off, lat_off, lon_off],
axis=1,
)
var[mask] = np.nan
temp_variable[self.df[["time", "lat", "lon"]].dropna().index] = var
variable = f"np_{time_off}_{lat_off}_{lon_off}_{variable}"
self.df[variable] = temp_variable
self.df[f"np_{time_off}_{lat_off}_{lon_off}_era"] = np.sqrt(
self.df[f"np_{time_off}_{lat_off}_{lon_off}_u10"] ** 2
+ self.df[f"np_{time_off}_{lat_off}_{lon_off}_v10"] ** 2
)
del var



def cube_era(
self, *, time_off=600, lat_off=0.5, lon_off=0.5, variables=["u10", "v10"]
):
"""Computes average of variables values of all ERA5 mesh points within a cube.
All mesh points within a defined latitude distance and longitude distance are kept, and within a maximum time offset.

Parameter
---------
time_off: 'int' or 'float'
Maximum time offset allowed for mesh point to be taken into consideration, in seconds.
lat_off : 'int' or 'float'
Maximum latitude offset allowed for mesh point to be taken into consideration, in degrees.
lon_off : 'int' or 'float'
Maximum longitude offset allowed for mesh point to be taken into consideration, in degrees.
variables : 'list' or None
List of variables from ERA5 to join to hydrophone position. None if you want all variables to be joined to data.

Returns
-------
None. Results are saved in self.df as 'cube_{time_off}_{lat_off}_{lon_off}_{variable}'
"""
self.load_era(method="cube")
temp_df = self.df[["time", "lat", "lon"]]
if variables == None:
variables = self.variables
for variable in variables:
temp_variable = np.full([len(self.timestamps)], np.nan)
var = np.load(
os.path.join(self.era_path, variable + "_" + self.dataset + ".npy")
)
pbar = tqdm(total=len(temp_df), position=0, leave=True)
for i, row in temp_df.iterrows():
if row.isnull().any():  # Skip iteration if there is a nan value
continue
time_index = np.where(
(abs(self.stamps["timestamps_epoch"] - row.time) <= time_off)
)[
0
]  # Get timestamps satisfyin time offset
if (
len(time_index) == 0
):  # Skip iteration if no weather source is within time offset for this timestep
pbar.update(1)
temp_variable[i] = np.nan
continue
lat_index = np.where(
(abs(self.stamps["latitude"] - row.lat) <= lat_off)
)[
0
]  # Get latitude indexes satisfyin lon offset
if (
len(lat_index) == 0
):  # Skip iteration if no weather source is within lat offset for this timestep
pbar.update(1)
temp_variable[i] = np.nan
continue
lon_index = np.where(
(abs(self.stamps["longitude"] - row.lon) <= lon_off)
)[
0
]  # Get longitude indexes satisfying lon offset
if (
len(lon_index) == 0
):  # Skip iteration if no weather source is within lon offset for this timestep
pbar.update(1)
temp_variable[i] = np.nan
continue
temp_value = []
for time_ind in time_index:
for lat_ind in lat_index:
for lon_ind in lon_index:
temp_value.append(
var[time_ind, lat_ind, lon_ind]
)  # Saving values satisfying the conditions
temp_variable[i] = np.nanmean(temp_value)
pbar.update(1)
variable = f"cube_{time_off}_{lat_off}_{lon_off}_{variable}"
self.df[variable] = temp_variable
self.df[f"cube_{time_off}_{lat_off}_{lon_off}_era"] = np.sqrt(
self.df[f"cube_{time_off}_{lat_off}_{lon_off}_u10"] ** 2
+ self.df[f"cube_{time_off}_{lat_off}_{lon_off}_v10"] ** 2
)
del var

def cylinder_era(self, *, time_off=600, r="depth", variables=["u10", "v10"]):
"""Computes average variables values of all ERA5 mesh points within a cylinder.

All mesh points within a defined distance are kept, and within a maximum time offset. Radius can vary as a function of depth.
90% of noise sources that can be heard at a depth D come from a surface area of (3D)²*pi.

Parameter
---------
time_off: 'int' or 'float'
Maximum time offset allowed for mesh point to be taken into consideration, in seconds.
r : 'int', 'float' or 'str'
Maximum distance in kilometer for which a mesh point is taken into consideration. 'depth' if that distance is computed with the hydrophone's depth.
variables : 'list' or None
List of variables from ERA5 to join to hydrophone position. None if you want all variables to be joined to data.

Returns
-------
None. Results are saved in self.df as 'cylinder_{time_off}_{lat_off}_{lon_off}_{variable}'
"""
self.load_era(method="cylinder")
temp_df = self.df[["time", "lat", "lon"]]
if isinstance(r, int) or isinstance(r, float):
radius = [r] * len(temp_df)
else:
radius = (
3 * self.df.depth
)  # Compute radius as a function of hydrophone's depth
if variables == None:
variables = self.variables
for variable in variables:
temp_variable = np.full([len(self.timestamps)], np.nan)
var = np.load(
os.path.join(self.era_path, variable + "_" + self.dataset + ".npy")
)
pbar = tqdm(total=len(temp_df), position=0, leave=True)
for i, row in temp_df.iterrows():
if row.isnull().any():  # Skip iteration if there is a nan value
continue
time_index = np.where(
(abs(self.stamps["timestamps_epoch"] - row.time) <= time_off)
)[
0
]  # Keep variable values within time offset
if len(time_index) == 0:
pbar.update(1)
temp_variable[i] = np.nan
continue
mask = np.full(
(var[0].shape), False
)  # Create mask for longitude latitude limits
for k, latitude in enumerate(self.stamps["latitude"]):
mask[
k,
np.where(
(
haversine(
latitude, row.lat, self.stamps["longitude"], row.lon
)
<= radius[i]
)
)[0],
] = True  # Keep elements within desired distance
if not mask.any():
pbar.update(1)
temp_variable[i] = np.nan
continue
temp_variable[i] = np.nanmean(np.nanmean(var[time_index], axis=0)[mask])
pbar.update(1)
variable = f"cylinder_{time_off}_{r}_{variable}"
self.df[variable] = temp_variable
self.df[f"cylinder_{time_off}_{r}_era"] = np.sqrt(
self.df[f"cylinder_{time_off}_{r}_u10"] ** 2
+ self.df[f"cylinder_{time_off}_{r}_v10"] ** 2
)
del var'''

'''


def distance_to_shore(self):
"""
Function that computes the distance between the hydrophone and the closest shore.
Precision is 0.04°, data is from NASA
Make sure to call format_gps before running method
"""
global dist2shore_ds
print("Loading distance to shore data...")
if self.local == True:
try:
path = read_config(resources.files("OSmOSE").joinpath("config.toml"))
dist2shore_ds = np.loadtxt(path.Auxiliary.shore_dist).astype("float16")
except FileNotFoundError:
print("Please modify path to dist2coast.txt file in .aux_path.txt")
return None
else:
dist2shore_ds = np.loadtxt(
"/home/datawork-osmose/dataset/auxiliary/dist2coast.txt"
).astype("float16")
if len(np.unique(self.df.lat)) == 1:
shore_distance = nearest_shore(self.longitude[:1], self.latitude[:1])
self.shore_distance = np.tile(shore_distance, len(self.timestamps)).astype(
"float16"
)
self.df["shore_dist"] = self.shore_distance
else:
# shore_distance =  np.full([len(self.timestamps)], np.nan)
# shore_distance[(~np.isnan(self.longitude)) | (~np.isnan(self.latitude))] = nearest_shore(self.longitude, self.latitude, 1)
shore_distance = nearest_shore(self.longitude, self.latitude)
self.shore_distance = shore_distance.astype("float16")
self.df["shore_dist"] = self.shore_distance

def wind_fetch(self):
"""
Method that computes wind fetch.
Algorithm is optimized to use closest shore distance to reduce computation time
"""
assert "u10" in [
elem[-3:] for elem in self.df.columns
], "To load wind fetch please load wind direction first"
print("Computing Wind Fetch")
u10_col = self.df.columns[
np.where(
(
np.array("-".join(elem[-3:] for elem in self.df.columns).split("-"))
== "u10"
)
)[0]
]
v10_col = self.df.columns[
np.where(
(
np.array("-".join(elem[-3:] for elem in self.df.columns).split("-"))
== "v10"
)
)[0]
]
total_wind_fetch = np.zeros(len(self.timestamps))

shore_distance = self.shore_distance
dir_u, dir_v = self.df[u10_col].to_numpy().reshape(
shore_distance.shape
), self.df[v10_col].to_numpy().reshape(shore_distance.shape)
dlon, dlat = get_wind_fetch(shore_distance, dir_u, dir_v)
lon, lat = mod_pos(self.longitude - dlon, "lon"), mod_pos(
self.latitude - dlat, "lat"
)
total_wind_fetch += shore_distance
shore_distance = nearest_shore(lon, lat)
counter = 0
with tqdm(total=50, position=0, leave=True) as pbar:
while (np.any(total_wind_fetch[shore_distance > 5] < 1600)) and (
counter < 50
):
dlon, dlat = get_wind_fetch(
shore_distance[(shore_distance > 5) & (total_wind_fetch < 1600)],
dir_u[(shore_distance > 5) & (total_wind_fetch < 1600)],
dir_v[(shore_distance > 5) & (total_wind_fetch < 1600)],
)
lon[(shore_distance > 5) & (total_wind_fetch < 1600)] = mod_pos(
lon[(shore_distance > 5) & (total_wind_fetch < 1600)] - dlon, "lon"
)
lat[(shore_distance > 5) & (total_wind_fetch < 1600)] = mod_pos(
lat[(shore_distance > 5) & (total_wind_fetch < 1600)] - dlat, "lat"
)
total_wind_fetch[
(shore_distance > 5) & (total_wind_fetch < 1600)
] += shore_distance[(shore_distance > 5) & (total_wind_fetch < 1600)]
shore_distance = nearest_shore(lon, lat)
counter += 1
pbar.update(1)
if counter != 50:
print("\nDone computing wind_fetch before the 50 iterations")
else:
print("\nWind fetch stopped computing at 50 iteration")
total_wind_fetch[total_wind_fetch > 1600] = 1600
self.wind_fetch = total_wind_fetch
self.df["wind_fetch"] = self.wind_fetch


def bathymetry(self):
if self.local == True:
try:
path = read_config(resources.files("OSmOSE").joinpath("config.toml"))
bathymetry_ds = nc.Dataset(path.Auxiliary.bathymetry)
except FileNotFoundError:
print(
"Please modify path to GEBCO_2022_sub_ice_topo.nc file in .aux_path.txt"
)
return None
else:
bathymetry_ds = nc.Dataset(
"/home/datawork-osmose/dataset/auxiliary/GEBCO_2022_sub_ice_topo.nc"
)
temp_lat, temp_lon = self.latitude[~np.isnan(self.latitude)].to_numpy().astype(
"float16"
), self.longitude[~np.isnan(self.longitude)].to_numpy().astype("float16")
pos_lat, pos_dist = j.nearest_point(
temp_lat, bathymetry_ds["lat"][:].astype("float16")
)
pos_lon, pos_dist = j.nearest_point(
temp_lon, bathymetry_ds["lon"][:].astype("float16")
)
bathymetry = np.full(len(self.df), np.nan)
bathymetry[~np.isnan(self.latitude)] = [
bathymetry_ds["elevation"][i, j] for i, j in zip(pos_lat, pos_lon)
]
self.df["bathy"] = bathymetry
del bathymetry_ds

'''

