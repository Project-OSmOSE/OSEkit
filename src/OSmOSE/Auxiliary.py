import cdsapi
import numpy as np
import pandas as pd
import os
import calendar
from glob import glob
from tqdm import tqdm
from pykdtree.kdtree import KDTree
import netCDF4 as nc
from typing import Union, Tuple, List
from datetime import datetime, date, timedelta
from OSmOSE.utils.timestamp_utils import check_epoch
from OSmOSE.config import *
from OSmOSE.Spectrogram import Spectrogram
from scipy import interpolate


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
		acoustic : bool = False,
		bathymetry : bool = True,
		distance_to_shore : bool = True,
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
		try :
			self.df = check_epoch(pd.read_csv(self.audio_path.joinpath('timestamp.csv')))
			#self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], format='%Y-%m-%dT%H:%M:%S.000000+0000')
			print(f"Current reference timestamp.csv has the following columns : {', '.join(self.df.columns)}")
		except FileNotFoundError :
			print('Dataset corresponding to analysis params was not found. Please call the build method first.\nParams are :')
			print('Dataset sampling rate : ', dataset_sr)
			print('Spectrogram duration : ', analysis_params['spectro_duration'])
			self.df = pd.DataFrame()
		self.acoustic, self.bathymetry, self.distance_to_shore = acoustic, bathymetry, distance_to_shore
		self.metadata = pd.read_csv(self._get_original_after_build().joinpath("metadata.csv"), header=0)
		self._depth, self._gps_coordinates = depth, gps_coordinates
		match depth :
			case True :
				try :
					self.depth = 'depth.csv'
				except FileNotFoundError: 
					print('depth file not found, defaulting to depth from metadata file')
					self.depth = self.metadata.depth.item()
			case int() | float() :
				self.depth = int(depth)
			case str() :
				self.depth = depth
			case _ :
				self.depth = self.metadata.depth.item()
		match gps_coordinates :
			case True :
				try:
					self.gps_coordinates = 'gps.csv'
				except FileNotFoundError:
					print('gps file not found, defaulting to gps data from metadata file')
					self.gps_coordinates = tuple(self.metadata[['lat','lon']].values[0])
			case list() | tuple() | str() :
				self.gps_coordinates = gps_coordinates
			case _:
				self.gps_coordinates = tuple(self.metadata[['lat','lon']].values[0])
		match era:
			case str() :
				self.era = era if os.path.isabs(era) else self.path.joinpath(OSMOSE_PATH.environment, era)
				assert os.path.isfile(self.era), 'ERA file was not found'
			case True:
				fns = glob(str(self.path.joinpath(OSMOSE_PATH.environment, '*nc')))
				assert len(fns) == 1, "Make sur there is one (and only one) nc file in your era folder"
				self.era = fns[0]		
			case _ :
				self.era = False
		self.other = other if other is not None else {}	
		if annotation : 
			self.other = {**self.other, **annotation}
		self.joined_path = self.path / OSMOSE_PATH.auxiliary / (self.audio_foldername + '.csv')

	
	def __str__(self):
		print(f'For the {self.name} dataset')
		elems = []
		if self._gps_coordinates :
			elems.append('gps')
		if self._depth :
			elems.append('depth')
		if self.era :
			elems.append('era')
		if self.acoustic : 
			elems.append('acoustic')
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

	def join_distance_to_shore(self):
		"""
		Function that computes the distance between the hydrophone and the closest shore.
		Precision is 0.04°, data is from NASA
		Make sure to call format_gps before running method
		"""
	
		dist2shore_ds = np.loadtxt("/home/datawork-osmose/dataset/auxiliary/dist2coast.txt").astype("float16")
		if len(np.unique(self.df.lat)) == 1:
			shore_distance = nearest_shore(dist2shore_ds, self.df.lat.iloc[:1], self.df.lon[:1])
			self.df['shore_distance'] = np.tile(shore_distance, len(self.df)).astype("float16")
		else:
			shore_distance = nearest_shore(dist2shore_ds, self.df.lon.to_numpy(), self.df.lat.to_numpy())
			shore_distance = shore_distance.astype("float16")
			self.df["shore_dist"] = shore_distance	


	def join_bathymetry(self):
		bathymetry_ds = nc.Dataset('/home/datawork-osmose/dataset/auxiliary/GEBCO_2022_sub_ice_topo.nc')
		latitude, longitude = self.df.lat.to_numpy(), self.df.lon.to_numpy()
		temp_lat, temp_lon = latitude[~np.isnan(latitude)], longitude[~np.isnan(longitude)]
		pos_lat, pos_dist = nearest_point(temp_lat, bathymetry_ds['lat'][:])
		pos_lon, pos_dist = nearest_point(temp_lon, bathymetry_ds['lon'][:])
		bathymetry = np.full(len(self.df), np.nan)
		bathymetry[~np.isnan(latitude)] = [bathymetry_ds['elevation'][i,j] for i,j in zip(pos_lat, pos_lon)] 
		self.df['bathy'] = bathymetry


	def join_acoustic(self, fcs = [], feature='welch'):
		'''
		This methods adds noise level at selected frequency to the joined dataframe
		Parameters :
			feature (str) : Type of processed features to fetch from : 'LTAS', 'spectrogram', 'welch'
		'''
		_noise_level, _time = [[] for fc in fcs], []
		full_band = []
		match feature :
			case 'welch':
				fns = glob(str(self.path_output_welch)+'/*')
				pbar = tqdm(fns)
				for fn in pbar :
					pbar.set_description(fn)
					_data = np.load(fn, allow_pickle = True)
					_time.extend(_data['Time'])
					for i, fc in enumerate(fcs) : 
						freq_ind = np.argmin(abs(_data['Freq']-fc))
						_noise_level[i].extend(np.log10(_data['Sxx'][:, freq_ind]))
					if 'full_band' not in self.df:
						full_band.extend(np.mean(np.log10(_data['Sxx']), axis = 1))
			case 'spectrogram' | 'LTAS':
				fns = glob(str(self.path_output_spectrogram_matrix)+'/*')
				pbar = tqdm(fns)
				for fn in pbar :
					pbar.set_description(fn)
					_data = np.load(fn, allow_pickle = True)
					_time.extend(pd.to_datetime(fn.split('/')[-1][:19], format = '%Y_%m_%dT%H_%M_%S'))
					for i, fc in enumerate(fcs) :
						freq_ind = np.argmin(abs(_data['Freq']-fc))
						_noise_level[i].extend(np.mean(_data['Sxx'][:, freq_ind]))
					if 'full_band' not in self.df :
						full_band.extend(np.mean(_data['Sxx'], axis = 1))
		if 'full_band' not in self.df :
			_temp = pd.DataFrame().from_dict({**{'timestamp':_time}, **{'full_band':full_band}, **{fcs[i] : _noise_level[i] for i in range(len(fcs))}}).sort_values('timestamp')
		else : 
			_temp = pd.DataFrame().from_dict({**{'timestamp':_time}, **{fcs[i] : _noise_level[i] for i in range(len(fcs))}}).sort_values('timestamp')
		_temp = check_epoch(_temp)
		self.df = pd.merge(self.df, _temp, on='epoch')


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
			_csv = check_epoch(pd.read_csv(self.other[key]))
			self.df[key] = interpolate.interp1d(_csv.epoch, _csv[key], bounds_error=False)(self.df.epoch)


	def interpolation_era(self):
		"""Computes interpolated variables values from ERA5 mesh points.
		ERA5 is suited for a scipy's 3D grid interpolation in time, latitude and longitude.
		As this method is quick, it is computed for all available variables.
		Results are saved in self.df as 'interp_{variable}'
		"""
		ds = nc.Dataset(self.era)
		variables = list(ds.variables.keys())[3:]
		era_time = pd.DataFrame(ds.variables['time'][:].data)
		era_datetime = era_time[0].apply(lambda x : datetime(1900,1,1)+timedelta(hours=int(x)))
		timestamps = era_datetime.apply(lambda x : x.timestamp()).to_numpy()
		
		pbar = tqdm(total=len(variables), position=0, leave=True)
		for variable in variables:
			pbar.update(1); pbar.set_description("Loading and formatting %s" % variable)
			self.df[variable] = interpolate.RegularGridInterpolator((timestamps, ds['latitude'][:], ds['longitude'][:]), ds[variable][:], bounds_error = False)((self.df.epoch, self.df.lat, self.df.lon))
		if 'u10' and 'v10' in self.df :
			self.df['era'] = np.sqrt(self.df.u10**2 + self.df.v10**2)
	
	def automatic_join(self):
		''' Automatically join all the available data'''
		if self._depth :
			self.join_depth()
		if self._gps_coordinates :
			self.join_gps()
		if self.bathymetry :
			self.join_bathymetry()
		if self.distance_to_shore:
			self.join_distance_to_shore()
		if self.era:
			self.interpolation_era()
		if self.acoustic : 
			self.join_acoustic()
		if self.other :
			self.join_other()


	def save_file(self, filename = None) :
		if filename : 
			self.df.to_csv(self.path.joinpath(OSMOSE_PATH.auxiliary, filename), index = None)
			self.joined_path = self.path.joinpath(OSMOSE_PATH.auxiliary, filename)
		else :
			if os.path.exists(self.joined_path):
				_existing = pd.read_csv(self.joined_path)
				try :
					_temp = pd.concat((_existing, self.df[self.df.columns.to_numpy()[[str(col) not in _existing.columns.to_numpy() for col in self.df.columns.to_numpy()]]]), axis = 1)
					_temp.to_csv(self.joined_path, index = None)
				except :
					print(f"Joined data corresponding to {self.audio_foldername} already exists. \nPlease enter filename to save data.")
			else :
				self.df.to_csv(self.joined_path, index = None)

def nearest_shore(dist2shore_ds, x,y):
	shore_distance = np.full([len(x)], np.nan)
	x, y = np.round(x*100//4*4/100+0.02, 2), np.round(y*100//4*4/100+0.02, 2)
	_x, _y = x[(~np.isnan(x)) | (~np.isnan(y))], y[(~np.isnan(x)) | (~np.isnan(y))]
	lat_ind = np.rint(-9000/0.04*(_y)+20245500).astype(int)
	lat_ind2 = np.rint(-9000/0.04*(_y-0.04)+20245500).astype(int)
	lon_ind = (_x/0.04+4499.5).astype(int)
	sub_dist2shore = np.stack([dist2shore_ds[ind1 : ind2]  for ind1, ind2 in zip(lat_ind, lat_ind2)])
	shore_temp = np.array([sub_dist2shore[i, lon_ind[i], -1] for i in range(len(lon_ind))])
	shore_distance[(~np.isnan(x)) | (~np.isnan(y))] = shore_temp
	return shore_distance

def nearest_point(data, var):
	tree = KDTree(var)
	neighbor_dists, neighbor_indices = tree.query(data)
	return neighbor_indices, neighbor_dists

def make_cds_file(key, udi, path):
    os.chdir(os.path.expanduser("~"))
    try:
        os.remove(".cdsapirc")
    except FileNotFoundError:
        pass

    cmd1 = "echo url: https://cds.climate.copernicus.eu/api/v2 >> .cdsapirc"
    cmd2 = "echo key: {}:{} >> .cdsapirc".format(udi, key)
    os.system(cmd1)
    os.system(cmd2)

    if path == None:
        try:
            os.mkdir("api")
        except FileExistsError:
            pass
        path_to_api = os.path.join(os.path.expanduser("~"), "api/")
    else:
        try:
            os.mkdir(os.path.join(path + "api"))
        except FileExistsError:
            pass
        path_to_api = os.path.join(path, "api")

    os.chdir(path_to_api)
    os.getcwd()


def return_cdsapi(filename, key, variables, years, months, days, hours, area):
    print("You have selected : \n")
    sel = [print(variables) for data in variables]
    print("\nfor the following times")
    print(f"Years : {years} \n Months : {months} \n Days : {days} \n Hours : {hours}")

    print(
        "\nYour boundaries are : North {}°, South {}°, East {}°, West {}°".format(
            area[0], area[2], area[3], area[1]
        )
    )

    filename = filename + ".nc"

    c = cdsapi.Client()

    if days == "all":
        days = [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
        ]
    if months == "all":
        months = [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ]
    if hours == "all":
        hours = [
            "00:00",
            "01:00",
            "02:00",
            "03:00",
            "04:00",
            "05:00",
            "06:00",
            "07:00",
            "08:00",
            "09:00",
            "10:00",
            "11:00",
            "12:00",
            "13:00",
            "14:00",
            "15:00",
            "16:00",
            "17:00",
            "18:00",
            "19:00",
            "20:00",
            "21:00",
            "22:00",
            "23:00",
        ]

    r = c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": variables,
            "year": years,
            "month": months,
            "day": days,
            "time": hours,
            "area": area,
            "format": "netcdf",
            "grid": [0.25, 0.25],
        },
        filename,
    )
    r.download(filename)
