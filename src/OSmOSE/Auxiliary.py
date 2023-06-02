from OSmOSE.Dataset import Dataset
from pathlib import Path
import numpy as np
import pandas as pd
import os, glob, sys
import scipy.signal as sg
from OSmOSE import jointure as j
from importlib import resources
import soundfile as sf
import grp
from tqdm import tqdm
import netCDF4 as nc
from OSmOSE import func_api
import calendar, time, datetime
from bisect import bisect_left
from OSmOSE.utils import read_config

def haversine(lat1, lat2, lon1, lon2):
	lat1 = np.array(lat1).astype('float64')
	lat2 = np.array(lat2).astype('float64')
	lon1 = np.array(lon1).astype('float64')
	lon2 = np.array(lon2).astype('float64')
	return 2*6371*np.arcsin(np.sqrt(np.sin((np.pi/180)*(lat2-lat1)/2)**2+np.cos((np.pi/180)*lat1)*np.cos((np.pi/180)*lat2)*np.sin((np.pi/180)*(lon2-lon1)/2)**2))

def get_nc_time(x) :
	try :
		return calendar.timegm(time.strptime(''.join(x.astype(str)), '%Y-%m-%dT%H:%M:%SZ'))
	except ValueError :
		return np.nan
	
def take_closest(array, elem):
    """
    Assumes array is sorted. Returns closest value to elem.
    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(array, elem)
    if pos == 0:
        return 0
    if pos == len(array):
        return len(array)-1
    return pos-1

def get_wind_fetch(dist, dir_u, dir_v):
	'''
	scale used for wind fetch is 0.04° = 5 km
	This ensures that shore distance in km yields a distance in ° that is less than the true distance.
	'''
	angle = np.arctan(dir_v/dir_u).astype('float16')
	return 0.04 / 5 * dist.astype('float16') * np.cos(angle), 0.04 / 5 * dist.astype('float16') * np.sin(angle)

def mod_pos(x, name):
	if name == 'lat':
		return (x+90)%180-90
	if name == 'lon':
		return (x+180)%360-180
            
def nearest_shore(x,y):
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

def numerical_gps(df):
	if len(df) == 6:
		return calendar.timegm(datetime.datetime(int(df['year']), int(df['month']), int(df['day']), int(df['hour']), int(df['minute']), int(df['second'])).timetuple())
	return calendar.timegm(datetime.datetime(int(df['year']), int(df['month']), int(df['day']), int(df['hour'])).timetuple())

def numerical_time(date):
	try:
		return calendar.timegm(time.strptime(str(date)[:-5], '%Y-%m-%dT%H:%M:%S'))
	except ValueError:
		return calendar.timegm(time.strptime(str(date)[:-4], '%Y%m%d%H'))

def window_rms(sig, window_size=512):
	sig_sqr = np.power(sig, 2)
	window = np.ones(window_size)/float(window_size)
	return np.sqrt(np.convolve(sig_sqr, window, 'same'))

def norm(sig):
	return (sig - np.nanmean(sig))/(np.nanstd(sig) + 1e-10)

def norm_uneven(sig1, sig2):
	concatenation = np.concatenate((sig1.flatten(), sig2.flatten()))
	mean, std = np.nanmean(concatenation), np.nanstd(concatenation)
	return (sig1 - mean)/(std + 1e-10), (sig2 - mean)/(std + 1e-10)

def inter_spl(spl, scale, freq):
	idx_low = len(spl[scale <= freq])
	idx_high = idx_low+1
	return (spl[idx_high] - spl[idx_low])/(scale[idx_high]-scale[idx_low])*(freq - scale[idx_low]) + spl[idx_low]

def download_era(ftime, longitude, latitude, era_path, filename) :
	print('\n DOWNLOADING ERA 5 DATA... \n')
	start_date, end_date = ftime[~np.isnan(ftime)][0], ftime[~np.isnan(ftime)][-1]
	start_date, end_date = time.gmtime(start_date), time.gmtime(end_date)
	udi = '165042'
	key = '868fa005-b5dc-4471-a8c2-862ac1bdd509'

	func_api.make_cds_file(key, udi,era_path)
	north_boundary, south_boundary = np.ceil(max(latitude[~np.isnan(latitude)])*4)/4+0.5, np.floor(min(latitude[~np.isnan(latitude)])*4)/4-0.5
	east_boundary, west_boundary = np.ceil(max(longitude[~np.isnan(longitude)])*4)/4+0.5, np.floor(min(longitude[~np.isnan(longitude)])*4)/4-0.5
	years = list(np.array(list(range(start_date.tm_year, end_date.tm_year+1))).astype(str))
	if start_date.tm_mon < end_date.tm_mon :
		months = list(np.array(list(range(start_date.tm_mon, end_date.tm_mon+1))).astype(str))
	else : 
		months = list(np.array(list(range(start_date.tm_mon, 13))).astype(str)) + list(np.array(list(range(1, end_date.tm_mon+1))).astype(str))
	if len(months) == 0:
		months = 'all'
	months = ['0' + month if len(month) == 1 else month for month in months]
	days = ['01', '02', '03','04', '05', '06','07', '08', '09','10', '11', '12','13', '14', '15','16', '17', '18', '19', '20', '21','22', '23', '24','25', '26', '27','28', '29', '30','31']
	hours = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00','09:00', '10:00', '11:00', '12:00', '13:00', '14:00','15:00', '16:00', '17:00','18:00', '19:00', '20:00','21:00', '22:00', '23:00']
	full_variables = ['10m_u_component_of_wind', '10m_v_component_of_wind', 'sea_surface_temperature', 'total_precipitation', 'total_cloud_cover', 'model_bathymetry']
	return func_api.final_creation(os.getcwd()+"/"+filename, filename, key, full_variables, years, months, days, hours, [north_boundary, west_boundary, south_boundary, east_boundary])	

get_cfosat_time = lambda x : calendar.timegm(time.strptime(x, '%Y%m%dT%H%M%S'))
get_datarmor_time = lambda x : calendar.timegm(time.strptime(x, '%Y-%m-%dT%H:%M:%S.000Z'))
get_era_time = lambda x : calendar.timegm(x.timetuple()) if isinstance(x, datetime.datetime) else x



class Variables():
	'''
	This class loads, formats and computes all available environmental data from a campaign.
	The class requires a path to the directory containing the campaign's WAV files, metadata.csv file, and timestamp.csv file.
	An optional path to a directory containing the campaign's GPS data is also accepted in case of a moving hydrophone.

	Attributes
	----------
    mpath : str, optional
        Path pointing to metadata.csv file
    tpath : str, optional
        Path pointing to dataframe containing time stamps at which to compute auxiliary data
    df : dataframe, optional
        Dataframe containing 'time' column with desired timestamps at which aux data will be computed
	'''
	
	def __init__(self, path, dataset, local = True):
		'''
		Initializes the Variables object.
		Parameters
		----------
		metadata_path : str
		Path to metadata.csv file.
		time_path : str
		Path to dataframe containing timestamps at which to compute auxiliary data.
		df : dataframe, optional
		Dataframe containing 'time' column with desired timestamps at which aux data will be computed. 
		Default is an empty dataframe.
		'''
		self.path = os.path.join(path, dataset)
		self.dataset = dataset
		if os.path.exists(os.path.join(self.path, 'data', 'auxiliary', 'instrument', 'gps_data.csv')):
			self.df = pd.read_csv(os.path.join(self.path, 'data', 'auxiliary', 'instrument', 'gps_data.csv'))[['time', 'depth', 'lat', 'lon']] 
			self.latitude, self.longitude = self.df['lat'], self.df['lon']
			self.timestamps = self.df['time']
			self.depth = self.df['depth']
		else :
			self.from_scratch()
		self.local = local

	def distance_to_shore(self):
		'''
		Function that computes the distance between the hydrophone and the closest shore.
		Precision is 0.04°, data is from NASA
		Make sure to call format_gps before running method
		'''
		global dist2shore_ds
		print('Loading distance to shore data...')
		if self.local == True :
			try :
				path = read_config(resources.files("OSmOSE").joinpath('config.toml'))
				dist2shore_ds = np.loadtxt(path.Auxiliary.shore_dist).astype('float16')
			except FileNotFoundError :
				print('Please modify path to dist2coast.txt file in .aux_path.txt')
				return None
		else :
			dist2shore_ds = np.loadtxt('/home/datawork-osmose/dataset/auxiliary/dist2coast.txt').astype('float16')
		if len(np.unique(self.df.lat)) == 1:
			shore_distance = nearest_shore(self.longitude[:1], self.latitude[:1])
			self.shore_distance = np.tile(shore_distance, len(self.timestamps)).astype('float16')
			self.df['shore_dist'] = self.shore_distance
		else :
			#shore_distance =  np.full([len(self.timestamps)], np.nan)
			#shore_distance[(~np.isnan(self.longitude)) | (~np.isnan(self.latitude))] = nearest_shore(self.longitude, self.latitude, 1)
			shore_distance = nearest_shore(self.longitude, self.latitude)
			self.shore_distance = shore_distance.astype('float16')
			self.df['shore_dist'] = self.shore_distance


	def wind_fetch(self):
		'''
		Method that computes wind fetch.
		Algorithm is optimized to use closest shore distance to reduce computation time
		'''
		assert 'u10' in [elem[-3:] for elem in self.df.columns], "To load wind fetch please load wind direction first"				
		print('Computing Wind Fetch')
		u10_col = self.df.columns[np.where((np.array('-'.join(elem[-3:] for elem in self.df.columns).split('-')) == 'u10'))[0]]
		v10_col = self.df.columns[np.where((np.array('-'.join(elem[-3:] for elem in self.df.columns).split('-')) == 'v10'))[0]]
		total_wind_fetch = np.zeros(len(self.timestamps))
		shore_distance = self.shore_distance
		dir_u, dir_v = self.df[u10_col].to_numpy().reshape(shore_distance.shape), self.df[v10_col].to_numpy().reshape(shore_distance.shape)
		dlon, dlat = get_wind_fetch(shore_distance, dir_u, dir_v)
		lon, lat = mod_pos(self.longitude-dlon, 'lon'), mod_pos(self.latitude-dlat, 'lat')
		total_wind_fetch += shore_distance
		shore_distance = nearest_shore(lon, lat)
		counter = 0
		with tqdm(total = 50, position = 0, leave = True) as pbar:
			while (np.any(total_wind_fetch[shore_distance > 5]<1600)) and (counter < 50):
				dlon, dlat = get_wind_fetch(shore_distance[(shore_distance > 5) & (total_wind_fetch < 1600)], dir_u[(shore_distance > 5) & (total_wind_fetch < 1600)], dir_v[(shore_distance > 5) & (total_wind_fetch < 1600)])
				lon[(shore_distance > 5) & (total_wind_fetch < 1600)] = mod_pos(lon[(shore_distance > 5) & (total_wind_fetch < 1600)] - dlon, 'lon')
				lat[(shore_distance > 5) & (total_wind_fetch < 1600)] = mod_pos(lat[(shore_distance > 5) & (total_wind_fetch < 1600)] - dlat, 'lat')
				total_wind_fetch[(shore_distance > 5) & (total_wind_fetch < 1600)] += shore_distance[(shore_distance > 5) & (total_wind_fetch < 1600)]
				shore_distance = nearest_shore(lon, lat)
				counter+=1
				pbar.update(1)
		if counter != 50:
			print('\nDone computing wind_fetch before the 50 iterations')
		else :
			print('\nWind fetch stopped computing at 50 iteration')
		total_wind_fetch[total_wind_fetch>1600] = 1600
		self.wind_fetch = total_wind_fetch
		self.df['wind_fetch'] = self.wind_fetch
		
	def bathymetry(self):
		if self.local == True :
			try :
				path = read_config(resources.files("OSmOSE").joinpath('config.toml'))
				bathymetry_ds = nc.Dataset(path.Auxiliary.bathymetry)
			except FileNotFoundError :
				print('Please modify path to GEBCO_2022_sub_ice_topo.nc file in .aux_path.txt')
				return None
		else :
			bathymetry_ds = nc.Dataset('/home/datawork-osmose/dataset/auxiliary/GEBCO_2022_sub_ice_topo.nc')
		temp_lat, temp_lon = self.latitude[~np.isnan(self.latitude)].to_numpy().astype('float16'), self.longitude[~np.isnan(self.longitude)].to_numpy().astype('float16')
		pos_lat, pos_dist = j.nearest_point(temp_lat, bathymetry_ds['lat'][:].astype('float16'))
		pos_lon, pos_dist = j.nearest_point(temp_lon, bathymetry_ds['lon'][:].astype('float16'))
		bathymetry = np.full(len(self.df), np.nan)
		bathymetry[~np.isnan(self.latitude)] = [bathymetry_ds['elevation'][i,j] for i,j in zip(pos_lat, pos_lon)] 
		self.df['bathy'] = bathymetry
		del bathymetry_ds

	def from_scratch(self, gps = None):
		'''
		Takes dictionary with lat, lon, depth of hydrophone and creates empty dataframe for class
		Objective is to build time with timestamps
		'''
		try : 
			dataset = Dataset(self.path, gps_coordinates=gps)
			metadata = pd.read_csv(dataset._get_original_after_build().joinpath("metadata.csv"))
		except FileNotFoundError:
			print('Could not find built dataset.')
			sys.exit()
		start_date, end_date = get_datarmor_time(metadata.start_date[0]), get_datarmor_time(metadata.end_date[0])
		self.timestamps = np.arange(start_date, end_date, 600)
		self.latitude, self.longitude = [metadata.lat[0]]*len(self.timestamps), [metadata.lon[0]]*len(self.timestamps)
		self.df = pd.DataFrame({'time': self.timestamps, 'lat':self.latitude, 'lon':self.longitude, 'depth':float('nan')}, index = [0])

	def join_auxiliary(self):
		print('\nJoining bathymetry data...')
		self.bathymetry()
		print('\nJoining shore distance data...')
		self.distance_to_shore()
		if 'era' in [elem[-3:] for elem in self.df.columns]:
			print('\nJoining wind fetch data...')
			self.wind_fetch()




class Weather(Variables):
	
	def __init__(self, path, dataset, local = True):
		super().__init__(path, dataset, local)		

	def join_era(self, *, method = 'interpolation', time_off = np.inf, lat_off = np.inf, lon_off = np.inf, r = np.inf, variables = ['u10', 'v10']):
		print(f'Joining ERA5 data using the {method} method.')
		if method == 'nearest':
			print(f'Offsets are :\n   time offset = {time_off}\n   latitude offset = {lat_off}\n   longitude offset = {lon_off}, \n   variables = {variables}')
			self.nearest_era(time_off=time_off, lat_off=lat_off, lon_off=lon_off, variables=variables)
		if method == 'interpolation':
			self.interpolation_era()
		if method == 'cube':
			print(f'Offsets are :\n   time offset = {time_off}\n   latitude offset = {lat_off}\n   longitude offset = {lon_off}, \n   variables = {variables}')
			self.cube_era(time_off=time_off, lat_off=lat_off, lon_off=lon_off, variables=variables)
		if method == 'cylinder':
			print(f'Offsets are :\n   time offset = {time_off}\n   radius = {r}, \n   variables = {variables}')
			self.cylinder_era(time_off=time_off, r=r, variables=variables)


	def load_era(self, method = 'other'):
		'''
		Method that uploads ERA5 data into a dataframe.
		Temporal and spatial interpolation or nearest point method is used to compute the single level at the desired longitude/latitude/time
		The upload method cannot be used on datarmor.
		Use function download_era in order to download desired data.
        Parameters
        ----------
        era_path : str
            Path to the directory containing the ERA data.
		filename : str
			Filename given to downloaded ERA data when using the Jupyter Notebook
		method : str
			{'interpolation', 'nearest'}. Method to use in order to join ERA data to hydrophone's position (defaults is 'interpolation'').
		time_off : float
			For nearest method, maximum time offset in seconds above which no ERA data is joined (default is np.inf).
		lat_off : float
			For nearest method, maximum latitude offset in degrees above which no ERA data is joined (default is np.inf).
		lon_off : float
			For nearest method, maximum longitude offset in degrees above which no ERA data is joined (default is np.inf).
        '''

		if os.path.exists(os.path.join(self.path, 'data', 'auxiliary', 'weather', 'era')):
			self.era_path = os.path.join(self.path, 'data', 'auxiliary', 'weather', 'era')
			downloaded_cds = os.path.join(self.era_path, self.dataset+'.nc')
			fh = nc.Dataset(downloaded_cds, mode='r')
			self.variables = list(fh.variables.keys())[3:]
		else :
			if not self.local :
				print("Please download ERA on local machine and upload it on Datarmor. \nSystem exit.")
				sys.exit("No ERA5 data")
			print('Trying to download ERA5 data...')
			self.variables = download_era(self.timestamps.to_numpy(), self.longitude, self.latitude, era_path, filename)

		if method == 'cube' or method == 'cylinder':
			self.stamps = np.load(os.path.join(self.era_path, 'stamps.npz'), allow_pickle = True)
		else :
			temp_stamps = np.load(os.path.join(self.era_path, 'stamps.npz'), allow_pickle=True)
			dates, lat, lon = temp_stamps['timestamps'], temp_stamps['latitude'], temp_stamps['longitude']
			stamps = np.array([[date, lat_val, lon_val] for date in dates for lat_val in lat for lon_val in lon])
			self.stamps = stamps.reshape(len(dates), len(lat), len(lon), 3)
		del temp_stamps, downloaded_cds, stamps

	def nearest_era(self, *, time_off = 600, lat_off = 0.5, lon_off = 0.5, variables = ['u10','v10']):
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
		self.load_era(method = 'nearest')
		data = self.df[['time', 'lat', 'lon']].dropna().to_numpy()
		self.stamps[:,:,:,0] = np.array(list(map(get_era_time, self.stamps[:,:,:,0].ravel()))).reshape(self.stamps[:,:,:,0].shape)
		pos_ind, pos_dist = j.nearest_point(data, self.stamps.reshape(-1, 3))
		if variables == 'None':
			variables = self.variables
		pbar = tqdm(total = len(variables), position = 0, leave = True)
		for variable in variables:
			pbar.update(1)
			pbar.set_description("Loading and formatting %s" % variable)
			temp_variable = np.full([len(self.timestamps)], np.nan)
			var = np.load(os.path.join(self.era_path, variable+'_'+self.dataset+'.npy'))
			var = var.flatten()[pos_ind]
			mask = np.any(abs(data - self.stamps.reshape(-1,3)[pos_ind]) > [time_off, lat_off, lon_off], axis = 1)
			var[mask] = np.nan
			temp_variable[self.df[['time', 'lat', 'lon']].dropna().index] = var
			variable = f'np_{time_off}_{lat_off}_{lon_off}_{variable}'
			self.df[variable] = temp_variable
		self.df[f'np_{time_off}_{lat_off}_{lon_off}_era'] = np.sqrt(self.df[f'np_{time_off}_{lat_off}_{lon_off}_u10']**2 + self.df[f'np_{time_off}_{lat_off}_{lon_off}_v10']**2)
		del var
	
	def interpolation_era(self):
		"""Computes interpolated variables values from ERA5 mesh points.

		ERA5 is suited for a scipy's 3D grid interpolation in time, latitude and longitude.
		As this method is quick, it is computed for all available variables.

		Parameter
		---------
		None.

		Returns
		-------
		None. Results are saved in self.df as 'interp_{variable}'
		"""
		self.load_era(method = 'interpolation')
		temp_lat = self.latitude[(~np.isnan(self.longitude)) | (~np.isnan(self.latitude))]    #Remove nan values from computation
		temp_lon = self.longitude[(~np.isnan(self.longitude)) | (~np.isnan(self.latitude))]
		temp_timestamps = self.timestamps[(~np.isnan(self.longitude)) | (~np.isnan(self.latitude))]
		pbar = tqdm(total = len(self.variables), position = 0, leave = True)
		for variable in self.variables:
			pbar.update(1)
			pbar.set_description("Loading and formatting %s" % variable)
			var = np.load(os.path.join(self.era_path, variable+'_'+self.dataset+'.npy'))
			interp = j.rect_interpolation_era(self.stamps, var)
			temp_variable =  np.full([len(self.timestamps)], np.nan)
			temp_variable[(~np.isnan(self.longitude)) | (~np.isnan(self.latitude))] = j.apply_interp(interp, temp_timestamps, temp_lat, temp_lon)
			variable = 'interp_'+variable
			self.df[variable] = temp_variable
		self.df['interp_era'] = np.sqrt(self.df.interp_u10**2 + self.df.interp_v10**2)
		del var

	def cube_era(self, *, time_off = 600, lat_off = 0.5, lon_off = 0.5, variables = ['u10', 'v10']):
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
		self.load_era(method = 'cube')
		temp_df = self.df[['time', 'lat', 'lon']]
		if variables == None:
			variables = self.variables
		for variable in  variables:
			temp_variable =  np.full([len(self.timestamps)], np.nan)
			var = np.load(os.path.join(self.era_path, variable+'_'+self.dataset+'.npy'))
			pbar = tqdm(total = len(temp_df), position = 0, leave = True)
			for i, row in temp_df.iterrows():
				if row.isnull().any():      #Skip iteration if there is a nan value
					continue
				time_index = np.where((abs(self.stamps['timestamps_epoch'] - row.time) <= time_off))[0]    #Get timestamps satisfyin time offset
				if len(time_index) == 0:    #Skip iteration if no weather source is within time offset for this timestep
					pbar.update(1)
					temp_variable[i] = np.nan
					continue
				lat_index = np.where((abs(self.stamps['latitude'] - row.lat) <= lat_off))[0]    #Get latitude indexes satisfyin lon offset
				if len(lat_index) == 0:     #Skip iteration if no weather source is within lat offset for this timestep
					pbar.update(1)
					temp_variable[i] = np.nan
					continue
				lon_index = np.where((abs(self.stamps['longitude'] - row.lon) <= lon_off))[0]   #Get longitude indexes satisfying lon offset
				if len(lon_index) == 0:     #Skip iteration if no weather source is within lon offset for this timestep
					pbar.update(1)
					temp_variable[i] = np.nan
					continue
				temp_value = []
				for time_ind in time_index:
					for lat_ind in lat_index:
						for lon_ind in lon_index :
							temp_value.append(var[time_ind, lat_ind, lon_ind])   #Saving values satisfying the conditions
				temp_variable[i] = np.nanmean(temp_value)
				pbar.update(1)
			variable = f'cube_{time_off}_{lat_off}_{lon_off}_{variable}'
			self.df[variable] = temp_variable
		self.df[f'cube_{time_off}_{lat_off}_{lon_off}_era'] = np.sqrt(self.df[f'cube_{time_off}_{lat_off}_{lon_off}_u10']**2+self.df[f'cube_{time_off}_{lat_off}_{lon_off}_v10']**2)
		del var
		
	def cylinder_era(self, *, time_off = 600, r = 'depth', variables = ['u10', 'v10']):
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
		self.load_era(method = 'cylinder')
		temp_df = self.df[['time', 'lat', 'lon']]
		if isinstance(r, int) or isinstance(r, float):
			radius = [r]*len(temp_df)
		else :
			radius = 3*self.df.depth       #Compute radius as a function of hydrophone's depth
		if variables == None:
			variables = self.variables
		for variable in  variables:
			temp_variable =  np.full([len(self.timestamps)], np.nan)
			var = np.load(os.path.join(self.era_path,variable+'_'+self.dataset+'.npy'))
			pbar = tqdm(total = len(temp_df), position = 0, leave = True)
			for i, row in temp_df.iterrows():
				if row.isnull().any():      #Skip iteration if there is a nan value
					continue
				time_index = np.where((abs(self.stamps['timestamps_epoch'] - row.time) <= time_off))[0]  #Keep variable values within time offset
				if len(time_index) == 0:
					pbar.update(1)
					temp_variable[i] = np.nan
					continue
				mask = np.full((var[0].shape), False)   #Create mask for longitude latitude limits
				for k, latitude in enumerate(self.stamps['latitude']):
					mask[k, np.where((haversine(latitude, row.lat, self.stamps['longitude'], row.lon) <= radius[i]))[0]] = True   #Keep elements within desired distance
				if not mask.any():
					pbar.update(1)
					temp_variable[i] = np.nan
					continue
				temp_variable[i] = np.nanmean(np.nanmean(var[time_index], axis = 0)[mask])
				pbar.update(1)
			variable = f'cylinder_{time_off}_{r}_{variable}'
			self.df[variable] = temp_variable
		self.df[f'cylinder_{time_off}_{r}_era'] = np.sqrt(self.df[f'cylinder_{time_off}_{r}_u10']**2+self.df[f'cylinder_{time_off}_{r}_v10']**2)
		del var

def check_era(path, dataset):
	if os.path.isdir(os.path.join(path, dataset, 'data', 'auxiliary', 'weather', 'era')):
		print("Era is already downloaded, you're free to continue")
		return None
	else:
		os.makedirs(os.path.join(path, dataset, 'data', 'auxiliary', 'weather', 'era'))
		print(f"Please upload ERA data in {os.path.join(path, 'weather', 'era')} \nIf you have not downloaded ERA data yet see https://github.com/Project-OSmOSE/osmose-datarmor/tree/package/local_notebooks")
		return None
