from Dataset import Dataset
import numpy as np
import pandas as pd
import os, glob, sys
import scipy.signal as sg
from OSmOSE import jointure as j
import soundfile as sf
from tqdm import tqdm
from netCDF4 import Dataset
from OSmOSE import func_api
import calendar, time, datetime
from bisect import bisect_left

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
	angle = np.arctan(dir_v/dir_u)
	return 0.04 / 5 * dist * np.cos(angle), 0.04 / 5 * dist * np.sin(angle)

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
	print('I made it here')
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
	
	def __init__(self, path, local = True):
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
		if os.path.exists(os.path.join(self.path, 'data', 'auxiliary', 'aux_data.csv')):
			self.df = pd.read_csv(os.path.join(self.path, 'data', 'auxiliary', 'aux_data.csv'))[['time', 'depth', 'lat', 'lon']] 
			self.latitude, self.longitude = self.df['lat'], self.df['lon']
			self.timestamps = self.df['time']
			self.depth = self.df['depth']
		self.local = local

	def shore_distance(self):
		'''
		Function that computes the distance between the hydrophone and the closest shore.
		Precision is 0.04°, data is from NASA
		Make sure to call format_gps before running method
		'''
		global dist2shore_ds
		print('\nLoading distance to shore data...')
		dist2shore_ds = np.loadtxt('/home/datawork-osmose/dataset/auxiliary/dist2coast.txt').astype('float32')
		if len(np.unique(self.df.lat)) == 1:
			shore_distance = nearest_shore(self.longitude[:1], self.latitude[:1])
			self.shore_distance = np.tile(shore_distance, len(self.ftime))
			self.df['shore_dist'] = self.shore_distance
		else :
			#shore_distance =  np.full([len(self.ftime)], np.nan)
			#shore_distance[(~np.isnan(self.longitude)) | (~np.isnan(self.latitude))] = nearest_shore(self.longitude, self.latitude, 1)
			shore_distance = nearest_shore(self.longitude, self.latitude)
			self.shore_distance = shore_distance
			self.df['shore_dist'] = self.shore_distance

	def wind_fetch(self):
		'''
		Method that computes wind fetch.
		Algorithm is optimized to use closest shore distance to reduce computation time
		'''
		assert 'interp_u10' in self.df, "To load wind fetch please load wind direction first"				
		print('\nComputing Wind Fetch')
		total_wind_fetch = np.zeros(len(self.ftime))
		dir_u, dir_v = self.df['interp_u10'].to_numpy(), self.df['interp_v10'].to_numpy()
		shore_distance = self.shore_distance
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
		global bathymetry_ds
		bathymetrie_ds = Dataset('/home/datawork-osmose/dataset/auxiliary/GEBCO_2022_sub_ice_topo.nc')
		temp_lat, temp_lon = self.latitude[~np.isnan(self.latitude)].to_numpy(), self.longitude[~np.isnan(self.longitude)].to_numpy()
		pos_lat, pos_dist = j.nearest_point(temp_lat, bathymetrie_ds['lat'][:])
		pos_lon, pos_dist = j.nearest_point(temp_lon, bathymetrie_ds['lon'][:])
		bathymetrie = np.full(len(self.df), np.nan)
		bathymetrie[~np.isnan(self.latitude)] = [bathymetrie_ds['elevation'][i,j] for i,j in zip(pos_lat, pos_lon)] 
		self.df['bathy'] = bathymetrie

	def from_scratch(self):
		'''
		Takes dictionary with lat, lon, depth of hydrophone and creates empty dataframe for class
		Objective is to build time with timestamps
		'''
		try : 
			metadata = pd.read_csv(Dataset._get_original_after_build().joinpath("metadata.csv"))
		except FileNotFoundError:
			print('Could not find built dataset.')
			sys.exit()
		start_date, end_date = get_datarmor_time(metadata.start_date), get_datarmor_time(metadata.end_date)
		self.timestamps = np.arange(start_date, end_date, 600)
		self.latitude, self.longitude = [metadata.lat]*len(self.timestamps), [metadata.lon]*len(self.timestamps)
		self.df = pd.DataFrame({'time': self.timestamsp, 'lat':self.latitude, 'lon':self.longitude, 'depth':float('nan')}, index = [0])

	@classmethod
	def from_fn(cls, path, dataset, *, local = True):
		return cls(path, dataset, local)



class Weather(Variables):
	
	def __init__(self, df, local = True):
		super().__init__(df, local)		
		
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
			fh = Dataset(downloaded_cds, mode='r')
			self.variables = list(fh.variables.keys())[3:]
		else :
			if not self.local :
				print("Please download ERA on local machine and upload it on Datarmor. \nSystem exit.")
				sys.exit()
			print('Trying to download ERA5 data...')
			self.variables = download_era(self.ftime.to_numpy(), self.longitude, self.latitude, era_path, filename)

		if method == 'cube' or method == 'cylinder':
			self.stamps = np.load(era_path+'stamps.npz', allow_pickle = True)
		else :
			temp_stamps = np.load(era_path + 'stamps.npz', allow_pickle=True)
			dates, lat, lon = temp_stamps['timestamps'], temp_stamps['latitude'], temp_stamps['longitude']
			stamps = np.array([[date, lat_val, lon_val] for date in dates for lat_val in lat for lon_val in lon])
			self.stamps = stamps.reshape(len(dates), len(lat), len(lon), 3)

	def nearest_era(self, *, time_off = 600, lat_off = 0.5, lon_off = 0.5, variables = ['u10','v10']):
		
		self.load_era(method = 'nearest')
		data = self.df[['time', 'lat', 'lon']].dropna().to_numpy()
		self.stamps[:,:,:,0] = np.array(list(map(get_era_time, self.stamps[:,:,:,0].ravel()))).reshape(self.stamps[:,:,:,0].shape)
		pos_ind, pos_dist = j.nearest_point(data, self.stamps.reshape(-1, 3))
		pbar = tqdm(total = len(self.variables), position = 0, leave = True)

		for variable in self.variables:
			pbar.update(1)
			pbar.set_description("Loading and formatting %s" % variable)
			temp_variable = np.full([len(self.ftime)], np.nan)
			var = np.load(self.era_path+variable+'_'+self.dataset+'.npy')
			var = var.flatten()[pos_ind]
			mask = np.any(abs(data - self.stamps.reshape(-1,3)[pos_ind]) > [time_off, lat_off, lon_off], axis = 1)
			var[mask] = np.nan
			temp_variable[self.df[['time', 'lat', 'lon']].dropna().index] = var
			variable = f'np_{time_off}_{lat_off}_{lon_off}_{variable}'
			self.df[variable] = temp_variable
		self.df[f'np_{time_off}_{lat_off}_{lon_off}_era'] = np.sqrt(self.df[f'np_{time_off}_{lat_off}_{lon_off}_u10']**2 + self.df[f'np_{time_off}_{lat_off}_{lon_off}_v10']**2)
		
	
	def interpolation_era(self):
		
		self.load_era(method = 'interpolation')
		temp_lat = self.latitude[(~np.isnan(self.longitude)) | (~np.isnan(self.latitude))]
		temp_lon = self.longitude[(~np.isnan(self.longitude)) | (~np.isnan(self.latitude))]
		temp_ftime = self.ftime[(~np.isnan(self.longitude)) | (~np.isnan(self.latitude))]
		pbar = tqdm(total = len(self.variables), position = 0, leave = True)
		for variable in self.variables:
			pbar.update(1)
			pbar.set_description("Loading and formatting %s" % variable)
			var = np.load(self.era_path+variable+'_'+self.dataset+'.npy')
			interp = j.rect_interpolation_era(self.stamps, var)
			temp_variable =  np.full([len(self.ftime)], np.nan)
			temp_variable[(~np.isnan(self.longitude)) | (~np.isnan(self.latitude))] = j.apply_interp(interp, temp_ftime, temp_lat, temp_lon)
			variable = 'interp_'+variable
			self.df[variable] = temp_variable
		self.df['interp_era'] = np.sqrt(self.df.interp_u10**2 + self.df.interp_v10**2)


	def cube_era(self, *, time_off = 600, lat_off = 0.5, lon_off = 0.5, variables = ['u10', 'v10']):

		self.load_era(method = 'cube')
		temp_df = self.df[['time', 'lat', 'lon']].dropna()
		if variables == None:
			variables = self.variables
		for variable in  variables:
			temp_variable =  np.full([len(self.ftime)], np.nan)
			var = np.load(self.era_path+variable+'_'+self.dataset+'.npy')
			pbar = tqdm(total = len(temp_df), position = 0, leave = True)
			for i, row in temp_df.iterrows():
				time_index = np.where((abs(self.stamps['timestamps_epoch'] - row.time) <= time_off))[0]
				if len(time_index) == 0:
					pbar.update(1)
					temp_variable[i] = np.nan
					continue
				lat_index = np.where((abs(self.stamps['latitude'] - row.lat) <= lat_off))[0]
				if len(lat_index) == 0:
					pbar.update(1)
					temp_variable[i] = np.nan
					continue
				lon_index = np.where((abs(self.stamps['longitude'] - row.lon) <= lon_off))[0]
				if len(lon_index) == 0:
					pbar.update(1)
					temp_variable[i] = np.nan
					continue
				temp_value = []
				for time_ind in time_index:
					for lat_ind in lat_index:
						for lon_ind in lon_index :
							temp_value.append(var[time_ind, lat_ind, lon_ind])
				temp_variable[i] = np.nanmean(temp_value)
				pbar.update(1)
			variable = f'cube_{time_off}_{lat_off}_{lon_off}_{variable}'
			self.df[variable] = temp_variable
		self.df[f'cube_{time_off}_{lat_off}_{lon_off}_era'] = np.sqrt(self.df[f'cube_{time_off}_{lat_off}_{lon_off}_u10']**2+self.df[f'cube_{time_off}_{lat_off}_{lon_off}_v10']**2)
		
		
	def cylinder_era(self, *, time_off = 600, r = 'depth', variables = ['u10', 'v10']):

		self.load_era(method = 'cylinder')
		temp_df = self.df[['time', 'lat', 'lon']].dropna()
		if isinstance(r, int) or isinstance(r, float):
			r = [r]*len(temp_df)
		else :
			r = 3*self.df.depth
		if variables == None:
			variables = self.variables
		for variable in  variables:
			temp_variable =  np.full([len(self.ftime)], np.nan)
			var = np.load(self.era_path+variable+'_'+self.dataset+'.npy')
			pbar = tqdm(total = len(temp_df), position = 0, leave = True)
			for i, row in temp_df.iterrows():
				time_index = np.where((abs(self.stamps['timestamps_epoch'] - row.time) <= time_off))[0]
				if len(time_index) == 0:
					pbar.update(1)
					temp_variable[i] = np.nan
					continue
				mask = np.full((var[0].shape), False)
				for k, latitude in enumerate(self.stamps['latitude']):
					mask[k, np.where((haversine(latitude, row.lat, self.stamps['longitude'], row.lon) <= r[i]))[0]] = True
				if not mask.any():
					pbar.update(1)
					temp_variable[i] = np.nan
					continue
				temp_variable[i] = np.nanmean(var[time_index][mask])
				pbar.update(1)
			variable = f'cylinder_{time_off}_{r}_{variable}'
			self.df[variable] = temp_variable
		self.df[f'cylinder_{time_off}_{r}_era'] = np.sqrt(self.df[f'cylinder_{time_off}_{r}_u10']**2+self.df[f'cylinder_{time_off}_{r}_v10']**2)
	
def check_era(path, dataset):
	if os.path.isdir(os.path.join(path, dataset, 'data', 'auxiliary', 'weather', 'era')):
		print("Era is already downloaded, you're free to continue")
		return None
	else:
		os.makedirs(os.path.join(path, dataset, 'data', 'auxiliary', 'weather', 'era'))
		print(f"Please upload ERA data in {os.path.join(path, 'weather', 'era')} \nIf you have not downloaded ERA data yet see https://github.com/Project-OSmOSE/osmose-datarmor/tree/package/local_notebooks")
		return None
