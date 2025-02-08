from datetime import date, datetime, timedelta
import math
import numpy as np
import os
import pandas as pd
import seals_utils
import time
import xarray as xr
from sphinx.ext import autodoc
#import wind_field_functions_L1_printFlag as wff

class SiteMapNcReader:
    """Class to read in a site map from a netCDF file."""
    
    def __init__(self, filename):
        """Create the SiteMapNcReader object and read in the given netCDF site map.
    
        Save the site map as an xarray as an instance variable.
        """
        
        self.filename = filename
    
        # Read the netCDF file into an xarray
        self.site_map = xr.open_dataset(self.filename)
        
        print('Just read in file', self.filename)
        #print(self.site_map)
        
    def get(self):
        """Return the site map as an xarray object
        """
        
        return self.site_map
        
class SensorReadingsNcReader:
    """Class to read in sensor readings from a netCDF file."""
    
    def __init__(self, filename):
        """Create the SensorReadingsNcReader object and read in the given netCDF file of sensor readings.
    
        Save the sensor readings as an xarray as an instance variable.
        """
        
        self.filename = filename
    
        # Read the netCDF file into an xarray
        self.sensor_readings = xr.open_dataset(self.filename)
        
        print('Just read in file', self.filename)
        #print(self.sensor_readings)
        
    def get(self):
        """Return the sensor readings as an xarray object."""
        
        return self.sensor_readings
    
def unify_time_coordinates(processed_data, sensor_names, time_increment, sensor_variables_interpolated_names):
    """Generate a sensor data array with uniform observation times across all of the sensors, using interpolation as needed."""
    
    num_sensors = len(sensor_names)
    unix_s_v = np.zeros(num_sensors)
    unix_e_v = np.zeros(num_sensors)
    
    # Determine the time range covered by ALL sensors.  (ie, the intersection of their ranges, not the union).
    for sensor_name_idx in range(num_sensors):
        sensor_name = sensor_names[sensor_name_idx]
        first_time_idx = 0
        
        # Get the last time index for this sensor
        last_time_idx = len(processed_data.processed[sensor_name]) - 1
        
        # Get the earliest and latest times of this sensor
        unix_s_v[sensor_name_idx] = time.mktime(processed_data.processed[sensor_name][first_time_idx]['time'].timetuple())
        unix_e_v[sensor_name_idx] = time.mktime(processed_data.processed[sensor_name][last_time_idx]['time'].timetuple())
        
    # Get the latest start and earliest end times over all the sensors.  This will be the range covered by ALL sensors.
    unix_min = np.max(unix_s_v)
    unix_max = np.min(unix_e_v)
    print('unix_min, unix_max, tdiff=', unix_min, unix_max, unix_max-unix_min)
    
    datetime_min = datetime.fromtimestamp(unix_min)
    datetime_max = datetime.fromtimestamp(unix_max)
    print('datetime_min=', datetime_min)
    print('datetime_max=', datetime_max)
    
    # Create unix time vector that spans this range
    unix_time_vector = np.arange(unix_min, unix_max + time_increment, time_increment)
    num_unix_times = len(unix_time_vector)
    print('num_unix_times=', num_unix_times)

    # Create a matching vector of python datetime objects
    datetime_vector = []
    for time_idx in range(num_unix_times):
        datetime_vector.append(datetime.fromtimestamp(unix_time_vector[time_idx]))
        
    # Initialize sensor data array with zeros
    num_interpolated_variables = len(sensor_variables_interpolated_names)
    sensor_data_array = np.zeros([num_sensors, num_interpolated_variables, num_unix_times]) # dimensions: sensors, variables, times
    print('type(sensor_data_array)=', type(sensor_data_array))
    print('sensor_data_array.shape=', sensor_data_array.shape)
    
    # Iterate over sensors again, 
    for sensor_name_idx in range(num_sensors):
        sensor_name = sensor_names[sensor_name_idx]
        
        print('sensor_name=', sensor_name)
        print('sensor_name_idx=', sensor_name_idx)
        
        sensor_time_len = len(processed_data.processed[sensor_name])
        print(sensor_time_len)
        
        unix_sensor_time_vector = np.zeros(sensor_time_len)
        print('unix_sensor_time_vector.shape=', unix_sensor_time_vector.shape)
        
        for time_idx in range(sensor_time_len):
            unix_sensor_time_vector[time_idx] = time.mktime(processed_data.processed[sensor_name][time_idx]['time'].timetuple())
            
        for variables_idx in range(num_interpolated_variables):
            variable = sensor_variables_interpolated_names[variables_idx]

            variable_pp = [ sub[variable] for sub in processed_data.processed[sensor_name] ]
            # print('len(variable_pp)=', len(variable_pp))
            variable_interpolated = np.interp(unix_time_vector, unix_sensor_time_vector, variable_pp)
            sensor_data_array[sensor_name_idx, variables_idx,:] = variable_interpolated

    # Return the interpolated data and the list of datetimes
    print('type(sensor_data_array)=', type(sensor_data_array))
    print('sensor_data_array.shape=', sensor_data_array.shape)
    print('len(datetime_vector)=', len(datetime_vector))
    return sensor_data_array, datetime_vector
    
def sensit_data_dir_from_datetime(datetime_obj, params):
    """Determine the date-specific SENSIT data directory from the given datetime.
    
    Determine the date from the datetime, make it a 2 digit year version, and use that and the sensit_path
    property to come up with the directory for the particular date.
    """
    
    year_str = str(datetime_obj.year - 2000)
    month_str = f'{datetime_obj.month:02}'
    day_str = f'{datetime_obj.day:02}'
    sensit_data_path = params['sensit_path']
    target_date_str = year_str + month_str + day_str # YYMMDD
    sensit_data_date_dir = os.path.join(sensit_data_path, target_date_str + " data")
    return sensit_data_date_dir
    
def trim_processed_data_for_time(sensit_data, start_datetime, end_datetime):
    """Remove the processed data that does not fall within the start_datetime and end_datetime.
    
    We don't need to do the same thing for the raw data, since the only thing we use that for is than
    lat/lon of the sensors.
    """
    
    print('Lengths of processed data BEFORE filtering for timespan:')
    for sensor in sensit_data.devices:
        print(sensor, ':', len(sensit_data.processed[sensor]))
    
    for sensor in sensit_data.devices:
        sensit_data.processed[sensor] = [x for x in sensit_data.processed[sensor] if x['time'] >= start_datetime and x['time'] <= end_datetime]
        
    print('Lengths of processed data AFTER filtering for timespan:')
    for sensor in sensit_data.devices:
        print(sensor, ':', len(sensit_data.processed[sensor]))
    
def append_processed_day2_to_day1(sensit_data_1, sensit_data_2):
    """Append the processed data from a second day to that of the first."""
    
    print('New lengths of processed data after append:')
    for sensor in sensit_data_1.devices:
        for idx in range(len(sensit_data_2.processed[sensor])):
            sensit_data_1.processed[sensor].append(sensit_data_2.processed[sensor][idx])
        print(sensor, ':', len(sensit_data_1.processed[sensor]))
    
class SensitData:
    """Class to encapsulate SENSIT data from one date directory of SENSIT data.
    
    The bulk of this class was provided by the SENSIT Technologies (gasleaksensors.com) in the form of the Data class in the file wind_field_functions_L1_printFlag.py.
    This version includes modification by NCAR for fixes, simplification (removal of things we're not using), and generalization.
    """
    
    def __init__(self, filepath, time_offset, met_sensors):
        """Initialize the SensitData object with data from a particular date directory of SENSIT data.
        
        Save data in the following instance variables:
        filepath - path of directory of a particular date's data
        raw - unprocessed data.  We use lat/lon from this.
        devices - list of device names
        processed - most of the data we use
        """
        self.filepath = filepath

        os.chdir(filepath)
        file_list = os.listdir()

        self.raw = {}
        self.devices = []

        for file in file_list:
            if ".csv" in file:
                params = file.split("_")
                sensor_name = "FMD" + str(params[3])
                # Add the sensor name to the list of them (instance variable "devices")
                self.devices.append(sensor_name)
                
                # Read in the raw data
                self.raw[sensor_name] = pd.read_csv(file, skiprows=2)

        self.processed = {}
        max_time = datetime(2000, 1, 1, 1, 1, 1)
        min_time = datetime(2100, 1, 1, 1, 1, 1)
    
        for fmd in self.devices:

            self.processed[fmd] = []

            baseline_chunk = []
            for i in range(1, 60):
                index = len(self.raw[fmd]) - i

                if fmd == "FMD1008":
                    tmp_methane = self.raw[fmd].loc[index, "LaserCellConcentration.Conc_cor"]
                else:
                    tmp_methane = self.raw[fmd].loc[index, "CH4"]

                baseline_chunk.append(tmp_methane)
            baseline = min(baseline_chunk)
            if baseline < 0:
                baseline = 0

            for i in range(1, len(self.raw[fmd])):
                index = len(self.raw[fmd])-i

                if fmd in met_sensors: # sonic anemometer
                    x_vector = -self.raw[fmd]['_3DMet.U'].interpolate(method='linear').loc[index]
                    y_vector = -self.raw[fmd]['_3DMet.V'].interpolate(method='linear').loc[index]
                else:
                    wind_dir = float(self.raw[fmd].loc[index, "WD"])
                    wind_speed = float(self.raw[fmd].loc[index, "WS"])
                    x_vector = -wind_speed * np.sin(wind_dir * np.pi / 180.0)
                    y_vector = -wind_speed * np.cos(wind_dir * np.pi / 180.0)

                # NOTE:  This hard-coded sensor ID should probably be either parameterized or eliminated,
                # pending answer from Sensit on why the special treatment of this sensor 
                if fmd == "FMD1008":
                    tmp_methane = self.raw[fmd].loc[index, "LaserCellConcentration.Conc_cor"]
                else:
                    tmp_methane = self.raw[fmd].loc[index, "CH4"]

                if tmp_methane - baseline < 10:
                    baseline = baseline + tmp_methane / 6 - baseline / 6
                     
                if tmp_methane - baseline > 0:
                    ch4_corr = tmp_methane - baseline
                else:
                    ch4_corr = 0
                  
                if fmd in met_sensors: # sonic anemometer
                    tmpw_value = self.raw[fmd].loc[index, "_3DMet.W"]
                    w_value = np.where(np.isnan(tmpw_value), 0.0, tmpw_value)
                else:
                    w_value = 0.0 #JAS ML-models can't handle nans #np.nan
                
                try:
                    save_dict = {
                        "name"      : fmd,
                        "time"      : datetime.strptime(self.raw[fmd].loc[index, "UTC Date Time"], "%d-%b-%Y %H:%M:%S") + timedelta(hours=time_offset),
                        "ws"        : self.raw[fmd].loc[index, "WS"],
                        "wd"        : self.raw[fmd].loc[index, "WD"],
                        "w_z"       : w_value,
                        "rh"        : self.raw[fmd].loc[index, "RH"],
                        "temp"      : self.raw[fmd].loc[index, "T"],
                        "w_x"       : x_vector,
                        "w_y"       : y_vector,
                        "ch4"       : ch4_corr  
                    }  
                    self.processed[fmd].append(save_dict)

                    if max_time < save_dict["time"]:
                        max_time = save_dict["time"]
                    if min_time > save_dict["time"]:
                        min_time = save_dict["time"]

                except Exception as e:
                    print(fmd + ": " + self.raw[fmd].loc[index, "Local Date Time"] + " " + e)
        
        for met_sensor in met_sensors:
            for i in range(1, len(self.processed[met_sensor])):
                ws_check_array = []
                wd_check_array = []
                for m in range(0, 5):
                    ws_check_array.append(self.processed[met_sensor][i-m]["ws"])
                    wd_check_array.append(self.processed[met_sensor][i-m]["wd"])
            
                wind_check = (len(set(ws_check_array)) <= 1 and len(set(wd_check_array)) <= 1)

                if (wind_check == True):
                    self.processed[met_sensor][i]["w_x"] = 0
                    self.processed[met_sensor][i]["w_y"] = 0

    
class sensor_readings_sensit_ADED_I_reader:
    """Class to read in SENSIT data from a directory for a particular time span."""
    
    def __init__(self, params):
        """Initialize the sensor_readings_sensit_reader object and read in the data.
        
        Save data in the following instance variables:
        params
        sensor_names
        sensor_positions_latlon
        time_unified_data
        datetimes
        """
        
        self.params = params
        
        # Determine target date(s) in the form YYMMDD.  May need to target two different dates, depending on the start
        # datetime and span.  SENSIT data in a particular date dir covers from that date at 5:00 to the next day at 5:00.  
        sensit_data_start_datetime_str = self.params['sensit_data_start_datetime']
        sensit_data_duration = self.params['sensit_data_duration']
        start_datetime = datetime.strptime(sensit_data_start_datetime_str, "%Y-%m-%d %H:%M:%S")
        end_datetime = start_datetime + timedelta(hours = sensit_data_duration)
        start_hour = start_datetime.hour
        if start_hour < 5:
            # Need to start with previous day directory.
            datetime_1 = start_datetime - timedelta(days = 1)
            # See if we need the day after that (date of start_datetime)) also.  Might not if start time + duration is less than 5 hours.
            if start_hour + sensit_data_duration > 5:
                # Yep, need another day
                datetime_2 = start_datetime
            else:
                datetime_2 = None
        else:
            # First date will be as given
            datetime_1 = start_datetime
            
            if start_hour + sensit_data_duration > 29: # 5 hours past the end of the day
                # Need another day
                datetime_2 = start_datetime + timedelta(days = 1)
            else:
                datetime_2 = None
        
        daylight = self.params['sensit_hour_offset']
        
        sensit_data_date_dir_1 = sensit_data_dir_from_datetime(datetime_1, self.params)
        print('sensit_data_date_dir_1', sensit_data_date_dir_1)
        sensit_data_1 = SensitData(sensit_data_date_dir_1, daylight, self.params['sensit_met_sensors'])
        # sensit_data_1 is an object of the class SensitData. It contains the data from one particular date directory 
        # under the overall SENSIT directory, which covers from that date at 5:00 to the next day at 5:00.
        # It contains the following properties:
        # filepath
        # raw                - Used in this method, but only for lat/lon sensor positions
        # devices            - Used in this method
        # processed          - Used in this method
        
        # Trim off the processed data that falls outside of the experiment's time span
        trim_processed_data_for_time(sensit_data_1, start_datetime, end_datetime)
        
        # Print some timestamps for the first sensor to observe time order (increasing)
        if len(sensit_data_1.processed[sensit_data_1.devices[0]]) >= 10:
            print('First 10 times of processed data for sensit_data_1,', sensit_data_1.devices[0] + ':')
            for idx in range(10):
                print(sensit_data_1.processed[sensit_data_1.devices[0]][idx]['time'])
        
        if datetime_2 is not None:
            sensit_data_date_dir_2 = sensit_data_dir_from_datetime(datetime_2, self.params)
            print('sensit_data_date_dir_2', sensit_data_date_dir_2)
            sensit_data_2 = SensitData(sensit_data_date_dir_2, daylight, self.params['sensit_met_sensors'])
            trim_processed_data_for_time(sensit_data_2, start_datetime, end_datetime)
            
            if len(sensit_data_2.processed[sensit_data_2.devices[0]]) >= 10:
                print('First 10 times of processed data for sensit_data_2,', sensit_data_2.devices[0] + ':')
                for idx in range(10):
                    print(sensit_data_2.processed[sensit_data_2.devices[0]][idx]['time'])
        
            # Now need to append the processed data of sensit_data_2 to that of sensit_data
            # Unlike the raw data, the processed data is actually in forward order, so that should
            # work nicely.
            append_processed_day2_to_day1(sensit_data_1, sensit_data_2)
        
        # Get list of sensor names.  Save it as an instance variable
        self.sensor_names = sensit_data_1.devices
        num_sensors = len(self.sensor_names)
        print('sensor_names=', self.sensor_names)
            
        # Get the sensor positions in lat/lon.  Save as instance variable
        self.sensor_positions_latlon = np.zeros([num_sensors, 2])
        for sensorIdx in range(0, num_sensors):
            self.sensor_positions_latlon[sensorIdx, 0] = sensit_data_1.raw[sensit_data_1.devices[sensorIdx]]['lat'].loc[0]
            self.sensor_positions_latlon[sensorIdx, 1] = sensit_data_1.raw[sensit_data_1.devices[sensorIdx]]['long'].loc[0]
        print('sensor_positions_latlon:')
        print(self.sensor_positions_latlon)
            
        # Unify the time coordinates equally across sensors, save as an instance variable
        self.time_unified_data, self.datetimes = unify_time_coordinates(sensit_data_1, self.sensor_names, params['sensit_time_increment'], params['sensit_vars_interpolated'])
        
        print('type(time_unified_data)=', type(self.time_unified_data))
        print('shape=', self.time_unified_data.shape)
        print('num elements=', len(self.time_unified_data))
        #print('time_unified_data=', self.time_unified_data)
        first_element = self.time_unified_data[0]
        print('type of first element=', type(first_element))
        print('length of first element=', len(first_element))
            
        return
        
class sensor_readings_sensit_ADED_II_reader:
    """Class to read in SENSIT data from a directory for a particular time span."""

    def __init__(self, params):
        """Initialize the sensor_readings_sensit_reader object and read in the data.
        
        Save data in the following instance variables:
        params
        sensor_names
        sensor_positions_latlon 
        time_unified_data
        datetimes
        """
        swap_FMD1001_CH4 = True
        median_scaling = False
        anomaly_filter = False
        anomalyThreshFactor = 25.0
        self.params = params
        
        sensit_data_path = params['sensit_path']
        #Create a list of files in the sensit_data_path directory
        file_list = os.listdir(sensit_data_path)
        raw = {}
        devices = []
        sensor_name=[]
        #Read in the various instrument raw data into a dictionary of pandas data frames
        for file in sorted(file_list):
          fullpath_file = os.path.join(sensit_data_path, file)
          print(fullpath_file)
          if ".csv" in file:
             parse_array = file.split("_")
             sensor_name = str(parse_array[0]) + str(parse_array[3])
             # Add the sensor name to the list of them (instance variable "devices")
             devices.append(sensor_name)

             # Read in the raw data 
             raw[sensor_name] = pd.read_csv(fullpath_file, skiprows=2)
        
        #Create lists of FMD and SPOD device names 
        FMD_names = [i for i in devices if "FMD" in i]
        SPOD_names = [i for i in devices if "SPOD" in i]

        #Prepare to process the device dataframes by creating a list of unecessary columns (variables) that can be removed
        #FMD-devices
        df=raw[FMD_names[0]]
        allcol = df.columns.tolist()
        keepcol = ['UTC Date Time', 'Local Date Time', 'CH4', 'WS', 'WD', 'lat', 'long']
        dropcol = [x for x in allcol if x not in keepcol]
        #SPOD-devices
        df_s=raw[SPOD_names[0]]
        allcol_spod = df_s.columns.tolist()
        keepcol_spod = ['UTC Date Time', 'Local Date Time', 'pid1_PPB_Calc', 'lat', 'long']
        dropcol_spod = [x for x in allcol_spod if x not in keepcol_spod]
       
        #Process the raw dataframes by removing uneccessary columns 
        dfs_FMD=[]
        dfs_SPOD=[]
        #FMD-devices
        for dev in FMD_names:
          df=raw[dev].drop(columns=dropcol)
          if swap_FMD1001_CH4:
            if dev == 'FMD1001':
               df['CH4'] = raw[dev]['LaserCellConcentration.Conc_cor']
          df['UTC Date Time']=pd.to_datetime(df['UTC Date Time'],format='%m/%d/%Y %I:%M:%S %p')
          df.set_index('UTC Date Time')
          dfs_FMD.append(df)
        #SPOD-devices
        for dev in SPOD_names:
          df=raw[dev].drop(columns=dropcol_spod)
          df['UTC Date Time']=pd.to_datetime(df['UTC Date Time'],format='%m/%d/%Y %I:%M:%S %p')
          df.set_index('UTC Date Time')
          dfs_SPOD.append(df)
         
        #Define the date-time window of interest
        sensit_data_start_datetime = params['sensit_data_start_datetime']
        sensit_data_duration = params['sensit_data_duration']
        exp_start = pd.to_datetime(sensit_data_start_datetime, format='%Y-%m-%d %H:%M:%S')
        exp_duration =  pd.Timedelta(sensit_data_duration, unit='h')
        exp_end = exp_start+exp_duration
        
        #Identify the last recorded (preceding the datetime window of interest) lat/lon for each sensor
        FMD_latlons=[]
        SPOD_latlons=[]
        for idx,df in enumerate(dfs_FMD):
          df_tmp = df.dropna(subset=['UTC Date Time','lat','long']).drop_duplicates(subset=['UTC Date Time']).set_index('UTC Date Time').drop(columns=['Local Date Time','CH4','WS','WD'])
          nearest=df_tmp.iloc[df_tmp.index.get_indexer([exp_start],method='ffill')]
          FMD_latlons.append(np.squeeze(nearest.to_numpy()))
          #print(FMD_latlons[-1])
        for idx,df in enumerate(dfs_SPOD):
          df_tmp = df.dropna(subset=['UTC Date Time','lat','long']).drop_duplicates(subset=['UTC Date Time']).set_index('UTC Date Time').drop(columns=['Local Date Time','pid1_PPB_Calc'])
          nearest=df_tmp.iloc[df_tmp.index.get_indexer([exp_start],method='ffill')]
          SPOD_latlons.append(np.squeeze(nearest.to_numpy()))
          #print(SPOD_latlons[-1])

        #Extract the portion of the dataset spanning the date-time window of interest
        df_exp=[]
        df_exp_spod=[]
        #FMD-devices
        fmd_meds=[]
        fmd_maxs=[]
        dev_rm=[]
        for idx,df in enumerate(dfs_FMD):
          df_e = df[(df['UTC Date Time'] >= exp_start) & (df['UTC Date Time'] <= exp_end)]
          dev_name = FMD_names[idx]
          if(df_e.empty):
            print(f"No data available from device: {dev_name}, omitting device.")
            dev_rm.append(dev_name)
          else:
            df_e.loc[df_e['WD'] == 375.0, 'WD'] = np.nan  ### When WS is below a minimum threshhold WD is set to 375, replace with nan
            if median_scaling or anomaly_filter:    #Find the median measured CH4 value across this device and append to the list for FMD devices
               dev_median = df_e['CH4'].median()
               fmd_meds.append(dev_median)
               dev_max =df_e['CH4'].max()
               fmd_maxs.append(dev_max)
               print(f"{dev_name} {dev_median}, {dev_max}")
            df_exp.append(df_e.drop(columns=['lat','long']))
        #Remove any devices that had no data
        for dev_name in dev_rm:
          idx=FMD_names.index(dev_name)
          FMD_names.remove(dev_name)
          FMD_latlons.pop(idx)
        print(f"{FMD_names}")
        print(f"{FMD_latlons}")
        if median_scaling or anomaly_filter:
          medians_list=[]  
          maxs_list=[]  
          for idx,dev_name in enumerate(FMD_names):
              if not(dev_name == "FMD1001"):
                  medians_list.append(fmd_meds[idx])
                  maxs_list.append(fmd_maxs[idx])
          fmd_avg_med = np.nanmean(np.asarray(medians_list))
          fmd_max_max = np.nanmax(np.asarray(maxs_list))
          print(f"avergaging medians_list: {medians_list}, fmd_avg_med = {fmd_avg_med}")
          print(f"taking max of maxs_list: {maxs_list}, fmd_max_max = {fmd_max_max}")
          
        ##SPOD-devices
        dev_rm=[]
        for idx,df_s in enumerate(dfs_SPOD):
          df_e_s = df_s[(df_s['UTC Date Time'] >= exp_start) & (df_s['UTC Date Time'] <= exp_end)]
          dev_name = SPOD_names[idx]
          if(df_e_s.empty):
            print(f"No data available from device: {dev_name}, omitting device.")
            dev_rm.append(dev_name)
          else:
            if median_scaling: #scale this SPOD CH4 obs by the ratio of this SPOD CH4 median to avg(all FMD median values)  
              dev_median = df_e_s['pid1_PPB_Calc'].median()
              scaleFactor = fmd_avg_med/dev_median
              print(f"{dev_name}: {dev_median}, scaleFactor = {scaleFactor}")
              if (scaleFactor > 0) and (scaleFactor < 1):  #Practically should only scale down and NEVER negative!
                scaledValues=df_e_s['pid1_PPB_Calc']*scaleFactor
                df_e_s.update({'pid1_PPB_Calc': scaledValues})
            if anomaly_filter:
              dev_mnab = np.nanmean(np.where(df_e_s['pid1_PPB_Calc']>1.1*fmd_avg_med,df_e_s['pid1_PPB_Calc'],np.nan)) #calculate mean above background
              if dev_mnab/fmd_avg_med > anomalyThreshFactor or (dev_mnab > 2.0*fmd_max_max and fmd_max_max > 5.0*fmd_avg_med):
                print(f"{dev_name} has suspicious values... setting to background")
                bgValues = fmd_avg_med + 0.0*df_e_s['pid1_PPB_Calc']
                df_e_s.update({'pid1_PPB_Calc': bgValues})
              

            df_exp_spod.append(df_e_s.drop(columns=['lat','long']))
        #Remove any devices that had no data
        for dev_name in dev_rm:
          idx=FMD_names.index(dev_name)
          SPOD_names.remove(dev_name)
          SPOD_latlons.pop(idx)

        #Regularize and interpolate the dataframes to be at 1Hz and spanning the entire datetime window without nans
        frequency = 's'
        new_index = pd.date_range(start=exp_start,end=exp_end,freq=frequency) #1 Hz datatime index spanning the full period of interest
        sensor_array_FMD = np.zeros(shape=(len(df_exp),4,new_index.shape[0]))
        sensor_array_SPOD = np.zeros(shape=(len(df_exp_spod),1,new_index.shape[0]))
        #FMD-devices
        dfcnt=0
        for df in df_exp:
           df_cols = df.columns 
           resampled_df = df.drop_duplicates(subset=['UTC Date Time']).set_index('UTC Date Time').resample(frequency).asfreq().reindex(new_index,method='bfill')
           resampled_df.interpolate(method='time',inplace=True)
           resampled_df.reset_index(inplace=True)
           resampled_df.columns=df_cols
           tmp_array = np.swapaxes(resampled_df.drop(columns=['UTC Date Time', 'Local Date Time']).to_numpy(),0,1)
           for varIndx in range(4):
               if varIndx == 0:  #CH4
                  sensor_array_FMD[dfcnt,varIndx,:] = tmp_array[varIndx,:]
               elif varIndx == 1:  # u
                  sensor_array_FMD[dfcnt,varIndx,:] = -tmp_array[1,:] * np.sin(np.radians(tmp_array[2,:]))
               elif varIndx == 2:  # v
                  sensor_array_FMD[dfcnt,varIndx,:] = -tmp_array[1,:] * np.cos(np.radians(tmp_array[2,:]))
               elif varIndx == 3:  # w
                  sensor_array_FMD[dfcnt,varIndx,:] = 0.0*tmp_array[1,:]
           dfcnt += 1
        #SPOD-devices
        dfcnt=0
        for df in df_exp_spod:
           df_spod_cols = df.columns 
           resampled_df = df.drop_duplicates(subset=['UTC Date Time']).set_index('UTC Date Time').resample(frequency).asfreq().reindex(new_index,method='bfill')
           resampled_df.interpolate(method='time',inplace=True)
           resampled_df.reset_index(inplace=True)
           resampled_df.columns=df_spod_cols
           sensor_array_SPOD[dfcnt,:,:] = np.swapaxes(resampled_df.drop(columns=['UTC Date Time', 'Local Date Time']).to_numpy(),0,1)
           dfcnt += 1

        self.datetimes = new_index
        self.time_unified_data = [sensor_array_FMD, sensor_array_SPOD]
        self.sensor_names = [FMD_names, SPOD_names]
        self.sensor_positions_latlon = [FMD_latlons, SPOD_latlons]

        return
