from sealsml.staticinference import specific_site_data_generation
import os
import numpy as np
import traceback
import xarray as xr
    
def generate_site_dataset(data_path, sitemap_path, potential_leaks_path, time_window_size, time_window_stride):
    """Generate a Dataset describing the site in the format needed by seals-ml models.
    
    Returns a "success" return code (True or False)
    """
    
    assert os.path.exists(data_path), f"File not found: {data_path}"
    assert os.path.exists(sitemap_path), f"File not found: {sitemap_path}"
    assert os.path.exists(potential_leaks_path), f"File not found: {potential_leaks_path}"
            
    print('MlModel, all 3 files exist (data, sitemap, potential leaks).')
        
    try:    
        # Get the site dataset using sealsml.staticinference's specific_site_data_generation method
        site_dataset = specific_site_data_generation(data_path, sitemap_path, potential_leaks_path, time_window_size = time_window_size, window_stride = time_window_stride)
        assert isinstance(site_dataset, xr.Dataset), "The object is not an xarray.Dataset"
        assert isinstance(site_dataset['encoder_input'], xr.DataArray), "The object is not an xarray.DataArray"
        assert isinstance(site_dataset['decoder_input'], xr.DataArray), "The object is not an xarray.DataArray"
            
        print('site_dataset is passing asserts.... good....')
    
        return True, site_dataset
    except:
        print('Exception in call to specific_site_data_generation (or following asserts)!')
        traceback.print_exc()
        return False, None
            
def package_ADED_I_data_for_SEALS(time_unified_data, sensor_names, met_sensor_names, site_map_dataset, sensor_positions_latlon, datetimes, params):
    """Package the given time-unified sensor data in the format needed by seals-ml models."""
    
    num_ch4_sensors = len(time_unified_data) # First dimension is over sensors
    num_met_sensors = len(met_sensor_names)
    location_dimension = 3
    met_velocities_dimension = 3
    time_dimension = time_unified_data.shape[2]
    print('num_ch4_sensors=', num_ch4_sensors)
    print('location_dimension=', location_dimension)
    print('met_velocities_dimension=', met_velocities_dimension)
    print('num_met_sensors=', num_met_sensors)
    print('time_dimension=', time_dimension)
    
    # find indices for the sensors that are meteorological (MET) sensors
    met_sensor_indices = [] # Start with empty list
    # Iterate over sensor names
    for sensor_name_idx in range(len(sensor_names)):
        this_sensor_name = sensor_names[sensor_name_idx]
        # Iterate over MET sensor names, looking for a match
        for met_sensor_name_idx in range(len(met_sensor_names)):
            this_met_sensor_name = met_sensor_names[met_sensor_name_idx]
            if (this_sensor_name == this_met_sensor_name):
                # Yep, this sensor name is in the list of MET sensors, so add it's index to the list of indices.
                met_sensor_indices.append(sensor_name_idx)
    print('met_sensor_indices=', met_sensor_indices)
    
    print('site map dataset=', site_map_dataset)
    
    # Grab the x,y and lat,lon values of the bottom plane (kDim=0)
    xPos_site = site_map_dataset.xPos.values[0,:,:]
    yPos_site = site_map_dataset.yPos.values[0,:,:]
    zPos_site = site_map_dataset.zPos.values[0,:,:] # Does using the bottom plane for this make sense?  Or is there a way to 
                                                    # find out which plane each sensor is at?
    lat_site = site_map_dataset.lat.values[:,:]
    lon_site = site_map_dataset.lon.values[:,:]
    
    # Get positions of MET sensors
    met_sensors_positions = np.zeros([num_met_sensors, location_dimension])
    for met_sensor_idx in range(num_met_sensors):
        ind_tt = met_sensor_indices[met_sensor_idx]
        lat_tt = sensor_positions_latlon[ind_tt][0]
        lon_tt = sensor_positions_latlon[ind_tt][1]
        diff_tt = np.sqrt(np.power(lat_site - lat_tt, 2.0) + np.power(lon_site - lon_tt, 2.0))
        ind_2d_tt = np.where(diff_tt == np.min(diff_tt, axis=(1,0)))
        print('Setting z position of MET sensor to 2.0 meters AGL')
        met_sensors_positions[met_sensor_idx,:] = [xPos_site[ind_2d_tt][0], yPos_site[ind_2d_tt][0], 2.0]
        
    # Get positions of CH4 (methane) sensors
    ch4_sensors_positions = np.zeros([num_ch4_sensors, location_dimension])
    for ch4_sensor_idx in range(num_ch4_sensors):
        lat_tt = sensor_positions_latlon[ch4_sensor_idx][0]
        lon_tt = sensor_positions_latlon[ch4_sensor_idx][1]
        diff_tt = np.sqrt(np.power(lat_site - lat_tt, 2.0) + np.power(lon_site-lon_tt, 2.0))
        ind_2d_tt = np.where(diff_tt==np.min(diff_tt, axis=(1,0)))
        print('Setting z position of CH4 sensor to 1.375 meters AGL')
        ch4_sensors_positions[ch4_sensor_idx,:] = [xPos_site[ind_2d_tt][0], yPos_site[ind_2d_tt][0], 1.375]
    
    # Get the meteorological (MET) sensor readings (wind velocities)
    met_velocities = np.zeros([num_met_sensors, met_velocities_dimension, time_dimension])
    for met_sensor_idx in range(num_met_sensors):
        ind_tt = met_sensor_indices[met_sensor_idx]
        met_velocities[met_sensor_idx,:,:] = time_unified_data[ind_tt, 1:4, :]
    
    # Get the methane sensor readings
    ch4_readings = np.zeros([num_ch4_sensors, time_dimension])
    for sensor_idx in range(num_ch4_sensors):
        ch4_readings[sensor_idx, :] = time_unified_data[sensor_idx, 0, :] * 1e-6 # PPM to kg/kg
        
    # Create the dataset object and put the data into it
    seals_std_dataset = xr.Dataset()
    seals_std_dataset['time'] = xr.DataArray(datetimes, dims=['time'])
    seals_std_dataset['metPos'] = xr.DataArray(met_sensors_positions, dims=['metSensors', 'locDim'])
    seals_std_dataset['metPos'].attrs['units'] = 'm'
    seals_std_dataset['metVels'] = xr.DataArray(met_velocities, dims=['metSensors', 'metVelsDim', 'time'])
    seals_std_dataset['metVels'].attrs['units'] = 'm/s'
    seals_std_dataset['CH4Pos'] = xr.DataArray(ch4_sensors_positions, dims=['CH4Sensors', 'locDim'])
    seals_std_dataset['CH4Pos'].attrs['units'] = 'm'
    seals_std_dataset['q_CH4'] = xr.DataArray(ch4_readings, dims=['CH4Sensors', 'time'])
    seals_std_dataset['q_CH4'].attrs['units'] = 'kg/kg'

    seals_std_dataset['CH4SensorsName'] = xr.DataArray(sensor_names, dims=['CH4Sensors'])
    seals_std_dataset['metSensorsName'] = xr.DataArray([sensor_names[idx] for idx in met_sensor_indices], dims=['metSensors'])
    
    print('seals_std_dataset=', seals_std_dataset)
    
    # Dump the dataset into a NetCDF file (if configured to)
    # For now anyway, ALWAYS save it, as the seals-ml method specific_site_data_generation reads it from file
    #if params['save_std_netcdf']:
    netcdf_out_path = params['std_netcdf_write_path']
    seals_std_dataset.to_netcdf(netcdf_out_path)
    
    return seals_std_dataset, netcdf_out_path

def package_ADED_II_data_for_SEALS(time_unified_data, sensor_names, unified_datetimes, site_map_dataset, sensor_positions_latlon, params):
    """Package the given time-unified sensor data in the format needed by seals-ml models."""
    
    num_ch4_sensors = 0
    for sensor_sets in range(len(time_unified_data)):
        num_ch4_sensors += time_unified_data[sensor_sets].shape[0] # First dimension is over sensors
    num_met_sensors = len(params['sensit_met_sensors'])
    met_sensor_names = params['sensit_met_sensors']
    location_dimension = 3
    met_velocities_dimension = 3
    time_dimension = unified_datetimes.shape[0]
    FMD_sensor_positions_latlon = sensor_positions_latlon[0]
    SPOD_sensor_positions_latlon = sensor_positions_latlon[1]
    print('num_ch4_sensors=', num_ch4_sensors)
    print('location_dimension=', location_dimension)
    print('met_velocities_dimension=', met_velocities_dimension)
    print('num_met_sensors=', num_met_sensors)
    print('time_dimension=', time_dimension)
    
    # find indices for the sensors that are meteorological (MET) sensors
    met_sensor_indices = [] # Start with empty list
    # Iterate over sensor names
    FMD_names = sensor_names[0]
    SPOD_names = sensor_names[1]
    for FMD_name_idx in range(len(FMD_names)): 
        this_sensor_name = FMD_names[FMD_name_idx]
        # Iterate over MET sensor names, looking for a match
        for met_sensor_name_idx in range(len(met_sensor_names)):
            this_met_sensor_name = met_sensor_names[met_sensor_name_idx]
            if (this_sensor_name == this_met_sensor_name):
                # Yep, this sensor name is in the list of MET sensors, so add it's index to the list of indices.
                met_sensor_indices.append(FMD_name_idx)
    print('met_sensor_indices=', met_sensor_indices)
    
    print('site map dataset=', site_map_dataset)
    # Grab the x,y and lat,lon values of the bottom plane (kDim=0)
    xPos_site = site_map_dataset.xPos.values[0,:,:]
    yPos_site = site_map_dataset.yPos.values[0,:,:]
    zPos_site = site_map_dataset.zPos.values[0,:,:] # Does using the bottom plane for this make sense?  Or is there a way to 
                                                    # find out which plane each sensor is at?
    lat_site = site_map_dataset.lat.values[:,:]
    lon_site = site_map_dataset.lon.values[:,:]

    # Get positions of MET sensors
    met_sensors_positions = np.zeros([num_met_sensors, location_dimension])
    for met_sensor_idx in range(num_met_sensors):
        ind_tt = met_sensor_indices[met_sensor_idx]
        lat_tt = FMD_sensor_positions_latlon[ind_tt][0]
        lon_tt = FMD_sensor_positions_latlon[ind_tt][1]
        diff_tt = np.sqrt(np.power(lat_site - lat_tt, 2.0) + np.power(lon_site - lon_tt, 2.0))
        ind_2d_tt = np.where(diff_tt == np.min(diff_tt, axis=(1,0)))
        print(f"met: min(diff_tt) = {np.min(diff_tt, axis=(1,0))}")
        print(f"met: (lat,lon) = ({FMD_sensor_positions_latlon[ind_tt][0]},{FMD_sensor_positions_latlon[ind_tt][1]})")
        print(f"closest i,j = {ind_2d_tt}")
        print('Setting x,y position of MET sensor to', xPos_site[ind_2d_tt][0], yPos_site[ind_2d_tt][0])
        print('Setting z position of MET sensor to 2.0 meters AGL')
        met_sensors_positions[met_sensor_idx,:] = [xPos_site[ind_2d_tt][0], yPos_site[ind_2d_tt][0], 2.0]
        
    # Get positions of CH4 (methane) sensors
    ch4_sensors_positions = np.zeros([num_ch4_sensors, location_dimension])
    for ch4_sensor_idx in range(num_ch4_sensors):
        if ch4_sensor_idx < len(FMD_names):
          lat_tt = FMD_sensor_positions_latlon[ch4_sensor_idx][0]
          lon_tt = FMD_sensor_positions_latlon[ch4_sensor_idx][1]
        else:
          lat_tt = SPOD_sensor_positions_latlon[ch4_sensor_idx-len(FMD_names)][0]
          lon_tt = SPOD_sensor_positions_latlon[ch4_sensor_idx-len(FMD_names)][1]
        diff_tt = np.sqrt(np.power(lat_site - lat_tt, 2.0) + np.power(lon_site-lon_tt, 2.0))
        ind_2d_tt = np.where(diff_tt==np.min(diff_tt, axis=(1,0)))
        print('Setting z position of CH4 sensor to 1.375 meters AGL')
        ch4_sensors_positions[ch4_sensor_idx,:] = [xPos_site[ind_2d_tt][0], yPos_site[ind_2d_tt][0], 1.375]
    
    # Get the meteorological (MET) sensor readings (wind velocities)
    met_velocities = np.zeros([num_met_sensors, met_velocities_dimension, time_dimension])
    for met_sensor_idx in range(num_met_sensors):
        ind_tt = met_sensor_indices[met_sensor_idx]
        met_velocities[met_sensor_idx,:,:] = time_unified_data[0][ind_tt, 1:4, :] #u,v,w in m/s
    
    # Get the methane sensor readings
    ch4_readings = np.zeros([num_ch4_sensors, time_dimension])
    for ch4_sensor_idx in range(num_ch4_sensors):
        if ch4_sensor_idx < len(FMD_names):
          ch4_readings[ch4_sensor_idx, :] = time_unified_data[0][ch4_sensor_idx, 0, :] * 1e-6 # PPM to kg/kg
        else:
          ch4_readings[ch4_sensor_idx, :] = time_unified_data[1][ch4_sensor_idx-len(FMD_names), 0, :] * 1e-6 # PPM to kg/kg
        
    # Create the dataset object and put the data into it
    seals_std_dataset = xr.Dataset()
    seals_std_dataset['time'] = xr.DataArray(unified_datetimes, dims=['time'])
    seals_std_dataset['metPos'] = xr.DataArray(met_sensors_positions, dims=['metSensors', 'locDim'])
    seals_std_dataset['metPos'].attrs['units'] = 'm'
    seals_std_dataset['metVels'] = xr.DataArray(met_velocities, dims=['metSensors', 'metVelsDim', 'time'])
    seals_std_dataset['metVels'].attrs['units'] = 'm/s'
    seals_std_dataset['CH4Pos'] = xr.DataArray(ch4_sensors_positions, dims=['CH4Sensors', 'locDim'])
    seals_std_dataset['CH4Pos'].attrs['units'] = 'm'
    seals_std_dataset['q_CH4'] = xr.DataArray(ch4_readings, dims=['CH4Sensors', 'time'])
    seals_std_dataset['q_CH4'].attrs['units'] = 'kg/kg'
    sensor_names = FMD_names+SPOD_names
    seals_std_dataset['CH4SensorsName'] = xr.DataArray(sensor_names, dims=['CH4Sensors'])
    seals_std_dataset['metSensorsName'] = xr.DataArray([FMD_names[idx] for idx in met_sensor_indices], dims=['metSensors'])
    
    print('seals_std_dataset=', seals_std_dataset)
    
    # Dump the dataset into a NetCDF file (if configured to)
    # For now anyway, ALWAYS save it, as the seals-ml method specific_site_data_generation reads it from file
    #if params['save_std_netcdf']:
    netcdf_out_path = params['std_netcdf_write_path']
    seals_std_dataset.to_netcdf(netcdf_out_path)
    
    return seals_std_dataset, netcdf_out_path

