***************
Parameters File
***************

The parameters file is in JSON format.  The repository includes an example called seals.params that can be used as a starting point, but will need to heavily modifified for your particular paths, files, etc.  The parameters are:

* mode -  "ARCHIVE" or "REALTIME", currently only ARCHIVE is supported.  REALTIME is a potential future capability.
* site_map_file - a netCDF file containing a 3D grid of the site.
* sensor_type - specifies the type of sensor input file:  "SENSIT" or "NETCDF".
* sensit_path - the path of the directory containing the SENSIT data. The directory contains daily subdirectories by the name "YYMMDD data" (yes, with a space), each containing multiple FMD_Data_Exportxxxxxxxxxxxxxxxxxxxxx.csv files, one file per device. Required when sensor_type is SENSIT. 
* sensit_hour_offset - time offset, in hours, to add to raw sensit data. Required when sensor_type is SENSIT.
* sensit_vars_interpolated - list of sensit variable names to be interpolated when unifying the time coordinates of the SENSIT readings. Required when sensor_type is SENSIT.
* sensit_met_sensors - list (often just one) of which of the SENSIT sensors has meteorological data (sensor ID(s)). Required when sensor_type is SENSIT.
* sensit_data_start_datetime - the datetime of the start of the period of SENSIT data to be processed. Required when sensor_type is SENSIT.
* sensit_data_duration - the duration of the period of SENSIT data to be processed.  The sum of this and sensit_data_start_datetime would give you the end datetime. Required when sensor_type is SENSIT.
* sensit_time_increment - the time increment, in seconds, by which to unify the time coordiantes of the SENSIT readings. Required when sensor_type is SENSIT.
* time_window_size - the size of the time window, in seconds, to use for either of the two Block Transformer models.  It should divisible by the block_size (part of the model).  If not, the framework software will adjust it (with warning) to the nearest multiple of block_size. Required for either of the Block Transformer models.
* time_window_stride - the window_stride value used in the call to the specific_site_data_generation function of sealsml.staticinference, which defines it as: stride width to slide window through time (index based).
* sensor_readings_file - the path of the sensor readings netCDF file when the sensor_type is NETCDF (not SENSIT) 
* model_file - the path to the .keras model file to be run.
* pot_leaks_file - the path to the netCDF file of potential leak locations to be evaluated.
* csv_output - whether or not to produce CSV output of the results of the run.  true or false.  This is currently the only output option.
* csv_location_output_filename - the path/filename of the file to be written with leak location information, when csv_output is true.
* csv_rate_output_filename - the path/filename of the file to be written with leak location information, when csv_output is true.  This may or may not be produced, depending on the model type.
* std_netcdf_write_path - the path/filename of the file to be written with the SEALS standard dataset needed by the specific_site_data_generation function of sealsml.staticinference.py, which defines it as: The dataset to be processed.
* siteds_netcdf_write_path - the path/filename of the file to be written with the site_dataset, which is the netCDF output of the specific_site_data_generation function of sealsml.staticinference.py, which defines it as: Encoder and Decoder in an xarray dataset.
