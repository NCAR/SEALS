import os, sys
import seals_enums as enums
import seals_utils as utils
import ml_model
import readers
import seals_utils
import traceback
import xarray as xr

class Controller:
    """Class to control the running of pre-trained Machine Learning (ML) models of the seals-ml github repository."""
    
    def __init__(self, params):
        """Create the instance of the Controller.
        
        Save the parameters object as an instance variable.
        """
        
        self.params = params
    
    def run_archive(self):
        """Run the controller and ML model in archive mode.
        
        This is used to process specific time ranges of sensor data that has been collected in the past.
        """
        
        print('Running in ARCHIVE mode.')
        archive_data_form = self.params['archive_data_form']
        # Read in the site map
        site_map_fn = self.params['site_map_file']
        if not os.path.exists(site_map_fn):
            print('Site map file does not exist:', site_map_fn)
            return False # Meaning NOT successful
        if site_map_fn[-3:] == '.nc':
            # Instantiate a site map reader for netCDF
            try:
                site_map_reader = readers.SiteMapNcReader(site_map_fn)
                site_map = site_map_reader.get()
                print('site_map:')
                print(site_map)
            except:
                print('Problem reading in site map', site_map_fn)
                traceback.print_exc()
                return False # Meaning NOT successful
                  
        if not archive_data_form == "standard_inference_dataset":  #Steps are required to process the archive data inputs 
          # Read in the sensor readings
          sensor_type = enums.SensorType(self.params['sensor_type'])
          if sensor_type == enums.SensorType.SENSIT_ADED_I:
              # Instantiate a sensor readings reader for SENSIT_ADED_I
              try:
                  sensor_readings_reader = readers.sensor_readings_sensit_ADED_I_reader(self.params)
              except:
                  print('Problem reading in SENSIT-ADED_II observations from dir:', self.params['sensit_path'])
                  traceback.print_exc()
                  return False # Meaning NOT successful
              # Package the sensor data into a Dataset in the standard format used by the SEALS software
              standard_input_dataset, standard_input_dataset_path = seals_utils.package_ADED_I_data_for_SEALS(sensor_readings_reader.time_unified_data, 
                                                                                                              sensor_readings_reader.sensor_names, 
                                                                                                              self.params['sensit_met_sensors'], 
                                                                                                              site_map, 
                                                                                                              sensor_readings_reader.sensor_positions_latlon,
                                                                                                              sensor_readings_reader.datetimes,
                                                                                                              self.params)
          elif sensor_type == enums.SensorType.SENSIT_ADED_II:
              # Instantiate a sensor readings reader for SENSIT_ADED_II
              try:
                  sensor_readings_reader = readers.sensor_readings_sensit_ADED_II_reader(self.params)
              except:
                  print('Problem reading in SENSIT-ADED_II observations from dir:', self.params['sensit_path'])
                  traceback.print_exc()
                  return False # Meaning NOT successful
              # Package the sensor data into a Dataset in the standard format used by the SEALS software
              standard_input_dataset, standard_input_dataset_path = seals_utils.package_ADED_II_data_for_SEALS(sensor_readings_reader.time_unified_data,
                                                                                                              sensor_readings_reader.sensor_names,
                                                                                                              sensor_readings_reader.datetimes,
                                                                                                              site_map, 
                                                                                                              sensor_readings_reader.sensor_positions_latlon,
                                                                                                              self.params)
          else:
              sensor_readings_fn = self.params['sensor_readings_file']
              if not os.path.exists(sensor_readings_fn):
                  print('Sensor readings file does not exist:', sensor_readings_fn)
                  return False # Meaning NOT successful
              if sensor_readings_fn[-3:] == '.nc':
                  # Instantiate a sensor readings reader for netCDF
                  try:
                      sensor_readings_reader = readers.SensorReadingsNcReader(sensor_readings_fn)
                      sensor_readings = sensor_readings_reader.get()
                      print('sensor_readings:')
                      print(sensor_readings)
                  except:
                      print('Problem reading in sensor readings', sensor_readings_fn)
                      traceback.print_exc()
                      return False # Meaning NOT successful
          
        else:    #A standard_input_dataset already exists and can be read directly
          sensor_readings_file = self.params['sensor_readings_file']
          standard_input_dataset = xr.open_dataset(sensor_readings_file)
          standard_input_dataset_path = sensor_readings_file

        ##### Given a standard_inference_dataset setup and run a trained model on the inference input data
        # Instantiate a model object
        model_file = self.params['model_file']
        model = ml_model.MlModel(model_file)
        
        # Get the site dataset
        pot_leaks_file = self.params['pot_leaks_file']
        time_window_size = self.params['time_window_size']
       
        print('time_window_size=', time_window_size)
        
        if model.model_type == enums.ModelType.BLOCK_TRANSFORMER_LEAK_LOC or model.model_type == enums.ModelType.TRANSFORMER_LEAK_RATE:        
            # If using a block transformer type model, make sure time_window_size is divisible by block_size
            print('block_size=', model.block_size)
            if time_window_size % model.block_size != 0:
                time_window_size = round(time_window_size / model.block_size) * model.block_size

                print('WARNING: time_window_size is NOT divisible by block_size.  Changing it to:', time_window_size)
            else:
                print('time_window_size is divisible by block_size.  GOOD!')
        
        time_window_stride = self.params['time_window_stride']
        success, site_dataset = utils.generate_site_dataset(standard_input_dataset_path, site_map_fn, pot_leaks_file, time_window_size, time_window_stride)
        if not success:
            return False
        else:
            siteds_netcdf_write_path = self.params['siteds_netcdf_write_path']
            site_dataset.to_netcdf(siteds_netcdf_write_path)

        # The two scaler files are expected to be in the same directory as the model file.
        model_dir, tail = os.path.split(self.params['model_file'])
        coord_scaler_file = os.path.join(model_dir, 'coord_scaler.json')
        sensor_scaler_file = os.path.join(model_dir, 'sensor_scaler.json')
        # The model specs file is also expected to be in the same directory as the model file.
        model_specs_file = os.path.join(model_dir, 'train.yml')
        # As is the model validation files CSV file.
        model_validation_files_path = os.path.join(model_dir, 'validation_files.csv')
        
        # Do pre-processing
        model.preprocess(site_dataset, coord_scaler_file, sensor_scaler_file, model_specs_file, model_validation_files_path)
        
        # Run the model
        ran_successfully = model.run()

        if ran_successfully:
            # Get the predictions
            self.location_predictions = model.get_location_predictions()
            self.rate_predictions = model.get_rate_predictions()
        
            # Produce whatever outputs are configured (not much to choose from at this point...)
            if self.params['csv_output']:
                self.write_csv_file(self.params['csv_location_output_filename'], self.location_predictions)
                if self.rate_predictions is not None: # May or may not exist depeding on the model type
                    self.write_csv_file(self.params['csv_rate_output_filename'], self.rate_predictions)
        
            return True
        else:
            return False
    
    def run_realtime(self):
        """Run the controller and ML model in real time mode.
        
        Intended future capability to run with sensor input stream(s) that are active in real time.
        """
        
        print('Running in REALTIME mode.')
        
        # To be added...
        
        return True
    
    def write_csv_file(self, output_fn, predictions):
        """Write the predictions generated by a run of a ML model to a CSV file."""
        
        print('Creating CSV file:', output_fn)
        with open(output_fn, 'w') as csv_file:
            for row in predictions:
                row_items = row.flatten() # Flatten the array of arrays with only one element each to just an array of elements
                row = str(row_items[0])
                for idx in range(1, len(row_items)):
                    row += ', ' + str(row_items[idx])
                csv_file.write(row + '\n')
        
        
        
