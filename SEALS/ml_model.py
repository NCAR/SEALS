import seals_enums as enums
from sealsml.keras import models # The actual Machine Learning models in the seals-m repository
from keras.models import load_model
from bridgescaler import load_scaler
import os
import pandas as pd
from sealsml.data import Preprocessor
from sealsml.backtrack import backtrack_preprocess, backtrack_scaleDataTuple, backtrack_unscaleDataTuple, mapPredlocsToClosestPL
import sealsml
import xarray as xr
import yaml
import traceback
import numpy as np

class MlModel:
    """Class to encapsulate the Machine Learning (ML) model."""
    
    def __init__(self, model_file_path):
        """Create the MlModel object, load the configured ML pre-trained model.
        
        This includes pre-processing and loading the scalers. 
        """
        
        # Load model
        print('Loading trained model from', model_file_path)
        self.model = load_model(model_file_path)
        if isinstance(self.model, sealsml.keras.models.BackTrackerDNN):
            print('It\'s of type sealsml.keras.models.BackTrackerDNN')
            self.model_type = enums.ModelType.BACKTRACKER
            self.model_file_path = model_file_path
            self.model_name = "backtracker"

        elif isinstance(self.model, sealsml.keras.models.LocalizedLeakRateBlockTransformer):
            print('It\'s of type sealsml.keras.models.LocalizedLeakRateBlockTransformer')
            self.model_type = enums.ModelType.BLOCK_TRANSFORMER_LEAK_LOC
            
            # Get the block size
            self.block_size = self.model.block_size
        elif isinstance(self.model, sealsml.keras.models.BlockTransformer):
            print('It\'s of type sealsml.keras.models.BlockTransformer')
            self.model_type = enums.ModelType.TRANSFORMER_LEAK_RATE
            
            # Get the block size
            self.block_size = self.model.block_size
        
            
    def preprocess(self, site_dataset, coord_scaler_file_path, sensor_scalers_file_path, model_specs_file_path, validation_files_path):
        """Do pre-processing.  
        
        This includes loading the scalers. 
        """
        
        print('Pre-processing...')
            
        # Instantiate a Preprocessor
        preprocessor = Preprocessor()
            
        if self.model_type == enums.ModelType.BACKTRACKER:

            #Load the config parameters
            with open(model_specs_file_path) as config_file:
                config = yaml.safe_load(config_file)
            # backtrack preprocess the data stream inputs
            x, speed, L_scale, H_scale, n_samples, n_pot_leaks, self.x_pot_leaks, self.y_pot_leaks, self.z_pot_leaks = backtrack_preprocess(site_dataset, **config[self.model_name]["preprocess"])
            print(self.x_pot_leaks)
            print(self.y_pot_leaks)

            # load the backtrack inputs scaler and scale the input data
            btin_scaler_path = f"{self.model_file_path.split('backtracker')[0]}bt_input_scaler.json"
            print(f"Loading scaler: {btin_scaler_path}")
            scaler = load_scaler(btin_scaler_path)
            (self.scaled_encoder,), scaler = backtrack_scaleDataTuple((x,), scaler=scaler)

        elif self.model_type == enums.ModelType.TRANSFORMER_LEAK_RATE or       \
             self.model_type == enums.ModelType.BLOCK_TRANSFORMER_LEAK_LOC:
            
            # Load the scaler files in to the Preprocessor
            preprocessor.load_scalers(coord_scaler_file_path, sensor_scalers_file_path)

            # Do the preprocessing
            self.scaled_encoder, self.scaled_decoder, self.encoder_mask, self.decoder_mask = preprocessor.preprocess(site_dataset['encoder_input'],
                                                                                                                     site_dataset['decoder_input'],
                                                                                                                     fit_scaler=False)
            
            
            # Some more asserts to make sure things are in order
            assert self.scaled_encoder.shape[:4] == site_dataset['encoder_input'].shape[:4]
            assert self.scaled_decoder.shape[:3] == site_dataset['decoder_input'].squeeze().shape[:3]
            assert self.encoder_mask.shape == (site_dataset['encoder_input'].shape[0], site_dataset['encoder_input'].shape[1])

            print('Encoder/decoder asserts are passing.')
            
            
    def run(self): # Returns True if successful, False if not
        """Run the ML model.
        
        Returns successful indicator (True or False).
        """
        
        # Run the model for predictions
        print('Running the model...')
        if self.model_type == enums.ModelType.BACKTRACKER:
          try:
            self.predictions = self.model.predict((self.scaled_encoder,)) # , batch_size=1024)
            btout_scaler_path = f"{self.model_file_path.split('backtracker')[0]}bt_output_scaler.json"
            print(f"Loading scaler: {btout_scaler_path}")
            scaler_y = load_scaler(btout_scaler_path)
            (self.predictions,) = backtrack_unscaleDataTuple((self.predictions,), scaler_y)
            #Map predicted locations to the closest pot. leak
            self.predictions = mapPredlocsToClosestPL(self.predictions, self.x_pot_leaks, self.y_pot_leaks, self.z_pot_leaks)
            print('predictions:')
            print(self.predictions)
            print('model run done.')
            return True # Successful
            
          except:
            print('Exception in model.predict():')
            traceback.print_exc()
            return False
        
        elif self.model_type == enums.ModelType.TRANSFORMER_LEAK_RATE or       \
             self.model_type == enums.ModelType.BLOCK_TRANSFORMER_LEAK_LOC:
          try:
            self.predictions = self.model.predict([self.scaled_encoder, self.scaled_decoder[..., 0:5], self.encoder_mask, self.decoder_mask]) # , batch_size=1024)
            # Could test model type here instead... as for block_transformer_leak_loc, predictions will be a tuple, but for
            # transformer_leak_rate it will be a numpy ndarray.
            if type(self.predictions) == tuple:
                print('predictions is a tuple of length', len(self.predictions))
                print('types:')
                for idx in range(len(self.predictions)):
                    print(type(self.predictions[idx]))
            else:
                print('predictions is of type', type(self.predictions))
                print('predictions.shape=', self.predictions.shape)
                
            print('predictions:')
            print(self.predictions)
            print('model run done.')
            return True # Successful
            
          except:
            print('Exception in model.predict():')
            traceback.print_exc()
            return False
        
            
    def get_location_predictions(self):
        """Get the predictions of the location.
        
        Called after running the model to return the resulting location predictions.        
        """
        
        if self.predictions is None:
            return
        # predictions might be a numpy ndarray (locations) or a tuple of them (location, rate)
        elif type(self.predictions) == tuple:
            return self.predictions[0]
        else:
            return self.predictions
            
    def get_rate_predictions(self):
        """Get the predictions of the leak rate.
        
        Called after running the model to return the resulting leak rate predictions.        
        """
        
        if self.predictions is None:
            return
        # predictions might be a numpy ndarray (location) or a tuple of them (location, rate)
        # BlockTransformer models just return location
        # LocalizedLeakRateBlockTransformer models return the tuple of location and rate
        elif type(self.predictions) == tuple:
            return self.predictions[1]
        else:
            return None

    def windRelCartToOrigCartesian(x, y, z, mean_wd, x0, y0, z0):
       # x0 -- original cartesian x-coordinate of the reference (met) sensor
       # y0 -- original cartesian y-coordinate of the reference (met) sensor
       # z0 -- original cartesian z-coordinate of the reference (met) sensor
       x_orig = x*np.cos(mean_wd) - y*np.sin(mean_wd) + x0
       y_orig = x*np.sin(mean_wd) + y*np.cos(mean_wd) + y0
       z_orig = z + z0
