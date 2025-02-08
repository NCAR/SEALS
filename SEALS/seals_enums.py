from enum import Enum

class Mode(str, Enum):
    """Enumeration to specify the run mode."""
    
    ARCHIVE = 'ARCHIVE'
    REALTIME = 'REALTIME'

# An enumeration to specify the model type
class ModelType(str, Enum):
    """Enumeration to specify the model type."""
    
    #NEURALNET = 'NEURALNET' # Is this the one that needs potential locations as input?  Assume so for now.
    #TRANSFORMER = 'TRANSFORMER'
    BACKTRACKER = 'backtracker'
    BLOCK_TRANSFORMER_LEAK_LOC = 'block_transformer_leak_loc'
    TRANSFORMER_LEAK_RATE = 'transformer_leak_rate'

class SensorType(str, Enum):
    """Enumeration to specify the type of sensor input data."""
    
    SENSIT_ADED_I = 'SENSIT_ADED_I' # A directory is needed for this case.
    SENSIT_ADED_II = 'SENSIT_ADED_II' # A directory is needed for this case.
    NETCDF = 'NETCDF' # A netCDF file is needed for this case.
