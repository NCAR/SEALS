#!/usr/bin/env python3

import os, sys
import json
import argparse
import time
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from enum import Enum
import seals_enums as enums
import seals_controller

def parse_args():
    """ parse the command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="JSON file with SEALS configuration parameter settings")
    args = parser.parse_args()
    return args

################## main()
if __name__ == '__main__':
    
    ########################################
    ### Parse the command line arguments ###
    ########################################
    args = parse_args()
    
    ########################################################
    ### Read the JSON file of configuration parameters   ###
    ########################################################
    with open(args.file) as file:
        params = json.loads(file.read())

    # Print input parameters
    print('Input parameters:')
    for key,value in params.items():
        print(f"   {key}: {value}")

    # Get the run mode from the params file
    run_mode = enums.Mode(params['mode'])
    
    # Instantiate the Controller
    controller = seals_controller.Controller(params)
    
    # Run in the specified mode
    if run_mode == enums.Mode.ARCHIVE:
        successful = controller.run_archive()
    elif run_mode == enums.Mode.REALTIME:
        successful = controller.run_realtime()
    else:
        print('Unknown mode: ', run_mode)
        
    if not successful:
        print('Run failed.')
        
    print('Done.')
    
    
