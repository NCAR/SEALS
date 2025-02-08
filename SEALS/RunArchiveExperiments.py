#!/usr/bin/env python3
'''
This script runs seals-ml experiments from one or more "test suites", each of which is defined by a list of 
experiments in the form of a CSV file, a name, a pre-trained model file, and a SENSIT data dir.

The CSV file is read, and each line is read for start datetime and duration.  The experiment is then run 
for that time span.  A customized params file is created for each experiment, and the CSV file of results
and log for each are saved.
'''
import os
import pandas as pd
import time

# Each test suite is a list of name, CSV file of experiments, model file, and SENSIT data directory.

ADEDI = True
if ADEDI:
    test_suites = [['1leakExp',
                'PATH_TO_ARCHIVE_EMISSIONS_METADATA/SENSIT_1leakExp_metadata_ADED1.csv', 
                'PATH_TO_TRAINED_MODEL/modelFile.keras',
                'PATH_TO_ARCHIVE_DATA/']
              ]
else:
    test_suites = [['1leakExp',
                'PATH_TO_ARCHIVE_EMISSIONS_METADATA/SENSIT_1leakExp_metadata_ADED1.csv', 
                'PATH_TO_TRAINED_MODEL/modelFile.keras',
                'PATH_TO_ARCHIVE_DATA/']
              ]


params_template_file = './seals.params.template'
params_files_dir = './test_suites_params'
logs_dir =         './test_suites_logs'
if ADEDI:
   results_dir =      'PATH_TO_WRITE_OUTPUT/'
else:
   results_dir =      'PATH_TO_WRITE_OUTPUT/'

def escape_slashes(path):
    """Escape slashes with backslashes so the path can be used in a sed command."""
    
    escaped_path = ''
    for ch in path:
        if ch == '/':
            escaped_path += '\/'
        else:
            escaped_path += ch
    return escaped_path

def create_dir_if_doesnt_exist(dir_name):
    """Create the directory if it doesn't already exist.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def create_experiment_params_file(suite_name, idx, model_file, sensit_dir, start_datetime, duration):
    """Create a params file for this specific test suite and index into its experiments file."""
    
    params_path = os.path.join(params_files_dir, suite_name + '_' + str(idx) + '.params')
    
    # Copy the template params file to this specific experiment's name
    command = 'cp ' + params_template_file + ' ' + params_path
    print(command)
    rc = os.system(command)
    if rc != 0:
        print('Failed to copy params template file to specific experiment instance!')
        return -1, None
        
    # Make substitutions in the params file    
    command = "sed -i 's/MODEL_FILE/" + escape_slashes(model_file) + "/g' " + params_path
    print(command)
    rc = os.system(command)
    if rc != 0:
        print('Failed to replace placeholder MODEL_FILE!')
        return -1, None
    
    command = "sed -i 's/SENSIT_PATH/" + escape_slashes(sensit_dir) + "/g' " + params_path
    rc = os.system(command)
    if rc != 0:
        print('Failed to replace placeholder SENSIT_PATH!')
        return -1, None
    
    command = "sed -i 's/START_DATETIME/" + start_datetime + "/g' " + params_path
    rc = os.system(command)
    if rc != 0:
        print('Failed to replace placeholder START_DATETIME!')
        return -1, None
    
    command = "sed -i 's/DURATION/" + str(duration) + "/g' " + params_path
    rc = os.system(command)
    if rc != 0:
        print('Failed to replace placeholder DURATION!')
        return -1, None
    
    output_path = os.path.join(results_dir, suite_name + '_' + str(idx))
    command = "sed -i 's/OUTPUT_PATH/" + escape_slashes(output_path) + "/g' " + params_path
    rc = os.system(command)
    if rc != 0:
        print('Failed to replace placeholder OUTPUT_PATH!')
        return -1
    
    create_dir_if_doesnt_exist(output_path)
    
    return 0, params_path
    

if __name__ == '__main__':
    """Run the experiments in the test_suites (may be only one test_suite).  
    
    Iterate over test suites, then experiments in it's CSV file.  Customize the params file for each, and
    save predictions (CSV file) and log for each.
    """

    for test_suite in test_suites:
        suite_name = test_suite[0]
        experiments_csv_file = test_suite[1]
        model_file = test_suite[2]
        sensit_dir = test_suite[3]
    
        # Use pandas to read the CSV file of experiments
        experiments = pd.read_csv(experiments_csv_file)
        print(experiments)
        num_experiments = experiments.shape[0]
    
        # Extract start datetime and durations columns (as lists)
        if ADEDI:
          start_datetime_strings = experiments['tc_ExpStartDatetime']
          duration_strings = experiments['tc_ExpDurationHrs']
        else:  # ADED_II
          experiments['UTCStart'] = pd.to_datetime(experiments['UTCStart'], format='%Y-%m-%d_%H:%M:%S')
          experiments['UTCEnd'] = pd.to_datetime(experiments['UTCEnd'], format='%Y-%m-%d_%H:%M:%S')  
          
    
        print('experiments_csv_file:', experiments_csv_file)
        print('model_file:', model_file)
        print('sensit_dir:', sensit_dir)
    
        failed_model_runs = 0
    
        for idx in range(num_experiments):
            if ADEDI:
              start_datetime_str = start_datetime_strings[idx]
              # Get rid of the +00:00 at the end
              start_datetime_str = start_datetime_str[:start_datetime_str.find('+')]
              duration_str = duration_strings[idx]
            else:
               start_datetime = experiments['UTCStart'].iloc[idx]
               end_datetime = experiments['UTCEnd'].iloc[idx]
               duration = pd.Timedelta(end_datetime-start_datetime,unit='h')
               print(f"start: {start_datetime}")
               print(f"end: {end_datetime}")
               print(f"dur: {duration}")
               duration_str = f"{duration.seconds/3600.00}"
               start_datetime_str = f"{start_datetime}"
    
            print('   idx:', idx)
            print('   start_datetime_str:', start_datetime_str)
            print('   duration_str:', duration_str)
            
            rc, params_file = create_experiment_params_file(suite_name, idx, model_file, sensit_dir, start_datetime_str, duration_str)
            log_file = os.path.join(logs_dir, suite_name + '_' + str(idx) + '.log')
            if rc == 0:
                print('Run the experiment...')
                command = './seals.py -f ' + params_file + ' &> ' + log_file
                print(command)
                rc = os.system(command)
                if rc != 0:
                    print('Run of model FAILED!')
                    failed_model_runs += 1
            else:
                print('Failed to create params file, cannot run experiment!')
                failed_model_runs += 1
            
        if failed_model_runs == 0:
            print('All experiments rans successfully.')
        else:
            print(failed_model_runs, 'experiments FAILED!')
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
