***********************************
Running Multiple Archive Experiments
***********************************

The framework includes a script, RunArchiveExperiments.py, that facilitates running one or more "test suites", each of which is defined by a list of experiments in the form of a CSV file, a name, a pre-trained model file, and a data dir.  The script takes no parameters.  The
test suites are defined in a list at the top of the file, so it is intended that you edit that in your own copy.  The script uses two columns from the CSV file, tc_ExpStartDatetime and tc_ExpDurationHrs, to define the time frame of the experiment.  Those and the other list values that define the test suite are used to create a custom params file for each experiment, using seals.params.template and making substitutions of placeholders using sed.  Each experiment is run, with its output files going to a distinct folder.  Since the script takes no parameters, it is simply run as:

./RunArchiveExperiments.py
