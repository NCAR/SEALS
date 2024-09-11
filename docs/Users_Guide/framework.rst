*********************
Running the Framework
*********************

1. Clone the seals-ml and SEALS repositories into a directory of you choice.
2. Create and activate the "seals" environment

* module load conda           (this may or may not be needed for your environment)
* mamba env create -f seals-ml/environment_gpu.yml
* conda activate seals

3. Edit SEALS/seals.params to reflect your environment, the model you want to run, etc.  See Section 2. Parameters File for details.
4. Run main program, giving the params file as an argument:

* In the SEALS sub-dir:
* ./seals.py -f seals.params
* You should see some fairly verbose output regarding the progress of the run.
