******
Output
******

There is currently only one option for output:  CSV.  For that, set the param csv_output to true. If set, a CSV file of predicted locations will be produced, and possibly also a CSV file of predicted flow rates, depending on the model used.  

There are two other files that are output, both of which are in netCDF format:

* The input dataset that is used by model is written to the path specified in the param std_netcdf_write_path.  This is an input parameter to sealsml.specific_site_data_generation.
* The site dataset, which is the ouput of sealsml.specific_site_data_generation, is written to the path specified in the param siteds_netcdf_write_path.
