# 2d-Pyclaw-Euler-Datagen
This repo shows an example of using the pyclaw 2d Euler solver to generate a dataset of either continuous (density field) or discrete (shock location) shock bubble interaction data.

The data generation code is in the 'datagen.ipynb' file, which shows an example of how to create a config object and pass it to the data generation class. By default, this creates 100 observation frames (can be changed in SBIConfig.py'. 

# File Structure
Alongside all the .ipynb and .py files (these should all be in the same folder/level of the file system), create a folder called 'Data' and a subfolder 'WarmStart' in 'Data', all data should be generated inside 'WarmStart', within folders named 'run_{number}'.
