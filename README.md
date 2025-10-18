# 2d-Pyclaw-Euler-Datagen
This repo shows an example of using the pyclaw 2d Euler solver to generate a dataset of either continuous (density field) or discrete (shock location) shock bubble interaction data.

The data generation code is in the 'datagen.ipynb' file, which shows an example of how to create a config object and pass it to the data generation class. By default, this creates 100 observation frames (can be changed in SBIConfig.py'. 

# Requirements
numpy>=1.21.0//
scipy>=1.7.0
clawpack>=5.8.0
pathos>=0.3.0
h5py>=3.0.0
tqdm>=4.62.0
tensorboardX>=2.4
torch>=1.10.0
matplotlib>=3.3.0

The Euler-2d Solver comes from Clawpack, from the PyClaw PDE solver. The documentation can be found here: https://www.clawpack.org/pyclaw/

# File Structure
Alongside all the .ipynb and .py files (these should all be in the same folder/level of the file system), create a folder called 'Data' and a subfolder 'WarmStart' in 'Data', all data should be generated inside 'WarmStart', within folders named 'run_{number}'.
