# Comparative study of various molecular representation for intermolecular interaction predictions
In this project we compare a number of molecular representations(vectorizers) 
to determine what is the most suitable way to represent a molecule as vector when intermolecular interactions 
are at most interest. In this study we use solvation energy as a target value and solvent and solute molecules as input.
The data is obtained from MNSol Database **putlink**.

# Repository structure
## [config.py](Solvation_1/config.py)
A file with some useful function used along all the project.
## [Training_files](Solvation_1/Trainig_files)
A folder with .py files each of which trains the network with some parameters.
## [Jupyter_examples](Solvation_1/Jupyter_examples)
A folder with .ipynb files that dublicate training files.
## [my_nets](Solvation_1/my_nets)
A package with some .py files to create and train networks
### &nbsp; &nbsp; [Create_dataset](Solvation_1/my_nets/Create_dataset.py)
&nbsp; &nbsp; &nbsp; A file that contains functions to create dataset using given vectorizers
### &nbsp; &nbsp; [net_func](Solvation_1/my_nets/net_func.py)
&nbsp; &nbsp; &nbsp; A file that contains functions train network and other useful functions
### &nbsp; &nbsp; [LinearNet](Solvation_1/my_nets/LinearNet.py)
&nbsp; &nbsp; &nbsp; A file that contains Linear Network used for training
### &nbsp; &nbsp; [ResNET](Solvation_1/my_nets/ResNET.py)
&nbsp; &nbsp; &nbsp; A file that contains 1D ResNET used for training
## [Vectorizers](Solvation_1/Vectorizers)
A package vectorizers.py that contains vectorizers functions used in this project
## [Tables](Solvation_1/Tables)
A folder with tables used for various functions and vectorizers
## [Preprocess](Solvation_1/Preprocess)
A folder with some files used to prepare data (tables, dicts, ...)

# Vectorizers info
## Solute_TESA
taken from MNSol database calculated parameter of Total Exposed Surface Area. More info (putlink)
## Solvent_Macro_props
properties of solvent: nD, alpha, beta, gamma, epsilon, phi, psi.
## MorganFingerprints
calculated morgan fingerprints (putlink)

# The End
