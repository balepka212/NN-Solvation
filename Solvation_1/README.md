# Comparative study of various molecular representation for intermolecular interaction predictions
In this project we compare a number of molecular representations(vectorizers) 
to determine what is the most suitable way to represent a molecule as vector when intermolecular interactions 
are at most interest. In this study we use solvation energy as a target value and solvent and solute molecules as input.
The data is obtained from MNSol Database **putlink**.

# Repository structure
## [config.py](config.py)
A file with some useful function used along all the project.
## [Training_files](Trainig_files)
A folder with .py files each of which trains the network with some parameters.
## [Jupyter_examples](Jupyter_examples)
A folder with .ipynb files that dublicate training files.
## [my_nets](my_nets)
A package with some .py files to create and train networks
### &nbsp; &nbsp; [Create_dataset](my_nets/Create_dataset.py)
&nbsp; &nbsp; &nbsp; A file that contains functions to create dataset using given vectorizers
### &nbsp; &nbsp; [net_func](my_nets/net_func.py)
&nbsp; &nbsp; &nbsp; A file that contains functions train network and other useful functions
### &nbsp; &nbsp; [LinearNet](my_nets/LinearNet.py)
&nbsp; &nbsp; &nbsp; A file that contains Linear Network used for training
### &nbsp; &nbsp; [ResNET](my_nets/ResNET.py)
&nbsp; &nbsp; &nbsp; A file that contains 1D ResNET used for training.
https://github.com/hsd1503/resnet1d/blob/master/util.py
## [Vectorizers](Vectorizers)
A package vectorizers.py that contains vectorizers functions used in this project
## [Tables](Tables)
A folder with tables used for various functions and vectorizers
## [Preprocess](Preprocess)
A folder with some files used to prepare data (tables, dicts, ...)

# Vectorizers info
## Solute_TESA
taken from MNSol database calculated parameter of Total Exposed Surface Area. More info (putlink)
## Solvent_Macro_props
properties of solvent: nD, alpha, beta, gamma, epsilon, phi, psi.
## MorganFingerprints
calculated morgan fingerprints (putlink)
# Table with Solvent-Solute Experiment links
## Linear

| Solvent➡️ <br/>⬇️Solute | Macro                                               | Morgan                                                | BoB                                             | BAT                                                         | SOAP | SLATM |
|-------------------------|-----------------------------------------------------|-------------------------------------------------------|-------------------------------------------------|-------------------------------------------------------------|------|-------|
| TESA                    | [Macro TESA](Run_results/Macro_TESA_Lin1)           | [Morgan TESA](Run_results/Morgan_2_124_TESA_Lin1)     | [BoB TESA](Run_results/BoB_TESA_Lin1)           | [BAT_TESA](Solvation_1/Run_results/BAT_TESA_Lin1)           |      |       |
| Classification          |                                                     |                                                       |                                                 |                                                             |      |       |
| Morgan                  | [Macro Morgan](Run_results/Macro_Morgan_2_124_Lin1) | [Morgan Morgan](Run_results/Macro_Morgan_2_124_Lin1b) | [BoB Morgan](Run_results/BoB_Morgan_2_124_Lin1) | [BAT_Morgan](Solvation_1/Run_results/BAT_Morgan_2_124_Lin2) |      |       |
| BoB                     | [Macro BoB](Run_results/Macro_BoB_Lin1)             | [Morgan BoB](Run_results/Macro_BoB_Lin2)              | [BoB BoB](Run_results/BoB_BoB_Lin2)             | [BAT_BoB](Solvation_1/Run_results/BAT_BoB_Lin1)             |      |       |
| BAT                     | [Macro_BAT](Solvation_1/Run_results/Macro_BAT_Lin1) | [Morgan_BAT](Solvation_1/Run_results/Morgan_BAT_Lin1) | [BoB_BAT](Solvation_1/Run_results/BoB_BAT_Lin1) | [BAT_BAT](Solvation_1/Run_results/BAT_BAT_Lin1)             |      |       |
| SOAP                    |                                                     |                                                       |                                                 |                                                             |      |       |
| SLATM                   |                                                     |                                                       |                                                 |                                                             |      |       |

## ResNET

| Solvent➡️ <br/>⬇️Solute | Macro                                               | Morgan                                               | BoB                                             | BAT                                                        | SOAP | SLATM |
|-------------------------|-----------------------------------------------------|------------------------------------------------------|-------------------------------------------------|------------------------------------------------------------|------|-------|
| TESA                    | [Macro TESA](Run_results/Macro_TESA_Res3)           | [Morgan TESA](Run_results/Morgan_2_124_TESA_Res1)    | [BoB TESA](Run_results/BoB_TESA_Res2)           | [BAT_TESA](Solvation_1/Run_results/BAT_TESA_Res1)          |      |       |
| Classification          |                                                     |                                                      |                                                 |                                                            |      |       |
| Morgan                  | [Macro Morgan](Run_results/Macro_Morgan_2_124_Res1) | [Morgan Morgan](Run_results/Macro_Morgan_2_124_Res1) | [BoB Morgan](Run_results/BoB_Morgan_2_124_Res2) | [BAT_Morgan](Solvation_1/Run_results/BAT_Morgan_2_124_Res2)|      |       |
| BoB                     | [Macro BoB](Run_results/Macro_BoB_Res1)             | [Morgan BoB](Run_results/Morgan_BoB_Res2)            | [BoB BoB](Run_results/BoB_BoB_Res2)             | [BAT_BoB](Solvation_1/Run_results/BAT_BoB_Res1)            |      |       |
| BAT                     | [Macro_BAT](Run_results/Macro_BAT_Res1)             | [Morgan_BAT](Run_results/Macro_BAT_Res1)             | [BoB_BAT](TBD)                                  | [BAT_BAT]                                                  |      |       |
| SOAP                    |                                                     |                                                      |                                                 |                                                            |      |       |
| SLATM                   |                                                     |                                                      |                                                 |                                                            |      |       |


# Vectorizers
### Morgan
Morgan molecule fingerprints.
pip install rdkit-pypi

### Macro Props
Macroscopic parameters of solvent.

### TESA
Total Exposed Surface Area.

### BoB
Bag of Bonds.
scipy install problems solved here:
https://stackoverflow.com/a/69710042/13835675



# The End
