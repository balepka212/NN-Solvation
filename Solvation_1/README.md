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

| Solvent➡️ <br/>⬇️Solute  | Blank                                         | Class                                               | Macro                                               | Morgan                                                | JustBonds                                      | BoB                                             | BAT                                                         | SOAP | SLATM |
|--------------------------|-----------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|-------------------------------------------------------|------------------------------------------------|-------------------------------------------------|-------------------------------------------------------------|------|-------|
| **Blank**                |                                               | [Class Blank](Run_results/Class_Blank_Lin1)         | [Macro Blank](Run_results/Macro_Blank_Lin1)         | [Morgan Blank](Run_results/Morgan_2_124_Blank_Lin1)   | [JB Blank](Run_results/JustBonds_Blank_Lin1)   | [BoB Blank](Run_results/BoB_Blank_Lin1)         | [BAT_Blank](Solvation_1/Run_results/BAT_Blank_Lin1)         |      |       |
| **Class**                | [Blank Class](Run_results/Blank_Class_Lin1)   | [Class Class](Run_results/Class_Class_Lin1)         | [Macro Class](Run_results/Macro_Class_Lin1)         | [Morgan Class](Run_results/Morgan_2_124_Class_Lin1)   | [JB Class](Run_results/JustBonds_Class_Lin1)   | [BoB Class](Run_results/BoB_Class_Lin1)         | [BAT_Class](Solvation_1/Run_results/BAT_Class_Lin1)         |      |       |
| **TESA**                 | [Blank TESA](Run_results/Blank_TESA_Lin1)     | [Class TESA](Run_results/Class_TESA_Lin1)           | [Macro TESA](Run_results/Macro_TESA_Lin1)           | [Morgan TESA](Run_results/Morgan_2_124_TESA_Lin1)     | [JB TESA](Run_results/JustBonds_TESA_Lin1)     | [BoB TESA](Run_results/BoB_TESA_Lin1)           | [BAT_TESA](Solvation_1/Run_results/BAT_TESA_Lin1)           |      |       |
| **Morgan**               | [Blank Morgan](Run_results/Blank_Morgan_Lin1) | [Class Morgan](Run_results/Class_Morgan_Lin1)       | [Macro Morgan](Run_results/Macro_Morgan_2_124_Lin1) | [Morgan Morgan](Run_results/Macro_Morgan_2_124_Lin1b) | [JB Morgan](Run_results/JustBonds_Morgan_Lin1) | [BoB Morgan](Run_results/BoB_Morgan_2_124_Lin1) | [BAT_Morgan](Solvation_1/Run_results/BAT_Morgan_2_124_Lin2) |      |       |
| **JustBonds**            | [Blank JB](Run_results/Blank_JustBonds_Lin1)  | [Class JB](Run_results/Class_JustBonds_Lin1)        | [Macro JB](Run_results/Macro_JustBonds_Lin1)        | [Morgan JB](Run_results/Morgan_JustBonds_Lin1)        | [JB JB](Run_results/JustBonds_JustBonds_Lin1)  | [BoB JB](Run_results/BoB_JustBonds_Lin1)        | [BAT JB](Run_results/BAT_JustBonds_Lin1)                    |      |       |
| **BoB**                  | [Blank BoB](Run_results/Blank_BoB_Lin1)       | [Class BoB](Run_results/Class_BoB_Lin1)             | [Macro BoB](Run_results/Macro_BoB_Lin1)             | [Morgan BoB](Run_results/Macro_BoB_Lin2)              | [JB BoB](Run_results/JustBonds_BoB_Lin1)       | [BoB BoB](Run_results/BoB_BoB_Lin2)             | [BAT_BoB](Solvation_1/Run_results/BAT_BoB_Lin1)             |      |       |
| **BAT**                  | [Blank BAT](Run_results/Blank_BAT_Lin1)       | [Class_BAT](Solvation_1/Run_results/Class_BAT_Lin1) | [Macro_BAT](Solvation_1/Run_results/Macro_BAT_Lin1) | [Morgan_BAT](Solvation_1/Run_results/Morgan_BAT_Lin1) | [JB BAT](Run_results/JustBonds_BAT_Lin1)       | [BoB_BAT](Solvation_1/Run_results/BoB_BAT_Lin1) | [BAT_BAT](Solvation_1/Run_results/BAT_BAT_Lin1)             |      |       |
| **SOAP**                 |                                               |                                                     |                                                     |                                                       |                                                |                                                 |                                                             |      |       |
| **SLATM**                |                                               |                                                     |                                                     |                                                       |                                                |                                                 |                                                             |      |       |

## ResNET

| Solvent➡️ <br/>⬇️Solute | Blank                                                | Class                                               | Macro                                               | Morgan                                               | JustBonds                                      | BoB                                             | BAT                                                         | SOAP | SLATM |
|------------------------|------------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|------------------------------------------------------|------------------------------------------------|-------------------------------------------------|-------------------------------------------------------------|------|-------|
| **Blank**              |                                                      | [Class Blank](Run_results/Class_Blank_Res1)         | [Macro Blank](Run_results/Macro_Blank_Res1)         | [Morgan Blank](Run_results/Morgan_2_124_Blank_Res1)  | [JB Blank](Run_results/JustBonds_Blank_Res1)   | [BoB Blank](Run_results/BoB_Blank_Res1)         | [BAT Blank](Run_results/BAT_Blank_Res1)                     |      |       |
| **Class**              | [Blank Class](Run_results/Blank_Class_Res1)          | [Class Class](Run_results/Class_Class_Res1)         | [Macro Class](Run_results/Macro_Class_Res1)         | [Morgan Class](Run_results/Morgan_2_124_Class_Res1)  | [JB Class](Run_results/JustBonds_Class_Res1)   | [BoB Class](Run_results/BoB_Class_Res1)         | [BAT_Class](Solvation_1/Run_results/BAT_Class_Res1)         |      |       |
| **TESA**               | [Blank TESA](Run_results/Blank_TESA_Res1)            | [Class TESA](Run_results/Class_TESA_Res1)           | [Macro TESA](Run_results/Macro_TESA_Res3)           | [Morgan TESA](Run_results/Morgan_2_124_TESA_Res1)    | [JB TESA](Run_results/JustBonds_TESA_Res1)     | [BoB TESA](Run_results/BoB_TESA_Res2)           | [BAT_TESA](Solvation_1/Run_results/BAT_TESA_Res1)           |      |       |
| **Morgan**             | [Blank Morgan](Run_results/Blank_Morgan_2_124_Res1)  | [Class Morgan](Run_results/Class_Morgan_Res1)       | [Macro Morgan](Run_results/Macro_Morgan_2_124_Res1) | [Morgan Morgan](Run_results/Macro_Morgan_2_124_Res1) | [JB Morgan](Run_results/JustBonds_Morgan_Res1) | [BoB Morgan](Run_results/BoB_Morgan_2_124_Res2) | [BAT_Morgan](Solvation_1/Run_results/BAT_Morgan_2_124_Res2) |      |       |
| **JustBonds**          | [Blank JB](Run_results/Blank_JustBonds_Res1)         | [Class JB](Run_results/Class_JustBonds_Res1)        | [Macro JB](Run_results/Macro_JustBonds_Res1)        | [Morgan JB](Run_results/Morgan_JustBonds_Res1)       | [JB JB](Run_results/JustBonds_JustBonds_Res1)  | [BoB JB](Run_results/Macro_JustBonds_Res1)      | [BAT JB](Run_results/Macro_JustBonds_Res1)                  |      |       |
| **BoB**                | [Blank BoB](Run_results/Blank_BoB_Res1)              | [Class BAT](Solvation_1/Run_results/Class_BoB_Res1) | [Macro BoB](Run_results/Macro_BoB_Res1)             | [Morgan BoB](Run_results/Morgan_BoB_Res2)            | [JB BoB](Run_results/JustBonds_BoB_Res1)       | [BoB BoB](Run_results/BoB_BoB_Res2)             | [BAT_BoB](Solvation_1/Run_results/BAT_BoB_Res1)             |      |       |
| **BAT**                | [Blank BAT](Run_results/Blank_BAT_Res1)              | [Class BAT](Solvation_1/Run_results/Class_BAT_Res1) | [Macro_BAT](Run_results/Macro_BAT_Res1)             | [Morgan_BAT](Run_results/Macro_BAT_Res1)             | [JB BAT](Run_results/JustBonds_BAT_Res1)       | [BoB_BAT](Run_results/BoB_BAT_Res1)             | [BAT_BAT](Solvation_1/Run_results/BAT_BAT_Res1)             |      |       |
| **SOAP**               |                                                      |                                                     |                                                     |                                                      |                                                |                                                 |                                                             |      |       |
| **SLATM**              |                                                      |                                                     |                                                     |                                                      |                                                |                                                 |                                                             |      |       |


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
