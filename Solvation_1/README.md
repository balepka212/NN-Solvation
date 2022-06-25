# Comparative study of various molecular representation for intermolecular interaction predictions
In this project we compare a number of molecular representations(vectorizers) 
to determine what is the most suitable way to represent a molecule as vector when intermolecular interactions 
are at most interest. In this study we use solvation energy as a target value and solvent and solute molecules as input.
The data is obtained from [MNSol Database](https://comp.chem.umn.edu/mnsol/). 
#### To MSU AI

## Training
The training data is written to Runs folder and the results are stored in Run_results including losses plot, normalization parameters, run_log and comments. The links to each result folder are presented [below](#table-with-solvent-solute-experiment-links)

#### Neural Networks
All training files are presented in [Training_files](Training_files) in the format Solvent_Solute_NN.
#### Kernel Ridge Regression
[KRR_training](Training_files/000_Consequent_KRR.py) - all KRR experiments are sequentially carried out in this file

## Other experiments
Experiments on another datasets ([Acree](https://doi.org/10.6084/m9.figshare.1572326.v1) and [FreeSolv](https://doi.org/10.1007/s10822-014-9747-x)), G<sub>solv</sub> distribution and feature permutation importance are presented in following [Jupyter Notebook](Examples/Foreign_Datasets_Feature_Permutations.ipynb).

# Repository structure
### [config.py](config.py)
A file with some useful function used along all the project.
### [Training_files](Training_files)
A folder with .py files each of which trains the network with some parameters. All KRR training is in 000_Consequent_KRR.py.
### [Jupyter_examples](Jupyter_examples)
A folder with .ipynb files that dublicate training files.
### [my_nets](my_nets)
A package with some .py files to create and train networks

&nbsp; &nbsp; [Create_dataset](my_nets/Create_dataset.py) - A file that contains functions to create dataset using given vectorizers

&nbsp; &nbsp; [net_func](my_nets/net_func.py) - A file that contains functions train network and other useful functions

&nbsp; &nbsp; [LinearNet](my_nets/LinearNet.py) - A file that contains Linear Network used for training


&nbsp; &nbsp; [ResNET](my_nets/ResNET.py)- A file that contains 1D ResNET used for training. The model is adopted from
https://github.com/hsd1503/resnet1d/blob/master/util.py
### [Vectorizers](Vectorizers)
A package vectorizers.py that contains vectorizers functions used in this project
### [Tables](Tables)
A folder with tables used for various functions and vectorizers
### [Preprocess](Preprocess)
A folder with some files used to prepare data (tables, dicts, ...)

# Vectorizers info
## Solute_TESA
taken from MNSol database calculated parameter of Total Exposed Surface Area. More info (putlink)
## Solvent_Macro_props
properties of solvent: nD, alpha, beta, gamma, epsilon, phi, psi.
## MorganFingerprints
calculated morgan fingerprints (putlink)
pip install rdkit-pypi


# Vectorizers
### Morgan
Morgan molecule fingerprints.

### Macro Props
Macroscopic parameters of solvent.

### TESA
Total Exposed Surface Area.

### BoB
Bag of Bonds.
scipy install problems solved here:
https://stackoverflow.com/a/69710042/13835675


# Table with Solvent-Solute Experiment links
## Kernel Ridge Regression

| Solvent➡️ <br/>⬇️Solute | Blank                                             | Class                                             | Macro                                              | Morgan                                               | JustBonds                                   | BoB                                           | BAT                                            | SOAP                                             |
|-------------------------|---------------------------------------------------|---------------------------------------------------|----------------------------------------------------|------------------------------------------------------|---------------------------------------------|-----------------------------------------------|------------------------------------------------|--------------------------------------------------|
| **Blank**               |                                                   | [Class Blank](Run_results/KRR/Class_Blank_KRR1)   | [Macro Blank](Run_results/KRR/Macro_Blank_KRR1)    | [Morgan Blank](Run_results/KRR/Morgan_Blank_KRR1)    | [JB Blank](Run_results/KRR/JB_Blank_KRR1)   | [BoB Blank](Run_results/KRR/BoB_Blank_KRR1)   | [BAT_Blank](Run_results/KRR/BAT_Blank_KRR1)    | [SOAP_Blank]( Run_results/KRR/SOAP_Blank_KRR1)   |
| **Class**               | [Blank Class](Run_results/KRR/Blank_Class_KRR1)   | [Class Class](Run_results/KRR/Class_Class_KRR1)   | [Macro Class](Run_results/KRR/Macro_Class_KRR1)    | [Morgan Class](Run_results/KRR/Morgan_Class_KRR1)    | [JB Class](Run_results/KRR/JB_Class_KRR1)   | [BoB Class](Run_results/KRR/BoB_Class_KRR1)   | [BAT_Class](Run_results/KRR/BAT_Class_KRR1)    | [SOAP_Class]( Run_results/KRR/SOAP_Class_KRR1)   |
| **TESA**                | [Blank TESA](Run_results/KRR/Blank_TESA_KRR1)     | [Class TESA](Run_results/KRR/Class_TESA_KRR1)     | [Macro TESA](Run_results/KRR/Macro_TESA_KRR1)      | [Morgan TESA](Run_results/KRR/Morgan_TESA_KRR1)      | [JB TESA](Run_results/KRR/JB_TESA_KRR1)     | [BoB TESA](Run_results/KRR/BoB_TESA_KRR1)     | [BAT_TESA](Run_results/KRR/BAT_TESA_KRR1)      | [SOAP_TESA]( Run_results/KRR/SOAP_TESA_KRR1)     |
| **Morgan**              | [Blank Morgan](Run_results/KRR/Blank_Morgan_KRR1) | [Class Morgan](Run_results/KRR/Class_Morgan_KRR1) | [Macro Morgan](Run_results/KRR/Macro_Morgan_KRR1)  | [Morgan Morgan](Run_results/KRR/Morgan_Morgan_KRR1)  | [JB Morgan](Run_results/KRR/JB_Morgan_KRR1) | [BoB Morgan](Run_results/KRR/BoB_Morgan_KRR1) | [BAT_Morgan](Run_results/KRR/BAT_Morgan_KRR1)  | [SOAP_Morgan](Run_results/KRR/SOAP_Morgan_KRR1)  |
| **JustBonds**           | [Blank JB](Run_results/KRR/Blank_JB_KRR1)         | [Class JB](Run_results/KRR/Class_JB_KRR1)         | [Macro JB](Run_results/KRR/Macro_JB_KRR1)          | [Morgan JB](Run_results/KRR/Morgan_JB_KRR1)          | [JB JB](Run_results/KRR/JB_JB_KRR1)         | [BoB JB](Run_results/KRR/BoB_JB_KRR1)         | [BAT JB](Run_results/KRR/BAT_JB_KRR1)          | [SOAP JB](Run_results/KRR/SOAP_JB_KRR1)          |
| **BoB**                 | [Blank BoB](Run_results/KRR/Blank_BoB_KRR1)       | [Class BoB](Run_results/KRR/Class_BoB_KRR1)       | [Macro BoB](Run_results/KRR/Macro_BoB_KRR1)        | [Morgan BoB](Run_results/KRR/Morgan_BoB_KRR1)        | [JB BoB](Run_results/KRR/JB_BoB_KRR1)       | [BoB BoB](Run_results/KRR/BoB_BoB_KRR1)       | [BAT_BoB](Run_results/KRR/BAT_BoB_KRR1)        | [SOAP_BoB](Run_results/KRR/SOAP_BoB_KRR1)        |
| **BAT**                 | [Blank BAT](Run_results/KRR/Blank_BAT_KRR1)       | [Class_BAT](Run_results/KRR/Class_BAT_KRR1)       | [Macro_BAT](Run_results/KRR/Macro_BAT_KRR1)        | [Morgan_BAT](Run_results/KRR/Morgan_BAT_KRR1)        | [JB BAT](Run_results/KRR/JB_BAT_KRR1)       | [BoB_BAT](Run_results/KRR/BoB_BAT_KRR1)       | [BAT_BAT](Run_results/KRR/BAT_BAT_KRR1)        | [SOAP_BAT](Run_results/KRR/SOAP_BAT_KRR1)        |
| **SOAP**                | [Blank SOAP](Run_results/KRR/Blank_SOAP_KRR1)     | [Class_SOAP](Run_results/KRR/Class_SOAP_KRR1)     | [Macro_SOAP](Run_results/KRR/Macro_SOAP_KRR1)      | [Morgan_SOAP](Run_results/KRR/Morgan_SOAP_KRR1)      | [JB SOAP](Run_results/KRR/JB_SOAP_KRR1)     | [BoB_SOAP](Run_results/KRR/BoB_SOAP_KRR1)     | [BAT_SOAP](Run_results/KRR/BAT_SOAP_KRR1)      | [SOAP_SOAP](Run_results/KRR/SOAP_SOAP_KRR1)      |



## Linear

| Solvent➡️ <br/>⬇️Solute | Blank                                                | Class                                                | Macro                                                      | Morgan                                                              | JustBonds                                              | BoB                                                    | BAT                                                    | SOAP                                                 |
|-------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------------|---------------------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|------------------------------------------------------|
| **Blank**               |                                                      | [Class Blank](Run_results/LinNet/Class_Blank_Lin1)   | [Macro Blank](Run_results/LinNet/Macro_Blank_Lin1)         | [Morgan Blank](Run_results/LinNet/Morgan_2_124_Blank_Lin1)          | [JB Blank](Run_results/LinNet/JustBonds_Blank_Lin1)    | [BoB Blank](Run_results/LinNet/BoB_Blank_Lin1)         | [BAT_Blank](Run_results/LinNet/BAT_Blank_Lin1)         | [SOAP_Blank]( Run_results/LinNet/SOAP_Blank_Lin1)    |
| **Class**               | [Blank Class](Run_results/LinNet/Blank_Class_Lin1)   | [Class Class](Run_results/LinNet/Class_Class_Lin1)   | [Macro Class](Run_results/LinNet/Macro_Class_Lin1)         | [Morgan Class](Run_results/LinNet/Morgan_Class_Lin1)                | [JB Class](Run_results/LinNet/JustBonds_Class_Lin1)    | [BoB Class](Run_results/LinNet/BoB_Class_Lin1)         | [BAT_Class](Run_results/LinNet/BAT_Class_Lin1)         | [SOAP_Class]( Run_results/LinNet/SOAP_Class_Lin1)    |
| **TESA**                | [Blank TESA](Run_results/LinNet/Blank_TESA_Lin1)     | [Class TESA](Run_results/LinNet/Class_TESA_Lin1)     | [Macro TESA](Run_results/LinNet/Macro_TESA_Lin1)           | [Morgan TESA](Run_results/LinNet/Morgan_2_124_TESA_Lin1)            | [JB TESA](Run_results/LinNet/JustBonds_TESA_Lin1)      | [BoB TESA](Run_results/LinNet/BoB_TESA_Lin1)           | [BAT_TESA](Run_results/LinNet/BAT_TESA_Lin1)           | [SOAP_TESA]( Run_results/LinNet/SOAP_TESA_Lin1)      |
| **Morgan**              | [Blank Morgan](Run_results/LinNet/Blank_Morgan_Lin1) | [Class Morgan](Run_results/LinNet/Class_Morgan_Lin1) | [Macro Morgan](Run_results/LinNet/Macro_Morgan_2_124_Lin1) | [Morgan Morgan](Run_results/LinNet/Morgan_2_124_Morgan_2_124_Lin1b) | [JB Morgan](Run_results/LinNet/JustBonds_Morgan_Lin1)  | [BoB Morgan](Run_results/LinNet/BoB_Morgan_2_124_Lin1) | [BAT_Morgan](Run_results/LinNet/BAT_Morgan_2_124_Lin2) | [SOAP_Morgan](Run_results/LinNet/SOAP_Morgan_Lin1)   |
| **JustBonds**           | [Blank JB](Run_results/LinNet/Blank_JustBonds_Lin1)  | [Class JB](Run_results/LinNet/Class_JustBonds_Lin1)  | [Macro JB](Run_results/LinNet/Macro_JustBonds_Lin1)        | [Morgan JB](Run_results/LinNet/Morgan_JustBonds_Lin1)               | [JB JB](Run_results/LinNet/JustBonds_JustBonds_Lin1)   | [BoB JB](Run_results/LinNet/BoB_JustBonds_Lin1)        | [BAT JB](Run_results/LinNet/BAT_JustBonds_Lin1)        | [SOAP JB](Run_results/LinNet/SOAP_JustBonds_Lin1)    |
| **BoB**                 | [Blank BoB](Run_results/LinNet/Blank_BoB_Lin1)       | [Class BoB](Run_results/LinNet/Class_BoB_Lin1)       | [Macro BoB](Run_results/LinNet/Macro_BoB_Lin1)             | [Morgan BoB](Run_results/LinNet/Morgan_BoB_Lin2)                    | [JB BoB](Run_results/LinNet/JustBonds_BoB_Lin1)        | [BoB BoB](Run_results/LinNet/BoB_BoB_Lin2)             | [BAT_BoB](Run_results/LinNet/BAT_BoB_Lin1)             | [SOAP_BoB](Run_results/LinNet/SOAP_BoB_Lin1)         |
| **BAT**                 | [Blank BAT](Run_results/LinNet/Blank_BAT_Lin1)       | [Class_BAT](Run_results/LinNet/Class_BAT_Lin1)       | [Macro_BAT](Run_results/LinNet/Macro_BAT_Lin1)             | [Morgan_BAT](Run_results/LinNet/Morgan_BAT_Lin1)                    | [JB BAT](Run_results/LinNet/JustBonds_BAT_Lin1)        | [BoB_BAT](Run_results/LinNet/BoB_BAT_Lin1)             | [BAT_BAT](Run_results/LinNet/BAT_BAT_Lin1)             | [SOAP_BAT](Run_results/LinNet/SOAP_BAT_Lin1)         |
| **SOAP**                | [Blank SOAP](Run_results/LinNet/Blank_SOAP_Lin1)     | [Class_SOAP](Run_results/LinNet/Class_SOAP_Lin1)     | [Macro_SOAP](Run_results/LinNet/Macro_SOAP_Lin1)           | [Morgan_SOAP](Run_results/LinNet/Morgan_SOAP_Lin1)                  | [JB SOAP](Run_results/LinNet/JustBonds_SOAP_Lin1)      | [BoB_SOAP](Run_results/LinNet/BoB_SOAP_Lin1)           | [BAT_SOAP](Run_results/LinNet/BAT_SOAP_Lin1)           | [SOAP_SOAP](Run_results/LinNet/SOAP_SOAP_Lin1)       |


## ResNET

| Solvent➡️ <br/>⬇️Solute | Blank                                                | Class                                                      | Macro                                                      | Morgan                                                             | JustBonds                                               | BoB                                                    | BAT                                                    | SOAP                                               |
|-------------------------|------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|--------------------------------------------------------------------|---------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|----------------------------------------------------|
| **Blank**               |                                                      | [Class Blank](Run_results/ResNet/Class_Blank_Res1)         | [Macro Blank](Run_results/ResNet/Macro_Blank_Res1)         | [Morgan Blank](Run_results/ResNet/Morgan_Blank_Res1)               | [JB Blank](Run_results/ResNet/JustBonds_Blank_Res1)     | [BoB Blank](Run_results/ResNet/BoB_Blank_Res1)         | [BAT Blank](Run_results/ResNet/BAT_Blank_Res1)         | [SOAP Blank](Run_results/ResNet/SOAP_Blank_Res1)   |
| **Class**               | [Blank Class](Run_results/ResNet/Blank_Class_Res1)   | [Class Class](Run_results/ResNet/Class_Class_Res1)         | [Macro Class](Run_results/ResNet/Macro_Class_Res1)         | [Morgan Class](Run_results/ResNet/Morgan_2_124_Class_Res1)         | [JB Class](Run_results/ResNet/JustBonds_Class_Res1)     | [BoB Class](Run_results/ResNet/BoB_Class_Res1)         | [BAT Class](Run_results/ResNet/BAT_Class_Res1)         | [SOAP Class](Run_results/ResNet/SOAP_Class_Res1)   |
| **TESA**                | [Blank TESA](Run_results/ResNet/Blank_TESA_Res1)     | [Class TESA](Run_results/ResNet/Class_TESA_Res1)           | [Macro TESA](Run_results/ResNet/Macro_TESA_Res1)           | [Morgan TESA](Run_results/ResNet/Morgan_2_124_TESA_Res1)           | [JB TESA](Run_results/ResNet/JustBonds_TESA_Res1)       | [BoB TESA](Run_results/ResNet/BoB_TESA_Res2)           | [BAT TESA](Run_results/ResNet/BAT_TESA_Res1)           | [SOAP TESA](Run_results/ResNet/SOAP_TESA_Res1)     |
| **Morgan**              | [Blank Morgan](Run_results/ResNet/Blank_Morgan_Res1) | [Class Morgan](Run_results/ResNet/Class_Morgan_2_124_Res1) | [Macro Morgan](Run_results/ResNet/Macro_Morgan_2_124_Res1) | [Morgan Morgan](Run_results/ResNet/Morgan_2_124_Morgan_2_124_Res1) | [JB Morgan](Run_results/ResNet/JustBonds_Morgan_Res1)   | [BoB Morgan](Run_results/ResNet/BoB_Morgan_2_124_Res2) | [BAT Morgan](Run_results/ResNet/BAT_Morgan_2_124_Res2) | [SOAP Morgan](Run_results/ResNet/SOAP_Morgan_Res1) |
| **JustBonds**           | [Blank JB](Run_results/ResNet/Blank_JustBonds_Res1)  | [Class JB](Run_results/ResNet/Class_JustBonds_Res1)        | [Macro JB](Run_results/ResNet/Macro_JustBonds_Res1)        | [Morgan JB](Run_results/ResNet/Morgan_JustBonds_Res1)              | [JB JB](Run_results/ResNet/JustBonds_JustBonds_Res1)    | [BoB JB](Run_results/ResNet/BoB_JustBonds_Res1)        | [BAT JB](Run_results/ResNet/BAT_JustBonds_Res1)        | [SOAP JB](Run_results/ResNet/SOAP_JB_Res1)         |
| **BoB**                 | [Blank BoB](Run_results/ResNet/Blank_BoB_Res1)       | [Class BoB](Run_results/ResNet/Class_BoB_Res1)             | [Macro BoB](Run_results/ResNet/Macro_BoB_Res1)             | [Morgan BoB](Run_results/ResNet/Morgan_BoB_Res2)                   | [JB BoB](Run_results/ResNet/JustBonds_BoB_Res1)         | [BoB BoB](Run_results/ResNet/BoB_BoB_Res2)             | [BAT BoB](Run_results/ResNet/BAT_BoB_Res1)             | [SOAP BoB](Run_results/ResNet/SOAP_BoB_Res1)       |
| **BAT**                 | [Blank BAT](Run_results/ResNet/Blank_BAT_Res1)       | [Class BAT](Run_results/ResNet/Class_BAT_Res1)             | [Macro_BAT](Run_results/ResNet/Macro_BAT_Res1)             | [Morgan_BAT](Run_results/ResNet/Morgan_BAT_Res1)                   | [JB BAT](Run_results/ResNet/JustBonds_BAT_Res1)         | [BoB BAT](Run_results/ResNet/BoB_BAT_Res1)             | [BAT BAT](Run_results/ResNet/BAT_BAT_Res1)             | [SOAP BAT](Run_results/ResNet/SOAP_BAT_Res1)       |
| **SOAP**                | [Blank SOAP](Run_results/ResNet/Blank_SOAP_Res1)     | [Class SOAP](Run_results/ResNet/Class_SOAP_Res1)           | [Macro_SOAP](Run_results/ResNet/Macro_SOAP_Res1)           | [Morgan_SOAP](Run_results/ResNet/Morgan_SOAP_Res1)                 | [JB SOAP](Run_results/ResNet/JustBonds_SOAP_Res1)       | [BoB SOAP](Run_results/ResNet/BoB_SOAP_Res1)           | [BAT SOAP](Run_results/ResNet/BAT_SOAP_Res1)           | [SOAP SOAP](Run_results/Bad/SOAP_SOAP_Res1)     |



# The End
