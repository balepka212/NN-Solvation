# Comparative study of various molecular representation for intermolecular interaction predictions
In this project we compare a number of molecular representations(vectorizers) 
to determine what is the most suitable way to represent a molecule as vector when intermolecular interactions 
are at most interest. In this study we use solvation energy as a target value and solvent and solute molecules as input.
The data is obtained from [MNSol Database](https://comp.chem.umn.edu/mnsol/). 
## Reproduce the results in figures
### Figure 2-4, Figures S1-S6. RMS scored on MNSol subsets.
To plot the RMS sublots run [plot_bar_charts.py](Preprocess/plot_bar_charts.py)
### Figure 5-8. Distributions and best models True-Pred on subsets
These figures are reproduced in following Jupyter Notebook: [Solvatum_models](Examples/Solvatum_models.ipynb)
### Figure 9. Feature Permutation Importance of KRR-JB-JB 
These figures are reproduced in following Jupyter Notebook: [Feature_Permutation_Importance](Examples/Feature_Permutation_Importance.ipynb)



## Load a particular model
To load a model trained on MNSol main subset use `get_trained_model`
```
from my_nets.net_func import get_trained_model

# choose solvent from ('blank', 'class', 'macro', 'macrox', 'morgan', 'mor2to20', 'jb', 'bob', 'bat', 'soap')
# choose solute from = ('blank', 'class', 'tesa', 'comp', 'morgan', 'mor2to20', 'jb', 'bob',  'bat', 'soap')

solvent = class
solute = jb

krr_model = get_trained_model('krr', solvent, solute, kernel='laplacian')
lin_model = get_trained_model('lin', solvent, solute)
res_model = get_trained_model('res', solvent, solute)
```
To implement trained JB models from this work on your dataset the bag sizes should be cropped. 
This feature is not available in the original [chemreps](https://github.com/chemreps/chemreps) package, 
yet [here](Tables/chemreps) are two modified files to be put in chemreps folder. 
To prepare your own data please refer to [this file](Preprocess/Solvatum_create_JB_dict.py)

## Training and Results
The training data is written to Runs folder (if you manually train a network) and the results are stored in Run_results
(due to large file sizes Run_results is available for manual download from 
[Yandex Disk](https://disk.yandex.com.tr/d/u9PkE_9iZovDiw)) including losses plot, 
normalization parameters, run_log and comments. The links to each result folder are presented
[below](#table-with-solvent-solute-experiment-links)

#### Neural Networks
All training files are presented in [Training_files](Training_files) in the format Solvent_Solute_NN.

#### Kernel Ridge Regression
[KRR_training](Training_files/000_Consequent_KRR.py) - all KRR experiments are sequentially carried out in this file. 
The results are available at

# Repository structure
### [config.py](config.py)
A file with some useful function used along all the project.
### [Training_files](Training_files)
A folder with .py files each of which trains the network with some parameters. All KRR training is in
[one file](Training_files/000_Consequent_KRR.py).

### [my_nets](my_nets)
A package with some .py files to create and train networks

&nbsp; &nbsp; [Create_dataset](my_nets/Create_dataset.py) - A file that contains functions to create dataset using given vectorizers

&nbsp; &nbsp; [net_func](my_nets/net_func.py) - A file that contains functions train network and other useful functions

&nbsp; &nbsp; [LinearNet](my_nets/LinearNet.py) - A file that contains Linear Network used for training

&nbsp; &nbsp; [ResNET](my_nets/ResNET.py)- A file that contains 1D ResNET used for training. The model is adopted from
[hsd1503](https://github.com/hsd1503/resnet1d/blob/master/util.py)
### [Vectorizers](Vectorizers)
A package vectorizers.py that contains vectorizers functions used in this project
### [Tables](Tables)
A folder with tables used for various functions and vectorizers
### [Preprocess](Preprocess)
A folder with some files used to prepare data (tables, dicts, ...)

## Vectorizers description {_tensor length_}
### Blank {_1_}
Vector of length 1 and value of 0 (vector = {0, }). 
We use it to represent solutes or solvents in reference experiments 
not to pass the model any information about the compound.

### Class {_83_}
Three layer classification, described in [MNSol Database](https://comp.chem.umn.edu/mnsol/). 

### Comp {_6_}
Some properties, calculated in gas-phase prior to SMD calculation, namely \
* μ, D (dipole moment)\
* dX, Å (size along the X axis)\
* dY, Å (size along the Y axis)\
* dZ, Å (size along the Z axis)\
* dE, (difference between LUMO and HOMO energies, calculated in gas-phase prior
to SMD calculation)\
* Molar mass, u

### TESA {_9_}
Only for solutes. 
Calculated by authors of the MNSol database total exposed surface area of molecule accessible to a solvent, 
based on overlapping atomic spheres. 
The accessible area for each type of atom in the compound is presented in the corresponding position. 
Detailed information present in [this paper](https://doi.org/10.1002/jcc.540160405). 

### Macro {_7_}
Only for solvents. 
Macroscopic properties of solvent taken from the MNSol database (sometimes called Abraham descriptors).
Namely: nD, alpha, beta, gamma, epsilon, phi, psi. Described in [MNSol Database](https://comp.chem.umn.edu/mnsol/).

### MacroX {_14_}

Combined Macro, Comp, and densities value for solvents. Density, g/cm3 (Obtained from [PubChem](https://pubchem.org)) 


### Morgan {_124_}
Circular fingerprints denoting molecular structure. 
The maximum radius of the substructure is 2, the maximum hash size is 124, 
thus many collisions of submolecules are present. 
In this method, the compound is represented by encoding the subsets of atoms presented in this molecule
calculated morgan fingerprints bit vector, described 
[here](https://towardsdatascience.com/a-practical-introduction-to-the-use-of-molecular-fingerprints-in-drug-discovery-7f15021be2b1)

If troubles with installation try

`pip install rdkit-pypi`

### Mor2to20 {_1441_}
Morgan with a radius of 2 and hash size of 2<sup>20</sup>. 
This gives a lot of sparse vectors, which were further reduced by deleting positions that have zeros for every compound in the dataset. Thus, reducing the length of the vector from 1048576 to 1441 with no collisions.

### BoB {_3098_}
Bag of Bonds.
Returns bag of bonds for a given compound. 
For each pair of atoms, a parameter derived from an interatomic distance is generated
(z<sub>i</sub>•z<sub>j</sub>/r<sub>ij</sub> for different atoms and 0.5•z<sup>2.4</sup> for a single atom)
and added to bag. 

scipy install problems solved here:
https://stackoverflow.com/a/69710042/13835675

### JB {_282_}
JustBonds. Bag of Bonds for bonded atoms only

On additional datasets modified JB was used. 
As some of the bags are larger than those in MNSol the bags were cropped to MNSol sizes. 
The closest atoms were put into the vector and the furthest were eliminated to match the bag sizes.

### BAT {_4558_}
Bonds-Angles-Torsions. 
BoB with the addition of values derived from angles of 3 consequently bonded atoms 
and torsions of 4 consequently bonded atoms

### SOAP {_4704_}
SOAP represents the local environment around a central atom by 
gaussian-smeared neighbour atom positions made rotationally invariant. 
The concept of this descriptor is described [here](https://doi.org/10.1103/PhysRevB.87.184115). 
We used dscribe.descriptors.SOAP python 
[package](https://github.com/SINGROUP/dscribe/tree/master/dscribe) to create molecule vectors.


# Table with Solvent-Solute Experiment links
Only works if Runs_folder is manually downloaded from [Yandex Disk](https://disk.yandex.com.tr/d/u9PkE_9iZovDiw)
## Kernel Ridge Regression

| Solvent➡️ <br/>⬇️Solute | Blank                                                 | Class                                                          | Macro                                                          | MacroX                                                               | Morgan                                                                 | Mor2to20                                                                   | JustBonds                                                | BoB                                                        | BAT                                                        | SOAP                                                         |
|-------------------------|-------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------|--------------------------------------------------------------------------|------------------------------------------------------------------------|-------------------------------------------------------------------------------|----------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|--------------------------------------------------------------|
| **Blank**               |                                                       | [Class Blank](Run_results/KRR/Class_Blank_KRR)                | [Macro Blank](Run_results/KRR/Macro_Blank_KRR)                | [MacroX Blank](Run_results/KRR/MacroX_Blank_KRR2)                | [Morgan Blank](Run_results/KRR/Morgan_Blank_KRR)                      | [Mor2to20 Blank](Run_results/KRR/Mor2to20_Blank_KRR2)                | [JB Blank](Run_results/KRR/JB_Blank_KRR)                | [BoB Blank](Run_results/KRR/BoB_Blank_KRR)                | [BAT_Blank](Run_results/KRR/BAT_Blank_KRR)                | [SOAP_Blank]( Run_results/KRR/SOAP_Blank_KRR)               |
| **Class**               | [Blank Class](Run_results/KRR/Blank_Class_KRR)        | [Class Class](Run_results/KRR/Class_Class_KRR)                | [Macro Class](Run_results/KRR/Macro_Class_KRR)                | [MacroX Class](Run_results/KRR/MacroX_Class_KRR2)                | [Morgan Class](Run_results/KRR/Morgan_Class_KRR)                      | [Mor2to20 Class](Run_results/KRR/Mor2to20_Class_KRR2)                | [JB Class](Run_results/KRR/JB_Class_KRR)                | [BoB Class](Run_results/KRR/BoB_Class_KRR)                | [BAT_Class](Run_results/KRR/BAT_Class_KRR)                | [SOAP_Class]( Run_results/KRR/SOAP_Class_KRR)               |
| **Comp**                | [Blank Comp](Run_results/KRR/Blank_Comp_KRR)          | [Class Comp](Run_results/KRR/Class_Comp_KRR2)                  | [Macro Comp](Run_results/KRR/Macro_Comp_KRR2)                  | [MacroX Comp](Run_results/KRR/MacroX_Comp_KRR2)                  | [Morgan Comp](Run_results/KRR/Morgan_Comp_KRR2)                        | [Mor2to20 Comp](Run_results/KRR/Mor2to20_Comp_KRR2)                  | [JB Comp](Run_results/KRR/JB_Comp_KRR2)                  | [BoB Comp](Run_results/KRR/BoB_Comp_KRR2)                  | [BAT Comp](Run_results/KRR/BAT_Comp_KRR2)                  | [SOAP Comp](Run_results/KRR/SOAP_Comp_KRR2)                  |
| **TESA**                | [Blank TESA](Run_results/KRR/Blank_TESA_KRR)          | [Class TESA](Run_results/KRR/Class_TESA_KRR)                  | [Macro TESA](Run_results/KRR/Macro_TESA_KRR)                  | [MacroX TESA](Run_results/KRR/MacroX_TESA_KRR2)                  | [Morgan TESA](Run_results/KRR/Morgan_TESA_KRR)                        | [Mor2to20 TESA](Run_results/KRR/Mor2to20_TESA_KRR2)                  | [JB TESA](Run_results/KRR/JB_TESA_KRR)                  | [BoB TESA](Run_results/KRR/BoB_TESA_KRR)                  | [BAT_TESA](Run_results/KRR/BAT_TESA_KRR)                  | [SOAP_TESA]( Run_results/KRR/SOAP_TESA_KRR)                 |
| **Morgan**              | [Blank Morgan](Run_results/KRR/Blank_Morgan_KRR)      | [Class Morgan](Run_results/KRR/Class_Morgan_KRR)              | [Macro Morgan](Run_results/KRR/Macro_Morgan_KRR)              | [MacroX Morgan](Run_results/KRR/MacroX_Morgan_KRR2)              | [Morgan Morgan](Run_results/KRR/Morgan_Morgan_KRR)                    | [Mor2to20 Morgan](Run_results/KRR/Mor2to20_Morgan_KRR2)              | [JB Morgan](Run_results/KRR/JB_Morgan_KRR)              | [BoB Morgan](Run_results/KRR/BoB_Morgan_KRR)              | [BAT_Morgan](Run_results/KRR/BAT_Morgan_KRR)              | [SOAP_Morgan](Run_results/KRR/SOAP_Morgan_KRR)              |
| **Mor2to20**         | [Blank Mor2to20](Run_results/KRR/Blank_Mor2to20_KRR) | [Class Mor2to20](Run_results/KRR/Class_Mor2to20_KRR2) | [Macro Mor2to20](Run_results/KRR/Macro_Mor2to20_KRR2) | [MacroX Mor2to20](Run_results/KRR/MacroX_Mor2to20_KRR2) | [Morgan Mor2to20](Run_results/KRR/Morgan_Mor2to20_KRR2) | [Mor2to20 Mor2to20](Run_results/KRR/Mor2to20_Mor2to20_KRR2) | [JB Mor2to20](Run_results/KRR/JB_Mor2to20_KRR2) | [BoB Mor2to20](Run_results/KRR/BoB_Mor2to20_KRR2) | [BAT Mor2to20](Run_results/KRR/BAT_Mor2to20_KRR2) | [SOAP Mor2to20](Run_results/KRR/SOAP_Mor2to20_KRR2) |                                                      |                                                             |                                                            |                                                                 |                                                                    |                                                                           |                                                          |                                                        |                                                        |                                                      |
| **JustBonds**           | [Blank JB](Run_results/KRR/Blank_JB_KRR)              | [Class JB](Run_results/KRR/Class_JB_KRR)                      | [Macro JB](Run_results/KRR/Macro_JB_KRR)                      | [MacroX JustBonds](Run_results/KRR/MacroX_JB_KRR2)               | [Morgan JB](Run_results/KRR/Morgan_JB_KRR)                            | [Mor2to20 JB](Run_results/KRR/Mor2to20_JB_KRR2)                      | [JB JB](Run_results/KRR/JB_JB_KRR)                      | [BoB JB](Run_results/KRR/BoB_JB_KRR)                      | [BAT JB](Run_results/KRR/BAT_JB_KRR)                      | [SOAP JB](Run_results/KRR/SOAP_JB_KRR)                      |
| **BoB**                 | [Blank BoB](Run_results/KRR/Blank_BoB_KRR)            | [Class BoB](Run_results/KRR/Class_BoB_KRR)                    | [Macro BoB](Run_results/KRR/Macro_BoB_KRR)                    | [MacroX BoB](Run_results/KRR/MacroX_BoB_KRR2)                    | [Morgan BoB](Run_results/KRR/Morgan_BoB_KRR)                          | [Mor2to20 BoB](Run_results/KRR/Mor2to20_BoB_KRR2)                    | [JB BoB](Run_results/KRR/JB_BoB_KRR)                    | [BoB BoB](Run_results/KRR/BoB_BoB_KRR)                    | [BAT_BoB](Run_results/KRR/BAT_BoB_KRR)                    | [SOAP_BoB](Run_results/KRR/SOAP_BoB_KRR)                    |
| **BAT**                 | [Blank BAT](Run_results/KRR/Blank_BAT_KRR)            | [Class_BAT](Run_results/KRR/Class_BAT_KRR)                    | [Macro_BAT](Run_results/KRR/Macro_BAT_KRR)                    | [MacroX BAT](Run_results/KRR/MacroX_BAT_KRR2)                    | [Morgan_BAT](Run_results/KRR/Morgan_BAT_KRR)                          | [Mor2to20 BAT](Run_results/KRR/Mor2to20_BAT_KRR2)                    | [JB BAT](Run_results/KRR/JB_BAT_KRR)                    | [BoB_BAT](Run_results/KRR/BoB_BAT_KRR)                    | [BAT_BAT](Run_results/KRR/BAT_BAT_KRR)                    | [SOAP_BAT](Run_results/KRR/SOAP_BAT_KRR)                    |
| **SOAP**                | [Blank SOAP](Run_results/KRR/Blank_SOAP_KRR)          | [Class_SOAP](Run_results/KRR/Class_SOAP_KRR)                  | [Macro_SOAP](Run_results/KRR/Macro_SOAP_KRR)                  | [MacroX SOAP](Run_results/KRR/MacroX_SOAP_KRR2)                  | [Morgan_SOAP](Run_results/KRR/Morgan_SOAP_KRR)                        | [Mor2to20 SOAP](Run_results/KRR/Mor2to20_SOAP_KRR2)                  | [JB SOAP](Run_results/KRR/JB_SOAP_KRR)                  | [BoB_SOAP](Run_results/KRR/BoB_SOAP_KRR)                  | [BAT_SOAP](Run_results/KRR/BAT_SOAP_KRR)                  | [SOAP_SOAP](Run_results/KRR/SOAP_SOAP_KRR)                  |



## Linear

| Solvent➡️ <br/>⬇️Solute | Blank                                                  | Class                                                             | Macro                                                             | MacroX                                                                  | Morgan                                                                    | Mor2to20                                                                      | JustBonds                                                          | BoB                                                           | BAT                                                           | SOAP                                                            |
|-------------------------|--------------------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------------|--------------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|-----------------------------------------------------------------|
| **Blank**               |                                                        | [Class Blank](Run_results/LinNet/Class_Blank_Lin)                | [Macro Blank](Run_results/LinNet/Macro_Blank_Lin)                | [MacroX Blank](Run_results/LinNet/MacroX_Blank_Lin)                | [Morgan Blank](Run_results/LinNet/Morgan_Blank_Lin)                | [Mor2to20 Blank](Run_results/LinNet/Mor2to20_Blank_Lin)                | [JB Blank](Run_results/LinNet/JB_Blank_Lin)                | [BoB Blank](Run_results/LinNet/BoB_Blank_Lin)                | [BAT_Blank](Run_results/LinNet/BAT_Blank_Lin)                | [SOAP_Blank]( Run_results/LinNet/SOAP_Blank_Lin)               |
| **Class**               | [Blank Class](Run_results/LinNet/Blank_Class_Lin)      | [Class Class](Run_results/LinNet/Class_Class_Lin)                | [Macro Class](Run_results/LinNet/Macro_Class_Lin)                | [MacroX Class](Run_results/LinNet/MacroX_Class_Lin)                | [Morgan Class](Run_results/LinNet/Morgan_Class_Lin)                      | [Mor2to20 Class](Run_results/LinNet/Mor2to20_Class_Lin)                | [JB Class](Run_results/LinNet/JB_Class_Lin)                | [BoB Class](Run_results/LinNet/BoB_Class_Lin)                | [BAT_Class](Run_results/LinNet/BAT_Class_Lin)                | [SOAP_Class]( Run_results/LinNet/SOAP_Class_Lin)               |
| **Comp**                | [Blank Comp](Run_results/LinNet/Blank_Comp_Lin)        | [Class Comp](Run_results/LinNet/Class_Comp_Lin)          | [Macro Comp](Run_results/LinNet/Macro_Comp_Lin)              | [MacroX Comp](Run_results/LinNet/MacroX_Comp_Lin)              | [Morgan Comp](Run_results/LinNet/Morgan_Comp_Lin)              | [Mor2to20 Comp](Run_results/LinNet/Mor2to20_Comp_Lin)              | [JB Comp](Run_results/LinNet/JB_Comp_Lin)              | [BoB Comp](Run_results/LinNet/BoB_Comp_Lin)              | [BAT Comp](Run_results/LinNet/BAT_Comp_Lin)              | [SOAP Comp](Run_results/LinNet/SOAP_Comp_Lin)              |
| **TESA**                | [Blank TESA](Run_results/LinNet/Blank_TESA_Lin)        | [Class TESA](Run_results/LinNet/Class_TESA_Lin)                  | [Macro TESA](Run_results/LinNet/Macro_TESA_Lin)                  | [MacroX TESA](Run_results/LinNet/MacroX_TESA_Lin)                  | [Morgan TESA](Run_results/LinNet/Morgan_TESA_Lin)                  | [Mor2to20 TESA](Run_results/LinNet/Mor2to20_TESA_Lin)                  | [JB TESA](Run_results/LinNet/JB_TESA_Lin)                  | [BoB TESA](Run_results/LinNet/BoB_TESA_Lin)                  | [BAT_TESA](Run_results/LinNet/BAT_TESA_Lin)                  | [SOAP_TESA]( Run_results/LinNet/SOAP_TESA_Lin)                 |
| **Morgan**              | [Blank Morgan](Run_results/LinNet/Blank_Morgan_Lin)    | [Class Morgan](Run_results/LinNet/Class_Morgan_Lin)              | [Macro Morgan](Run_results/LinNet/Macro_Morgan_Lin)        | [MacroX Morgan](Run_results/LinNet/MacroX_Morgan_Lin)        | [Morgan Morgan](Run_results/LinNet/Morgan_Morgan_Linb)       | [Mor2to20 Morgan](Run_results/LinNet/Mor2to20_Morgan_Lin)        | [JB Morgan](Run_results/LinNet/JB_Morgan_Lin)              | [BoB Morgan](Run_results/LinNet/BoB_Morgan_Lin)        | [BAT_Morgan](Run_results/LinNet/BAT_Morgan_Lin2)        | [SOAP_Morgan](Run_results/LinNet/SOAP_Morgan_Lin)              |
| **Mor2to20**         | [Blank Mor2to20](Run_results/LinNet/Blank_Mor2to20_Lin) | [Class Mor2to20](Run_results/LinNet/Class_Mor2to20_Lin) | [Macro Mor2to20](Run_results/LinNet/Macro_Mor2to20_Lin) | [MacroX Mor2to20](Run_results/LinNet/MacroX_Mor2to20_Lin) | [Morgan Mor2to20](Run_results/LinNet/Morgan_Mor2to20_Lin) | [Mor2to20 Mor2to20](Run_results/LinNet/Mor2to20_Mor2to20_Lin) | [JB Mor2to20](Run_results/LinNet/JB_Mor2to20_Lin) | [BoB Mor2to20](Run_results/LinNet/BoB_Mor2to20_Lin) | [BAT Mor2to20](Run_results/LinNet/BAT_Mor2to20_Lin) | [SOAP Mor2to20](Run_results/LinNet/SOAP_Mor2to20_Lin) |                                                      |                                                             |                                                            |                                                                 |                                                                    |                                                                           |                                                          |                                                        |                                                        |                                                      |
| **JustBonds**           | [Blank JB](Run_results/LinNet/Blank_JB_Lin)            | [Class JB](Run_results/LinNet/Class_JB_Lin)               | [Macro JB](Run_results/LinNet/Macro_JB_Lin)               | [MacroX JustBonds](Run_results/LinNet/MacroX_JB_Lin)        | [Morgan JB](Run_results/LinNet/Morgan_JB_Lin)                     | [Mor2to20 JB](Run_results/LinNet/Mor2to20_JB_Lin)               | [JB JB](Run_results/LinNet/JB_JB_Lin)               | [BoB JB](Run_results/LinNet/BoB_JB_Lin)               | [BAT JB](Run_results/LinNet/BAT_JB_Lin)               | [SOAP JB](Run_results/LinNet/SOAP_JB_Lin)               |
| **BoB**                 | [Blank BoB](Run_results/LinNet/Blank_BoB_Lin)          | [Class BoB](Run_results/LinNet/Class_BoB_Lin)                    | [Macro BoB](Run_results/LinNet/Macro_BoB_Lin)                    | [MacroX BoB](Run_results/LinNet/MacroX_BoB_Lin)                    | [Morgan BoB](Run_results/LinNet/Morgan_BoB_Lin2)                          | [Mor2to20 BoB](Run_results/LinNet/Mor2to20_BoB_Lin)                    | [JB BoB](Run_results/LinNet/JB_BoB_Lin)                    | [BoB BoB](Run_results/LinNet/BoB_BoB_Lin2)                    | [BAT_BoB](Run_results/LinNet/BAT_BoB_Lin)                    | [SOAP_BoB](Run_results/LinNet/SOAP_BoB_Lin)                    |
| **BAT**                 | [Blank BAT](Run_results/LinNet/Blank_BAT_Lin)          | [Class_BAT](Run_results/LinNet/Class_BAT_Lin)                    | [Macro_BAT](Run_results/LinNet/Macro_BAT_Lin)                    | [MacroX BAT](Run_results/LinNet/MacroX_BAT_Lin)                    | [Morgan_BAT](Run_results/LinNet/Morgan_BAT_Lin)                          | [Mor2to20 BAT](Run_results/LinNet/Mor2to20_BAT_Lin)                    | [JB BAT](Run_results/LinNet/JB_BAT_Lin)                    | [BoB_BAT](Run_results/LinNet/BoB_BAT_Lin)                    | [BAT_BAT](Run_results/LinNet/BAT_BAT_Lin)                    | [SOAP_BAT](Run_results/LinNet/SOAP_BAT_Lin)                    |
| **SOAP**                | [Blank SOAP](Run_results/LinNet/Blank_SOAP_Lin)        | [Class_SOAP](Run_results/LinNet/Class_SOAP_Lin)                  | [Macro_SOAP](Run_results/LinNet/Macro_SOAP_Lin)                  | [MacroX SOAP](Run_results/LinNet/MacroX_SOAP_Lin)                  | [Morgan_SOAP](Run_results/LinNet/Morgan_SOAP_Lin)                        | [Mor2to20 SOAP](Run_results/LinNet/Mor2to20_SOAP_Lin)                  | [JB SOAP](Run_results/LinNet/JB_SOAP_Lin)                  | [BoB_SOAP](Run_results/LinNet/BoB_SOAP_Lin)                  | [BAT_SOAP](Run_results/LinNet/BAT_SOAP_Lin)                  | [SOAP_SOAP](Run_results/LinNet/SOAP_SOAP_Lin)                  |


## ResNET

| Solvent➡️ <br/>⬇️Solute | Blank                                                               | Class                                                               | Macro                                                                | MacroX                                                                  | Morgan                                                                     | Mor2to20                                                                      | JustBonds                                                          | BoB                                                            | BAT                                                            | SOAP                                                              |
|-------------------------|---------------------------------------------------------------------|---------------------------------------------------------------------|----------------------------------------------------------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------|-------------------------------------------------------------------|
| **Blank**               |                                                                     | [Class Blank](Run_results/ResNet/Class_Blank_Res)                  | [Macro Blank](Run_results/ResNet/Macro_Blank_Res)                   | [MacroX Blank](Run_results/ResNet/MacroX_Blank_Res)                | [Morgan Blank](Run_results/ResNet/Morgan_Blank_Res)                       | [Mor2to20 Blank](Run_results/ResNet/Mor2to20_Blank_Res)                | [JB Blank](Run_results/ResNet/JB_Blank_Res)                | [BoB Blank](Run_results/ResNet/BoB_Blank_Res)                 | [BAT Blank](Run_results/ResNet/BAT_Blank_Res)                 | [SOAP Blank](Run_results/ResNet/SOAP_Blank_Res)                  |
| **Class**               | [Blank Class](Run_results/ResNet/Blank_Class_Res)                  | [Class Class](Run_results/ResNet/Class_Class_Res)                  | [Macro Class](Run_results/ResNet/Macro_Class_Res)                   | [MacroX Class](Run_results/ResNet/MacroX_Class_Res)                | [Morgan Class](Run_results/ResNet/Morgan_Class_Res)                 | [Mor2to20 Class](Run_results/ResNet/Mor2to20_Class_Res)                | [JB Class](Run_results/ResNet/JB_Class_Res)                | [BoB Class](Run_results/ResNet/BoB_Class_Res)                 | [BAT Class](Run_results/ResNet/BAT_Class_Res)                 | [SOAP Class](Run_results/ResNet/SOAP_Class_Res)                  |
| **Comp**                | [Blank Comp](Run_results/ResNet/Blank_Comp_Res)                | [Class Comp](Run_results/ResNet/Class_Comp_Res)            | [Macro Comp](Run_results/ResNet/Macro_Comp_Res)                 | [MacroX Comp](Run_results/ResNet/MacroX_Comp_Res)              | [Morgan Comp](Run_results/ResNet/Morgan_Comp_Res)               | [Mor2to20 Comp](Run_results/ResNet/Mor2to20_Comp_Res)              | [JB Comp](Run_results/ResNet/JB_Comp_Res)              | [BoB Comp](Run_results/ResNet/BoB_Comp_Res)               | [BAT Comp](Run_results/ResNet/BAT_Comp_Res)               | [SOAP Comp](Run_results/ResNet/SOAP_Comp_Res)                |
| **TESA**                | [Blank TESA](Run_results/ResNet/Blank_TESA_Res)                    | [Class TESA](Run_results/ResNet/Class_TESA_Res)                    | [Macro TESA](Run_results/ResNet/Macro_TESA_Res)                     | [MacroX TESA](Run_results/ResNet/MacroX_TESA_Res)                  | [Morgan TESA](Run_results/ResNet/Morgan_TESA_Res)                   | [Mor2to20 TESA](Run_results/ResNet/Mor2to20_TESA_Res)                  | [JB TESA](Run_results/ResNet/JB_TESA_Res)                  | [BoB TESA](Run_results/ResNet/BoB_TESA_Res2)                   | [BAT TESA](Run_results/ResNet/BAT_TESA_Res)                   | [SOAP TESA](Run_results/ResNet/SOAP_TESA_Res)                    |
| **Morgan**              | [Blank Morgan](Run_results/ResNet/Blank_Morgan_Res)                | [Class Morgan](Run_results/ResNet/Class_Morgan_Res)          | [Macro Morgan](Run_results/ResNet/Macro_Morgan_Res)           | [MacroX Morgan](Run_results/ResNet/MacroX_Morgan_Res)        | [Morgan Morgan](Run_results/ResNet/Morgan_Morgan_Res)         | [Mor2to20 Morgan](Run_results/ResNet/Mor2to20_Morgan_Res)        | [JB Morgan](Run_results/ResNet/JB_Morgan_Res)              | [BoB Morgan](Run_results/ResNet/BoB_Morgan_Res2)         | [BAT Morgan](Run_results/ResNet/BAT_Morgan_Res2)         | [SOAP Morgan](Run_results/ResNet/SOAP_Morgan_Res)                |
| **Mor2to20**         | [Blank Mor2to20](Run_results/ResNet/Blank_Mor2to20_Res)   | [Class Mor2to20](Run_results/ResNet/Class_Mor2to20_Res)   | [Macro Mor2to20](Run_results/ResNet/Macro_Mor2to20_Res)    | [MacroX Mor2to20](Run_results/ResNet/MacroX_Mor2to20_Res) | [Morgan Mor2to20](Run_results/ResNet/Morgan_Mor2to20_Res)  | [Mor2to20 Mor2to20](Run_results/ResNet/Mor2to20_Mor2to20_Res) | [JB Mor2to20](Run_results/ResNet/JB_Mor2to20_Res) | [BoB Mor2to20](Run_results/ResNet/BoB_Mor2to20_Res)  | [BAT Mor2to20](Run_results/ResNet/BAT_Mor2to20_Res)  | [SOAP Mor2to20](Run_results/ResNet/SOAP_Mor2to20_Res)   |                                                      |                                                             |                                                            |                                                                 |                                                                    |                                                                           |                                                          |                                                        |                                                        |                                                      |
| **JustBonds**           | [Blank JB](Run_results/ResNet/Blank_JB_Res)                 | [Class JB](Run_results/ResNet/Class_JB_Res)                 | [Macro JB](Run_results/ResNet/Macro_JB_Res)                  | [MacroX JustBonds](Run_results/ResNet/MacroX_JB_Res)        | [Morgan JB](Run_results/ResNet/Morgan_JB_Res)                      | [Mor2to20 JB](Run_results/ResNet/Mor2to20_JB_Res)               | [JB JB](Run_results/ResNet/JB_JB_Res)               | [BoB JB](Run_results/ResNet/BoB_JB_Res)                | [BAT JB](Run_results/ResNet/BAT_JB_Res)                | [SOAP JB](Run_results/ResNet/SOAP_JB_Res)                        |
| **BoB**                 | [Blank BoB](Run_results/ResNet/Blank_BoB_Res)                      | [Class BoB](Run_results/ResNet/Class_BoB_Res)                      | [Macro BoB](Run_results/ResNet/Macro_BoB_Res)                       | [MacroX BoB](Run_results/ResNet/MacroX_BoB_Res)                    | [Morgan BoB](Run_results/ResNet/Morgan_BoB_Res2)                           | [Mor2to20 BoB](Run_results/ResNet/Mor2to20_BoB_Res)                    | [JB BoB](Run_results/ResNet/JB_BoB_Res)                    | [BoB BoB](Run_results/ResNet/BoB_BoB_Res2)                     | [BAT BoB](Run_results/ResNet/BAT_BoB_Res)                     | [SOAP BoB](Run_results/ResNet/SOAP_BoB_Res)                      |
| **BAT**                 | [Blank BAT](Run_results/ResNet/Blank_BAT_Res)                      | [Class BAT](Run_results/ResNet/Class_BAT_Res)                      | [Macro_BAT](Run_results/ResNet/Macro_BAT_Res)                       | [MacroX BAT](Run_results/ResNet/MacroX_BAT_Res)                    | [Morgan_BAT](Run_results/ResNet/Morgan_BAT_Res)                           | [Mor2to20 BAT](Run_results/ResNet/Mor2to20_BAT_Res)                    | [JB BAT](Run_results/ResNet/JB_BAT_Res)                    | [BoB BAT](Run_results/ResNet/BoB_BAT_Res)                     | [BAT BAT](Run_results/ResNet/BAT_BAT_Res)                     | [SOAP BAT](Run_results/ResNet/SOAP_BAT_Res)                      |
| **SOAP**                | [Blank SOAP](Run_results/ResNet/Blank_SOAP_Res)                    | [Class SOAP](Run_results/ResNet/Class_SOAP_Res)                    | [Macro_SOAP](Run_results/ResNet/Macro_SOAP_Res)                     | [MacroX SOAP](Run_results/ResNet/MacroX_SOAP_Res)                  | [Morgan_SOAP](Run_results/ResNet/Morgan_SOAP_Res)                         | [Mor2to20 SOAP](Run_results/ResNet/Mor2to20_SOAP_Res)                  | [JB SOAP](Run_results/ResNet/JB_SOAP_Res)                  | [BoB SOAP](Run_results/ResNet/BoB_SOAP_Res)                   | [BAT SOAP](Run_results/ResNet/BAT_SOAP_Res)                   | [SOAP SOAP](Run_results/Bad/SOAP_SOAP_Res)                       |



# The End
