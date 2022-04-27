from collections import OrderedDict

from tqdm import tqdm

from Solvation_1.config import project_path
import chemreps
import pickle as pkl
from Solvation_1.Vectorizers.vectorizers import get_sdf_file
import torch
from rdkit import Chem


with open(project_path('Solvation_1/Tables/Solvents_Solutes.pkl'), 'rb') as f:
    Solvents, Solutes = pkl.load(f)

with open('/Users/balepka/PycharmProjects/msuAI/Solvation_1/Tables/MNSol_bags4.pkl', 'rb') as f:
    my_bags = pkl.load(f)

BoB_bags, BoB_sizes = my_bags[0].copy()

BoB_data = {}
for compound in Solvents+Solutes:
    print(compound)
    mol_path = get_sdf_file(compound)
    BoB_array = chemreps.bag_of_bonds.bag_of_bonds(project_path(mol_path), BoB_bags, BoB_sizes)
    out = torch.tensor(BoB_array)
    BoB_data[compound] = out

with open('/Users/balepka/PycharmProjects/msuAI/Solvation_1/Tables/BoB_dict.pkl', 'wb') as f:
    pkl.dump(BoB_data, f)



