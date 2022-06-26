from collections import OrderedDict

from tqdm import tqdm

from config import project_path
import chemreps
import pickle as pkl
from Vectorizers.vectorizers import get_sdf_file
import torch
from rdkit import Chem


with open(project_path('Tables/Solvents_Solutes.pkl'), 'rb') as f:
    Solvents, Solutes = pkl.load(f)

with open('/Users/balepka/PycharmProjects/msuAI/Tables/MNSol_bags4.pkl', 'rb') as f:
    my_bags = pkl.load(f)

BAT_bags, BAT_sizes = my_bags[1].copy()

BAT_data = {}
for compound in Solvents+Solutes:
    print(compound)
    mol_path = get_sdf_file(compound)
    BAT_array = chemreps.bat.bat(project_path(mol_path), BAT_bags, BAT_sizes)
    out = torch.tensor(BAT_array)
    BAT_data[compound] = out

with open('/Users/balepka/PycharmProjects/msuAI/Tables/BAT_dict.pkl', 'wb') as f:
    pkl.dump(BAT_data, f)



