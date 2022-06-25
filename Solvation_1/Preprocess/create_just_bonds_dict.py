import torch
from chemreps.bagger import BagMaker
from chemreps.just_bonds import bonds

from Solvation_1.Vectorizers.vectorizers import get_sdf_file
from Solvation_1.config import project_path
import pickle as pkl

dataset = project_path('Solvation_1/Tables/Reserve/Sdf')
bagger = BagMaker('JustBonds', dataset)

with open(project_path('Solvation_1/Tables/Solvents_Solutes.pkl'), 'rb') as f:
    Solvents, Solutes = pkl.load(f)

JB_data = {}
for compound in Solvents+Solutes:
    mol_path = get_sdf_file(compound)
    JB_array = bonds(project_path(mol_path), bagger.bags, bagger.bag_sizes)
    out = torch.tensor(JB_array)
    JB_data[compound] = out

with open('/Users/balepka/PycharmProjects/msuAI/Solvation_1/Tables/just_bonds_dict.pkl', 'wb') as f:
    pkl.dump(JB_data, f)