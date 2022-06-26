import torch
from chemreps.bagger import BagMaker
from chemreps.bat import bat
from Vectorizers.vectorizers import get_sdf_file
from config import project_path
import pickle as pkl

dataset = project_path('Tables/Reserve/Sdf')
bagger = BagMaker('BAT', dataset)

with open(project_path('Tables/Solvents_Solutes.pkl'), 'rb') as f:
    Solvents, Solutes = pkl.load(f)

BAT_data = {}
for compound in Solvents+Solutes:
    mol_path = get_sdf_file(compound)
    BAT_array = bat(project_path(mol_path), bagger.bags, bagger.bag_sizes)
    out = torch.tensor(BAT_array)
    BAT_data[compound] = out

with open('/Users/balepka/PycharmProjects/msuAI/Tables/BAT_cr_dict.pkl', 'wb') as f:
    pkl.dump(BAT_data, f)