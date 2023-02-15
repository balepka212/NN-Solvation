from chemreps.bagger import BagMaker
from chemreps.just_bonds import bonds

from create_JB_dict import create_tensor
from Vectorizers.vectorizers import get_sdf_file
from config import project_path
import pickle as pkl
import os



with open(project_path('Tables/Solvatum/Solvents_Solutes.pkl'), 'rb') as f:
    Solvents, Solutes = pkl.load(f)

with open(project_path('Tables/Solvatum/Filenames.pkl'), 'rb') as f:
    Filenames = pkl.load(f)


dataset = project_path('Tables/MNSol/Sdf')
bagger = BagMaker('JustBonds', dataset)

the_dict = {}
for compound in Solvents+Solutes:
    mol_path = os.path.join('Tables', 'Solvatum', 'sdfs', Filenames[compound]+'.sdf')
    the_dict[compound] = create_tensor(mol_path, bagger=bagger, from_sdf=True, crop=True)


with open(project_path('Tables/Solvatum/JB_dict.pkl'), 'wb') as f:
    pkl.dump(the_dict, f)