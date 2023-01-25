import pickle as pkl
from config import project_path
from create_Morgan_2_124_dict import create_tensor

with open(project_path('Tables/Solvatum/Smiles_dict.pkl'), 'rb') as f:
    smiles_dict = pkl.load(f)

the_dict = {}
for solute, smiles in smiles_dict.items():
    the_dict[solute] = create_tensor(smiles, (2, 124, False), from_smiles=True)

with open(project_path('Tables/Solvatum/Morgan_dict.pkl'), 'wb') as f:
    pkl.dump(the_dict, f)