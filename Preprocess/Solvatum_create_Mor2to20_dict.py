import pickle as pkl
from config import project_path
from create_Morgan_r_nbit_dict import create_tensor
from tqdm import tqdm

with open(project_path(f'Tables/short2long_fp_2_1048576_dict.pkl'),
          'rb') as f:
    short2long_fp = pkl.load(f)

non_zero_bits = set(short2long_fp.values())

with open(project_path('Tables/Solvatum/Smiles_dict.pkl'), 'rb') as f:
    smiles_dict = pkl.load(f)

the_dict = {}

for solute, smiles in tqdm(smiles_dict.items()):
    the_dict[solute], _all_bi = create_tensor(smiles, (2, 2**20, False), non_zero_bits, from_smiles=True, fantom_all_bi=True)

with open(project_path('Tables/Solvatum/Mor2to20_dict.pkl'), 'wb') as f:
    pkl.dump(the_dict, f)