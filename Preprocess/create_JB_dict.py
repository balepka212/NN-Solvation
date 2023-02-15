import torch
from chemreps.bagger import BagMaker
from chemreps.just_bonds import bonds

from Vectorizers.vectorizers import get_sdf_file
from config import project_path
import pickle as pkl


def create_tensor(compound, from_sdf=False, bagger='Tables/Reserve/Sdf', crop=False):
    if bagger is str:
        dataset = project_path(bagger)
        bagger = BagMaker('JustBonds', dataset)

    if from_sdf:
        mol_path = compound
    else:
        mol_path = get_sdf_file(compound)
    JB_array = bonds(project_path(mol_path), bagger.bags, bagger.bag_sizes, crop=crop)
    out = torch.tensor(JB_array)
    return out


if __name__ == '__main__':
    dataset = project_path('Tables/MNSol/Sdf')
    bagger = BagMaker('JustBonds', dataset)

    with open(project_path('Tables/Solvents_Solutes.pkl'), 'rb') as f:
        Solvents, Solutes = pkl.load(f)

    JB_data = {}
    for compound in Solvents+Solutes:
        JB_data[compound] = create_tensor(compound, bagger=bagger)

    with open('/Tables/JB_dict.pkl', 'wb') as f:
        pkl.dump(JB_data, f)