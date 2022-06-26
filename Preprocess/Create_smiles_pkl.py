from config import *
import pickle as pkl


def create_smiles_dict(solvent_table='Tables/Reserve/SMILES_solvents_all.txt',
                       solute_table='Tables/Reserve/SMILES_solutes_all_3.tsv'):
    """TODO description
     create dictionary from tables of i, compound, smiles
     solvent_table: table with solvents
     solute_table: table with solutes"""
    smiles_dict = {}

    def make_dict(path):
        with open(project_path(path)) as f:
            for line in f:
                i, compound, smiles = line.split('\t')
                smiles = smiles.strip()
                print(i, compound.strip('"'), smiles)
                # print(compound)

                smiles_dict[compound.strip('"').replace(' ', '')] = smiles

    make_dict(solvent_table)
    make_dict(solute_table)
    # print(f'3: {smiles_dict["water"]}')

    return smiles_dict


get_SMILES = create_smiles_dict()
with open((project_path('Tables/get_SMILES.pkl')), 'wb') as g:
    pkl.dump(get_SMILES, g)
