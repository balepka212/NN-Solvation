import torch
import pickle as pkl
from rdkit import Chem


def test_sp(solvent, args=None):
    return torch.tensor(len(solvent))


def test_up(solute, args=None):
    return torch.tensor(len(solute))


def solute_TESA(solute, args):
    """TODO add description
        df is pd dataframe which is df3"""
    df, *args = args
    row = df[df['SoluteName'] == solute][:1]
    out = row[row.columns[20:29]]
    out = torch.tensor(out.values, dtype=torch.float)
    return out


def solvent_macro_props1(solvent, args):
    """TODO add description
        table is pd dataframe"""
    table, *args = args
    row = table[table['Name'] == solvent]
    out = row[row.columns[2:]]
    out = torch.tensor(out.values, dtype=torch.float)
    return out


def get_smiles(compound, args=('../Solvation_1/Tables/get_SMILES.pkl',)):
    """TODO add description"""
    with open(args[0], 'rb') as f:
        dictionary = pkl.load(f)
    return dictionary[compound]


