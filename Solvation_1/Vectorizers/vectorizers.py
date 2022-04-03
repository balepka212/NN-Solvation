import torch
import pickle as pkl
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger


def test_sp(solvent, args=None, params=None):
    """
        Test solvent vectorizer. Returns length of given string.

        Parameters
        ----------
        solvent: str
            solvent to be vectorized
        args: None
            not needed here
        params: None
            not needed here
        """

    return torch.tensor(len(solvent))


def test_up(solute, args=None, params=None):
    """
        Test solute vectorizer. Returns length of given string.

        Parameters
        ----------
        solute: str
            solute to be vectorized
        args: None
            not needed here
        params: None
            not needed here
        """

    return torch.tensor(len(solute))


def solute_TESA(solute, args, params=None):
    """
    TESA vectorizer.

    Returns Total Exposed Surface Area of solute. Data is obtained from MNSol database.

    Parameters
    ----------
    solute: str
        solute to be vectorized
    args: [pd.table]
        df3 database where 20-28 columns are TESA
    params: None
        not needed here
    """

    df, *args = args
    row = df[df['SoluteName'] == solute][:1]    # get the row with desired Solute
    out = row[row.columns[20:29]]   # get TESA data
    out = torch.tensor(out.values, dtype=torch.float)
    return out


def solvent_macro_props1(solvent, args, params=None):
    """
    Solvent Macro Properties vectorizer.

    Returns a vector of properties: nD, alpha, beta, gamma, epsilon, phi, psi. Data is obtained from MNSol database.

    Parameters
    ----------
    solvent: str
        solvent to be vectorized
    args: [pd.table]
        database where 2-... columns are properties of solvent. column 'Name' contains solvent
    params: None
        not needed here
    """

    table, *args = args
    row = table[table['Name'] == solvent]   # get the row with desired Solute
    out = row[row.columns[2:]]  # get Macro Properties data
    out = torch.tensor(out.values, dtype=torch.float)
    return out


def get_smiles(compound, args=('../Solvation_1/Tables/get_SMILES.pkl',), params=None):
    """
    Returns SMILES notation of given compound.

    Parameters
    ----------
    compound: str
        compound to get smiles from
    args: (dict,)
        tuple with dictionary of structure {compound: smiles}
    params: None
        not needed here
    """

    with open(args[0], 'rb') as f:
        dictionary = pkl.load(f)    # load SMILES dictionary
    return dictionary[compound.replace(' ', '')]


def morgan_fingerprints(compound, args, params: (int, int, bool) = (2, 124, False)):
    """
    Morgan Fingerprints vectorizer.

    Computes molecule fingerprints:
        Assigns each atom with an identifier.
        Updates each atom’s identifiers based on its neighbours to some order (radius).
        Removes duplicates.

    Parameters
    ----------
    compound: str
        compound to be vectorized
    args: None
        not needed here
    params: (int, int, bool)
        (radius, nBits, useChirality)
    """

    RDLogger.DisableLog('rdApp.*')  # disables WARNING of not removing H-atoms
    radius, nBits, chiral = params
    smiles = get_smiles(compound)   # get compound SMILES notation
    mol = Chem.MolFromSmiles(smiles)   # get compound molecule instance
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=chiral, radius=radius, nBits=nBits)  # get fingerprints
    out = torch.tensor(fp, dtype=torch.float)
    out = torch.unsqueeze(out, dim=0)   # add dummy dimension to match other tensors shape
    return out


# Vectorizers map to be put in SS_Dataset class
copy_of_vectorizers_map = {
            'solvent_macro_props1': {'func': solvent_macro_props1, 'formats': ['tsv'],
                                     'paths': ['Solvation_1/Tables/Solvent_properties3.tsv'], 'params': None},
            'solute_TESA': {'func': solute_TESA, 'formats': ['feather'],
                            'paths': ['Solvation_1/Tables/df3_3'], 'params': None},
            'test_sp': {'func': test_sp, 'formats': [], 'paths': [], 'params': None},
            'test_up': {'func': test_up, 'formats': [], 'paths': [], 'params': None},
            'Morgan_fp_2_124': {'func': morgan_fingerprints, 'formats': [], 'paths': [], 'params': [2, 124, False]},
        }
