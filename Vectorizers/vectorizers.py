from collections import OrderedDict
from functools import lru_cache

import chemreps
import torch
import pickle as pkl
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from config import project_path


@lru_cache(maxsize=1000)
def get_dictionary(name: str):
    """
    Returns dictionary desired dictionary.

    Parameters
    ----------
    name: str
        name of dictionary
    """
    path_map = {
        'blank': 'Tables/blank_dict.pkl',
        'smiles': 'Tables/get_SMILES.pkl',
        'classification': 'Tables/Classification_dict.pkl',
        'class': 'Tables/Classification_dict.pkl',
        'handlefile': 'Tables/file_handles.pkl',
        'bobsizes': 'Tables/MNSol_bags4.pkl',
        'bob': 'Tables/BoB_dict.pkl',
        'bagofbonds': 'Tables/BoB_dict.pkl',
        'bat': 'Tables/BAT_dict.pkl',
        'tesa': 'Tables/TESA_dict.pkl',
        'solutetesa': 'Tables/TESA_dict.pkl',
        'morgan': 'Tables/Morgan_2_124_dict.pkl',
        'morgan2124': 'Tables/Morgan_2_124_dict.pkl',
        'morgan124': 'Tables/Morgan_2_124_dict.pkl',
        'morgan21048576': 'Tables/Morgan_2_1048576_dict.pkl',
        'morgan22to20': 'Tables/Morgan_2_1048576_dict.pkl',
        'morgan2to20': 'Tables/Morgan_2_1048576_dict.pkl',
        'mor2to20': 'Tables/Morgan_2_1048576_dict.pkl',
        'macro': 'Tables/Macro_dict.pkl',
        'solventmacroprops': 'Tables/Macro_dict.pkl',
        'macroextra': 'Tables/Macro_extra_dict.pkl',
        'macrox': 'Tables/Macro_extra_dict.pkl',
        'solventtosolute': 'Tables/Reserve/solute_named_solvents.pkl',
        'ss': 'Tables/Solvents_Solutes.pkl',
        'justbonds': 'Tables/just_bonds_dict.pkl',
        'jb': 'Tables/just_bonds_dict.pkl',
        'soap': 'Tables/SOAP_dict.pkl',
        'slatm': None,
        'smdfilename': 'Tables/SMD_filenames_dict.pkl',
        'computedprops': 'Tables/Computed_props_dict.pkl',
        'comp': 'Tables/Computed_props_dict.pkl',
        'computed': 'Tables/Computed_props_dict.pkl'
    }
    new_name = str(name.lower())
    new_name = ''.join(new_name.split())
    new_name = ''.join(new_name.split('_'))

    with open(project_path(path_map[new_name]), 'rb') as f:
        dictionary = pkl.load(f)  # load dictionary
    return dictionary


def get_sample(compound: str, method: str):
    """
        Vectorizer function.

        Returns a vector from cached dict.

        Parameters
        ----------
        compound: str
            solute to be vectorized
        method: str
            vectorization method to be used
        """
    out = get_dictionary(method)[compound]  # get vector with cached dict
    out = torch.unsqueeze(out, dim=0)  # add dummy dimension to match other tensors shape
    return out


def get_handle_file(compound, args=('Tables/MNSol/xyz_files', 'Tables/MNSol/all_solutes'),
                    params=None, classic_xyz=True):
    """
        Returns a path for xyz file.

        Parameters
        ----------
        compound: str
            compound to find xyz file for
        args: (str, )
            tuple with path to xyz_files folder
        params: None
            not needed here
        classic_xyz: bool
            True if classic xyz, False if MNSol xyz file
        """
    xyz_path, xyz_mnsol_path, *args = args
    dictionary = get_dictionary('handlefile')
    path = xyz_path if classic_xyz else xyz_mnsol_path
    return path + '/' + dictionary[compound] + '.xyz'


def get_sdf_file(compound, args=('Tables/Reserve/Sdf',), params=None):
    """
        Returns a path to sdf file.

        Parameters
        ----------
        compound: str
            compound to find sdf file for
        args: (str, )
            tuple with path to sdf_files folder
        params: None
            not needed here
        """
    sdf_path, *args = args
    dictionary = get_dictionary('handlefile')
    return sdf_path + '/' + dictionary[compound] + '.sdf'


# CAN BE DELETED?


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
    row = table[table['Name'] == solvent]  # get the row with desired Solute
    out = row[row.columns[2:]]  # get Macro Properties data
    out = torch.tensor(out.values, dtype=torch.float)
    return out


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
    smiles = get_dictionary('smiles')[compound]  # get compound SMILES notation
    mol = Chem.MolFromSmiles(smiles)  # get compound molecule instance
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=chiral, radius=radius, nBits=nBits)  # get fingerprints
    out = torch.tensor(fp, dtype=torch.float)
    out = torch.unsqueeze(out, dim=0)  # add dummy dimension to match other tensors shape
    return out


def solute_TESA(solute, args, params=None):
    """
        TESA vectorizer.

        Returns Total Exposed Surface Area of solute. Data is obtained from MNSol database.

        Parameters
        ----------
        solute: str
            solute to be vectorized
        args: None
            not needed here
        params: None
            not needed here
        """
    out = get_dictionary('tesa')[solute]  # get TESA with cached dict
    out = torch.unsqueeze(out, dim=0)  # add dummy dimension to match other tensors shape
    return out


def classification(compound, args=None, params=None):
    """
        Classification vectorizer.

        From MNSol Database three layer - classification, one-hotted and flattened.

        Parameters
        ----------
        compound: str
            solute to be vectorized
        args: None
            not needed here
        params: None
            not needed here
        """
    out = get_dictionary('classification')[compound]  # get classification with cached dict
    out = torch.unsqueeze(out, dim=0)  # add dummy dimension to match other tensors shape
    return out


def bag_of_bonds(compound, args=None, params=None):
    out = get_dictionary('bob')[compound]  # get BAT with cached dict
    out = torch.unsqueeze(out, dim=0)  # add dummy dimension to match other tensors shape
    return out


def BAT(compound, args=None, params=None):
    out = get_dictionary('bat')[compound]  # get BAT with cached dict
    out = torch.unsqueeze(out, dim=0)  # add dummy dimension to match other tensors shape
    return out


def Capitalized_name(name:str):
    if name in ('blank', 'class', 'comp', 'macro', 'morgan', 'mor2to20'):
        return name.capitalize()
    elif name in ('tesa', 'jb', 'bat', 'soap'):
        return name.upper()
    elif name == 'bob':
        return 'BoB'
    elif name == 'macrox':
        return 'MacroX'

nicknames = {'blank': 'blank',
             'classification': 'class',
             'class': 'class',

             'tesa': 'tesa',
             'solutetesa': 'tesa',
             'computedprops': 'comp',
             'comp': 'comp',
             'computed': 'comp',

             'macro': 'macro',
             'solventmacroprops': 'macro',
             'macroextra': 'macrox',
             'macrox': 'macrox',

             'morgan': 'morgan',
             'morgan2124': 'morgan',
             'morgan124': 'morgan',
             'morgan21048576': 'mor2to20',
             'morgan22to20': 'mor2to20',
             'morgan2to20': 'mor2to20',
             'mor2to20': 'mor2to20',

             'justbonds': 'jb',
             'jb': 'jb',
             'bob': 'bob',
             'bagofbonds': 'bob',
             'bat': 'bat',

             'soap': 'soap'}


def split_folder(folder):
    folder = folder.lower()
    for morgan in ('morgan_2_124', 'morgan_124', 'morgan124'):
        folder = folder.replace(morgan, 'morgan')
    for mor2to20 in ('morgan_2_1048576', 'morgan_1048576', 'morgan_2_2to20', 'morgan_2to20'):
        folder = folder.replace(mor2to20, 'mor2to20')
    solvent, solute, model, *_ = folder.split('_')
    solvent = nicknames[solvent]
    solute = nicknames[solute]
    return solvent, solute, model



def parse_formula(formula: str):
    by_el = []
    the_el = ''
    for char in formula:
        if char.isupper():
            by_el.append(the_el)
            the_el = ''
        the_el += char
    by_el.append(the_el)
    by_el.remove('')
    the_dict = {}
    for element in by_el:
        if '_{' in element:
            el, number = element.strip('}').split('_{', 1)
            the_dict[el] = int(number)
        elif any([char.isdigit() for char in element]):
            for i, char in enumerate(element):
                if char.isdigit():
                    split_i = i
                    break
            el = element[:i]
            number = int(element[i:])
            the_dict[el] = number
        else:
            the_dict[element] = 1

    return the_dict

# Vectorizers map to be put in SS_Dataset class
copy_of_vectorizers_map = {
    'solvent_macro_props1': {'func': solvent_macro_props1, 'formats': ['tsv'],
                             'paths': ['Tables/Solvent_properties3.tsv'], 'params': None},
    'solute_TESA': {'func': solute_TESA, 'formats': ['feather'],
                    'paths': ['Tables/df3_3'], 'params': None},
    'classification': {'func': classification, 'formats': [], 'paths': [], 'params': None},
    'test_sp': {'func': test_sp, 'formats': [], 'paths': [], 'params': None},
    'test_up': {'func': test_up, 'formats': [], 'paths': [], 'params': None},
    'Morgan_fp_2_124': {'func': morgan_fingerprints, 'formats': [], 'paths': [], 'params': [2, 124, False]},
    'bag_of_bonds': {'func': bag_of_bonds, 'formats': [], 'paths': [], 'params': None},
    'BAT': {'func': BAT, 'formats': [], 'paths': [], 'params': None},
}

#
# @lru_cache(maxsize=1000)
# def get_smiles(path: str = 'Tables/get_SMILES.pkl'):
#     """
#     Returns SMILES dictionary.
#
#     Parameters
#     ----------
#     path: str
#         path to SMILES dict in pkl
#     """
#
#     with open(project_path(path), 'rb') as f:
#         dictionary = pkl.load(f)  # load SMILES dictionary
#     return dictionary

#
# @lru_cache(maxsize=1000)
# def get_handle_file_dict(path: str = 'Tables/file_handles.pkl'):
#     """
#     Returns HandleFile dictionary.
#
#     Parameters
#     ----------
#     path: str
#         path to HandleFile dict in pkl
#     """
#
#     with open(project_path(path), 'rb') as f:
#         dictionary = pkl.load(f)  # load HandleFile dictionary
#     return dictionary

#
# @lru_cache(maxsize=1000)
# def get_bob_sizes(path: str = 'Tables/MNSol_bags1.pkl'):
#     """
#     Returns Bag and Bag sizes for BoB.
#
#     Parameters
#     ----------
#     path: str
#         path to BoB params in pkl
#
#     Returns
#     ----------
#     path: list(list, )
#         [[Bags:dict, Bag_sizes:dict],]
#     """
#
#     with open(project_path(path), 'rb') as f:
#         dictionary = pkl.load(f)  # load HandleFile dictionary
#     return dictionary


#
# @lru_cache(maxsize=1000)
# def get_BoB_dict(path: str = 'Tables/BoB_dict.pkl'):
#     """
#     Returns dictionary of compound-BoB vector.
#
#     Parameters
#     ----------
#     path: str
#         path to BoB dict in pkl
#     """
#
#     with open(project_path(path), 'rb') as f:
#         dictionary = pkl.load(f)  # load BoB dictionary
#     return dictionary

#
# @lru_cache(maxsize=1000)
# def get_BAT_dict(path: str = 'Tables/BAT_dict.pkl'):
#     """
#     Returns dictionary of compound-BAT vector.
#
#     Parameters
#     ----------
#     path: str
#         path to BAT dict in pkl
#     """
#
#     with open(project_path(path), 'rb') as f:
#         dictionary = pkl.load(f)  # load BAT dictionary
#     return dictionary


#
# def get_smiles(compound, args=('Tables/get_SMILES.pkl',), params=None):
#     """
#     Returns SMILES notation of given compound.
#
#     Parameters
#     ----------
#     compound: str
#         compound to get smiles from
#     args: (dict,)
#         tuple with dictionary of structure {compound: smiles}
#     params: None
#         not needed here
#     """
#     path = project_path(args[0])
#     dictionary = read_smiles(path)
#     return dictionary[compound.replace(' ', '')]
#


#
# def solute_TESA(solute, args, params=None):
#     """
#     TESA vectorizer.
#
#     Returns Total Exposed Surface Area of solute. Data is obtained from MNSol database.
#
#     Parameters
#     ----------
#     solute: str
#         solute to be vectorized
#     args: [pd.table]
#         df3 database where 20-28 columns are TESA
#     params: None
#         not needed here
#     """
#
#     df, *args = args
#     row = df[df['SoluteName'] == solute][:1]  # get the row with desired Solute
#     out = row[row.columns[20:29]]  # get TESA data
#     out = torch.tensor(out.values, dtype=torch.float)
#     return out
