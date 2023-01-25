from pyarrow import feather
import torch
import pickle as pkl

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from Vectorizers.vectorizers import get_dictionary
from config import project_path


with open(project_path('Tables/Solvents_Solutes.pkl'), 'rb') as f:
    Solvents, Solutes = pkl.load(f)


def create_tensor(compound, params):
    RDLogger.DisableLog('rdApp.*')  # disables WARNING of not removing H-atoms
    radius, nBits, chiral = params
    smiles = get_dictionary('smiles')[compound]  # get compound SMILES notation
    mol = Chem.MolFromSmiles(smiles)  # get compound molecule instance
    bitInfo={}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=chiral, radius=radius, nBits=nBits, bitInfo=bitInfo)  # get fingerprints
    out = torch.tensor(fp, dtype=torch.float)
    return out, bitInfo




df3 = feather.read_feather('/Users/balepka/PycharmProjects/msuAI/Tables/df3_3')
data_dict = {}
bitInfo_dict = {}
for compound in Solutes+Solvents:
    fp, bitInfo = create_tensor(compound, (2, 124, False))
    data_dict[compound] = fp
    bitInfo_dict[compound] = bitInfo

with open('/Users/balepka/PycharmProjects/msuAI/Tables/Morgan_2_124_bitInfo_dict.pkl', 'wb') as f:
    pkl.dump(data_dict, f)
with open('/Users/balepka/PycharmProjects/msuAI/Tables/Only_fp_bitInfo_dict.pkl', 'wb') as f:
    pkl.dump(bitInfo_dict, f)



#
#
# def morgan_fingerprints(compound, args, params: (int, int, bool) = (2, 124, False)):
#     """
#     Morgan Fingerprints vectorizer.
#
#     Computes molecule fingerprints:
#         Assigns each atom with an identifier.
#         Updates each atom’s identifiers based on its neighbours to some order (radius).
#         Removes duplicates.
#
#     Parameters
#     ----------
#     compound: str
#         compound to be vectorized
#     args: None
#         not needed here
#     params: (int, int, bool)
#         (radius, nBits, useChirality)
#     """
#
#     RDLogger.DisableLog('rdApp.*')  # disables WARNING of not removing H-atoms
#     radius, nBits, chiral = params
#     smiles = get_dictionary('smiles')[compound]  # get compound SMILES notation
#     mol = Chem.MolFromSmiles(smiles)  # get compound molecule instance
#     fp = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=chiral, radius=radius, nBits=nBits)  # get fingerprints
#     out = torch.tensor(fp, dtype=torch.float)
#     out = torch.unsqueeze(out, dim=0)  # add dummy dimension to match other tensors shape
#     return out
