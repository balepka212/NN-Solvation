import tqdm
from pyarrow import feather
import torch
import pickle as pkl

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import _getMorganEnv

from Vectorizers.vectorizers import get_dictionary
from config import project_path
r = 2
nbit = 2**20

with open(project_path('Tables/Solvents_Solutes.pkl'), 'rb') as f:
    Solvents, Solutes = pkl.load(f)


def create_tensor(compound, params, non_zero_bits, from_smiles=False, fantom_all_bi=False):
    if fantom_all_bi:
        all_bi = {}
    RDLogger.DisableLog('rdApp.*')  # disables WARNING of not removing H-atoms
    radius, nBits, chiral = params
    if from_smiles:
        smiles = compound
    else:
        smiles = get_dictionary('smiles')[compound]  # get compound SMILES notation
    mol = Chem.MolFromSmiles(smiles)  # get compound molecule instance
    bitInfo = {}
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, bitInfo=bitInfo,
                                                        invariants=AllChem.GetConnectivityInvariants(mol,
                                                                                                     includeRingMembership=False),
                                                        useBondTypes=False, useChirality=chiral)
    bits = [i for i, x in enumerate(list(fingerprint)) if x]
    fps = {}
    for bitId in bits:
        atomId, radius = bitInfo[bitId][0]

        menv = _getMorganEnv(mol, atomId, radius, baseRad=0.3, aromaticColor=(0.9, 0.9, 0.2), ringColor=(0.8, 0.8, 0.8),
                             centerColor=(0.6, 0.6, 0.9), extraColor=(0.9, 0.9, 0.9))
        fps[bitId] = menv

        if bitId not in all_bi:
            all_bi[bitId] = {'smiles': [], 'submol': []}

        fragment_smiles = Chem.MolToSmiles(menv.submol)
        if fragment_smiles not in all_bi[bitId]['smiles']:
            all_bi[bitId]['smiles'].append(fragment_smiles)
            all_bi[bitId]['submol'].append(menv.submol)

    short_fp = [x for i, x in enumerate(fingerprint) if i in non_zero_bits]
    out = torch.tensor(short_fp, dtype=torch.float)
    if fantom_all_bi:
        return out, all_bi
    return out


# df3 = feather.read_feather('/Users/balepka/PycharmProjects/msuAI/Tables/df3_3')
if __name__ == '__main__':
    data_dict = {}
    non_zero_bits = set()
    for compound in tqdm.tqdm(Solutes+Solvents):
        RDLogger.DisableLog('rdApp.*')  # disables WARNING of not removing H-atoms
        radius, nBits, chiral = r, nbit, False
        smiles = get_dictionary('smiles')[compound]  # get compound SMILES notation
        mol = Chem.MolFromSmiles(smiles)  # get compound molecule instance
        bitInfo = {}
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, bitInfo=bitInfo,
                                                            invariants=AllChem.GetConnectivityInvariants(mol,
                                                                                                         includeRingMembership=False),
                                                            useBondTypes=False, useChirality=chiral)
        for i, bit in enumerate(fingerprint):
            if bit:
                non_zero_bits.add(i)

    short2long_fp = {}
    for i, x in enumerate(sorted(non_zero_bits)):
        short2long_fp[i] = x

    all_bi = {}
    for compound in tqdm.tqdm(Solutes+Solvents):
        data_dict[compound] = create_tensor(compound, (r, nbit, False), non_zero_bits)

    with open(f'/Users/balepka/PycharmProjects/msuAI/Tables/Morgan_{str(int(r))}_{str(int(nbit))}_dict.pkl', 'wb') as f:
        pkl.dump(data_dict, f)
    with open(f'/Users/balepka/PycharmProjects/msuAI/Tables/short2long_fp_{str(int(r))}_{str(int(nbit))}_dict.pkl', 'wb') as f:
        pkl.dump(short2long_fp, f)
    with open(f'/Users/balepka/PycharmProjects/msuAI/Tables/fp_bits_{str(int(r))}_{str(int(nbit))}.pkl', 'wb') as f:
        pkl.dump(all_bi, f)
