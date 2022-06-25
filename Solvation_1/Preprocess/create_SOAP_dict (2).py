import ase.io.sdf
import torch
from Solvation_1.Vectorizers.vectorizers import get_sdf_file
from Solvation_1.config import project_path
import pickle as pkl
from dscribe.descriptors import SOAP

with open(project_path('Solvation_1/Tables/Solvents_Solutes.pkl'), 'rb') as f:
    Solvents, Solutes = pkl.load(f)

species = ['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Se', 'Br', 'I']
rcut = 3.
nmax = 4
lmax = 3
# Setting up the SOAP descriptor
soap = SOAP(
    species=species,
    periodic=False,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
    sparse=False,
    average='inner'
)

SOAP_data={}
for compound in Solvents+Solutes:

    # Molecule created as an ASE.Atoms
    sdf_path = get_sdf_file(compound)
    mol = ase.io.sdf.read_sdf(project_path(sdf_path))

    # Create SOAP output for the system
    soap_mol = soap.create(mol)
    out = torch.tensor(soap_mol, dtype=torch.float16)
    SOAP_data[compound] = out

    if soap_mol.shape[0] != 4704:
        print(compound)
        print(soap_mol)
        print(soap_mol.shape)



with open('/Users/balepka/PycharmProjects/msuAI/Solvation_1/Tables/SOAP_dict.pkl', 'wb') as f:
    pkl.dump(SOAP_data, f)