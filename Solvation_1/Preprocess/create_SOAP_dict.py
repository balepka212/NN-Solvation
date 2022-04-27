import torch
from chemreps.bagger import BagMaker
from chemreps.bat import bat
from Solvation_1.Vectorizers.vectorizers import get_sdf_file
from Solvation_1.config import project_path
import pickle as pkl
from dscribe.descriptors import SOAP

with open(project_path('Solvation_1/Tables/Solvents_Solutes.pkl'), 'rb') as f:
    Solvents, Solutes = pkl.load(f)



species = ["H", "C", "O", "N"]
rcut = 6.0
nmax = 8
lmax = 6

# Setting up the SOAP descriptor
soap = SOAP(
    species=species,
    periodic=False,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
)

with open('/Users/balepka/PycharmProjects/msuAI/Solvation_1/Tables/SOAP_dict.pkl', 'wb') as f:
    pkl.dump(SOAP_data, f)