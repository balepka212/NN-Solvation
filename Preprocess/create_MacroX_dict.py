import numpy as np
import pandas as pd
import torch
import pickle as pkl
from config import project_path
from Vectorizers.vectorizers import get_dictionary
import os
import periodictable as pt


def create_tensor(solvent, args):
    table, *args = args
    row = table[table['Name'] == solvent]  # get the row with desired Solvent
    out = row[row.columns[2:]]  # get Macro Properties data
    out = out.values
    # SMD_filename = get_dictionary('smd_filename')[solvent]
    filepath = os.path.join(project_path('/SMD_inputs/Solvs'), 'GAS', solvent+'-GAS.out')
    with open(filepath) as f:
        for i, line in enumerate(f.readlines()):
            if line.startswith('Magnitude (Debye)'):
                _header, dipole_moment = line.split(':')
                dipole_moment = float(dipole_moment.strip())
                print(f'{solvent}: {dipole_moment} debye')

            elif line.startswith('CARTESIAN COORDINATES (ANGSTROEM)'):
                coord_i = i+2
            elif line.startswith('ORBITAL ENERGIES'):
                orbital_i = i+4
    with open(filepath) as f:
        whole_file = f.readlines()
        # getting molecular size
        Xs, Ys, Zs = [], [], []
        mass = 0.
        for line in whole_file[coord_i:]:
            if not line.strip():
                break
            element, x, y, z = line.split()
            mass += pt.elements.symbol(element).mass
            Xs.append(float(x))
            Ys.append(float(y))
            Zs.append(float(z))
        dX = max(Xs) - min(Xs)
        dY = max(Ys) - min(Ys)
        dZ = max(Zs) - min(Zs)

        # getting LUMO-HOMO
        for line in whole_file[orbital_i:]:
            No, Occ, _E_Eh, E_eV = [float(x) for x in line.split()]
            if Occ == 0.:
                dE = E_eV-HOMO
                break
            HOMO = E_eV

    out = np.append(out, [dipole_moment, dX, dY, dZ, dE, mass])
    out = torch.tensor(out, dtype=torch.float)
    return out.squeeze()


if __name__ == '__main__':
    with open(project_path('Tables/Solvents_Solutes.pkl'), 'rb') as f:
        Solvents, Solutes = pkl.load(f)

    table1 = pd.read_table(project_path('Tables/Solvent_properties3.tsv'))
    data_dict = {}
    for compound in Solvents:
        data_dict[compound] = create_tensor(compound, (table1,))

    #
    with open('/Tables/MacroX_dict.pkl', 'wb') as f:
        pkl.dump(data_dict, f)
#
#
# def solvent_macro_props1(solvent, args, params=None):
#     """
#     Solvent Macro Properties vectorizer.
#
#     Returns a vector of properties: nD, alpha, beta, gamma, epsilon, phi, psi. Data is obtained from MNSol database.
#
#     Parameters
#     ----------
#     solvent: str
#         solvent to be vectorized
#     args: [pd.table]
#         database where 2-... columns are properties of solvent. column 'Name' contains solvent
#     params: None
#         not needed here
#     """
#
#     table, *args = args
#     row = table[table['Name'] == solvent]  # get the row with desired Solute
#     out = row[row.columns[2:]]  # get Macro Properties data
#     out = torch.tensor(out.values, dtype=torch.float)
#     return out
