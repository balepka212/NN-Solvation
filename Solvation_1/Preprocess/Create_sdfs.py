import os
import pickle

import periodictable as pt
from rdkit import Chem
from Solvation_1.Vectorizers.vectorizers import get_dictionary, get_handle_file
from Solvation_1.config import project_path
import pickle as pkl

with open(project_path('Solvation_1/Tables/Solvents_Solutes.pkl'), 'rb') as f:
    Solvents, Solutes = pickle.load(f)

atom_dict = pt.elements.__dict__['_element']
print(atom_dict[8].covalent_radius)
el_to_n_dict = {}
n_to_el_dict = {}

for N, element in atom_dict.items():
    el_to_n_dict[element.symbol] = int(N)
    n_to_el_dict[N] = element.symbol

print(el_to_n_dict)


class Atom:
    def __init__(self, atom_number, x, y, z):
        self.atomic_number = atom_number
        self.xyz = [float(r) for r in (x, y, z)]
        self.xyz_id = None
        self.smiles_id = None
        self.element = n_to_el_dict[self.atomic_number]
        self.radius = atom_dict[atom_number].covalent_radius


class my_molecule:
    def __init__(self, smiles, path, classic_xyz=True):
        self.classic_xyz = classic_xyz
        self.atoms = []
        self.smiles = smiles
        self.xyz_path = path
        self.read_xyz()
        mol1 = Chem.MolFromSmiles(self.smiles)
        mol2 = Chem.AddHs(mol1)
        self.mol = mol2
        smiles_bond_dict = {}
        for b in self.mol.GetBonds():
            bond_name = ''.join(sorted((b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol())))
            if bond_name in smiles_bond_dict:
                smiles_bond_dict[bond_name] += 1
            else:
                smiles_bond_dict[bond_name] = 1
        self.smiles_bond_dict = smiles_bond_dict

        for at in self.atoms:
            pass

    def read_xyz(self):
        with open(self.xyz_path) as f:
            # TODO understand format of file
            for i, line in enumerate(f):
                if self.classic_xyz:
                    if i == 0:
                        try:
                            n_str = int(line)
                        except ValueError:
                            print('no int')
                    elif i == 1:
                        comment = line
                        self.comment = comment
                    else:
                        element, x, y, z = line.strip().split()
                        atom_number = el_to_n_dict[element]
                        at = Atom(atom_number, x, y, z)
                        at.xyz_id = i - 2
                        self.atoms.append(at)
                else:
                    if i == 0:
                        try:
                            handle, formula, name, identifier = line.split()
                        except ValueError:
                            print('Bad')
                    elif i == 1:
                        comment = line
                        self.comment = comment
                    elif i == 2:
                        pass
                    else:
                        atom_number, x, y, z = line.strip().split()
                        at = Atom(atom_number, x, y, z)
                        at.xyz_id = i - 3
                        self.atoms.append(at)

    def estimate_bonds(self):
        bond_list = []
        bond_dict = {}
        for i, x in enumerate(self.atoms):
            # print(f'{i} = {x.xyz_id}:{x.element}')
            for j, y in enumerate(self.atoms):
                if 'OO' in self.smiles and x.element == 'O' and y.element == 'O':
                    factor = 1.2
                elif '[O-][N+](=O)C' in self.smiles and \
                        ((x.element == 'C' and y.element == 'N') or (x.element == 'N' and y.element == 'C')):
                    factor = 1.1
                else:
                    factor = 1.05
                if i < j and is_bond(x, y, factor=factor):
                    bond_list.append((i, j))
                    bond_name = ''.join(sorted((x.element, y.element)))
                    if bond_name in bond_dict:
                        bond_dict[bond_name] += 1
                    else:
                        bond_dict[bond_name] = 1
        self.bond_list = bond_list
        self.xyz_bond_dict = bond_dict

    def check_bond_dicts(self, verbose=False):
        same = (self.smiles_bond_dict == self.xyz_bond_dict)
        if verbose and not same:
            print(f'smiles: {self.smiles_bond_dict}')
            print(f'xyz:    {self.xyz_bond_dict}')
        return same


def distance(a1: Atom, a2: Atom):
    x1, y1, z1 = a1.xyz
    x2, y2, z2 = a2.xyz
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5


def is_bond(a1: Atom, a2: Atom, factor=1.05):
    d = distance(a1, a2)
    threshold = factor * (a1.radius + a2.radius)
    if d < threshold:
        return True
    return False


output_folder = '/Users/balepka/Yandex.Disk-isaevvv@my.msu.ru.localized/Study/Neural Network/Tables/Sdf'
for m_name in Solvents:
    # print(m_name)
    m_smiles = get_dictionary('smiles')[m_name]
    m_path = project_path(get_handle_file(m_name))
    m = my_molecule(m_smiles, m_path)
    m.estimate_bonds()
    if m.check_bond_dicts(verbose=False):
        test_output = os.path.join(output_folder, get_dictionary('handlefile')[m_name] + '.sdf')
        # print(f'All good: {m_name}')
    else:
        print(f"BAD: {m_name} -> {get_dictionary('handlefile')[m_name]}")
        print(f'xyz:   {m.xyz_bond_dict}')
        print(f'smiles:{m.smiles_bond_dict}')
        test_output = os.path.join(output_folder, 'bad', get_dictionary('handlefile')[m_name] + '.sdf')
    with open(test_output, 'w') as f:
        f.write(m_name + '\n')
        f.write('Bonds Generated from xyz' + '\n')
        f.write('' + '\n')
        f.write(''.join([str(x).rjust(3) for x in (len(m.atoms), len(m.bond_list), 0, ' ', 0, 0, 0, 0, 0)]) +
                str(0).rjust(6) + str('V2000').rjust(6) + '\n')
        for a in m.atoms:
            f.write(''.join([f"{r:.7}".rjust(10) for r in a.xyz]) + ' ' +
                    a.element.ljust(3) + ''.join([str(x).rjust(3) for x in [0, ] * 12]) + '\n')
        for i, j in m.bond_list:
            f.write(''.join([str(x).rjust(3) for x in [i + 1, j + 1, 8, 0, 0, 0, 0]]) + '\n')
        f.write('M  END')

#
# bond_list = []
# bond_dict = {}
# for i, x in enumerate(ethanol.atoms):
#     print(f'{i} = {x.xyz_id}:{x.element}')
#     for j, y in enumerate(ethanol.atoms):
#         if i<j and is_bond(x,y):
#             bond_list.append((i,j))
#             bond_name = ''.join(sorted((x.element, y.element)))
#             if bond_name in bond_dict:
#                 bond_dict[bond_name] += 1
#             else:
#                 bond_dict[bond_name] = 1
# print(bond_list)
# print(bond_dict)
# smiles_bond_dict = {}
# for b in ethanol.mol.GetBonds():
#     bond_name = ''.join(sorted((b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol())))
#     if bond_name in smiles_bond_dict:
#         smiles_bond_dict[bond_name] += 1
#     else:
#         smiles_bond_dict[bond_name] = 1
# print(smiles_bond_dict)
# print(smiles_bond_dict == bond_dict)
#


# def parse_formula(formula):
#     ordered_elements = ('H', 'C', 'N', 'O', 'F', 'P', 'S', 'CL', 'SE', 'BR', 'I')
#     ar = []
#     current = True
#     for char in formula:
#         # print(f'{char} {current} is: {char.isdigit()}')
#         if char.isdigit() is current:
#             ar[-1]+=char.upper()
#         else:
#             ar.append(char.upper())
#             current = not current
#     # print(ar)
#     elements = dict([(ar[i], ar[i+1]) for i in range(0, len(ar), 2)])
#     # print(elements)
#     form = ''
#     for el in ordered_elements:
#         if el in elements:
#             form+=el
#             form+=elements[el]
#     return form
