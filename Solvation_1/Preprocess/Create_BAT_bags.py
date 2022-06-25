from collections import OrderedDict

from tqdm import tqdm

from Solvation_1.config import project_path
import chemreps
import pickle as pkl
from Solvation_1.Vectorizers.vectorizers import get_sdf_file, get_dictionary
import torch
from rdkit import Chem

with open(project_path('Solvation_1/Tables/Solvents_Solutes.pkl'), 'rb') as f:
    Solvents, Solutes = pkl.load(f)


def parse_formula(formula: str):
    form = {}
    previous = None
    numbers = ''
    element = ''
    for char in formula:
        if char.isupper():
            if previous == 'lower' or previous == 'upper':
                form[element] = 1
            elif previous == 'num':
                form[element] = int(numbers)
                numbers = ''
            element = char
            previous = 'upper'
        elif char.isdigit():
            numbers += char
            previous = 'num'
        else:
            element += char
            previous = 'lower'
    if numbers:
        form[element] = int(numbers)
    else:
        form[element] = 1
    return form


def parse_bag(bag: str):
    parsed = []
    element = ''
    for char in bag:
        if char.isupper():
            parsed.append(element)
            element = char
        else:
            element += char
    parsed.append(element)
    parsed.remove('')
    return parsed


#
# ordered_elements = ('', 'H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Se', 'Br', 'I')
# # el_props = {'':50, 'H': 50, 'C':30, 'N':20, 'O':20, 'F':5, 'Si':3, 'P':5, 'S':5, 'Cl':3, 'Se':3, 'Br':3, 'I':3}
# BAT_bags = {}
# BAT_sizes = {}
#
# for e1 in ordered_elements:
#     for e2 in ordered_elements:
#         for e3 in ordered_elements:
#             for e4 in ordered_elements:
#                 name = ''.join((e1, e2, e3, e4))
#                 BAT_bags[name] = []
#                 BAT_sizes[name] = 0
#
# BAT_bags2 = {}
# for x in sorted(BAT_bags.keys()):
#     BAT_bags2[x] = []
# BAT_bags = BAT_bags2
# BAT_bags.pop('')
# BAT_sizes.pop('')
#
# for compound in (Solvents+Solutes):
#     smiles = get_dictionary('smiles')[compound]
#     mol = Chem.MolFromSmiles(smiles)
#     formula = Chem.AllChem.CalcMolFormula(mol)
#     # print(f'{compound}: {parse_formula(formula)}')
#     form_dict = parse_formula(formula)
#     for bag in BAT_bags:
#         bat_list = parse_bag(bag)
#         size = 1
#         for el in bat_list:
#             try:
#                 size *= form_dict[el]
#             except KeyError:
#                 size = 0
#         if BAT_sizes[bag] < size:
#             BAT_sizes[bag] = size
#
# BAT_nz_sizes = {}
# for key, value in BAT_sizes.items():
#     if value != 0:
#         BAT_nz_sizes[key] = value

# def sort_el(bag: str):
#     bag_list = parse_bag(bag)
#     number = 0
#     for i, el in enumerate(bag_list):
#         N = ordered_elements.index(el)
#         number += 100**(4-i)*N
#     return number

# print(BAT_nz_sizes)
# BAT_sizes2 = {}
# BAT_bags2 = {}
# for bag in sorted(BAT_nz_sizes.keys()):
#     BAT_sizes2[bag] = BAT_nz_sizes[bag]
#     BAT_bags2[bag] = []
# BAT_sizes = OrderedDict(BAT_sizes2)
# BAT_bags = OrderedDict(BAT_bags2)

# with open('/Users/balepka/PycharmProjects/msuAI/Solvation_1/Tables/MNSol_bags2.pkl', 'rb') as f:
#     my_bags = pkl.load(f)
#
# # my_bags.append((BAT_bags, BAT_sizes))
# #
# # with open('/Users/balepka/PycharmProjects/msuAI/Solvation_1/Tables/MNSol_bags2.pkl', 'wb') as f:
# #     pkl.dump(my_bags, f)
#
# BAT_bags, BAT_sizes = my_bags[1]
#
# BAT_data = {}
# for compound in Solvents[:5]:
#     print(compound)
#     mol_path = get_sdf_file(compound)
#     BAT_array = chemreps.bat.bat(project_path(mol_path), BAT_bags, BAT_sizes)
#     # BoB_array = chemreps.bag_of_bonds.bag_of_bonds(project_path(mol_path), bags1, bag_sizes1)
#     out = torch.tensor(BAT_array)
#     BAT_data[compound] = out
#
# with open('/Users/balepka/PycharmProjects/msuAI/Solvation_1/Tables/BAT_dict.pkl', 'wb') as f:
#     pkl.dump(BAT_data, f)


Atom_bags = {}
Bond_bags = {}
Angle_bags = {}
Torsion_bags = {}

for compound in tqdm(Solvents + Solutes):
    smiles = get_dictionary('smiles')[compound]
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    print(compound)


    def num_list_to_sym(my_list):
        ats = []
        for n in my_list:
            at1 = mol.GetAtomWithIdx(int(n))
            ats.append(at1.GetSymbol())
        ats2 = list(reversed(ats))
        ats = sorted((ats, ats2))[0]
        answer = ''.join(ats)
        return answer


    # SINGLE ATOMS
    single_atoms = {}
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        try:
            single_atoms[sym] += 1
        except KeyError:
            single_atoms[sym] = 1
    for key, value in single_atoms.items():
        try:
            Atom_bags[key] = max(Atom_bags[key], value)
        except KeyError:
            Atom_bags[key] = value

    bonds = []
    bond_dict = {}
    for b in mol.GetBonds():
        a1 = b.GetBeginAtom()
        a2 = b.GetEndAtom()
        sym1 = a1.GetSymbol()
        sym2 = a2.GetSymbol()
        bonds.append(sorted((a1.GetIdx(), a2.GetIdx())))
        try:
            bond_dict[''.join(sorted((sym1, sym2)))] += 1
        except KeyError:
            bond_dict[''.join(sorted((sym1, sym2)))] = 1
    for key, value in bond_dict.items():
        try:
            Bond_bags[key] = max(Bond_bags[key], value)
        except KeyError:
            Bond_bags[key] = value

    # print('b')
    # print(bonds)
    # print(bond_dict)

    # ANGLES
    angles = []
    angle_dict = {}


    def get_bonded_atoms(atom_idx, bonds_list):
        bonded = []
        for x, y in bonds_list:
            if atom_idx == x:
                bonded.append(y)
            elif atom_idx == y:
                bonded.append(x)
        return bonded


    def get_angles(bond1, bond_list):
        bond_list1 = bond_list.copy()
        bond_list1.remove(bond1)
        for i, x in enumerate(bond1):
            bonded = get_bonded_atoms(x, bond_list1)
            for at in bonded:
                if i:
                    angle = bond1 + [at, ]
                else:
                    angle = [at, ] + bond1
                angle2 = list(reversed(angle))
                angle = sorted((angle, angle2))[0]
                if angle not in angles:
                    angles.append(angle)
                    angle_bag = num_list_to_sym(angle)
                    try:
                        angle_dict[angle_bag] += 1
                    except KeyError:
                        angle_dict[angle_bag] = 1


    for bond in bonds:
        get_angles(bond, bonds)

    for key, value in angle_dict.items():
        try:
            Angle_bags[key] = max(Angle_bags[key], value)
        except KeyError:
            Angle_bags[key] = value

    # print('ang')
    # print(angles)
    # print(angle_dict)

    # TORSION
    torsions = []
    torsion_dict = {}


    def get_torsion(angle1, bond_list):
        bond_list1 = bond_list.copy()
        b1 = list(sorted(angle1[1:]))
        b2 = list(sorted(angle1[:-1]))
        bond_list1.remove(b1)
        bond_list1.remove(b2)
        for i, x in enumerate(angle1):
            if not i == 1:
                bonded = get_bonded_atoms(x, bond_list1)
                for at in bonded:
                    if i:
                        torsion = angle1 + [at, ]
                    else:
                        torsion = [at, ] + angle1
                    torsion2 = list(reversed(torsion))
                    torsion = sorted((torsion, torsion2))[0]
                    if torsion not in torsions:
                        torsions.append(torsion)
                        torsion_bag = num_list_to_sym(torsion)
                        try:
                            torsion_dict[torsion_bag] += 1
                        except KeyError:
                            torsion_dict[torsion_bag] = 1
                        x1, x2, x3, x4 = parse_bag(torsion_bag)
                        if x1 == x4 and x2 != x3:
                            torsion_bag = ''.join((x4, x3, x2, x1))
                            try:
                                torsion_dict[torsion_bag] += 1
                            except KeyError:
                                torsion_dict[torsion_bag] = 1


    for angle in angles:
        get_torsion(angle, bonds)

    for key, value in torsion_dict.items():
        try:
            Torsion_bags[key] = max(Torsion_bags[key], value)
        except KeyError:
            Torsion_bags[key] = value

# Chem.Atom.G
# Chem.Bond.GetEndAtom()
# Chem.Mol

print('atoms')
print(Atom_bags)
print('bonds')
print(Bond_bags)
print('angles')
print(Angle_bags)
print('torsion')
print(Torsion_bags)
print(sum([Torsion_bags[x] for x in Torsion_bags]))
together = sum([sum([dict1[x] for x in dict1]) for dict1 in (Atom_bags, Bond_bags, Angle_bags, Torsion_bags)])
print(together)

# bag_sizes = {}
# bags = {}
# for key in sorted(list(Bag_sizes1.keys())):
#     bag_sizes[key] = Bag_sizes1[key]
#     bags[key] = []
# bag_sizes = OrderedDict(bag_sizes)
# bags = OrderedDict(bags)

with open('/Users/balepka/PycharmProjects/msuAI/Solvation_1/Tables/MNSol_bags2.pkl', 'rb') as f:
    my_bags = pkl.load(f)

BoB_pkl = my_bags[0]
BAT_pkl = BoB_pkl
BAT_bags, BAT_sizes = BAT_pkl

for dict1 in (Angle_bags, Torsion_bags):
    for key, value in dict1.items():
        BAT_sizes[key] = value

bag_sizes = {}
bags = {}
for key in sorted(list(BAT_sizes.keys())):
    bag_sizes[key] = BAT_sizes[key]
    bags[key] = []
bag_sizes = OrderedDict(bag_sizes)
bags = OrderedDict(bags)

my_bags = my_bags[:1]
my_bags.append((bags, bag_sizes))

with open('/Users/balepka/PycharmProjects/msuAI/Solvation_1/Tables/MNSol_bags3.pkl', 'wb') as f:
    pkl.dump(my_bags, f)
