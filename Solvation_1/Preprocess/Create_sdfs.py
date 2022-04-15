import periodictable as pt
from rdkit import Chem

atom_dict = pt.elements.__dict__['_element']
# print(atom_dict)
el_to_n_dict = {}
for N, element in atom_dict.items():
    el_to_n_dict[element] = int(N)


class Atom:
    def __init__(self, atom_number, x, y, z):
        self.atomic_number = atom_number
        self.xyz = (x, y, z)
        self.xyz_id = None
        self.smiles_id = None
        self.element = atom_dict[self.atomic_number]


class my_molecule:
    def __init__(self, smiles, path, classic_xyz=True):
        self.classic_xyz = classic_xyz
        self.atoms = []
        self.smiles = smiles
        self.xyz_path = path
        self.read_xyz()
        self.mol = Chem.MolFromSmiles(self.smiles)

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
                        at.xyz_id = i-3
                        self.atoms.append(at)



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
