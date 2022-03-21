# import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import Dataset
# import torch.nn.functional as F
from Solvation_1.Vectorizers.vectorizers import *
from Solvation_1.config import *


def create_SS_table(df3):
    Solvents = dict(df3['Solvent'].value_counts().items())
    Solutes = dict(df3['SoluteName'].value_counts().items())
    from collections import OrderedDict

    Solvents = OrderedDict(df3['Solvent'].value_counts().items())
    Solutes = OrderedDict(df3['SoluteName'].value_counts().items())
    table_SS = pd.DataFrame(index=Solutes, columns=Solvents)

    for solute in Solutes:
        for solvent in Solvents:
            SS_row = df3.loc[df3['Solvent'] == solvent].loc[df3['SoluteName'] == solute]['DeltaGsolv']
            if SS_row.empty:
                pass
            else:
                table_SS[solvent][solute] = SS_row.item()

    return table_SS


class SS_Dataset(Dataset):

    def __init__(self, ss_table, solvent_vect, solute_vect, transform=None):

        self.vectorizers_map = {
            'solvent_macro_props1': {'func': solvent_macro_props1, 'formats': ['tsv'],
                                     'paths': ['Solvation_1/Tables/Solvent_properties1.tsv']},
            'solute_TESA': {'func': solute_TESA, 'formats': ['feather'],
                            'paths': ['Solvation_1/Tables/df3_f']},
            'test_sp': {'func': test_sp, 'formats': [], 'paths': []},
            'test_up': {'func': test_up, 'formats': [], 'paths': []},
        }

        self.table = ss_table
        self.solvent_vect = solvent_vect
        self.solute_vect = solute_vect
        self.data = []
        self.transform = transform
        if self.table.index.name != 'Unnamed: 0':
            self.table = self.table.set_index('Unnamed: 0')

        for solvent in self.table.columns.tolist():
            for solute in self.table.index.tolist():
                G_solv = self.table[solvent][solute]
                # print(G_solv, solvent, solute)
                if not pd.isna(G_solv):
                    self.data.append((solvent, solute, G_solv))
        # print(f'len data {len(self.data)}')

        self.solvent_func = self.vectorizers_map[self.solvent_vect]['func']
        self.solute_func = self.vectorizers_map[self.solute_vect]['func']
        self.solvent_args = []
        self.solute_args = []
        for form, path in zip(self.vectorizers_map[self.solvent_vect]['formats'],
                              self.vectorizers_map[self.solvent_vect]['paths']):
            self.solvent_args.append(read_format(form)(project_path(path)))
        for form, path in zip(self.vectorizers_map[self.solute_vect]['formats'],
                              self.vectorizers_map[self.solute_vect]['paths']):
            self.solute_args.append(read_format(form)(project_path(path)))

    def __getitem__(self, i):
        solvent, solute, G_solv = self.data[i]
        X1 = self.solvent_func(solvent, self.solvent_args)
        X2 = self.solute_func(solute, self.solute_args)
        X = torch.cat((X1, X2), 1)
        y = torch.tensor(G_solv)
        return X.float(), y.float()

    def __len__(self):
        return len(self.data)
