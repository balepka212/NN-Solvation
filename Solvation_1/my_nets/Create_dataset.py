from torch.utils.data import Dataset
from Solvation_1.Vectorizers.vectorizers import *
from Solvation_1.config import *


def create_SS_table(df3):
    """ TODO description"""
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
    """ TODO description"""

    def __init__(self, ss_table, solvent_vect, solute_vect, normalize=(False, False, False),
                 full_data='Solvation_1/Tables/Entire_table3.tsv', show_norm_params=True):

        self.vectorizers_map = {
            'solvent_macro_props1': {'func': solvent_macro_props1, 'formats': ['tsv'],
                                     'paths': ['Solvation_1/Tables/Solvent_properties3.tsv']},
            'solute_TESA': {'func': solute_TESA, 'formats': ['feather'],
                            'paths': ['Solvation_1/Tables/df3_3']},
            'test_sp': {'func': test_sp, 'formats': [], 'paths': []},
            'test_up': {'func': test_up, 'formats': [], 'paths': []},
        }

        self.table = ss_table
        self.solvent_vect = solvent_vect
        self.solute_vect = solute_vect
        self.data = []

        if self.table.index.name != 'Unnamed: 0':
            self.table = self.table.set_index('Unnamed: 0')

        for solvent in self.table.columns.tolist():
            for solute in self.table.index.tolist():
                G_solv = self.table[solvent][solute]
                if not pd.isna(G_solv):
                    self.data.append((solvent, solute, G_solv))

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

        self.normalize = normalize
        self.full_data = full_data
        if True in self.normalize:
            # norm_parameters
            norm_data = []
            norm_table = pd.read_table(project_path(self.full_data))
            if norm_table.index.name != 'Unnamed: 0':
                norm_table = norm_table.set_index('Unnamed: 0')
            for solvent in norm_table.columns.tolist():
                for solute in norm_table.index.tolist():
                    G_solv = norm_table[solvent][solute]
                    if not pd.isna(G_solv):
                        norm_data.append((solvent, solute, G_solv))

            for i in range(len(norm_data)):
                solvent, solute, G_solv = norm_data[i]
                X1 = self.solvent_func(solvent, self.solvent_args)
                X2 = self.solute_func(solute, self.solute_args)
                y = torch.tensor(G_solv)
                if i == 0:
                    X_solvent, X_solute, y_all = X1, X2, y
                else:
                    X_solvent = torch.vstack((X_solvent, X1))
                    X_solute = torch.vstack((X_solute, X2))
                    y_all = torch.vstack((y_all, y))

            self.X_solvent_norm = torch.std_mean(X_solvent, dim=0)
            self.X_solute_norm = torch.std_mean(X_solute, dim=0)
            self.y_norm = torch.std_mean(y_all, dim=0)
            self.norm_params = {'Solvent': self.X_solvent_norm, 'Solute': self.X_solute_norm, 'G': self.y_norm}
            if show_norm_params:
                print(f'length-> S: {len(X_solvent)}, U: {len(X_solute)}, G: {len(y_all)}')
                print(f'S: {self.X_solvent_norm}')
                print(f'U: {self.X_solute_norm}')
                print(f'G: {self.y_norm}')

    def __getitem__(self, i):
        solvent, solute, G_solv = self.data[i]
        X1 = self.solvent_func(solvent, self.solvent_args)
        X2 = self.solute_func(solute, self.solute_args)

        G_n, solvent_n, solute_n = self.normalize
        if solvent_n:
            std, mean = self.X_solvent_norm
            X1 = (X1 - mean) / std

        if solute_n:
            std, mean = self.X_solute_norm
            X2 = (X2 - mean) / std

        G_solv = torch.tensor(G_solv)
        if G_n:
            std, mean = self.y_norm
            G_solv = (G_solv - mean) / std

        X = torch.cat((X1, X2), 1)
        return X.float(), G_solv.float()

    def __len__(self):
        return len(self.data)
