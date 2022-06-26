import random

from torch.utils.data import Dataset
from Vectorizers.vectorizers import *
from config import *


def create_SS_table(df3):
    """
    Returns a table of Solvent-Solute pairs deltaG values from given pd.DataFrame

    Here the df3 is MNSol database without charged species and solvent mixtures

    Parameters
    ----------
    df3: pd.DataFrame
        A dataframe that contains deltaG data and columns of 'Solvent' and 'SoluteName'
    """

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
    """
        Class represent Solvent-Solute dataset.

        ...

        Attributes
        ----------
        vectorizers_map : dict
            A dict that contains necessary parameters for vectorizers
        table : pd.DataFrame
            A table of solvent-solute values of deltaG
        solvent_vect : str
            name of vectorizer used to represent solvent
        solute_vect : str
            name of vectorizer used to represent solute
        data : list
            list of (solvent, solute, data) tuples
        solvent_func : function
            A function representing a vectorizer for solvent
        solute_func : function
            A function representing a vectorizer for solute
        solvent_args : list
            A list of arguments for function representing a vectorizer for solvent
        solute_args : list
            A list of arguments for function representing a vectorizer for solute
        solvent_params : list
            A list of parameters for function representing a vectorizer for solvent
        solute_params : list
            A list of parameters for function representing a vectorizer for solute
        normalize : (bool, bool, bool)
            A tuple of three bools showing if normalization is required for solvent, solute and G_solv respectively
        full_data : str
            A path to entire table. Used for normalization
        X_solvent_norm : (tensor, tensor)
            A tuple representing std and mean tensors for solvent
        X_solute_norm : (tensor, tensor)
            A tuple representing std and mean tensors for solute
        y_norm : (tensor, tensor)
            A tuple representing std and mean tensors for G_true
        norm_params : dict
            A dict in which normalization parameters are stored.
            'Solvent' for solvent, 'Solute' for solute and 'G' for G_true

        Methods
        -------
        says(sound=None)
            Prints the animals name and what sound it makes
        """

    def __init__(self, ss_table, solvent_vect, solute_vect, normalize=(False, False, False),
                 full_data='Tables/Entire_table3.tsv', show_norm_params=True):
        # A dict for vectorizers necessary data
        self.vectorizers_map = {
            'test_sp': 'testsp',
            'test_up': 'testup',
            'blank': 'blank',
            'classification': 'classification',
            'class':'classification',
            'solvent_macro_props1': 'macro',
            'macro': 'macro',
            'solute_TESA': 'tesa',
            'tesa': 'tesa',
            'Morgan_fp_2_124': 'morgan2124',
            'morgan': 'morgan2124',
            'just_bonds':'justbonds',
            'jb':'justbonds',
            'bag_of_bonds': 'bob',
            'bob': 'bob',
            'BAT': 'bat',
            'slatm': 'slatm',
            'soap': 'soap',

        }
        # self.vectorizers_map = {
        #     'solvent_macro_props1': {'func': solvent_macro_props1, 'formats': ['tsv'],
        #                              'paths': ['Tables/Solvent_properties3.tsv'], 'params': None},
        #     'solute_TESA': {'func': solute_TESA, 'formats': ['feather'],
        #                     'paths': ['Tables/df3_3'], 'params': None},
        #     'classification': {'func': classification, 'formats': [],
        #                        'paths': [], 'params': None},
        #     'test_sp': {'func': test_sp, 'formats': [], 'paths': [], 'params': None},
        #     'test_up': {'func': test_up, 'formats': [], 'paths': [], 'params': None},
        #     'Morgan_fp_2_124': {'func': morgan_fingerprints, 'formats': [], 'paths': [], 'params': [2, 124, False]},
        #     'bag_of_bonds': {'func': bag_of_bonds, 'formats': [], 'paths': [], 'params': None},
        #     'BAT': {'func': BAT, 'formats': [], 'paths': [], 'params': None},
        # }

        self.table = ss_table
        try:
            self.solvent_vect = self.vectorizers_map[solvent_vect]
        except KeyError:
            self.solvent_vect = solvent_vect
        try:
            self.solute_vect = self.vectorizers_map[solute_vect]
        except KeyError:
            self.solute_vect = solute_vect
        self.data = []

        # Set a column with Solutes as index column
        if self.table.index.name != 'Unnamed: 0':
            self.table = self.table.set_index('Unnamed: 0')
        # Create data strings
        for solvent in self.table.columns.tolist():
            for solute in self.table.index.tolist():
                G_solv = self.table[solvent][solute]
                if not pd.isna(G_solv):
                    self.data.append((solvent, solute, G_solv))

        # # Preparing vectorizers parameters
        # self.solvent_func = self.vectorizers_map[self.solvent_vect]['func']
        # self.solute_func = self.vectorizers_map[self.solute_vect]['func']
        # self.solvent_args = []
        # self.solute_args = []
        # for form, path in zip(self.vectorizers_map[self.solvent_vect]['formats'],
        #                       self.vectorizers_map[self.solvent_vect]['paths']):
        #     self.solvent_args.append(read_format(form)(project_path(path)))
        # for form, path in zip(self.vectorizers_map[self.solute_vect]['formats'],
        #                       self.vectorizers_map[self.solute_vect]['paths']):
        #     self.solute_args.append(read_format(form)(project_path(path)))
        # self.solvent_params = self.vectorizers_map[self.solvent_vect]['params']
        # self.solute_params = self.vectorizers_map[self.solute_vect]['params']

        # Check if any normalization is needed
        self.normalize = normalize
        self.full_data = full_data
        if True in self.normalize:
            G_n, solvent_n, solute_n = self.normalize
            # norm_parameters
            norm_data = []
            # norm parameters are calculated from all the available data
            norm_table = pd.read_table(project_path(self.full_data))

            # Set a column with Solutes as index column
            if norm_table.index.name != 'Unnamed: 0':
                norm_table = norm_table.set_index('Unnamed: 0')
            # Create list of solvent, solute, G_solv values
            for solvent in norm_table.columns.tolist():
                for solute in norm_table.index.tolist():
                    G_solv = norm_table[solvent][solute]
                    if not pd.isna(G_solv):
                        norm_data.append((solvent, solute, G_solv))

            # Create X and y tensors
            for i in range(len(norm_data)):
                solvent, solute, G_solv = norm_data[i]
                X1 = get_sample(solvent, self.solvent_vect)
                X2 = get_sample(solute, self.solute_vect)
                y = torch.tensor(G_solv)

                # Create initial tensors
                if i == 0:
                    X_solvent, X_solute, y_all = X1, X2, y
                # Stack all the tensors together
                else:
                    X_solvent = torch.vstack((X_solvent, X1))
                    X_solute = torch.vstack((X_solute, X2))
                    y_all = torch.vstack((y_all, y))

            def std_mean_no_zero(tensor, dim=0):
                """ Prevents std to be 0. Replaces 0 std with 1."""
                s1, m1 = torch.std_mean(tensor, dim=dim)
                s2 = s1 + (s1 == 0) * 1
                return tuple((s2, m1))

            self.X_solvent_norm = std_mean_no_zero(X_solvent, dim=0)
            self.X_solute_norm = std_mean_no_zero(X_solute, dim=0)
            self.y_norm = std_mean_no_zero(y_all, dim=0)
            self.norm_params = {'Solvent': self.X_solvent_norm, 'Solute': self.X_solute_norm, 'G': self.y_norm}

            # If needed norm parameters will be printed
            if show_norm_params:
                print(f'length check-> Solvent: {len(X_solvent)}, Solute: {len(X_solute)}, G_solv: {len(y_all)}\n')
                if solvent_n:
                    print(f'Solvent\n std: {self.X_solvent_norm[0]} \n mean: {self.X_solvent_norm[1]}')
                if solute_n:
                    print(f'Solute\n std: {self.X_solute_norm[0]} \n mean: {self.X_solute_norm[1]}')
                if G_n:
                    print(f'G_solv\n std: {self.y_norm[0]} \n mean: {self.y_norm[1]}\n')

        # Create X and y for ML
        X, y = [], []
        for i in range(self.__len__()):
            XXX, yyy = self.__getitem__(i)
            X.append(XXX.squeeze().tolist())
            y.append(yyy.squeeze().tolist())
        self.X = X
        self.y = y

    def __getitem__(self, i):
        solvent, solute, G_solv = self.data[i]
        # Create X and y tensors
        X1 = get_sample(solvent, self.solvent_vect)
        X2 = get_sample(solute, self.solute_vect)

        # Apply normalization if required
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

    def train_val_split(self, split):
        """
            Splits into

            Parameters
            ----------
            split: Union(float, int, Iterable)
                float - fraction of train data, the rest is val
                int - number of train samples, the rest is val
                Iterable:
                    (float, float) - fraction of train, fraction of val
                    (int, int) - number of train and val samples
            """
        if type(split) is float:
            t_length = round(self.__len__()*split//1)
        elif type(split) is int:
            t_length = split
        else:
            try:
                l = len(split)
                if l == 2:
                    t, v = split
                    if type(t) is float and type(v) is float:
                        assert sum(split) == 1, 'Sum is not 1'
                        t_length = round(self.__len__() * t // 1)
                    elif type(t) is int and type(v) is int:
                        assert sum(split) == self.__len__(), 'Sum is not equal to length'
                        t_length = t
                    else:
                        print('Weird types')

                else:
                    print('Wrong length of split: should be 2.')
            except TypeError:
                print('Split has no length')

        if t_length:
            v_length = self.__len__() - t_length
            indexes = list(range(self.__len__()))
            random.shuffle(indexes)
            t_ind = indexes[:t_length]
            v_ind = indexes[t_length:]
            train_dataset = SS_subset(self, t_ind)
            val_dataset = SS_subset(self, v_ind)
        return train_dataset, val_dataset


class SS_subset(Dataset):
    def __init__(self, parent_dataset: SS_Dataset, indexes):
        self.data = [(parent_dataset.__getitem__(i)) for i in indexes]
        self.X_solvent_norm = parent_dataset.X_solvent_norm
        self.X_solute_norm = parent_dataset.X_solute_norm
        self.y_norm = parent_dataset.y_norm
        self.norm_params = parent_dataset.norm_params
        self.normalize = parent_dataset.normalize
        X, y = [], []
        for XXX, yyy in self.data:
            X.append(XXX.squeeze().tolist())
            y.append(yyy.squeeze().tolist())
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
       XXX, yyy = self.data[i]
       return XXX, yyy
