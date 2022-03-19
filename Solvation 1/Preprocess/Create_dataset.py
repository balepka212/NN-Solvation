import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F

def create_df(data_file = r'/Users/balepka/Yandex.Disk.localized/Study/Lab/Neural Network/MNSol-v2009_energies_v2.tsv',
              solvent_props_file = r'/Users/balepka/Yandex.Disk.localized/Study/Lab/Neural Network/MNSolDatabase_v2012/Solvent_properties.tsv'):
    # Reading data from file
    with open(filename) as f:
        t = 0
        data = pd.read_table(f)
        df1 = pd.DataFrame(data)
    # Deleting charges species
    df2 = df1.loc[df1['Charge'].isin([0])]
    # Deleting solvent mixtures
    with open(solvent_props_file) as f:
        data = pd.read_table(f, header=1)
        solvent_props = pd.DataFrame(data)
    names = []
    for name, count in df2['Solvent'].value_counts().items():
        row = solvent_props.loc[solvent_props['Name'] == name]
        values = np.array(row[['nD', 'alpha', 'beta', 'gamma', 'epsilon', 'phi', 'psi']])
        # print(f'{name} -> {count} -> {values.shape[0]}')
        if values.shape[0] == 0:
            verboseprint(name)
            names.append(name)
    df3 = df2.loc[~df2['Solvent'].isin(names)]
    df3 = df3[df3.SoluteName != 'water dimer']
    return df3

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

    # table_SS.to_csv('/Users/balepka/Yandex.Disk.localized/Study/Lab/Neural Network/Dataset preparation/table_SS.csv')

    # table_SS.head()
    # table_SS.columns.tolist()
    # table_SS.index.tolist()
    # for solvent in table_SS.columns.tolist():
    #     for solute in table_SS.index.tolist():
    #         G_solv = table_SS[solvent][solute]
    #         if not pd.isna(G_solv):
    #             verboseprint(f's:{solvent} - {solute}: {G_solv}')
    #         else:
    #             verboseprint(f'NaN:{solvent} - {solute}: {G_solv}')


#
# filename = r'/Users/balepka/Yandex.Disk.localized/Study/Lab/Neural Network/MNSol-v2009_energies_v2.tsv'
# with open(filename) as f:
#     t = 0
#     data = pd.read_table(f)
#     df1 = pd.DataFrame(data)
#
# df2 = df1.loc[df1['Charge'].isin([0])]

# Deleting solvent mixtures
# filename2 = r'/Users/balepka/Yandex.Disk.localized/Study/Lab/Neural Network/MNSolDatabase_v2012/Solvent_properties.tsv'
# with open(filename2) as f:
#     data = pd.read_table(f, header=1)
#     solvent_props = pd.DataFrame(data)
# names = []
# for name, count in df2['Solvent'].value_counts().items():
#     row = solvent_props.loc[solvent_props['Name'] == name]
#     values = np.array(row[['nD', 'alpha', 'beta', 'gamma', 'epsilon', 'phi', 'psi']])
#     # print(f'{name} -> {count} -> {values.shape[0]}')
#     if values.shape[0] == 0:
#         print(name)
#         names.append(name)
# df3 = df2.loc[~df2['Solvent'].isin(names)]
# df3 = df3[df3.SoluteName != 'water dimer']

#Creating table solvent - solute
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

table_SS.to_csv('/Users/balepka/Yandex.Disk.localized/Study/Lab/Neural Network/Dataset preparation/table_SS.csv')

# table_SS.head()
table_SS.columns.tolist()
table_SS.index.tolist()
for solvent in table_SS.columns.tolist():
    for solute in table_SS.index.tolist():
        G_solv = table_SS[solvent][solute]
        if not pd.isna(G_solv):
            verboseprint(f's:{solvent} - {solute}: {G_solv}')
        else:
            verboseprint(f'NaN:{solvent} - {solute}: {G_solv}')



## Creating SS Dataset
from torch.utils.data import Dataset


class SS_Dataset(Dataset):
    def __init__(self, table, solvent_props, solute_props, args = ((), ()), transform=None):
        self.table = table
        self.solvent_props = solvent_props
        self.solute_props = solute_props
        self.data = []
        self.transform = transform
        self.table = self.table.set_index('Unnamed: 0')
        self.sp, self.up = args
        for solvent in self.table.columns.tolist():
            for solute in self.table.index.tolist():
                G_solv = self.table[solvent][solute]
                if not pd.isna(G_solv):
                    self.data.append((solvent, solute, G_solv))

    def __getitem__(self, i):
        solvent, solute, G_solv = self.data[i]
        X1 = self.solvent_props(solvent, self.sp)
        X2 = self.solute_props(solute, self.up)
        #check dim
        # print(f'X1 - {X1.shape}')
        # print(f'X2 - {X2.shape}')
        verboseprint(f'X1 {X1.dtype}]')
        verboseprint(f'X2 {X2.dtype}]')
        X = torch.cat((X1, X2), 1)
        y = torch.tensor(G_solv)
        return X.float(), y.float()

    def __len__(self):
        return len(self.data)


table_v1 = pd.read_csv(
    '/Users/balepka/Yandex.Disk.localized/Study/Lab/Neural Network/Dataset preparation/table_SS_v1/table_SS_v1.csv')
