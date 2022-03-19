import torch
# import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
# import torch.nn.functional as F
from pyarrow import feather

def create_df(data_file = r'/Users/balepka/Yandex.Disk.localized/Study/Lab/Neural Network/MNSol-v2009_energies_v2.tsv',
              solvent_props_file = r'/Users/balepka/Yandex.Disk.localized/Study/Lab/Neural Network/MNSolDatabase_v2012/Solvent_properties.tsv'):
    # Reading data from file
    with open(data_file) as f:
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
            # print(name)
            names.append(name)
    df3 = df2.loc[~df2['Solvent'].isin(names)]
    df3 = df3[df3.SoluteName != 'water dimer']
    return df3

df3 = create_df()
feather.write_feather(df3, '../Tables/df3_f')


# feather.read_feather('example_feather')


