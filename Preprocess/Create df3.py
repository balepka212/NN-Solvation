import pandas as pd
import numpy as np
from pyarrow import feather


def create_df(data_file=r'/Users/balepka/PycharmProjects/msuAI/Tables/MNSol_alldata.txt',
              solvent_props_file=r'/Users/balepka/PycharmProjects/msuAI/Tables/Solvent_properties3.tsv'):
    """TODO description"""
    # Reading data from file
    with open(data_file) as f:
        data = pd.read_table(f)
        df1 = pd.DataFrame(data)
    # Deleting charges species
    print(f'df1: {len(df1)}')

    df2 = df1.loc[df1['Charge'].isin([0])]
    print(f'df2: {len(df2)}')

    # Deleting solvent mixtures
    with open(solvent_props_file) as f:
        data = pd.read_table(f, header=0)
        solvent_props = pd.DataFrame(data)
    names = []
    c, c1 = 0, 0
    for name, count in df2['Solvent'].value_counts().items():
        # print(name, count)
        row = solvent_props.loc[solvent_props['Name'] == name]
        values = np.array(row[['nD', 'alpha', 'beta', 'gamma', 'epsilon', 'phi', 'psi']])
        # print(f'{name} -> {count} -> {values.shape[0]}')
        if values.shape[0] == 0:
            # print(name)
            names.append(name)
            # print(f'– {name} {count}')
            c += count
        else:
            # print(values)
            # print(f'+ {name} {count}')
            c1 += count
    # print(c)
    df3 = df2.loc[~df2['Solvent'].isin(names)]
    print(f'df3.0: {len(df3)}')

    df3 = df3[df3.SoluteName != 'waterdimer']
    print(f'df3: {len(df3)}')
    return df3


if __name__ == "__main__":
    my_df = create_df()
    feather.write_feather(my_df, '../Tables/df3_3')
