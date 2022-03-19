import pandas as pd
import torch
from pyarrow import feather

solute_TESA_df = feather.read_feather('../Solvation_1/Tables/df3_f')
solvent_macro_props_table = pd.read_table('../Solvation_1/Tables/Solvent_properties1.tsv')

def test_sp(solvent):
    return torch.tensor(len(solvent))


def test_up(solute):
    return torch.tensor(len(solute))


def solute_TESA(solute, df=solute_TESA_df):
    '''df is pd dataframe which is df3'''
    row = df[df['SoluteName'] == solute][:1]
    out = row[row.columns[17:26]]
    out = torch.tensor(out.values, dtype=torch.float)
    # print(f'type TESA: {out.dtype}')
    return out


def solvent_macro_props1(solvent, table=solvent_macro_props_table):
    '''table is pd dataframe'''
    print(f'{table}')
    row = table[table['Name'] == solvent]
    print(f'row {row}')
    out = row[row.columns[2:]]
    out = torch.tensor(out.values, dtype=torch.float)
    return out


