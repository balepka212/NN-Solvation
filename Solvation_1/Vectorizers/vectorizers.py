import pandas as pd
import torch
from pyarrow import feather

# solute_TESA_df = feather.read_feather('../../Solvation_1/Tables/df3_f')
# solvent_macro_props_table = pd.read_table('../../Solvation_1/Tables/Solvent_properties1.tsv')
# print(solvent_macro_props_table)

def test_sp(solvent, args=None):
    return torch.tensor(len(solvent))


def test_up(solute, args=None):
    return torch.tensor(len(solute))


def solute_TESA(solute, args):
    '''df is pd dataframe which is df3'''
    df, *args = args
    row = df[df['SoluteName'] == solute][:1]
    out = row[row.columns[17:26]]
    out = torch.tensor(out.values, dtype=torch.float)
    # print(f'type TESA: {out.dtype}')
    return out


def solvent_macro_props1(solvent, args):
    '''table is pd dataframe'''
    # print(f'{table}')
    table, *args = args
    row = table[table['Name'] == solvent]
    # print(f'row {row}')
    out = row[row.columns[2:]]
    out = torch.tensor(out.values, dtype=torch.float)
    return out


