

def test_sp(solvent):
    return torch.tensor(len(solvent))


def test_up(solute):
    return torch.tensor(len(solute))


def solute_TESA(solute, df):
    row = df[df['SoluteName'] == solute][:1]
    out = row[row.columns[17:26]]
    out = torch.tensor(out.values, dtype=torch.float)
    # print(f'type TESA: {out.dtype}')
    return out


def solvent_macro_props1(solvent, path_to_table):
    table = pd.read_table(path_to_table)
    row = table[table['Name'] == solvent]
    out = row[row.columns[2:]]
    out = torch.tensor(out.values, dtype=torch.float)
    return out


