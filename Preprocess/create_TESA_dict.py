from pyarrow import feather
import torch
import pickle as pkl
from config import project_path


with open(project_path('Tables/Solvents_Solutes.pkl'), 'rb') as f:
    Solvents, Solutes = pkl.load(f)


def create_tensor(solute, args):
    df, *args = args
    row = df[df['SoluteName'] == solute][:1]  # get the row with desired Solute
    out = row[row.columns[20:29]]  # get TESA data
    out = torch.tensor(out.values, dtype=torch.float)
    return out.squeeze()


df3 = feather.read_feather('/Users/balepka/PycharmProjects/msuAI/Tables/df3_3')
data_dict = {}
for compound in Solutes:
    data_dict[compound] = create_tensor(compound, (df3,))

with open('/Users/balepka/PycharmProjects/msuAI/Tables/TESA_dict.pkl', 'wb') as f:
    pkl.dump(data_dict, f)
