import pandas as pd
from pyarrow import feather
import torch
import pickle as pkl
from Solvation_1.config import project_path


with open(project_path('Solvation_1/Tables/Solvents_Solutes.pkl'), 'rb') as f:
    Solvents, Solutes = pkl.load(f)


def create_tensor(solvent, args):
    table, *args = args
    row = table[table['Name'] == solvent]  # get the row with desired Solvent
    out = row[row.columns[2:]]  # get Macro Properties data
    out = torch.tensor(out.values, dtype=torch.float)
    return out.squeeze()


table1 = pd.read_table(project_path('Solvation_1/Tables/Solvent_properties3.tsv'))
data_dict = {}
for compound in Solvents:
    data_dict[compound] = create_tensor(compound, (table1,))


with open('/Users/balepka/PycharmProjects/msuAI/Solvation_1/Tables/Macro_dict.pkl', 'wb') as f:
    pkl.dump(data_dict, f)
#
#
#
# def solvent_macro_props1(solvent, args, params=None):
#     """
#     Solvent Macro Properties vectorizer.
#
#     Returns a vector of properties: nD, alpha, beta, gamma, epsilon, phi, psi. Data is obtained from MNSol database.
#
#     Parameters
#     ----------
#     solvent: str
#         solvent to be vectorized
#     args: [pd.table]
#         database where 2-... columns are properties of solvent. column 'Name' contains solvent
#     params: None
#         not needed here
#     """
#
#     table, *args = args
#     row = table[table['Name'] == solvent]  # get the row with desired Solute
#     out = row[row.columns[2:]]  # get Macro Properties data
#     out = torch.tensor(out.values, dtype=torch.float)
#     return out