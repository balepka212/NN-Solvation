import torch
from config import project_path
import pickle as pkl

with open(project_path('Tables/Solvents_Solutes.pkl'), 'rb') as f:
    Solvents, Solutes = pkl.load(f)

blank_data = {}
for compound in Solvents+Solutes:
    out = torch.tensor([0.0,])
    blank_data[compound] = out

with open('/Users/balepka/PycharmProjects/msuAI/Tables/blank_dict.pkl', 'wb') as f:
    pkl.dump(blank_data, f)