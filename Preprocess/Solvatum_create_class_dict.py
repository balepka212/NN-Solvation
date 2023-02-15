import pickle as pkl
from config import project_path
from create_Class import create_tensor

with open(project_path('Tables/Solvatum/pre_class_dict.pkl'), 'rb') as f:
    pre_class_dict = pkl.load(f)

with open(project_path('Tables/Classification_dict.pkl'), 'rb') as f:
    Classes_dict = pkl.load(f)

the_dict = {}
for solvent, value in pre_class_dict.items():
    if value in Classes_dict:
        the_dict[solvent] = Classes_dict[value]
    else:
        the_dict[solvent] = create_tensor(value)


with open(project_path('Tables/Solvatum/Classification_dict.pkl'), 'wb') as f:
    pkl.dump(the_dict, f)