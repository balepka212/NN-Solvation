from pyarrow import feather
from collections import OrderedDict
import torch
import pickle as pkl

from Vectorizers.vectorizers import get_dictionary
from config import project_path


def get_classes(solute, args):
    df, *args = args
    row = df[df['SoluteName'] == solute][:1]  # get the row with desired Solute
    cl = row[row.columns[6:9]]  # get Classification
    return cl.squeeze().values


def create_tensor(classes):
    # out = torch.tensor(out.values, dtype=torch.float)
    # print(cl.squeeze().values)
    class_dict = OrderedDict(
        {1: 3, 2: 19, 3: 8, 4: 5, 5: 2, 6: 3, 7: 3, 8: 0, 9: 3, 10: 7, 11: 4, 12: 2, 13: 7, 14: 2, 15: 0})
    class_dict2 = OrderedDict({2: {1: 6, 2: 2, 4: 2}, 3: {3: 2}, 4: {1: 2}})

    empty = [0, ] * 83
    x, y, z = classes
    x_n = 0
    for k, v in class_dict.items():
        if k == x:
            empty[x_n] = 1
            break
        else:
            x_n += v
            x_n += 1
    y_n = x_n
    if y != 0:
        if x in class_dict2:
            for k, v in class_dict2[x].items():
                if k < y:
                    y_n += v
        y_n = y_n + y
        empty[y_n] = 1
        if z != 0:
            z_n = y_n + z
            empty[z_n] = 1
    empty = torch.tensor(empty, dtype=torch.float)
    return empty


def tensor_to_classes(tensor):
    FIRST, SECOND, THIRD = 0, 0, 0
    class_dict = OrderedDict(
        {1: 3, 2: 19, 3: 8, 4: 5, 5: 2, 6: 3, 7: 3, 8: 0, 9: 3, 10: 7, 11: 4, 12: 2, 13: 7, 14: 2, 15: 0})
    class_dict2 = OrderedDict({2: {1: 6, 2: 2, 4: 2}, 3: {3: 2}, 4: {1: 2}})
    firsts =[0, 4, 24, 33, 39, 42, 46, 50, 51, 55, 63, 68, 71, 79, 82]
    # FIRST LEVEL
    for i, f in enumerate(firsts, start=1):
        if tensor[f]:
            FIRST = i
            break

    # SECOND LEVEL
    if FIRST in class_dict2:
        i = 1
        s = firsts[FIRST-1] + 1
        while s <= firsts[FIRST-1]+class_dict[FIRST]:
            if tensor[s]:
                SECOND = i
                #THIRD LEVEL
                if i in class_dict2[FIRST]:
                    s2 = s
                    for j in range(1, class_dict2[FIRST][SECOND]+1):
                        if tensor[s2+j]:
                            THIRD = j
                            break
                break
            if i in class_dict2[FIRST]:
                s += (class_dict2[FIRST][i]+1)
            else:
                s += 1
            i += 1


    else:
        for i, s in enumerate(range(firsts[FIRST-1]+1, firsts[FIRST-1] +  class_dict[FIRST]+1), start=1):
            if tensor[s]:
                SECOND = i

    return FIRST, SECOND, THIRD






if __name__ == '__main__':

    with open(project_path('Tables/Solvents_Solutes.pkl'), 'rb') as f:
        Solvents, Solutes = pkl.load(f)

    df3 = feather.read_feather('/Users/balepka/PycharmProjects/msuAI/Tables/df3_3')
    data_dict = {}
    for compound in Solutes:
        data_dict[compound] = create_tensor(get_classes(compound, (df3,)))

    #solvent that present in solutes


    for solvent in Solvents:
        compound = get_dictionary('solventtosolute')[solvent]
        if compound != 'NOFILE':
            data_dict[solvent] = create_tensor(get_classes(compound, (df3,)))

    # missed solvents
    missed_solvent_dict = {
        'decalin': (2, 1, 3),
        'tributylphosphate': (11, 1, 0),
        'chlorohexane': (6, 1, 0),
        'dibromoethane': (7, 1, 0),
        'hexadecyliodide': (15, 0, 0),
        'ethoxybenzene': (2, 3, 0),
        'phenylether': (2, 3, 0),
        'onitrotoluene': (4, 2, 0),
        'isopropyltoluene': (2, 1, 6),
        'secbutylbenzene': (2, 1, 6),
        'bromooctane': (7, 1, 0)
        }

    for solvent, cl in missed_solvent_dict.items():
        data_dict[solvent] = create_tensor(cl)

    with open('/Tables/Acree/Classification.pkl', 'wb') as f:
        pkl.dump(data_dict, f)
