from collections import OrderedDict

from config import project_path
import torch
import os
import pickle as pkl

output = '/Users/balepka/Downloads/best_krr_v2.txt'

def split_solute_solvent(filename: str):
    filename = filename.split('_KRR')[0]
    filename = filename.replace('JustBonds', 'JB')
    filename = filename.replace('Morgan_2_124', 'Morgan')
    args = filename.split('_')

    if len(args) == 2:
        solvent, solute = args[0], args[1]
        return solvent, solute

    elif filename.startswith('Morgan_2_2to20'):
        solvent = 'Morgan_2_2to20'
        solute = filename[15:]
        return solvent, solute

    elif filename.endswith('Morgan_2_2to20'):
        solute = 'Morgan_2_2to20'
        solvent = filename[:15]
        return solvent, solute
    #
    # elif args[0].lower() == 'morgan':
    #     solvent = '_'.join(args[:3])
    #     if args[3].lower() == 'morgan':
    #         solute = '_'.join(args[3:6])
    #         return solvent, solute
    #     solute = args[3]
    #     return solvent, solute
    #
    # else:
    #     solvent = args[0]
    #     solute = '_'.join(args[1:4])
    #     return solvent, solute



best_data = {}
directory = project_path('Run_results/KRR')
for folder in os.listdir(directory):
    # print(folder)
    if ('KRR2' in folder) or ('KRR1' in folder):
    # print(f'{folder}')
        file_path = os.path.join(directory, folder, 'all_curves.pkl')
        solvent, solute = split_solute_solvent(folder)
        try:
            with open(file_path, 'rb') as f:
                # print(file_path)
                all_curves = pkl.load(f)
                val = all_curves['val']
                best = {}
                for kernel, data in val.items():
                    best[kernel] = data[-1]
                sorted_best = sorted(best.keys(), key=lambda x: best[x])
                best_kernel = sorted_best[0]
                best_data[folder] = {'S_vect': solvent, 'U_vect': solute, 'kernel': best_kernel}
                for dataset in ('train', 'val', 'solvent', 'solute'):
                    best_data[folder][dataset] = all_curves[dataset][best_kernel][-1]

        except FileNotFoundError:
            print(f'FNF {folder}')
        except NotADirectoryError:
            print(f'NAD {folder}')



with open(output, 'w') as f:
    f.write('\t'.join(('folder', 'S_vect', 'U_vect', 'kernel', 'train', 'val', 'solvent', 'solute')) + '\n')
    for folder, losses in best_data.items():
        f.write(folder + '\t' +
                '\t'.join(str(x) for x in (losses['S_vect'],
                                                   losses['U_vect'], losses['kernel'])) + '\t' +
                '\t'.join((str(float(x)) for x in (losses['train'],
                losses['val'], losses['solvent'], losses['solute']))) + '\n')

with open(project_path('Other/best_KRR5.pkl'), 'wb') as f:
    pkl.dump(best_data, f)

print(best_data)

