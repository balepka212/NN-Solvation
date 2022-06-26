from collections import OrderedDict

from config import project_path
import torch
import os
import pickle as pkl

output = '/Users/balepka/Downloads/best_krr_v1.txt'



best_data = {}
directory = project_path('Run_results')
for folder in os.listdir(directory):
    if 'KRR1' in folder:
    # print(f'{folder}')
        file_path = os.path.join(directory, folder, 'all_curves.pkl')
        solvent, solute, _ = folder.split('_')
        try:
                with open(file_path, 'rb') as f:
                    all_curves = pkl.load(f)
                    val = all_curves['val']
                    best = {}
                    for kernel, data in val.items():
                        best[kernel] = data[-1]
                    sorted_best = sorted(best.keys(), key=lambda x: best[x])
                    best_kernel = sorted_best[0]
                    best_data[folder] = {'S_vect': solvent, 'U_vect': solute, 'kernel':best_kernel}
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

with open(project_path('Other/best_KRR2.pkl'), 'wb') as f:
    pkl.dump(best_data, f)


