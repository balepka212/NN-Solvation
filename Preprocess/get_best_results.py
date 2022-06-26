from config import project_path
import torch
import os

output = '/Users/balepka/Downloads/best_losses.txt'


best_data = {}
directory = project_path('Runs')
for folder in os.listdir(directory):
    # print(f'{folder}')
    file_path = os.path.join(directory, folder, 'best', 'best_val_model.pt')
    try:
        with open(file_path, 'rb') as f:
            checkpoint = torch.load(f)
            print(f'{folder} -> {checkpoint["losses"]}')
            best_data[folder] = checkpoint['losses']
    except FileNotFoundError:
        best_dir = os.path.join(directory, folder, 'best')
        try:
            paths = sorted(os.listdir(best_dir),
                           key=lambda x: os.path.getctime(os.path.join(best_dir, x)))
            file_path = os.path.join(best_dir, paths[-1])
            with open(file_path, 'rb') as f:
                try:
                    checkpoint = torch.load(f)
                    print(f'{folder} -> {checkpoint["losses"]}')
                    best_data[folder] = checkpoint['losses']
                except Exception as e:
                    print(e.args)
        except FileNotFoundError:
            print(f'No BEST in {folder}')
    except NotADirectoryError:
        print(f'BAD {folder}')

with open(output, 'w') as f:
    f.write('\t'.join(('folder', 'train', 'val', 'solvent', 'solute')) + '\n')
    for folder, losses in best_data.items():
        f.write(folder + '\t' + '\t'.join((str(float(x)) for x in (losses['train'],
                losses['val'], losses['solvent'], losses['solute']))) + '\n')

