from create_xyz import xyz_from_file
from config import project_path
import os

directory = os.fsencode(project_path('Tables/Reserve/all_solutes'))

for i, file in enumerate(os.listdir(directory)):
    filename = os.fsdecode(file)
    if filename.endswith(".xyz"):
        # print(f'{i}: {filename}')
        file_path = project_path('Tables/Reserve/all_solutes/'+filename)
        xyz_from_file(file_path, project_path('Tables/Reserve/xyz_files'), copy=False)
        # print(os.path.join(directory, filename))

        continue
    else:
        continue

# file_path = '/Users/balepka/Yandex.Disk.localized/Study/Lab/Neural Network/MNSolDatabase_v2012/all_solutes/i133.xyz'
# out_folder = '/Users/balepka/Downloads'
#
# xyz_from_file(file_path, out_folder)





