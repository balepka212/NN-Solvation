import periodictable as pt
from pyarrow import feather
from Solvation_1.config import *



def xyz_from_file(file, output, copy=True):
    """TODO description"""
    atom_dict = pt.elements.__dict__['_element']
    with open(file, 'r') as f:
        lines = f.readlines()
        n_lines = len(lines)
        # print(n_lines)

        out_name = output+('/copy_' if copy else '/')+file.split('/')[-1]
        with open(out_name, 'w+') as out:
            out.write(str(n_lines-3)+'\n')

        for i, line in enumerate(lines):
            # print(line.split('   '))
            # print(i)
            if i == 0:
                with open(out_name, 'a') as out:
                    out.write(out_name.split('/')[-1]+'\n')
            elif i == 1:
                pass
            elif i == 2:
                pass
            else:
                N, x, y, z = line.split()
                atom = atom_dict[int(N)]
                with open(out_name, 'a') as out:
                    out.write(f'{atom}   {x}   {y}   {z}\n')


def xyz_from_name(name, output, df3_path='Solvation_1/Tables/df3_3', files_folder='Solvation_1/Tables/Reserve/all_solutes'):
    df3 = feather.read_feather(project_path(df3_path))
    file_handle = df3[df3.SoluteName == name]['FileHandle'].tolist()[0]
    file_path = project_path(files_folder+'/'+file_handle+'.xyz')
    xyz_from_file(file_path, output)


#
# file_path = '/Users/balepka/Yandex.Disk.localized/Study/Lab/Neural Network/MNSolDatabase_v2012/all_solutes/0045eth.xyz'
# out_folder = '/Users/balepka/Downloads'
# #
# xyz_from_file(file_path, out_folder)

