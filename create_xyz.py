def make_xyz(file, output):
    atom_dict = {1:'H', 6:'C', 7:'N', 8:'O', 15:'P', 16:'S', 17:'Cl'}
    with open(file, 'r') as f:
        lines = f.readlines()
        n_lines = len(lines)
        print(n_lines)
        out_name = output+'/copy_'+file.split('/')[-1]
        with open(out_name, 'w+') as out:
            out.write(str(n_lines-3)+'\n')

        for i,line in enumerate(lines):
            print(line.split('   '))
            print(i)
            if i == 0:
                with open(out_name, 'a') as out:
                    out.write(out_name.split('/')[-1]+'\n')
            elif i == 1:
                pass
            elif i == 2:
                pass
            else:
                N, x, y, z = line.split('   ')
                atom = atom_dict[int(N)]
                with open(out_name, 'a') as out:
                    out.write(f'{atom}   {x}   {y}   {z}')


file = '/Users/balepka/Yandex.Disk.localized/Study/Lab/Neural Network/MNSolDatabase_v2012/all_solutes/i133.xyz'
output = '/Users/balepka/Downloads'

make_xyz(file, output)