'''#!/usr/bin/python'''
import sys, string, os, glob
import periodictable as pt
from math import *
from pathlib import Path
import pandas as pd
from config import project_path
from Vectorizers.vectorizers import get_handle_file
import pickle as pkl

# The input files for geometry/sp will be created by using that script
# sys.argv[1] = xyz files

# Main program starts here
def create_inp(filename, solvent=None, folder='/'):
    with open(project_path(filename)) as f:
        DFT = 'B3LYP'
        # coordinates

        coord = f.readlines()  # read every line

        d = coord[0].split()
        if len(d) == 1:
            MOL = True
        else:
            MOL = False

        if MOL:
            coord = coord[2:]

        coordinates = []
        odd = False

        def odd_elec(element):
            el = pt.elements.symbol(element)
            if el.number % 2:
                return True
            else:
                return False

        for i in coord:
            i = i.split()
            if len(i) > 3:
                coordinates.append(i)
                if odd_elec(i[0]):
                    odd = not odd

    #################################
    # system requeriments
    #################################
    cores = 16
    mem = 2500
    ######### Multiplecity

    # mult = sys.argv[1][1:2]
    # charge = int(sys.argv[1][3:5])
    if odd:
        mult = '2'
        print(filename)
    else:
        mult = '1'

    charge = '00'

    # all molecule
    f_name = os.path.basename(filename)[:-4]
    if solvent is None:
        Path(project_path(os.path.join(folder, 'GAS'))).mkdir(parents=True, exist_ok=True)
        filepath = os.path.join(folder, 'GAS', 'm' + str(mult) + '_00_' + f_name + '-' + 'GAS' + '.inp')
        with open(project_path(filepath), 'w') as fil:

            fil.write('%s\n' % ('# put your comments here'))
            fil.write('%s\n' % ('%pal nprocs ' + str(cores) + ' end'))
            fil.write('''! ''' + DFT + ''' def2-svp D3BJ SP DEFGRID3  
            %scf MaxIter 800 end
            ''')
            fil.write('%s\n' % ('%MaxCore ' + str(mem)))
            fil.write('* xyz ' + str(charge) + ' ' + str(mult) + '\n')
            for i in range(len(coordinates)):
                fil.write(
                    '%2s %16s %16s %16s\n' % (coordinates[i][0], coordinates[i][1], coordinates[i][2], coordinates[i][3]))
            fil.write('*\n')

            fil.write('\n' % ())
            fil.write('\n' % ())
            fil.write('\n' % ())
            fil.close()

    # all molecule
    else:
        solv = solvent
        with open(project_path('Tables/orca_matching_solvent2.pkl'), 'rb') as orca_solv:
            solvent_orca_dict = pkl.load(orca_solv)
        Path(project_path(os.path.join(folder, solv))).mkdir(parents=True, exist_ok=True)
        filepath = os.path.join(folder, solv, 'm' + str(mult) + '_00_' + f_name + '_' + solv + '.inp')
        with open(project_path(filepath), 'w') as fil:

            fil.write('%s\n' % ('# put your comments here'))
            fil.write('%s\n' % ('%pal nprocs ' + str(cores) + ' end'))
            fil.write('''! ''' + DFT + ''' def2-svp D3BJ SP DEFGRID3  
            %scf MaxIter 800 end
            ''')
            fil.write('''%cpcm
            smd true
            SMDsolvent "''' + solvent_orca_dict[solv] + '''"
            end\n''')
            fil.write('%s\n' % ('%MaxCore ' + str(mem)))
            fil.write('* xyz ' + str(charge) + ' ' + str(mult) + '\n')
            for i in range(len(coordinates)):
                fil.write(
                    '%2s %16s %16s %16s\n' % (coordinates[i][0], coordinates[i][1], coordinates[i][2], coordinates[i][3]))
            fil.write('*\n')

            fil.write('\n' % ())
            fil.write('\n' % ())
            fil.write('\n' % ())
            fil.close()


# #
# with open(project_path('Tables/Solvents_Solutes.pkl'), 'rb') as f:
#     Solvents, Solutes = pkl.load(f)
#
# for compound in Solutes:
#     mol_path = get_handle_file(compound)
#     print(f'{compound}:  {mol_path}')
#     create_inp(mol_path, folder='Tables/SMD/')
#
# the_table = pd.read_table(project_path('Tables/SS_table_v3.tsv'))
#
# the_table = the_table.rename(columns={'Unnamed: 0': 'Solute'})
# the_table = the_table.set_index('Solute')
# for solvent in the_table.columns:
#     for solute in the_table.index:
#         data = the_table[solvent][solute]
#         if pd.notna(data):
#             xyz = get_handle_file(solute)
#             create_inp(xyz, solvent=solvent, folder='Tables/SMD/')