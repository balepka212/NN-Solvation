import pandas as pd
from config import project_path
import sys, string, os, glob
import periodictable as pt
from math import *
from pathlib import Path
import pandas as pd
from config import project_path
from Vectorizers.vectorizers import get_handle_file
# with open(project_path('Tables/Entire_table3.tsv')) as f:
#     true_table = pd.read_table(f)
#     true_table = true_table.set_index('Unnamed: 0')
#
# SMD_table = true_table.copy()
# for S in true_table:
#     for U in true_table.index:
#         if true_table[S][U]:
#             print(f'{true_table[S][U]}: {S=}  {U=}')
#         else:
#             print(f'NaN {S=}  {U=}')

# with open('/Users/balepka/Yandex.Disk-isaevvv@my.msu.ru.localized/My Articles/NN_Solvation/SMD_MAIN/GAS/m1_00_0151phy-GAS.out') as f:
#     for line in f.readlines():
#         if line.startswith('FINAL SINGLE POINT ENERGY'):
#             print(line.split()[-1])
#         elif line.startswith('Magnitude (Debye)'):
#             print(line.split()[-1])

the_table = pd.read_table(project_path('Tables/Entire_table3.tsv'))

the_table = the_table.rename(columns={'Unnamed: 0': 'Solute'})
the_table = the_table.set_index('Solute')
SMD_solv_table = the_table.copy()
SMD_results_table = the_table.copy()

solute_table = pd.read_table(project_path('Tables/MNSol/solute_test_table_v1.tsv'))
solvent_table = pd.read_table(project_path('Tables/MNSol/solvent_test_table_v1.tsv'))

folder = 'SMD_inputs/SMD_all'

for solvent in the_table.columns:
    for solute in the_table.index:
        data = the_table[solvent][solute]
        if pd.notna(data):
            xyz = get_handle_file(solute)
            Path(project_path(os.path.join(folder, solvent))).mkdir(parents=True, exist_ok=True)
            f_name = os.path.basename(xyz)[:-4]
            filepath = os.path.join(folder, solvent, 'm1_00_' + f_name + '_' + solvent + '.out')
            gas_filepath = os.path.join(folder, 'GAS','m1_00_' + f_name + '-' + 'GAS' + '.out')
            with open(project_path(gas_filepath)) as f1:
                for line in f1.readlines():
                    if line.startswith('FINAL SINGLE POINT ENERGY'):
                        Gas_energy = float(line.split()[-1])
            try:
                with open(project_path(filepath)) as f:
                    for line in f.readlines():
                        if line.startswith('FINAL SINGLE POINT ENERGY'):
                            Solv_energy = float(line.split()[-1])
                            SMD_solv_table[solvent][solute] = Solv_energy
                            SMD_results_table[solvent][solute] = \
                                (Solv_energy-Gas_energy)*627.51  # in kcal/mol -> normilized std mean

            except FileNotFoundError:
                print(f_name, solvent)


SMD_results_table.to_csv(project_path('SMD_inputs/SMD_results.tsv'), sep="\t")
SMD_solv_table.to_csv(project_path('SMD_inputs/SMD_solv.tsv'), sep="\t")



