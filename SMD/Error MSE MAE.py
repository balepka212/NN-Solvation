import pandas as pd
from config import project_path


the_table = pd.read_table(project_path('Tables/Entire_table3.tsv'))
train_table = pd.read_table(project_path('Tables/SS_table_v3.tsv'))
solute_table = pd.read_table(project_path('Tables/solute_test_table_v3.tsv'))
solvent_table = pd.read_table(project_path('Tables/solvent_test_table_v3.tsv'))
SMD_table = pd.read_table(project_path('SMD_inputs/SMD_results.tsv'))

for t in (the_table, train_table, solute_table, solvent_table, SMD_table):
    t.rename(columns={'Unnamed: 0': 'Solute'}, inplace=True)
    t.set_index('Solute', inplace=True)
    print(f'{sum(t.count())}')

MSE={'train':0, 'solute':0, 'solvent':0 }
MAE={'train':0, 'solute':0, 'solvent':0, }
count ={'train':0, 'solute':0, 'solvent':0, }
for solvent in the_table.columns:
    for solute in the_table.index:
        data = the_table[solvent][solute]
        if pd.notna(data):
            dataset = None
            if pd.notna(train_table[solvent][solute]):
                dataset = 'train'
            elif pd.notna(solute_table[solvent][solute]):
                dataset = 'solute'
            elif pd.notna(solvent_table[solvent][solute]):
                dataset = 'solvent'

            SMD_G = SMD_table[solvent][solute]
            abs_value = abs(data-SMD_G)
            sq_value = abs_value**2
            MAE[dataset] += abs_value
            MSE[dataset] += sq_value
            count[dataset] +=1

for dataset in ('train', 'solute', 'solvent'):
    MSE[dataset] /= count[dataset]
    MAE[dataset] /= count[dataset]

