import numpy as np
import pickle as pkl
from config import project_path
import pandas as pd
from tqdm import tqdm

S_vects = ('blank', 'class', 'macro', 'macrox', 'morgan', 'mor2to20', 'jb', 'bob', 'bat', 'soap')
U_vects = ('blank', 'class', 'tesa', 'comp', 'morgan', 'mor2to20', 'jb', 'bob', 'bat', 'soap')
template_df = pd.DataFrame(columns=S_vects, index=U_vects)

pred_tables = {}
with open(project_path(f'Tables/predicted_tables_KRR1.pkl'), 'rb') as f:
    pred_tables = pkl.load(f)

dataset_tables = {'main': pd.read_table(project_path('Tables/SS_table_v3.tsv'), index_col=0),
                  'solvent': pd.read_table(project_path('Tables/solvent_test_table_v3.tsv'), index_col=0),
                  'solute': pd.read_table(project_path('Tables/solute_test_table_v3.tsv'), index_col=0)}

scores = {}
for S_vect in S_vects:
    for U_vect in U_vects:
        if (S_vect, U_vect) not in pred_tables:
            continue
        print(S_vect, U_vect)
        pred_table = pred_tables[(S_vect, U_vect)]
        scores[(S_vect, U_vect)] = {}
        for name, table in dataset_tables.items():
            mae = 0
            mse = 0
            number = 0
            G_preds = []
            scores[(S_vect, U_vect)][name] = {}
            for solvent in table.columns:
                for solute in table.index:
                    G_true = table[solvent][solute]
                    if not np.isnan(G_true):
                        G_pred = pred_table[solvent][solute]
                        dG = G_true-G_pred
                        mae += abs(dG)
                        mse += dG*dG
                        number += 1
                        G_preds.append(G_pred)
            scores[(S_vect, U_vect)][name]['number'] = number
            scores[(S_vect, U_vect)][name]['mae'] = mae/number
            scores[(S_vect, U_vect)][name]['mse'] = mse/number
            scores[(S_vect, U_vect)][name]['rms'] = (mse/number)**0.5
            scores[(S_vect, U_vect)][name]['pred'] = G_preds

        # with open(project_path('Tables/Scores_Res.pkl'), 'wb') as f:
        #     pkl.dump(scores, f)


# true mean smd test

for test in ('true', 'mean', 'smd'):
    print(f'{test} test')
    pred_table = pred_tables[test]
    scores[test] = {}
    for name, table in dataset_tables.items():
        mae = 0
        mse = 0
        number = 0
        G_preds = []
        scores[test][name] = {}
        for solvent in table.columns:
            for solute in table.index:
                G_true = table[solvent][solute]
                if not np.isnan(G_true):
                    G_pred = pred_table[solvent][solute]
                    dG = G_true-G_pred
                    mae += abs(dG)
                    mse += dG*dG
                    number += 1
                    G_preds.append(G_pred)
        scores[test][name]['number'] = number
        scores[test][name]['mae'] = mae/number
        scores[test][name]['mse'] = mse/number
        scores[test][name]['rms'] = (mse/number)**0.5
        scores[test][name]['pred'] = G_preds

# Pearson corr
for model in scores:
    for subset in scores[model]:
        preds = scores[model][subset]['pred']
        trues = scores['true'][subset]['pred']
        if len(preds) != len(trues):
            print(f'LEN!!!   {model}')
        rho = np.corrcoef(preds, trues)
        scores[model][subset]['rho'] = rho[0][1]

with open(project_path('Tables/Scores_KRR.pkl'), 'wb') as f:
    pkl.dump(scores, f)
