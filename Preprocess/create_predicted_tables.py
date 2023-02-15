import os
from torch.utils.data import DataLoader
from my_nets.Create_dataset import SS_Dataset
from Vectorizers.vectorizers import get_dictionary, get_sample
from my_nets.LinearNet import LinearNet3
from my_nets.ResNET import ResNet1D
import numpy as np
import pickle as pkl
from config import project_path
from my_nets.net_func import single_sample_loader, nn, load_ckp
import pandas as pd
import torch
from tqdm import tqdm
from Vectorizers.vectorizers import split_folder


# predicted_tables = {}
with open(project_path('Tables/predicted_tables_Lin.pkl'), 'rb') as f:
    predicted_tables = pkl.load(f)

for folder in tqdm(sorted(os.listdir(project_path('Run_results/LinNet')))):
    if folder.startswith('.'):
        continue
    S_vect, U_vect, *_ = split_folder(folder)

    if (S_vect, U_vect) in predicted_tables:
        continue

    # create dataset
    lr = 1e-5
    solvent_vectorizer = S_vect
    solute_vectorizer = U_vect
    norm_bools = (True, True, True)
    # Res_Dict = {'base_filters': 2, 'kernel_size': 3, 'stride': 2, 'groups': 1, 'n_block': 3, 'n_classes': 1,
    #             'use_bn': True, 'use_do': True, 'verbose': False}
    epochs = 10000

    comments = f"""solute: {solvent_vectorizer}
                    solute: {solute_vectorizer}
                    norm: {norm_bools}
                    learning rate: {lr}
                    epochs: {epochs}
                """
    try:
        with open(project_path(f'Run_results/LinNet/{folder}/norm_params.pkl'), 'rb') as f:
            norm_params = pkl.load(f)
    except EOFError:
        print(f'norm fail: {folder}')
        continue

    entire_table = pd.read_table(project_path('Tables/Entire_table3.tsv'), index_col=0)

    table_v1 = pd.read_table(project_path('Tables/SS_table_v3.tsv'))
    dataset = SS_Dataset(table_v1, solvent_vectorizer, solute_vectorizer, normalize=norm_bools, norm_params=norm_params)
    len_data = dataset.__len__()
    val_data = len_data // 10

    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len_data - val_data, val_data])

    solvent_table = pd.read_table(project_path('Tables/solvent_test_table_v3.tsv'))
    solute_table = pd.read_table(project_path('Tables/solute_test_table_v3.tsv'))
    # solvent_test_dataset = SS_Dataset(solvent_table, solvent_vectorizer, solute_vectorizer,
    #                                   normalize=norm_bools, show_norm_params=False)
    # solute_test_dataset = SS_Dataset(solute_table, solvent_vectorizer, solute_vectorizer,
    #                                  normalize=norm_bools, show_norm_params=False)

    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # solvent_test_loader = DataLoader(solvent_test_dataset, batch_size=46, shuffle=False)
    # solute_test_loader = DataLoader(solute_test_dataset, batch_size=64, shuffle=False)

    # load model
    # in_feat = next(iter(dataset))[0].shape[0]  # get dimensions of input vector
    verbose = False

    in_feat = next(iter(dataset))[0].shape[-1]

    model = LinearNet3(in_features=in_feat)
    loss_function = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model, optimizer, *args = load_ckp(f'Runs/{folder}/best/best_val_model.pt', model, optimizer)

    # loop through all solvents and solutes
    G_std, G_mean = norm_params['G']

    predicted_table = entire_table.copy(deep=True)
    model.eval()

    for solvent in entire_table.columns:
        for solute in entire_table.index:
            loader = single_sample_loader(solvent, solute, solvent_vectorizer, solute_vectorizer, table=entire_table,
                                          norm_params=norm_params, nan_is_inf=True)
            for vector, G_true in loader:
                G_pred = model(vector)
                predicted_table[solvent][solute] = G_pred * G_std + G_mean

    predicted_tables[(solvent_vectorizer, solute_vectorizer)] = predicted_table

    with open(project_path('Tables/predicted_tables_Lin.pkl'), 'wb') as f:
        pkl.dump(predicted_tables, f)


true_Gs = pd.read_table(project_path('Tables/Entire_table3.tsv'), index_col=0)
predicted_tables['true'] = true_Gs
SMD_results = pd.read_table(project_path('SMD_inputs/SMD_results.tsv'), index_col=0)
predicted_tables['smd'] = SMD_results
mean_table = true_Gs.copy(deep=True)


with open(project_path('Run_results/LinNet/BAT_BAT_Lin1/norm_params.pkl'), 'rb') as f:
    norm_params = pkl.load(f)

_std, mean = norm_params['G']
print(mean)
for solvent in mean_table.columns:
    for solute in mean_table.index:
        mean_table[solvent][solute] = mean
predicted_tables['mean'] = mean_table

with open(project_path('Tables/predicted_tables_Lin.pkl'), 'wb') as f:
    pkl.dump(predicted_tables, f)