# import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from rdkit import Chem
from Solvation_1.Vectorizers.vectorizers import get_smiles
from Solvation_1.my_nets.Create_dataset import SS_Dataset
from torch.utils.data import DataLoader
from Solvation_1.config import *
import shutil
import os
import errno
import matplotlib.pyplot as plt
import numpy as np


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    print(f_path)
    # if it is the best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to the best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    val_loss_min = checkpoint['losses']['val']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], val_loss_min.item()


def validate(model, loader):
    """TODO write all descriptions"""
    total = 0
    all_MSE = 0
    loss = nn.MSELoss(reduction='sum')
    with torch.no_grad():
        for vector, G_true in loader:
            vector, G_true = vector.to('cpu'), G_true.to('cpu')
            model.to('cpu')
            outputs = model(vector)
            total += G_true.size(0)
            all_MSE += loss(outputs.squeeze(), G_true.squeeze())

    return all_MSE / len(loader.dataset)


def train(model, train_loader, val_loader, solvent_test_loader, solute_test_loader,  loss_function, optimizer,
          epochs=10, device='cpu', ckp_path='Run_1', every_n=5, val_loss_min_input=1e+16):
    """TODO write all descriptions"""
    val_loss_min = val_loss_min_input
    run_folder = project_path('Solvation_1/Runs/' + ckp_path)
    try:
        os.makedirs(run_folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    with open(run_folder + '/run_log.tsv', 'w+') as f:
        f.write('\t'.join(str(x) for x in ('epoch', 'train', 'val', 'solvent', 'solute')) + '\n')

    for epoch in range(epochs):
        hist_loss = 0
        for vector, G_true in tqdm(train_loader):  # get batch
            vector, G_true = vector.to(device), G_true.to(device)
            model.to(device)
            outputs = model(vector)  # call forward inside
            # print(f'out: {outputs.shape}')
            # print(f'G: {G_true.shape}')

            loss = loss_function(outputs.squeeze(), G_true.squeeze())  # calculate loss
            loss.backward()  # calculate gradients
            optimizer.step()  # performs a single optimization step (parameter update).
            optimizer.zero_grad()  # sets the gradients of all optimized tensors to zero.

            hist_loss += loss.item()  # For stat only
        train_loss = hist_loss/len(train_loader.dataset)

        val_loss = validate(model, val_loader)
        solvent_test_loss = validate(model, solvent_test_loader)
        solute_test_loss = validate(model, solute_test_loader)

        with open(run_folder + '/run_log.tsv', 'a') as f:
            f.write('\t'.join(str(float(x)) for x in
                              (epoch, train_loss, val_loss, solvent_test_loss, solute_test_loss))+'\n')

        if not epoch % every_n:
            checkpoint = {
                'epoch': epoch,
                'losses': {'train': train_loss,
                           'val': val_loss,
                           'solvent': solvent_test_loss,
                           'solute': solute_test_loss},
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            checkpoint_path = project_path('Solvation_1/Runs/' + ckp_path + '/ep_' + str(epoch) + '.pt')
            best_model_path = project_path('Solvation_1/Runs/' + ckp_path + '/ep_' + str(epoch) + '_best.pt')
            save_ckp(checkpoint, False, checkpoint_path, best_model_path)

            if val_loss <= val_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(val_loss_min, val_loss))
                # save checkpoint as best model
                save_ckp(checkpoint, True, checkpoint_path, best_model_path)
                val_loss_min = val_loss

            print(f'epoch {epoch} -> {val_loss}')

    return val_loss_min


def beautiful_sample(model, solvent, solute):
    """TODO description"""
    solvent_smiles = get_smiles(solvent)
    solute_smiles = get_smiles(solute)
    solvent_mol = Chem.MolFromSmiles(solvent_smiles)
    solute_mol = Chem.MolFromSmiles(solute_smiles)
    print(f'solvent {solvent}')
    print(f'solute {solute}')
    display(Chem.Draw.MolsToGridImage((solvent_mol, solute_mol)))

    entire = pd.read_table(project_path('Solvation_1/Tables/Entire_table3.tsv'))
    entire = entire.set_index('Unnamed: 0')
    table_sample = entire[[solvent]].loc[[solute]]
    sample_ds = SS_Dataset(table_sample, 'solvent_macro_props1', 'solute_TESA')
    sample_loader = DataLoader(sample_ds)

    with torch.no_grad():
        model.eval()
        for vector, G_true in sample_loader:
            G_pred = model(vector)
            print(f'predicted {G_pred.squeeze()}, true {G_true.squeeze()}')


def plot_losses(file_path):
    """TODO description
        plots loss dynamics upon epochs:
        train, val, solvent test and solute test MSEs"""
    losses_dict = {}
    losses = []
    with open(file_path, 'r') as f:
        for line in f:
            losses.append(line.split('\t'))
    data = list(zip(*losses[1:]))
    # ar = np.array(list(int(float(x)) for x in data[0]))
    # print(ar)
    # print(list(ar[(ar % xtick_distance == 0)]))
    # plt.xticks(list(ar[(ar % xtick_distance == 0)]))
    plt.figure(figsize=(12, 8))
    for i, column in enumerate(losses[0]):
        column = column.strip()
        losses_dict[column] = list(float(x) for x in data[i])
        if i != 0:
            plt.subplot(220+i)
            plt.title(column)
            plt.plot(losses_dict['epoch'], losses_dict[column])
        else:
            losses_dict[column] = list(int(x) for x in losses_dict[column])
