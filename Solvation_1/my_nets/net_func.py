import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from rdkit import Chem
from Solvation_1.Vectorizers.vectorizers import get_smiles
from Solvation_1.my_nets.Create_dataset import SS_Dataset
from torch.utils.data import DataLoader
from Solvation_1.config import *
import shutil

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
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
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
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()


def validate(model, val_loader):
    """TODO write all descriptions"""
    total = 0
    all_MSE = 0
    loss = nn.MSELoss(reduction='sum')
    with torch.no_grad():
        for vector, G_true in val_loader:
            vector, G_true = vector.to('cpu'), G_true.to('cpu')
            model.to('cpu')
            outputs = model(vector)
            total += G_true.size(0)
            all_MSE += loss(outputs.squeeze(), G_true.squeeze())

    return all_MSE / len(val_loader.dataset)


def train(model, train_loader, val_loader, loss_function, optimizer, epochs=10, device='cpu'):
    """TODO write all descriptions"""
    for epoch in range(epochs):
        hist_loss = 0
        for vector, G_true in tqdm(train_loader):  # get batch
            vector, G_true = vector.to(device), G_true.to(device)
            model.to(device)
            outputs = model(vector)  # call forward inside
            # print(f'out: {outputs.shape}')
            # print(f'G: {G_true.shape}')

            loss = loss_function(outputs.squeeze(), G_true)  # calculate loss
            loss.backward()  # calculate gradients
            optimizer.step()  # performs a single optimization step (parameter update).
            optimizer.zero_grad()  # sets the gradients of all optimized tensors to zero.

            hist_loss += loss.item()  # For stat only
        train_loss = hist_loss/len(train_loader.dataset)

        val_loss = validate(model, val_loader)


        checkpoint = {
            'epoch': epoch,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        print(f'epoch {epoch} -> {accuracy}')

    return val_loss


def beautiful_sample(model, solvent, solute):
    solvent_smiles = get_smiles(solvent)
    solute_smiles = get_smiles(solute)
    solvent_mol = Chem.MolFromSmiles(solvent_smiles)
    solute_mol = Chem.MolFromSmiles(solute_smiles)
    print(f'solvent {solvent}')
    print(f'solute {solute}')
    display(Chem.Draw.MolsToGridImage((solvent_mol, solute_mol)))

    entire = pd.read_table('Tables/Entire_table.tsv')
    entire = entire.set_index('Unnamed: 0')
    table_sample = entire[[solvent]].loc[[solute]]
    sample_ds = SS_Dataset(table_sample, 'solvent_macro_props1', 'solute_TESA')
    sample_loader = DataLoader(sample_ds)

    with torch.no_grad():
        model.eval()
        for vector, G_true in sample_loader:
            G_pred = model(vector)
            print(f'predicted {G_pred.squeeze()}, true {G_true.squeeze()}')

