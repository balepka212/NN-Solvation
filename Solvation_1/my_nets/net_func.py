import torch
import torch.nn as nn
from rdkit import Chem
from Solvation_1.Vectorizers.vectorizers import get_smiles
from Solvation_1.my_nets.Create_dataset import SS_Dataset
from torch.utils.data import DataLoader
from Solvation_1.config import *
import shutil
import os
import errno
import matplotlib.pyplot as plt


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    Saves a checkpoint of training model.

    If it has minimal val loss saves it as best checkpoint.

    Parameters
    ----------
    state:
        checkpoint we want to save
    is_best: bool
        is this the best checkpoint; min validation loss
    checkpoint_path: str
        path to save checkpoint
    best_model_path: str
        path to save best model
    """

    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # print(f_path)
    # if it is the best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to the best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    Loads a checkpoint of trained model.

    Parameters
    ----------
    checkpoint_fpath: str
        path to save checkpoint
    model: NN model
        A model that we want to load checkpoint parameters into
    optimizer: optimizer
        An optimizer we defined in previous training
    """

    # load checkpoint
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
    """
    Validates model on given dataloader.

    Parameters
    ----------
    model: NN model
        A model that we want to validate
    loader: dataloader
        A dataloader on which to make validation
    """

    total = 0
    all_MSE = 0
    # using reduction 'sum' to calculate mean MSE
    loss = nn.MSELoss(reduction='sum')
    with torch.no_grad():
        for vector, G_true in loader:
            # make sure all vectors and model on same device
            vector, G_true = vector.to('cpu'), G_true.to('cpu')
            model.to('cpu')
            outputs = model(vector)
            total += G_true.size(0)
            # squeeze vectors to prevent dummy dimension problems
            all_MSE += loss(outputs.squeeze(), G_true.squeeze())

    return all_MSE/len(loader.dataset)


def train(model, train_loader, val_loader, solvent_test_loader, solute_test_loader,  loss_function, optimizer,
          epochs=10, device='cpu', ckp_path='Run_1', every_n=5, val_loss_min_input=1e+16):
    """
    Trains an NN model.

    Parameters
    ----------
    model: NN model
        A model to be trained
    train_loader: dataloader
        train dataloader on which model trains
    val_loader: dataloader
        val dataloader on which model validates its MSE score
    solvent_test_loader: dataloader
        test dataloader on which model validates its MSE score towards rare solutes
    solute_test_loader: dataloader
        test dataloader on which model validates its MSE score towards rare solvents
    loss_function: function
        a function to be used ass loss
    optimizer: optimizer
        An optimizer we are going to train with
    epochs: int
        number of epochs to train the model
    device: str
        A string representing the device to be trained on
    ckp_path: str
        A path in project to save checkpoints, losses data and norm_parameters
    every_n: int
        How often to save a model checkpoint
    val_loss_min_input: float
        current minimal val loss value
    """

    # make sure val_loss_min is set to checkpoint value if model is loaded from checkpoint
    val_loss_min = val_loss_min_input
    run_folder = project_path('Solvation_1/Runs/' + ckp_path)
    # create a run folder and best subfolder if they do not exist
    for folder in (run_folder, run_folder+'/best'):
        try:
            os.makedirs(folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # Header row for losses log file
    with open(run_folder + '/run_log.tsv', 'w+') as f:
        f.write('\t'.join(str(x) for x in ('epoch', 'train', 'val', 'solvent', 'solute')) + '\n')

    # loop through epochs
    for epoch in range(epochs):
        # create hist_loss to calculate overall MSE
        hist_loss = 0
        for vector, G_true in train_loader:  # get batch
            # make sure all vectors and model on same device
            vector, G_true = vector.to(device), G_true.to(device)
            model.to(device)
            outputs = model(vector)  # call forward inside
            # print(f'out: {outputs.shape}')
            # print(f'G: {G_true.shape}')

            # squeeze vectors to prevent dummy dimension problems
            loss = loss_function(outputs.squeeze(), G_true.squeeze())  # calculate loss
            loss.backward()  # calculate gradients
            optimizer.step()  # performs a single optimization step (parameter update).
            optimizer.zero_grad()  # sets the gradients of all optimized tensors to zero.

            hist_loss += loss.item()  # For stat only
        train_loss = hist_loss/len(train_loader.dataset)  # calculated train loss

        # calculate val and test losses
        val_loss = validate(model, val_loader)
        solvent_test_loss = validate(model, solvent_test_loader)
        solute_test_loss = validate(model, solute_test_loader)

        # write losses to log file
        with open(run_folder + '/run_log.tsv', 'a') as f:
            f.write('\t'.join(str(float(x)) for x in
                              (epoch, train_loss, val_loss, solvent_test_loss, solute_test_loss))+'\n')

        # check if epoch is needed to be saved. Parameter every_n is needed/
        if not epoch % every_n:
            # create checkpoint
            checkpoint = {
                'epoch': epoch,
                'losses': {'train': train_loss,
                           'val': val_loss,
                           'solvent': solvent_test_loss,
                           'solute': solute_test_loss},
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            # save checkpoint
            checkpoint_path = project_path('Solvation_1/Runs/' + ckp_path + '/ep_' + str(epoch) + '.pt')
            best_model_path = project_path('Solvation_1/Runs/' + ckp_path + '/best/ep_' + str(epoch) + '_best.pt')
            save_ckp(checkpoint, False, checkpoint_path, best_model_path)

            if val_loss <= val_loss_min:
                print(f'epoch {epoch}: val loss ({val_loss_min} -> {val_loss}). Saving model')
                # save checkpoint as best model
                save_ckp(checkpoint, True, checkpoint_path, best_model_path)
                val_loss_min = val_loss
            else:
                print(f'epoch {epoch}: val loss ({val_loss_min} -> {val_loss})')
            # print(f'epoch {epoch}: val loss {val_loss}')
    shutil.copyfile(run_folder + '/run_log.tsv', project_path('Solvation_1/Run_results/'+ckp_path+'/run_log.tsv'))

    return val_loss_min


def beautiful_sample(model, solvent, solute, normalize=(True, True, True)):
    """
    Evaluates true and predicted value of delta_G for given solvent and solute using given model.
    Shows solvent and solute structures.

    Parameters
    ----------
    model: NN model
        A model to use for prediction
    solvent: str
        name of solvent
    solute: str
        name of solute
    normalize: (bool, bool, bool)
        A tuple of three bools showing if normalization is required for solvent, solute and G_solv respectively
    """

    solvent_smiles = get_smiles(solvent)     # get solvent SMILES
    solute_smiles = get_smiles(solute)  # get solute SMILES
    solvent_mol = Chem.MolFromSmiles(solvent_smiles)    # create solvent molecule instance
    solute_mol = Chem.MolFromSmiles(solute_smiles)  # create solute molecule instance
    print(f'solvent {solvent}')
    print(f'solute {solute}')
    display(Chem.Draw.MolsToGridImage((solvent_mol, solute_mol)))   # display solvent and solute structures

    entire = pd.read_table(project_path('Solvation_1/Tables/Entire_table3.tsv'))    # load table with all species
    # Set a column with Solutes as index column
    if entire.index.name != 'Unnamed: 0':
        entire = entire.set_index('Unnamed: 0')
    table_sample = entire[[solvent]].loc[[solute]]    # leave only cell with needed solvent and solute
    sample_ds = SS_Dataset(table_sample, 'solvent_macro_props1', 'solute_TESA',
                           normalize=normalize, show_norm_params=False)     # create sample dataset
    sample_loader = DataLoader(sample_ds)   # create sample dataloader

    # predict delta_G value
    with torch.no_grad():
        model.eval()
        for vector, G_true in sample_loader:
            std, mean = sample_ds.norm_params['G']      # load std and mean parameters
            G_pred = model(vector)
            print(f'predicted {(G_pred*std+mean).squeeze()}, true {(G_true*std+mean).squeeze()}')


def plot_losses(file_path):
    """
    Plots losses from losses log file.

    Parameters
    ----------
    file_path: str
        path to losses log file
    """

    losses_dict = {}
    losses = []
    # collect losses from log file
    with open(file_path, 'r') as f:
        for line in f:
            losses.append(line.split('\t'))
    data = list(zip(*losses[1:]))   # make data suitable for pyplot
    # ar = np.array(list(int(float(x)) for x in data[0]))
    # print(ar)
    # print(list(ar[(ar % xtick_distance == 0)]))
    # plt.xticks(list(ar[(ar % xtick_distance == 0)]))

    plt.figure(figsize=(12, 8))  # change figure size
    for i, column in enumerate(losses[0]):
        column = column.strip()
        losses_dict[column] = list(float(x) for x in data[i])
        if i != 0:
            # if not epoch column
            plt.subplot(220+i)  # define subplot integer from enumerate index i
            plt.title(column)   # graph titles
            plt.plot(losses_dict['epoch'], losses_dict[column])
        else:
            # if epoch column - make it integer
            losses_dict[column] = list(int(x) for x in losses_dict[column])
    plt.savefig(file_path.rsplit('/', maxsplit=1)[0] + '/losses_plot.png')
    plt.show()
