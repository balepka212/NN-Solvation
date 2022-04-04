from torch.utils.data import DataLoader
# from Solvation_1.my_nets.Create_dataset import *
# from Solvation_1.Vectorizers.vectorizers import *
from Solvation_1.my_nets.LinearNet import LinearNet3
from Solvation_1.my_nets.ResNET import ResNet1D, MyConv1dPadSame, MyMaxPool1dPadSame, BasicBlock
from Solvation_1.my_nets.net_func import *
import pickle as pkl


def Experiment(runs_folder='Example_Lin1',
               net='Lin',
               lr=1e-5,
               solvent_vectorizer='solvent_macro_props1',
               solute_vectorizer='solute_TESA',
               norm_bools=(True, True, True),
               net_dict=None,
               epochs=20
               ):
    """Main function to perform experiment.

     Parameters
    ----------
    runs_folder: str
        Name of folder all the files will be stored
    net: 'Lin' or 'Res'
        Which network to use: LinearNet3 or 1dResNET
    lr: float
        Learning rate value
    solvent_vectorizer: str
        Name of vectorizer for solvent
    solute_vectorizer: str
        Name of vectorizer for solute
    norm_bools: (bool, bool, bool)
        Whether apply normalization to solvent vector, solute vector and G_true vector respectively
    Net_Dict: dict
        Kwargs for Network. Especially ResNET
    epochs: int
        number of epochs to train
    """

    comments = f"""solute: {solvent_vectorizer}
     solute: {solute_vectorizer}
     norm: {norm_bools}
     learning rate: {lr}
     Net_Dict: {net_dict}
     epochs: {epochs}"""

    # Create dataset
    table_v1 = pd.read_table(project_path('Solvation_1/Tables/SS_table_v3.tsv'))
    dataset = SS_Dataset(table_v1, solvent_vectorizer, solute_vectorizer, normalize=norm_bools)
    len_data = dataset.__len__()
    val_data = len_data // 10
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len_data - val_data, val_data])

    solvent_table = pd.read_table(project_path('Solvation_1/Tables/solvent_test_table_v3.tsv'))
    solute_table = pd.read_table(project_path('Solvation_1/Tables/solute_test_table_v3.tsv'))
    solvent_test_dataset = SS_Dataset(solvent_table, solvent_vectorizer, solute_vectorizer,
                                      normalize=norm_bools, show_norm_params=False)
    solute_test_dataset = SS_Dataset(solute_table, solvent_vectorizer, solute_vectorizer,
                                     normalize=norm_bools, show_norm_params=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    solvent_test_loader = DataLoader(solvent_test_dataset, batch_size=46, shuffle=False)
    solute_test_loader = DataLoader(solute_test_dataset, batch_size=64, shuffle=False)

    print(f'train length: {len(train_loader.dataset)}')
    print(f'val length: {len(val_loader.dataset)}')
    print(f'solute test length: {len(solute_test_loader.dataset)}')
    print(f'solvent test length: {len(solvent_test_loader.dataset)}\n')

    for folder in ('Solvation_1/Runs/', 'Solvation_1/Run_results/'):
        try:
            os.makedirs(project_path(folder + runs_folder))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        with open(project_path(folder + runs_folder + '/comments.txt'), 'w') as f:
            f.write(comments)
        with open(project_path(folder + runs_folder + '/norm_params.pkl'), 'wb+') as f:
            pkl.dump(dataset.norm_params, f)

    # Train Network
    x, y = next(iter(dataset))
    print(f'Shape of input: {x.shape}\n')

    if net == 'Res':
        in_feat = next(iter(dataset))[0].shape[0]  # get input tensor size
        model = ResNet1D(in_feat, **net_dict)
    elif net == 'Lin':
        in_feat = next(iter(dataset))[0].shape[-1]  # get size of input vector to create a Linear Network
        model = LinearNet3(in_features=in_feat)

    loss_function = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    print('Training')
    MSE = train(model, train_loader, val_loader, solvent_test_loader, solute_test_loader, loss_function,
                optimizer, epochs=epochs, ckp_path=runs_folder)
    print('Finished training!')

    plot_losses(project_path('Solvation_1/Run_results/' + runs_folder + '/run_log.tsv'))
    return MSE
